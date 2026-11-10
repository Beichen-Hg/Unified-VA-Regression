"""Training recipe utilities: LLRD, EMA, partial freeze."""
import torch
import copy
from typing import Optional
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_llrd_param_groups(encoder, head_modules: list,
                            lr: float, weight_decay: float,
                            llrd_factor: float = 0.95) -> list:
    """Build parameter groups with layer-wise learning rate decay for XLM-R.

    Encoder layers get decaying LR: layer_i_lr = lr * factor^(n_layers - i).
    Head modules get the base lr.
    Embeddings get the smallest LR.

    Args:
        encoder: The encoder wrapper (has .model attribute for XLM-R)
        head_modules: List of head nn.Modules (va_head, proto_head, etc.)
        lr: Base learning rate
        weight_decay: Weight decay
        llrd_factor: Decay factor per layer (e.g. 0.95)

    Returns:
        List of param group dicts for optimizer
    """
    param_groups = []
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    # Get the underlying transformer model
    model = encoder.backbone if hasattr(encoder, 'backbone') else encoder

    # Find encoder layers
    layers = None
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        layers = model.encoder.layer
    elif hasattr(model, 'layers'):
        layers = model.layers

    if layers is None:
        logger.warning("LLRD: cannot find encoder layers, falling back to flat LR")
        all_params = list(encoder.parameters())
        for m in head_modules:
            if m is not None:
                all_params += list(m.parameters())
        return [{"params": all_params, "lr": lr, "weight_decay": weight_decay}]

    n_layers = len(layers)

    # Embeddings: lowest LR
    emb_lr = lr * (llrd_factor ** (n_layers + 1))
    emb_params_decay = []
    emb_params_nodecay = []
    for name, param in model.named_parameters():
        if "embeddings" in name:
            if any(nd in name for nd in no_decay):
                emb_params_nodecay.append(param)
            else:
                emb_params_decay.append(param)

    if emb_params_decay:
        param_groups.append({"params": emb_params_decay, "lr": emb_lr,
                             "weight_decay": weight_decay})
    if emb_params_nodecay:
        param_groups.append({"params": emb_params_nodecay, "lr": emb_lr,
                             "weight_decay": 0.0})

    # Encoder layers: decaying LR
    for i, layer in enumerate(layers):
        layer_lr = lr * (llrd_factor ** (n_layers - i))
        decay_params = []
        nodecay_params = []
        for name, param in layer.named_parameters():
            if any(nd in name for nd in no_decay):
                nodecay_params.append(param)
            else:
                decay_params.append(param)
        if decay_params:
            param_groups.append({"params": decay_params, "lr": layer_lr,
                                 "weight_decay": weight_decay})
        if nodecay_params:
            param_groups.append({"params": nodecay_params, "lr": layer_lr,
                                 "weight_decay": 0.0})

    # Pooler + any remaining encoder params (base LR)
    assigned = set()
    for pg in param_groups:
        for p in pg["params"]:
            assigned.add(id(p))
    remaining = [p for p in encoder.parameters() if id(p) not in assigned]
    if remaining:
        param_groups.append({"params": remaining, "lr": lr,
                             "weight_decay": weight_decay})

    # Head modules: base LR
    for m in head_modules:
        if m is not None:
            head_decay = []
            head_nodecay = []
            for name, param in m.named_parameters():
                if any(nd in name for nd in no_decay):
                    head_nodecay.append(param)
                else:
                    head_decay.append(param)
            if head_decay:
                param_groups.append({"params": head_decay, "lr": lr,
                                     "weight_decay": weight_decay})
            if head_nodecay:
                param_groups.append({"params": head_nodecay, "lr": lr,
                                     "weight_decay": 0.0})

    n_params = sum(len(pg["params"]) for pg in param_groups)
    logger.info(f"LLRD: {n_layers} layers, factor={llrd_factor}, "
                f"emb_lr={emb_lr:.2e}, top_layer_lr={lr * llrd_factor:.2e}, "
                f"head_lr={lr:.2e}, {n_params} param tensors in {len(param_groups)} groups")
    return param_groups


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of parameters. Call update() after each optimizer
    step. Use apply_shadow() before eval, restore() after eval.
    """

    def __init__(self, modules: list, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for m in modules:
            if m is None:
                continue
            for name, param in m.named_parameters():
                if param.requires_grad:
                    key = f"{id(m)}_{name}"
                    self.shadow[key] = param.data.clone()
        self._modules = [m for m in modules if m is not None]
        logger.info(f"EMA: tracking {len(self.shadow)} params, decay={decay}")

    def update(self):
        """Update shadow params with current params."""
        for m in self._modules:
            for name, param in m.named_parameters():
                if param.requires_grad:
                    key = f"{id(m)}_{name}"
                    self.shadow[key].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """Replace model params with shadow (for eval)."""
        for m in self._modules:
            for name, param in m.named_parameters():
                if param.requires_grad:
                    key = f"{id(m)}_{name}"
                    self.backup[key] = param.data.clone()
                    param.data.copy_(self.shadow[key])

    def restore(self):
        """Restore original params (after eval)."""
        for m in self._modules:
            for name, param in m.named_parameters():
                if param.requires_grad:
                    key = f"{id(m)}_{name}"
                    if key in self.backup:
                        param.data.copy_(self.backup[key])
        self.backup.clear()


def freeze_encoder_layers(encoder, freeze_until: int = 0):
    """Freeze encoder layers 0..freeze_until-1. Unfreeze the rest.

    Args:
        encoder: Encoder wrapper with .model attribute
        freeze_until: Freeze layers [0, freeze_until). 0 = no freeze.
    """
    if freeze_until <= 0:
        return

    model = encoder.backbone if hasattr(encoder, 'backbone') else encoder

    # Freeze embeddings
    for name, param in model.named_parameters():
        if "embeddings" in name:
            param.requires_grad = False

    # Freeze layers
    layers = None
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        layers = model.encoder.layer
    elif hasattr(model, 'layers'):
        layers = model.layers

    if layers is None:
        logger.warning("freeze_encoder_layers: cannot find layers")
        return

    n_frozen = 0
    for i, layer in enumerate(layers):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False
            n_frozen += 1

    total = len(layers)
    logger.info(f"Partial freeze: {n_frozen}/{total} layers frozen, "
                f"{total - n_frozen} layers trainable")
