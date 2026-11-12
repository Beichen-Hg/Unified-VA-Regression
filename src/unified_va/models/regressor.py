"""A single backbone/readout implementation for all three training stages."""

import weakref
import torch
from torch import nn
from .conditional_adapter import ConditionalAdapter
from .heads import LinearHead, DualHeadMLP, ConditionClassifier


class VARegressor(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone, self.cfg = backbone, cfg
        hidden = backbone.config.hidden_size
        self.pooling = cfg["small_llm"]["pooling"]
        if self.pooling not in ("last_token", "gated_last_mean") or cfg[
            "head_type"
        ] not in ("linear", "dual_mlp"):
            raise ValueError("Unsupported regression pooling or readout")
        head = cfg.get("dual_head_mlp", {})
        self.head = (
            LinearHead(hidden)
            if cfg["head_type"] == "linear"
            else DualHeadMLP(
                hidden,
                head.get("shared_hidden", 512),
                head.get("branch_hidden", 256),
                head.get("dropout", 0.1),
            )
        )
        self.feature_dropout = nn.Dropout(cfg["small_llm"]["feature_dropout"])
        self.pool_gate = (
            nn.Linear(2 * hidden, 1) if self.pooling == "gated_last_mean" else None
        )
        self.backbone_conditional_adapters = []
        self.auxiliary = None
        self._attention_mask = None
        if cfg["experiment"]["stage"] == "stage2":
            settings = cfg["backbone_conditional_adapter"]
            if len(backbone.layers) < settings["end_layer"]:
                raise ValueError(
                    "The backbone must provide decoder layers 24 through 35"
                )
            owner = weakref.ref(self)
            for index in range(settings["start_layer"], settings["end_layer"]):
                adapter = ConditionalAdapter(
                    hidden,
                    settings["bottleneck_dim"],
                    settings["embed_dim"],
                    settings["dropout"],
                    settings["soft_condition_hidden"],
                )
                layer = backbone.layers[index]
                layer.add_module("adapter", adapter)

                def refine(module, inputs, output, adapter=adapter):
                    mask = owner()._attention_mask
                    if isinstance(output, tuple):
                        return (adapter(output[0], mask), *output[1:])
                    return adapter(output, mask)

                layer.register_forward_hook(refine)
                self.backbone_conditional_adapters.append(adapter)
            self.auxiliary = ConditionClassifier(
                settings["langs"], settings["domains"], settings["embed_dim"]
            )
        self.freeze_lower_layers(cfg["freeze_until"])

    def freeze_lower_layers(self, boundary=24):
        self.backbone.requires_grad_(False)
        for layer in self.backbone.layers[boundary:]:
            layer.requires_grad_(True)
        if hasattr(self.backbone, "norm"):
            self.backbone.norm.requires_grad_(True)

    def forward(self, input_ids, attention_mask):
        # Retain this mask through non-reentrant recomputation during backward.
        # Training performs backward before starting the next forward.
        self._attention_mask = attention_mask
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        lengths = attention_mask.long().sum(1).clamp_min(1) - 1
        last = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        if self.pooling == "gated_last_mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            gate = torch.sigmoid(
                self.pool_gate(
                    torch.cat([last, mean], -1).to(self.pool_gate.weight.dtype)
                )
            ).to(last.dtype)
            last = last + gate * (mean - last)
        return self.head(self.feature_dropout(last))

    def condition_embeddings(self, *, detach=False):
        values = [a.condition_embeddings for a in self.backbone_conditional_adapters]
        if not values or any(v is None for v in values):
            raise ValueError("A Stage2 forward must precede condition diagnostics")
        result = tuple(torch.stack([v[i] for v in values]).mean(0) for i in (0, 1))
        return tuple(v.detach() for v in result) if detach else result


def build(cfg):
    from transformers import AutoModel, AutoTokenizer

    settings = cfg["small_llm"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_name"], local_files_only=True, revision=settings["revision"]
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer needs a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = getattr(torch, settings["torch_dtype"])
    backbone = AutoModel.from_pretrained(
        settings["model_name"],
        local_files_only=True,
        revision=settings["revision"],
        torch_dtype=dtype,
        attn_implementation=settings["attn_implementation"],
    )
    if settings["gradient_checkpointing"]:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=settings["gradient_checkpointing_kwargs"]
        )
    return VARegressor(backbone, cfg), tokenizer
