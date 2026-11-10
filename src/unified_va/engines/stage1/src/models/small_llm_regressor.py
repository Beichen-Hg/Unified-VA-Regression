"""Decoder-only small LLM backbone for VA regression."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.models.heads import build_va_head
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_torch_dtype(name: str | None):
    if not name or name == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported torch_dtype: {name}")
    return mapping[name]


class SmallLLMRegressor(nn.Module):
    """Regression wrapper over a decoder-only LLM backbone."""

    def __init__(self, backbone: nn.Module, head: nn.Module,
                 pooling: str = "last_token", feature_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pooling = pooling
        self.hidden_dim = backbone.config.hidden_size
        self.feature_dropout = nn.Dropout(feature_dropout) if feature_dropout > 0 else nn.Identity()
        self.pool_gate = None
        if self.pooling == "gated_last_mean":
            self.pool_gate = nn.Linear(self.hidden_dim * 2, 1)

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def _last_token_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]

    def _pool_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return self._mean_pool(hidden, attention_mask)
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "gated_last_mean":
            last = self._last_token_pool(hidden, attention_mask)
            mean = self._mean_pool(hidden, attention_mask)
            gate_in = torch.cat([last, mean], dim=-1)
            gate_in = gate_in.to(self.pool_gate.weight.dtype)
            gate = torch.sigmoid(self.pool_gate(gate_in)).to(last.dtype)
            return last + gate * (mean - last)
        return self._last_token_pool(hidden, attention_mask)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        pooled = self._pool_hidden(hidden, attention_mask)
        return self.feature_dropout(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        features = self.encode(input_ids, attention_mask)
        head_param = next(self.head.parameters(), None)
        if head_param is not None and features.dtype != head_param.dtype:
            features = features.to(head_param.dtype)
        return self.head(features)

    def trainable_ratio(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_ratio": trainable / max(total, 1),
        }


def _infer_peft_task_type(name: str):
    from peft import TaskType
    if name.lower() in {"causal_lm", "causallm"}:
        return TaskType.CAUSAL_LM
    return TaskType.FEATURE_EXTRACTION


def build_small_llm_regressor(cfg: dict):
    """Build decoder-only regressor + tokenizer from config."""
    llm_cfg = cfg.get("small_llm", {})
    model_name = (
        llm_cfg.get("model_path")
        or llm_cfg.get("model_name")
        or cfg.get("encoder", {}).get("model_name")
    )
    if not model_name:
        raise ValueError("small_llm.model_name/model_path is required")

    common_kwargs = {
        "trust_remote_code": llm_cfg.get("trust_remote_code", False),
        "local_files_only": llm_cfg.get("local_files_only", False),
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs = dict(common_kwargs)
    dtype = _resolve_torch_dtype(llm_cfg.get("torch_dtype", "auto"))
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    attn_impl = llm_cfg.get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    backbone = AutoModel.from_pretrained(model_name, **model_kwargs)
    if getattr(backbone.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        backbone.config.pad_token_id = tokenizer.pad_token_id
    if getattr(backbone.config, "use_cache", None) is not None:
        backbone.config.use_cache = False
    if llm_cfg.get("gradient_checkpointing", False) and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        from peft import LoraConfig, get_peft_model

        peft_cfg = LoraConfig(
            r=lora_cfg.get("rank", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=_infer_peft_task_type(lora_cfg.get("task_type", "feature_extraction")),
        )
        backbone = get_peft_model(backbone, peft_cfg)

    head = build_va_head(cfg, backbone.config.hidden_size)
    model = SmallLLMRegressor(
        backbone=backbone,
        head=head,
        pooling=llm_cfg.get("pooling", "last_token"),
        feature_dropout=llm_cfg.get("feature_dropout", cfg.get("head_dropout", 0.0)),
    )
    stats = model.trainable_ratio()
    logger.info(
        "SmallLLM backbone ready: model=%s hidden=%s trainable=%s/%s (%.4f)",
        model_name,
        backbone.config.hidden_size,
        stats["trainable_params"],
        stats["total_params"],
        stats["trainable_ratio"],
    )
    return model, tokenizer, backbone.config.hidden_size


def freeze_small_llm_layers(model: SmallLLMRegressor, freeze_until: int = 0):
    """Freeze decoder layers [0, freeze_until)."""
    if freeze_until <= 0:
        return

    backbone = model.backbone if hasattr(model, "backbone") else model
    base_model = getattr(backbone, "model", backbone)
    layers = None
    if hasattr(base_model, "layers"):
        layers = base_model.layers
    elif hasattr(base_model, "decoder") and hasattr(base_model.decoder, "layers"):
        layers = base_model.decoder.layers

    for attr in ("embed_tokens", "wte", "embeddings"):
        emb = getattr(base_model, attr, None)
        if emb is not None:
            for param in emb.parameters():
                param.requires_grad = False

    if layers is None:
        logger.warning("freeze_small_llm_layers: cannot find decoder layers")
        return

    n_frozen = 0
    for i, layer in enumerate(layers):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False
            n_frozen += 1
    logger.info("SmallLLM partial freeze: %s/%s layers frozen", n_frozen, len(layers))
