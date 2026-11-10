"""Decoder-only small LLM backbone for VA regression."""
from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Mistral3Config, Mistral3Model, __version__ as TRANSFORMERS_VERSION

try:
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
except ImportError:  # pragma: no cover - depends on transformers version
    Qwen3_5TextConfig = None
    Qwen3_5TextModel = None

from src.models.backbone_conditional_adapter import (
    ConditionalDecoderLayerWrapper,
    FactorizedConditionalResidualAdapter,
    PostCheckpointRepairResidualAdapter,
)
from src.models.heads import build_va_head
from src.models.heads import infer_group_values_from_cfg
from src.models.text_only_protocol import validate_text_only_cfg, validate_text_only_model
from src.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_loader_revisions(cfg: dict, model_name: str) -> dict:
    """Resolve explicit loader pins without contacting a repository or hashing files."""
    llm_cfg = cfg.get("small_llm", {}) or {}
    model_revision = llm_cfg.get("model_revision", llm_cfg.get("revision"))
    tokenizer_revision = llm_cfg.get("tokenizer_revision", llm_cfg.get("revision"))
    matched = bool((cfg.get("matched_protocol", {}) or {}).get("enabled", False))
    path = Path(model_name).expanduser()
    is_local = path.is_dir()
    looks_local = path.is_absolute() or model_name.startswith(("./", "../", "~/"))
    if matched and looks_local and not is_local:
        raise FileNotFoundError(f"matched-protocol model snapshot directory is missing: {path}")
    if matched:
        for name, value in (("model_revision", model_revision), ("tokenizer_revision", tokenizer_revision)):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
                raise ValueError(f"matched protocol requires an explicit 40-character commit pin for small_llm.{name}")
    return {
        "model_name": model_name,
        "source_kind": "local_directory" if is_local else "remote_repository",
        "requested_model_revision": model_revision,
        "requested_tokenizer_revision": tokenizer_revision,
    }


def _resolved_loader_commit(obj, model_name: str) -> str | None:
    config = getattr(obj, "config", None)
    init_kwargs = getattr(obj, "init_kwargs", {}) or {}
    value = getattr(config, "_commit_hash", None) or init_kwargs.get("_commit_hash")
    if value:
        return str(value)
    # Hugging Face cache snapshot directory names are immutable commit IDs.
    path = Path(model_name)
    if path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", path.name):
        return path.name
    return None


def validate_matched_condition_cfg(cfg: dict) -> dict:
    contract = dict(cfg.get("matched_protocol", {}) or {})
    if not contract.get("enabled", False):
        return {}
    arm = str(contract.get("arm", "")).upper()
    if arm not in {"S", "C", "M", "T"}:
        raise ValueError("matched_protocol.arm must be S, C, M or T")
    adapter = cfg.get("backbone_conditional_adapter", {}) or {}
    if arm in {"S", "C"} and adapter.get("enabled", False):
        raise ValueError(f"matched arm {arm} must not include a conditional adapter")
    if arm == "M":
        if not adapter.get("enabled", False) or adapter.get("condition_source") != "metadata":
            raise ValueError("matched arm M requires explicit metadata-conditioned adapters")
        if cfg.get("disable_explicit_metadata", False):
            raise ValueError("matched arm M cannot disable its declared metadata inputs")
        if not isinstance(contract.get("metadata_source"), str) or not contract["metadata_source"].strip():
            raise ValueError("matched arm M requires a recorded matched_protocol.metadata_source")
    if arm == "T":
        validate_text_only_cfg(cfg, required=True)
        contract["condition_representation"] = "continuous_soft_text"
        contract["auxiliary_label_metric"] = "condition_representation_label_predictability"
    contract["arm"] = arm
    return contract


def _ensure_qwen35_text_classes():
    global Qwen3_5TextConfig, Qwen3_5TextModel
    if Qwen3_5TextConfig is not None and Qwen3_5TextModel is not None:
        return Qwen3_5TextConfig, Qwen3_5TextModel
    try:
        from transformers import Qwen3_5TextConfig as _Qwen3_5TextConfig, Qwen3_5TextModel as _Qwen3_5TextModel
        Qwen3_5TextConfig = _Qwen3_5TextConfig
        Qwen3_5TextModel = _Qwen3_5TextModel
    except ImportError:
        try:
            from transformers.models.qwen3_5.configuration_qwen3_5 import (
                Qwen3_5TextConfig as _Qwen3_5TextConfig,
            )
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5TextModel as _Qwen3_5TextModel,
            )
            Qwen3_5TextConfig = _Qwen3_5TextConfig
            Qwen3_5TextModel = _Qwen3_5TextModel
        except ImportError:
            pass
    return Qwen3_5TextConfig, Qwen3_5TextModel


def _filter_callable_kwargs(fn, kwargs: dict | None) -> dict:
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


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


def _load_model_config_dict(model_name: str) -> dict | None:
    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text())
    except Exception:
        return None


def _has_local_remote_code_files(model_name: str) -> bool:
    model_dir = Path(model_name)
    if not model_dir.is_dir():
        return False
    patterns = ("*.py", "modules.json", "configuration*.py", "modeling*.py")
    for pattern in patterns:
        if any(model_dir.glob(pattern)):
            return True
    return False


def _is_local_mistral3_dir(model_name: str) -> bool:
    cfg = _load_model_config_dict(model_name)
    return isinstance(cfg, dict) and str(cfg.get("model_type", "")).strip().lower() == "mistral3"


def _is_local_qwen35_dir(model_name: str) -> bool:
    cfg = _load_model_config_dict(model_name)
    return isinstance(cfg, dict) and str(cfg.get("model_type", "")).strip().lower() == "qwen3_5"


def _build_mistral3_config(model_name: str) -> Mistral3Config:
    cfg = _load_model_config_dict(model_name)
    if not isinstance(cfg, dict):
        raise ValueError(f"expected local mistral3 config at {model_name}/config.json")
    cfg = dict(cfg)
    text_cfg = dict(cfg.get("text_config", {}) or {})
    # transformers 4.57 can construct Mistral3Config, but the downloaded
    # Ministral 3 configs still expose an internal `ministral3` text_config
    # that is not in AutoConfig's mapping. Normalize it to the supported
    # `ministral` text model before instantiation.
    if str(text_cfg.get("model_type", "")).strip().lower() == "ministral3":
        text_cfg["model_type"] = "ministral"
    if text_cfg.get("sliding_window", None) is None:
        text_cfg["sliding_window"] = 4096
    cfg["text_config"] = text_cfg
    return Mistral3Config.from_dict(cfg)


def _build_qwen35_text_config(model_name: str) -> "Qwen3_5TextConfig":
    qwen35_text_config_cls, _ = _ensure_qwen35_text_classes()
    if qwen35_text_config_cls is None:
        raise ValueError(
            "Qwen3.5 support requires a newer transformers build with Qwen3_5TextConfig/Qwen3_5TextModel"
        )
    cfg = _load_model_config_dict(model_name)
    if not isinstance(cfg, dict):
        raise ValueError(f"expected local qwen3.5 config at {model_name}/config.json")
    text_cfg = dict(cfg.get("text_config", {}) or {})
    if not text_cfg:
        raise ValueError(f"expected text_config in {model_name}/config.json for qwen3.5 text-only loading")
    return qwen35_text_config_cls.from_dict(text_cfg)


def resolve_small_llm_regressor_model_name(cfg: dict) -> str:
    llm_cfg = cfg.get("small_llm", {}) or {}
    model_name = (
        llm_cfg.get("model_path")
        or llm_cfg.get("model_name")
        or cfg.get("encoder", {}).get("model_name")
    )
    if not model_name:
        raise ValueError("small_llm.model_name/model_path is required")
    return str(model_name)


def inspect_small_llm_regressor_compatibility(cfg: dict) -> dict:
    model_name = resolve_small_llm_regressor_model_name(cfg)
    llm_cfg = cfg.get("small_llm", {}) or {}
    report = {
        "supported": True,
        "model_name": model_name,
        "backbone_family": "generic",
        "transformers_version": TRANSFORMERS_VERSION,
    }
    if not _is_local_qwen35_dir(model_name):
        return report

    model_cfg = _load_model_config_dict(model_name) or {}
    qwen35_text_config_cls, qwen35_text_model_cls = _ensure_qwen35_text_classes()
    has_qwen35_text_config = qwen35_text_config_cls is not None
    has_qwen35_text_model = qwen35_text_model_cls is not None
    has_local_remote_code_files = _has_local_remote_code_files(model_name)
    report.update(
        {
            "backbone_family": "qwen3_5",
            "model_transformers_version": model_cfg.get("transformers_version"),
            "has_qwen35_text_config": has_qwen35_text_config,
            "has_qwen35_text_model": has_qwen35_text_model,
            "has_local_remote_code_files": has_local_remote_code_files,
        }
    )
    if has_qwen35_text_config and has_qwen35_text_model:
        return report

    message = (
        "Unsupported local Qwen3.5 regressor config for the current environment: "
        f"model={model_name} requires transformers with Qwen3_5TextConfig/Qwen3_5TextModel, "
        f"but current transformers={TRANSFORMERS_VERSION}"
    )
    expected_transformers = model_cfg.get("transformers_version")
    if expected_transformers:
        message += f" and the local snapshot declares transformers_version={expected_transformers}"
    message += "."
    if llm_cfg.get("trust_remote_code", False):
        if has_local_remote_code_files:
            message += (
                " The local snapshot includes remote-code files, but this regressor path has no tested "
                "fallback without native Qwen3.5 text classes."
            )
        else:
            message += (
                " The local snapshot does not include remote-code Python files, so trust_remote_code "
                "cannot provide a fallback."
            )
    elif has_local_remote_code_files:
        message += (
            " Enabling trust_remote_code alone is still not sufficient for this regressor path without "
            "native Qwen3.5 text classes."
        )
    message += " Use a newer transformers build or switch this config to a backbone supported by the current environment."
    report["supported"] = False
    report["message"] = message
    return report


def validate_small_llm_regressor_compatibility(cfg: dict) -> dict:
    report = inspect_small_llm_regressor_compatibility(cfg)
    if not report.get("supported", True):
        raise ValueError(report["message"])
    return report


class PrePoolRepairAdapter(nn.Module):
    """Small token-space repair adapter applied before pooling."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
        residual_scale: float = 0.35,
        zero_init: bool = True,
        detach_input: bool = True,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.detach_input = bool(detach_input)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int(bottleneck_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(bottleneck_dim), hidden_dim),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_hidden = hidden_states.detach() if self.detach_input else hidden_states
        delta = self.net(base_hidden.to(next(self.net.parameters()).dtype)).to(base_hidden.dtype)
        delta = torch.tanh(delta) * max(self.residual_scale, 0.0)
        if attention_mask is not None:
            delta = delta * attention_mask.unsqueeze(-1).to(delta.dtype)
        return base_hidden + delta, delta


class TopBlockRepairAdapter(nn.Module):
    """Transformer-like token mixing repair block applied on the repaired path only."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int = 8,
        ff_mult: float = 2.0,
        dropout: float = 0.1,
        residual_scale: float = 0.2,
        detach_input: bool = True,
    ):
        super().__init__()
        ff_hidden = max(int(hidden_dim * float(ff_mult)), hidden_dim)
        self.detach_input = bool(detach_input)
        self.residual_scale = float(residual_scale)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=max(int(num_heads), 1),
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, hidden_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_hidden = hidden_states.detach() if self.detach_input else hidden_states
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask <= 0

        work_dtype = self.norm1.weight.dtype
        work_hidden = base_hidden.to(work_dtype)
        norm_hidden = self.norm1(work_hidden)
        attn_out, _ = self.self_attn(
            norm_hidden,
            norm_hidden,
            norm_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        mixed_hidden = work_hidden + attn_out * self.residual_scale
        ff_out = self.ff(self.norm2(mixed_hidden))
        repaired_hidden = mixed_hidden + ff_out * self.residual_scale
        repaired_hidden = repaired_hidden.to(base_hidden.dtype)
        delta = repaired_hidden - base_hidden
        if attention_mask is not None:
            delta = delta * attention_mask.unsqueeze(-1).to(delta.dtype)
            repaired_hidden = base_hidden + delta
        return repaired_hidden, delta


class TextRoutingDetector(nn.Module):
    """Predict soft routing posteriors from full-text hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_lang_classes: int,
        num_domain_classes: int,
        detector_hidden: int = 128,
        dropout: float = 0.1,
        pooling: str = "masked_mean",
        temperature: float = 1.0,
        detach_input: bool = True,
    ):
        super().__init__()
        self.pooling = str(pooling or "masked_mean").strip().lower()
        self.temperature = float(temperature)
        self.detach_input = bool(detach_input)
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int(detector_hidden)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lang_head = nn.Linear(int(detector_hidden), int(num_lang_classes))
        self.domain_head = nn.Linear(int(detector_hidden), int(num_domain_classes))

    def _pool_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            if self.pooling == "last_token":
                return hidden_states[:, -1]
            return hidden_states.mean(dim=1)
        if self.pooling == "last_token":
            lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, lengths]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_hidden = hidden_states.detach() if self.detach_input else hidden_states
        pooled = self._pool_hidden(base_hidden, attention_mask)
        work_dtype = self.trunk[0].weight.dtype
        trunk_hidden = self.trunk(pooled.to(work_dtype))
        lang_logits = self.lang_head(trunk_hidden)
        domain_logits = self.domain_head(trunk_hidden)
        return (
            lang_logits.to(base_hidden.dtype),
            domain_logits.to(base_hidden.dtype),
            pooled.to(base_hidden.dtype),
        )


class SmallLLMRegressor(nn.Module):
    """Regression wrapper over a decoder-only LLM backbone."""

    def __init__(self, backbone: nn.Module, head: nn.Module,
                 pooling: str = "last_token", feature_dropout: float = 0.0,
                 backbone_conditional_adapter_cfg: dict | None = None,
                 post_ckpt_repair_adapter_cfg: dict | None = None,
                 pre_pool_repair_cfg: dict | None = None,
                 top_block_repair_cfg: dict | None = None,
                 text_routing_detector_cfg: dict | None = None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pooling = pooling
        self.hidden_dim = backbone.config.hidden_size
        self.feature_dropout = nn.Dropout(feature_dropout) if feature_dropout > 0 else nn.Identity()
        self.pool_gate = None
        self._backbone_batch_lang = None
        self._backbone_batch_domain = None
        self._backbone_attention_mask = None
        self.backbone_conditional_adapters: list[FactorizedConditionalResidualAdapter] = []
        self.post_ckpt_repair_adapters: list[PostCheckpointRepairResidualAdapter] = []
        self._post_ckpt_repair_adapter_scale = 1.0
        self._ema_teacher_adapters = []
        self.pre_pool_repair = None
        self.top_block_repair = None
        self.text_routing_detector = None
        self._text_routing_detector_temperature = 1.0
        self._last_feature_views: dict[str, torch.Tensor] = {}
        if self.pooling == "gated_last_mean":
            self.pool_gate = nn.Linear(self.hidden_dim * 2, 1)
        if pre_pool_repair_cfg and pre_pool_repair_cfg.get("enabled", False):
            self.pre_pool_repair = PrePoolRepairAdapter(
                hidden_dim=self.hidden_dim,
                bottleneck_dim=int(pre_pool_repair_cfg.get("bottleneck_dim", 128)),
                dropout=float(pre_pool_repair_cfg.get("dropout", 0.1)),
                residual_scale=float(pre_pool_repair_cfg.get("residual_scale", 0.35)),
                zero_init=bool(pre_pool_repair_cfg.get("zero_init", True)),
                detach_input=bool(pre_pool_repair_cfg.get("detach_input", True)),
            )
            logger.info(
                "Pre-pool repair enabled: bottleneck=%s residual_scale=%.3f zero_init=%s detach_input=%s",
                int(pre_pool_repair_cfg.get("bottleneck_dim", 128)),
                float(pre_pool_repair_cfg.get("residual_scale", 0.35)),
                bool(pre_pool_repair_cfg.get("zero_init", True)),
                bool(pre_pool_repair_cfg.get("detach_input", True)),
            )
        if top_block_repair_cfg and top_block_repair_cfg.get("enabled", False):
            self.top_block_repair = TopBlockRepairAdapter(
                hidden_dim=self.hidden_dim,
                num_heads=int(top_block_repair_cfg.get("num_heads", 8)),
                ff_mult=float(top_block_repair_cfg.get("ff_mult", 2.0)),
                dropout=float(top_block_repair_cfg.get("dropout", 0.1)),
                residual_scale=float(top_block_repair_cfg.get("residual_scale", 0.2)),
                detach_input=bool(top_block_repair_cfg.get("detach_input", True)),
            )
            logger.info(
                "Top-block repair enabled: heads=%s ff_mult=%.2f residual_scale=%.3f detach_input=%s",
                int(top_block_repair_cfg.get("num_heads", 8)),
                float(top_block_repair_cfg.get("ff_mult", 2.0)),
                float(top_block_repair_cfg.get("residual_scale", 0.2)),
                bool(top_block_repair_cfg.get("detach_input", True)),
            )
        if text_routing_detector_cfg and text_routing_detector_cfg.get("enabled", False):
            if not hasattr(head, "lang_to_idx") or not hasattr(head, "domain_to_idx"):
                raise ValueError(
                    "text_routing_detector requires a factorized head with lang/domain class maps"
                )
            self.text_routing_detector = TextRoutingDetector(
                hidden_dim=self.hidden_dim,
                num_lang_classes=len(getattr(head, "lang_to_idx")),
                num_domain_classes=len(getattr(head, "domain_to_idx")),
                detector_hidden=int(text_routing_detector_cfg.get("detector_hidden", 128)),
                dropout=float(text_routing_detector_cfg.get("dropout", 0.1)),
                pooling=str(text_routing_detector_cfg.get("pooling", "masked_mean")),
                temperature=float(text_routing_detector_cfg.get("temperature", 1.0)),
                detach_input=bool(text_routing_detector_cfg.get("detach_input", True)),
            )
            self._text_routing_detector_temperature = float(
                text_routing_detector_cfg.get("temperature", 1.0)
            )
            logger.info(
                "Text routing detector enabled: pooling=%s hidden=%s temperature=%.3f detach_input=%s",
                str(text_routing_detector_cfg.get("pooling", "masked_mean")),
                int(text_routing_detector_cfg.get("detector_hidden", 128)),
                float(text_routing_detector_cfg.get("temperature", 1.0)),
                bool(text_routing_detector_cfg.get("detach_input", True)),
            )
        if backbone_conditional_adapter_cfg and backbone_conditional_adapter_cfg.get("enabled", False):
            self._attach_backbone_conditional_adapters(
                backbone_conditional_adapter_cfg,
                post_ckpt_repair_adapter_cfg=post_ckpt_repair_adapter_cfg,
            )
        elif post_ckpt_repair_adapter_cfg and post_ckpt_repair_adapter_cfg.get("enabled", False):
            raise ValueError(
                "post_ckpt_repair_adapter currently requires backbone_conditional_adapter.enabled=true "
                "so W1 checkpoint keys remain compatible"
            )

    def _resolve_decoder_layers(self):
        backbone = self.backbone if hasattr(self, "backbone") else self
        candidates = [
            backbone,
            getattr(backbone, "model", None),
            getattr(backbone, "language_model", None),
        ]
        decoder = getattr(backbone, "get_decoder", None)
        if callable(decoder):
            try:
                candidates.append(decoder())
            except Exception:
                pass
        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "layers"):
                return candidate.layers
            if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
                return candidate.model.layers
            if hasattr(candidate, "decoder") and hasattr(candidate.decoder, "layers"):
                return candidate.decoder.layers
        return None

    def _attach_backbone_conditional_adapters(
        self,
        cfg: dict,
        *,
        post_ckpt_repair_adapter_cfg: dict | None = None,
    ):
        layers = self._resolve_decoder_layers()
        if layers is None:
            raise ValueError("backbone_conditional_adapter enabled but decoder layers were not found")
        start_layer = max(int(cfg.get("start_layer", max(len(layers) - 8, 0))), 0)
        end_layer = int(cfg.get("end_layer", len(layers)))
        end_layer = min(max(end_layer, start_layer), len(layers))
        repair_cfg = post_ckpt_repair_adapter_cfg or {}
        repair_enabled = bool(repair_cfg.get("enabled", False))
        repair_start_layer = max(int(repair_cfg.get("start_layer", max(end_layer - 2, start_layer))), start_layer)
        repair_end_layer = int(repair_cfg.get("end_layer", end_layer))
        repair_end_layer = min(max(repair_end_layer, repair_start_layer), end_layer)
        use_composition = bool(cfg.get("use_composition", False))
        composition_start_layer = max(int(cfg.get("composition_start_layer", start_layer)), start_layer)
        composition_end_layer = int(cfg.get("composition_end_layer", end_layer))
        composition_end_layer = min(max(composition_end_layer, composition_start_layer), end_layer)
        langs = cfg.get("langs", [])
        domains = cfg.get("domains", [])
        if not langs or not domains:
            raise ValueError("backbone_conditional_adapter requires non-empty langs and domains")
        for idx in range(start_layer, end_layer):
            layer_use_composition = use_composition and composition_start_layer <= idx < composition_end_layer
            adapter = FactorizedConditionalResidualAdapter(
                hidden_dim=self.hidden_dim,
                langs=langs,
                domains=domains,
                embed_dim=int(cfg.get("embed_dim", 16)),
                bottleneck_dim=int(cfg.get("bottleneck_dim", 64)),
                dropout=float(cfg.get("dropout", 0.1)),
                use_shared=bool(cfg.get("use_shared", True)),
                use_composition=layer_use_composition,
                composition_bottleneck_dim=cfg.get("composition_bottleneck_dim"),
                composition_dropout=cfg.get("composition_dropout"),
                composition_gate_bias=float(cfg.get("composition_gate_bias", 0.0)),
                composition_gate_max=float(cfg.get("composition_gate_max", 1.0)),
                composition_residual_scale=float(cfg.get("composition_residual_scale", 1.0)),
                composition_state_aware=bool(cfg.get("composition_state_aware", False)),
                composition_state_hidden=int(cfg.get("composition_state_hidden", 64)),
                shared_residual_scale=float(cfg.get("shared_residual_scale", 1.0)),
                shared_branch_dropout=float(cfg.get("shared_branch_dropout", 0.0)),
                confidence_gated_composition=bool(cfg.get("confidence_gated_composition", False)),
                confidence_gate_scale=float(cfg.get("confidence_gate_scale", 1.0)),
                confidence_gate_bias=float(cfg.get("confidence_gate_bias", 0.0)),
                confidence_gate_min=float(cfg.get("confidence_gate_min", 0.0)),
                lang_alias_map=cfg.get("lang_alias_map", {}),
                domain_alias_map=cfg.get("domain_alias_map", {}),
                condition_source=str(cfg.get("condition_source", "metadata")),
                soft_condition_hidden=int(cfg.get("soft_condition_hidden", 128)),
                soft_condition_pooling=str(cfg.get("soft_condition_pooling", "mean")),
            )
            repair_adapter = None
            if repair_enabled and repair_start_layer <= idx < repair_end_layer:
                repair_adapter = PostCheckpointRepairResidualAdapter(
                    hidden_dim=self.hidden_dim,
                    bottleneck_dim=int(repair_cfg.get("bottleneck_dim", 128)),
                    gate_hidden=int(repair_cfg.get("gate_hidden", 64)),
                    dropout=float(repair_cfg.get("dropout", 0.1)),
                    residual_scale=float(repair_cfg.get("residual_scale", 1.0)),
                    zero_init=bool(repair_cfg.get("zero_init", True)),
                    gate_bias_init=float(repair_cfg.get("gate_bias_init", -4.0)),
                )
            layers[idx] = ConditionalDecoderLayerWrapper(
                layers[idx],
                adapter,
                self,
                repair_adapter=repair_adapter,
                adapter_index=len(self.backbone_conditional_adapters),
            )
            self.backbone_conditional_adapters.append(adapter)
            if repair_adapter is not None:
                self.post_ckpt_repair_adapters.append(repair_adapter)
        logger.info(
            "Backbone conditional adapters enabled: layers=%s..%s count=%s bottleneck=%s composition=%s comp_layers=%s..%s comp_bottleneck=%s comp_gate_max=%.3f comp_scale=%.3f comp_state_aware=%s comp_state_hidden=%s",
            start_layer,
            end_layer - 1,
            len(self.backbone_conditional_adapters),
            int(cfg.get("bottleneck_dim", 64)),
            use_composition,
            composition_start_layer,
            composition_end_layer - 1,
            int(cfg.get("composition_bottleneck_dim", cfg.get("bottleneck_dim", 64))),
            float(cfg.get("composition_gate_max", 1.0)),
            float(cfg.get("composition_residual_scale", 1.0)),
            bool(cfg.get("composition_state_aware", False)),
            int(cfg.get("composition_state_hidden", 64)),
        )
        if repair_enabled:
            logger.info(
                "Post-ckpt repair adapters enabled: layers=%s..%s count=%s bottleneck=%s gate_hidden=%s",
                repair_start_layer,
                repair_end_layer - 1,
                len(self.post_ckpt_repair_adapters),
                int(repair_cfg.get("bottleneck_dim", 128)),
                int(repair_cfg.get("gate_hidden", 64)),
            )

    def set_backbone_conditioning(
        self,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        self._backbone_batch_lang = batch_lang
        self._backbone_batch_domain = batch_domain
        self._backbone_attention_mask = attention_mask

    def clear_backbone_conditioning(self):
        self._backbone_batch_lang = None
        self._backbone_batch_domain = None
        self._backbone_attention_mask = None

    def get_backbone_conditioning(self):
        return self._backbone_batch_lang, self._backbone_batch_domain

    def get_backbone_attention_mask(self):
        return self._backbone_attention_mask

    def set_post_ckpt_repair_adapter_scale(self, scale: float):
        self._post_ckpt_repair_adapter_scale = max(float(scale), 0.0)

    def get_post_ckpt_repair_adapter_scale(self) -> float:
        return float(self._post_ckpt_repair_adapter_scale)

    def set_backbone_runtime_controls(
        self,
        *,
        shared_residual_scale: float | None = None,
        shared_branch_dropout: float | None = None,
        composition_gate_max: float | None = None,
        composition_residual_scale: float | None = None,
    ) -> None:
        for adapter in self.backbone_conditional_adapters:
            adapter.set_runtime_controls(
                shared_residual_scale=shared_residual_scale,
                shared_branch_dropout=shared_branch_dropout,
                composition_gate_max=composition_gate_max,
                composition_residual_scale=composition_residual_scale,
            )

    def clear_backbone_runtime_controls(self) -> None:
        for adapter in self.backbone_conditional_adapters:
            adapter.clear_runtime_controls()

    def set_ema_teacher_adapters(self, adapters) -> None:
        self._ema_teacher_adapters = list(adapters or [])

    def get_ema_teacher_adapter(self, index: int | None):
        if index is None or index < 0 or index >= len(self._ema_teacher_adapters):
            return None
        return self._ema_teacher_adapters[index]

    def backbone_adapter_usage(self) -> dict[str, float]:
        if not self.backbone_conditional_adapters and not self.post_ckpt_repair_adapters:
            return {}
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for adapter in self.backbone_conditional_adapters:
            for key, value in adapter.gate_stats().items():
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        for adapter in self.post_ckpt_repair_adapters:
            for key, value in adapter.gate_stats().items():
                totals[f"post_ckpt_{key}"] = totals.get(f"post_ckpt_{key}", 0.0) + float(value)
                counts[f"post_ckpt_{key}"] = counts.get(f"post_ckpt_{key}", 0) + 1
        return {
            f"backbone_conditional_{key}_gate_mean": totals[key] / max(counts[key], 1)
            for key in totals
        }

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

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_feature_views = {}
        self.set_backbone_conditioning(
            batch_lang=batch_lang,
            batch_domain=batch_domain,
            attention_mask=attention_mask,
        )
        try:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            self.clear_backbone_conditioning()
        hidden = outputs.last_hidden_state
        if self.text_routing_detector is not None:
            (
                routing_lang_logits,
                routing_domain_logits,
                routing_detector_pooled,
            ) = self.text_routing_detector(hidden, attention_mask=attention_mask)
            self._last_feature_views.update(
                {
                    "routing_lang_logits": routing_lang_logits,
                    "routing_domain_logits": routing_domain_logits,
                    "routing_detector_pooled": routing_detector_pooled,
                    "routing_temperature": torch.tensor(
                        self._text_routing_detector_temperature,
                        device=hidden.device,
                        dtype=hidden.dtype,
                    ),
                }
            )
        clean_pooled = self._pool_hidden(hidden, attention_mask)
        clean_features = self.feature_dropout(clean_pooled)
        if self.pre_pool_repair is None and self.top_block_repair is None:
            return clean_features

        repaired_hidden = hidden
        repair_delta = torch.zeros_like(hidden)
        if self.pre_pool_repair is not None:
            repaired_hidden, repair_delta = self.pre_pool_repair(
                repaired_hidden,
                attention_mask=attention_mask,
            )
        if self.top_block_repair is not None:
            repaired_hidden, top_delta = self.top_block_repair(
                repaired_hidden,
                attention_mask=attention_mask,
            )
            repair_delta = repair_delta + top_delta
        repaired_pooled = self._pool_hidden(repaired_hidden, attention_mask)
        repaired_features = self.feature_dropout(repaired_pooled)
        self._last_feature_views = {
            "clean_pooled": clean_features,
            "repaired_pooled": repaired_features,
            "pre_pool_repair_delta": repair_delta,
        }
        return clean_features

    def apply_head(
        self,
        features: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
        **head_kwargs,
    ):
        if "repaired_features" not in head_kwargs:
            repaired_features = self._last_feature_views.get("repaired_pooled")
            if repaired_features is not None:
                head_kwargs["repaired_features"] = repaired_features
        if "routing_lang_logits" not in head_kwargs:
            routing_lang_logits = self._last_feature_views.get("routing_lang_logits")
            if routing_lang_logits is not None:
                head_kwargs["routing_lang_logits"] = routing_lang_logits
        if "routing_domain_logits" not in head_kwargs:
            routing_domain_logits = self._last_feature_views.get("routing_domain_logits")
            if routing_domain_logits is not None:
                head_kwargs["routing_domain_logits"] = routing_domain_logits
        if "routing_temperature" not in head_kwargs:
            routing_temperature = self._last_feature_views.get("routing_temperature")
            if routing_temperature is not None:
                head_kwargs["routing_temperature"] = float(routing_temperature.detach().cpu().item())
        head_param = next(self.head.parameters(), None)
        if head_param is not None and features.dtype != head_param.dtype:
            features = features.to(head_param.dtype)
        if hasattr(self.head, "forward_with_metadata"):
            filtered = _filter_callable_kwargs(self.head.forward_with_metadata, head_kwargs)
            return self.head.forward_with_metadata(
                features,
                batch_lang=batch_lang,
                batch_domain=batch_domain,
                **filtered,
            )
        filtered = _filter_callable_kwargs(self.head.forward, head_kwargs)
        return self.head(features, **filtered)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
        **head_kwargs,
    ):
        contract = getattr(self, "matched_protocol", {})
        if contract.get("arm") == "M":
            expected = input_ids.size(0)
            if batch_lang is None or batch_domain is None:
                raise ValueError("matched M inference requires the declared language and domain metadata")
            if len(batch_lang) != expected or len(batch_domain) != expected:
                raise ValueError("matched M metadata must contain one language and domain value per record")
        elif contract.get("arm") == "T":
            # Labels may supervise training separately; they are never VA
            # forward inputs for the text-conditioned arm.
            batch_lang, batch_domain = None, None
        features = self.encode(
            input_ids,
            attention_mask,
            batch_lang=batch_lang,
            batch_domain=batch_domain,
        )
        return self.apply_head(features, batch_lang=batch_lang, batch_domain=batch_domain, **head_kwargs)

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
    validate_text_only_cfg(cfg)
    condition_contract = validate_matched_condition_cfg(cfg)
    llm_cfg = cfg.get("small_llm", {})
    skip_small_llm_preflight = str(os.environ.get("SKIP_SMALL_LLM_PREFLIGHT", "0")).strip().lower() in {
        "1", "true", "yes"
    }
    compatibility = (
        inspect_small_llm_regressor_compatibility(cfg)
        if skip_small_llm_preflight
        else validate_small_llm_regressor_compatibility(cfg)
    )
    model_name = compatibility["model_name"]
    loading_provenance = resolve_loader_revisions(cfg, model_name)

    common_kwargs = {
        "trust_remote_code": llm_cfg.get("trust_remote_code", False),
        "local_files_only": llm_cfg.get("local_files_only", False),
    }
    tokenizer_kwargs = dict(common_kwargs)
    if loading_provenance["requested_tokenizer_revision"] is not None:
        tokenizer_kwargs["revision"] = loading_provenance["requested_tokenizer_revision"]
    if _is_local_mistral3_dir(model_name):
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs = dict(common_kwargs)
    if loading_provenance["requested_model_revision"] is not None:
        model_kwargs["revision"] = loading_provenance["requested_model_revision"]
    dtype = _resolve_torch_dtype(llm_cfg.get("torch_dtype", "auto"))
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    attn_impl = llm_cfg.get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    if _is_local_mistral3_dir(model_name):
        model_kwargs["config"] = _build_mistral3_config(model_name)
        backbone = Mistral3Model.from_pretrained(model_name, **model_kwargs)
    elif _is_local_qwen35_dir(model_name):
        _, qwen35_text_model_cls = _ensure_qwen35_text_classes()
        if qwen35_text_model_cls is None:
            raise ValueError(
                "Qwen3.5 support requires a newer transformers build with Qwen3_5TextModel"
            )
        model_kwargs["config"] = _build_qwen35_text_config(model_name)
        backbone = qwen35_text_model_cls.from_pretrained(model_name, **model_kwargs)
    else:
        backbone = AutoModel.from_pretrained(model_name, **model_kwargs)
    loading_provenance["resolved_model_commit"] = _resolved_loader_commit(backbone, model_name)
    loading_provenance["resolved_tokenizer_commit"] = _resolved_loader_commit(tokenizer, model_name)
    loading_provenance["resolved_commit_note"] = (
        "Loader-reported commit or cache snapshot directory; null means not reported. "
        "Requested revisions are recorded separately and are not inferred verification."
    )
    if getattr(backbone.config, "hidden_size", None) is None:
        text_cfg = getattr(backbone.config, "text_config", None)
        text_hidden = getattr(text_cfg, "hidden_size", None)
        if text_hidden is not None:
            backbone.config.hidden_size = int(text_hidden)
    if getattr(backbone.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        backbone.config.pad_token_id = tokenizer.pad_token_id
    if getattr(backbone.config, "use_cache", None) is not None:
        backbone.config.use_cache = False
    if llm_cfg.get("gradient_checkpointing", False) and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

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
    conditional_cfg = cfg.get("backbone_conditional_adapter", {})
    if conditional_cfg.get("enabled", False):
        conditional_cfg = dict(conditional_cfg)
        shared_reg_cfg = dict(cfg.get("shared_branch_regularization", {}) or {})
        if shared_reg_cfg.get("enabled", False):
            conditional_cfg.setdefault("shared_residual_scale", float(shared_reg_cfg.get("shared_residual_scale", 1.0)))
            conditional_cfg.setdefault("shared_branch_dropout", float(shared_reg_cfg.get("shared_branch_dropout", 0.0)))
        confidence_cfg = dict(cfg.get("confidence_gated_composition", {}) or {})
        if confidence_cfg.get("enabled", False):
            conditional_cfg.setdefault("confidence_gated_composition", True)
            conditional_cfg.setdefault("confidence_gate_scale", float(confidence_cfg.get("gate_scale", 1.0)))
            conditional_cfg.setdefault("confidence_gate_bias", float(confidence_cfg.get("gate_bias", 0.0)))
            conditional_cfg.setdefault("confidence_gate_min", float(confidence_cfg.get("gate_min", 0.0)))
        conditional_cfg["langs"] = [
            str(v).strip() for v in conditional_cfg.get("langs", []) if str(v).strip()
        ] or infer_group_values_from_cfg(cfg, "lang")
        conditional_cfg["domains"] = [
            str(v).strip() for v in conditional_cfg.get("domains", []) if str(v).strip()
        ] or infer_group_values_from_cfg(cfg, "domain")
    repair_adapter_cfg = cfg.get("post_ckpt_repair_adapter", {})
    pre_pool_repair_cfg = cfg.get("pre_pool_repair", {})
    top_block_repair_cfg = cfg.get("top_block_repair", {})
    text_routing_detector_cfg = cfg.get("text_routing_detector", {})
    model = SmallLLMRegressor(
        backbone=backbone,
        head=head,
        pooling=llm_cfg.get("pooling", "last_token"),
        feature_dropout=llm_cfg.get("feature_dropout", cfg.get("head_dropout", 0.0)),
        backbone_conditional_adapter_cfg=conditional_cfg,
        post_ckpt_repair_adapter_cfg=repair_adapter_cfg,
        pre_pool_repair_cfg=pre_pool_repair_cfg,
        top_block_repair_cfg=top_block_repair_cfg,
        text_routing_detector_cfg=text_routing_detector_cfg,
    )
    model.loading_provenance = loading_provenance
    model.matched_protocol = condition_contract
    if cfg.get("text_only_inference", False):
        validate_text_only_model(model)
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
    candidates = [
        backbone,
        getattr(backbone, "model", None),
        getattr(backbone, "language_model", None),
    ]
    decoder_getter = getattr(backbone, "get_decoder", None)
    if callable(decoder_getter):
        try:
            candidates.append(decoder_getter())
        except Exception:
            pass

    layers = None
    embed_modules = []
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "layers"):
            layers = candidate.layers
        elif hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            layers = candidate.model.layers
        elif hasattr(candidate, "decoder") and hasattr(candidate.decoder, "layers"):
            layers = candidate.decoder.layers
        for attr in ("embed_tokens", "wte", "embeddings"):
            emb = getattr(candidate, attr, None)
            if emb is not None:
                embed_modules.append(emb)
        if layers is not None:
            break

    for emb in embed_modules:
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
