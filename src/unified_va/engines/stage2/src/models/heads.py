"""Model heads for DimABSA2026 VA regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_group_values_from_cfg(cfg: dict, key: str) -> list[str]:
    """Infer known language/domain values from config when not provided explicitly."""
    values = set()
    direct = cfg.get(key)
    if isinstance(direct, str) and direct.strip():
        values.add(direct.strip())
    aux_cfg = cfg.get("auxiliary_data", {})
    for source in aux_cfg.get("sources", []) or []:
        value = str(source.get(key, "")).strip()
        if value:
            values.add(value)
    return sorted(values)


class _GroupEncodingMixin:
    @staticmethod
    def _canonicalize_value(value, alias_map: dict[str, str]) -> str:
        value = str(value or "__unk__").strip()
        if not value:
            value = "__unk__"
        return alias_map.get(value, value)

    def _build_value_to_idx(self, values: list[str] | None, alias_map: dict[str, str] | None = None) -> dict[str, int]:
        alias_map = alias_map or {}
        out = {"__unk__": 0}
        for value in values or []:
            canon = self._canonicalize_value(value, alias_map)
            if canon not in out:
                out[canon] = len(out)
        return out

    def encode_group_ids(
        self,
        langs: list[str] | torch.Tensor | None,
        domains: list[str] | torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(langs):
            lang_ids = langs.to(device=device, dtype=torch.long)
        else:
            lang_ids = torch.tensor(
                [
                    self.lang_to_idx.get(
                        self._canonicalize_value(v, getattr(self, "lang_alias_map", {})),
                        0,
                    )
                    for v in (langs or [])
                ],
                dtype=torch.long,
                device=device,
            )
        if torch.is_tensor(domains):
            domain_ids = domains.to(device=device, dtype=torch.long)
        else:
            domain_ids = torch.tensor(
                [
                    self.domain_to_idx.get(
                        self._canonicalize_value(v, getattr(self, "domain_alias_map", {})),
                        0,
                    )
                    for v in (domains or [])
                ],
                dtype=torch.long,
                device=device,
            )
        return lang_ids, domain_ids


class SoftConditionGroupClassifier(nn.Module, _GroupEncodingMixin):
    """Auxiliary classifier over soft lang/domain condition embeddings."""

    def __init__(
        self,
        embed_dim: int,
        langs: list[str],
        domains: list[str],
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lang_alias_map: dict[str, str] | None = None,
        domain_alias_map: dict[str, str] | None = None,
    ):
        super().__init__()
        self.lang_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (lang_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.domain_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (domain_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.lang_to_idx = self._build_value_to_idx(langs, self.lang_alias_map)
        self.domain_to_idx = self._build_value_to_idx(domains, self.domain_alias_map)
        self.lang_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.lang_to_idx)),
        )
        self.domain_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.domain_to_idx)),
        )

    def forward(
        self,
        lang_emb: torch.Tensor,
        domain_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lang_dtype = next(self.lang_head.parameters()).dtype
        domain_dtype = next(self.domain_head.parameters()).dtype
        lang_logits = self.lang_head(lang_emb.to(lang_dtype))
        domain_logits = self.domain_head(domain_emb.to(domain_dtype))
        return lang_logits, domain_logits

    def num_lang_classes(self) -> int:
        return len(self.lang_to_idx)

    def num_domain_classes(self) -> int:
        return len(self.domain_to_idx)


class VARegressionHead(nn.Module):
    """2-dim regression head for valence/arousal prediction."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        out = self.fc(h).clamp(1.0, 9.0)
        return (out * 100).round() / 100


class VARegressionMLPHead(nn.Module):
    """Deep MLP head for VA regression: Linear → ReLU → Dropout → Linear."""
    def __init__(self, hidden_dim: int, head_hidden: int = 256, head_dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        out = self.mlp(h).clamp(1.0, 9.0)
        return (out * 100).round() / 100


class VADualHeadMLP(nn.Module):
    """Shared bottleneck + separate V/A MLP branches for LLM regression."""
    def __init__(self, hidden_dim: int, shared_hidden: int = 512,
                 branch_hidden: int = 256, head_dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, shared_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.v_branch = nn.Sequential(
            nn.Linear(shared_hidden, branch_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(branch_hidden, 1),
        )
        self.a_branch = nn.Sequential(
            nn.Linear(shared_hidden, branch_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(branch_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        shared = self.shared(h)
        v = self.v_branch(shared)
        a = self.a_branch(shared)
        return torch.cat([v, a], dim=-1)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        out = self.forward(h).clamp(1.0, 9.0)
        return (out * 100).round() / 100


class MetadataConditionedHeadAdapter(nn.Module, _GroupEncodingMixin):
    """Lightweight metadata-conditioned wrapper over an existing VA head.

    This keeps a shared base head intact while allowing language/domain metadata
    to inject a small zero-initialized residual into the pooled representation.
    """

    def __init__(
        self,
        base_head: nn.Module,
        hidden_dim: int,
        langs: list[str] | None = None,
        domains: list[str] | None = None,
        embed_dim: int = 16,
        conditioner_hidden: int = 128,
        dropout: float = 0.1,
        learned_gate: bool = True,
        lang_alias_map: dict[str, str] | None = None,
        domain_alias_map: dict[str, str] | None = None,
    ):
        super().__init__()
        self.base_head = base_head
        self.hidden_dim = int(hidden_dim)
        self.learned_gate = bool(learned_gate)
        self.lang_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (lang_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.domain_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (domain_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.lang_to_idx = self._build_value_to_idx(langs, self.lang_alias_map)
        self.domain_to_idx = self._build_value_to_idx(domains, self.domain_alias_map)
        self.lang_embedding = nn.Embedding(len(self.lang_to_idx), embed_dim, padding_idx=0)
        self.domain_embedding = nn.Embedding(len(self.domain_to_idx), embed_dim, padding_idx=0)

        self.trunk = nn.Sequential(
            nn.Linear(self.hidden_dim + 2 * embed_dim, conditioner_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(conditioner_hidden, conditioner_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(conditioner_hidden, self.hidden_dim)
        self.gate_head = nn.Linear(conditioner_hidden, 1) if self.learned_gate else None

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)

    def conditioned_features(
        self,
        source_features: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch_lang is None:
            batch_lang = ["__unk__"] * source_features.size(0)
        if batch_domain is None:
            batch_domain = ["__unk__"] * source_features.size(0)
        lang_ids, domain_ids = self.encode_group_ids(batch_lang, batch_domain, device=source_features.device)
        lang_emb = self.lang_embedding(lang_ids).to(source_features.dtype)
        domain_emb = self.domain_embedding(domain_ids).to(source_features.dtype)
        z = self.trunk(torch.cat([source_features, lang_emb, domain_emb], dim=-1))
        delta = self.delta_head(z)
        if self.gate_head is not None:
            gate = torch.sigmoid(self.gate_head(z)).to(delta.dtype)
            delta = gate * delta
        return source_features + delta

    def forward_with_metadata(
        self,
        source_features: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_head(self.conditioned_features(source_features, batch_lang, batch_domain))

    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_metadata(source_features, None, None)

    def predict(self, source_features: torch.Tensor) -> torch.Tensor:
        conditioned = self.conditioned_features(source_features, None, None)
        if hasattr(self.base_head, "predict"):
            return self.base_head.predict(conditioned)
        out = self.base_head(conditioned).clamp(1.0, 9.0)
        return (out * 100).round() / 100

    def regularization_loss(self) -> torch.Tensor | None:
        if hasattr(self.base_head, "regularization_loss"):
            return self.base_head.regularization_loss()
        return None


class MetadataConditionedOutputResidualHead(nn.Module, _GroupEncodingMixin):
    """Metadata-conditioned output residual with separate V/A corrections."""

    def __init__(
        self,
        base_head: nn.Module,
        hidden_dim: int,
        langs: list[str] | None = None,
        domains: list[str] | None = None,
        embed_dim: int = 16,
        conditioner_hidden: int = 128,
        dropout: float = 0.1,
        learned_gate: bool = True,
        per_dim_gate: bool = True,
        gate_bias_init: float = 0.0,
        detach_base_prediction: bool = True,
        clamp_delta: float = 0.5,
        bound_mode: str = "tanh",
        lang_alias_map: dict[str, str] | None = None,
        domain_alias_map: dict[str, str] | None = None,
    ):
        super().__init__()
        self.base_head = base_head
        self.hidden_dim = int(hidden_dim)
        self.learned_gate = bool(learned_gate)
        self.per_dim_gate = bool(per_dim_gate)
        self.detach_base_prediction = bool(detach_base_prediction)
        self.clamp_delta = float(clamp_delta)
        self.bound_mode = str(bound_mode).strip().lower()
        self.lang_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (lang_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.domain_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (domain_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.lang_to_idx = self._build_value_to_idx(langs, self.lang_alias_map)
        self.domain_to_idx = self._build_value_to_idx(domains, self.domain_alias_map)
        self.lang_embedding = nn.Embedding(len(self.lang_to_idx), embed_dim, padding_idx=0)
        self.domain_embedding = nn.Embedding(len(self.domain_to_idx), embed_dim, padding_idx=0)

        self.trunk = nn.Sequential(
            nn.Linear(self.hidden_dim + 2 + 2 * embed_dim, conditioner_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(conditioner_hidden, conditioner_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(conditioner_hidden, 2)
        gate_dim = 2 if self.per_dim_gate else 1
        self.gate_head = nn.Linear(conditioner_hidden, gate_dim) if self.learned_gate else None

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, float(gate_bias_init))

    def _base_forward(
        self,
        source_features: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(self.base_head, "forward_with_metadata"):
            return self.base_head.forward_with_metadata(
                source_features,
                batch_lang=batch_lang,
                batch_domain=batch_domain,
            )
        return self.base_head(source_features)

    def _bound_delta(self, delta_raw: torch.Tensor) -> torch.Tensor:
        if self.bound_mode == "clip":
            return torch.clamp(delta_raw, -self.clamp_delta, self.clamp_delta)
        scale = max(self.clamp_delta, 1e-9)
        return torch.tanh(delta_raw / scale) * scale

    def forward_with_metadata(
        self,
        source_features: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_pred = self._base_forward(
            source_features,
            batch_lang=batch_lang,
            batch_domain=batch_domain,
        )
        pred_input = base_pred.detach() if self.detach_base_prediction else base_pred
        if batch_lang is None:
            batch_lang = ["__unk__"] * source_features.size(0)
        if batch_domain is None:
            batch_domain = ["__unk__"] * source_features.size(0)
        lang_ids, domain_ids = self.encode_group_ids(batch_lang, batch_domain, device=source_features.device)
        lang_emb = self.lang_embedding(lang_ids).to(source_features.dtype)
        domain_emb = self.domain_embedding(domain_ids).to(source_features.dtype)
        z = self.trunk(torch.cat([source_features, pred_input, lang_emb, domain_emb], dim=-1))
        delta = self._bound_delta(self.delta_head(z))
        if self.gate_head is not None:
            gate = torch.sigmoid(self.gate_head(z)).to(delta.dtype)
            if gate.shape[-1] == 1:
                gate = gate.expand_as(delta)
            delta = gate * delta
        return base_pred + delta

    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_metadata(source_features, None, None)

    def predict(self, source_features: torch.Tensor) -> torch.Tensor:
        out = self.forward_with_metadata(source_features, None, None).clamp(1.0, 9.0)
        return (out * 100).round() / 100

    def regularization_loss(self) -> torch.Tensor | None:
        if hasattr(self.base_head, "regularization_loss"):
            return self.base_head.regularization_loss()
        return None


class FactorizedOutputResidualHead(nn.Module, _GroupEncodingMixin):
    """Factorized output-space residual head.

    This keeps the stage1 base predictor as the main path and learns only a
    routed residual in output space:

      y_final = y_base + (shared + lang + domain + composition)

    The residual branches are zero-initialized so the module starts as an
    identity mapping on top of the frozen or mostly frozen stage1 predictor.
    """

    def __init__(
        self,
        base_head: nn.Module,
        hidden_dim: int,
        langs: list[str] | None = None,
        domains: list[str] | None = None,
        *,
        embed_dim: int = 16,
        shared_hidden: int = 128,
        branch_hidden: int = 128,
        composition_hidden: int = 128,
        dropout: float = 0.1,
        use_shared: bool = True,
        use_lang_branch: bool = True,
        use_domain_branch: bool = True,
        use_composition: bool = True,
        condition_source: str = "metadata",
        soft_condition_hidden: int = 128,
        detach_base_prediction: bool = True,
        clamp_delta: float = 1.0,
        bound_mode: str = "tanh",
        gate_bias_init: float = 0.0,
        composition_gate_bias: float = 0.0,
        composition_gate_max: float = 1.0,
        composition_residual_scale: float = 1.0,
        composition_activation_threshold: float = 0.0,
        hard_routing_threshold: float = 0.0,
        feature_repair_enabled: bool = False,
        feature_repair_hidden: int = 256,
        feature_repair_scale: float = 0.5,
        feature_repair_zero_init: bool = True,
        feature_repair_hard_only: bool = False,
        repair_probe_enabled: bool = False,
        repair_probe_hidden: int = 128,
        lang_alias_map: dict[str, str] | None = None,
        domain_alias_map: dict[str, str] | None = None,
    ):
        super().__init__()
        self.base_head = base_head
        self.hidden_dim = int(hidden_dim)
        self.use_shared = bool(use_shared)
        self.use_lang_branch = bool(use_lang_branch)
        self.use_domain_branch = bool(use_domain_branch)
        self.use_composition = bool(use_composition)
        self.detach_base_prediction = bool(detach_base_prediction)
        self.clamp_delta = float(clamp_delta)
        self.bound_mode = str(bound_mode).strip().lower()
        self.condition_source = str(condition_source or "metadata").strip().lower()
        if self.condition_source not in {"metadata", "soft_text"}:
            raise ValueError(f"unsupported condition_source: {self.condition_source}")
        self.soft_condition_hidden = int(soft_condition_hidden)
        self.composition_gate_max = float(composition_gate_max)
        self.composition_residual_scale = float(composition_residual_scale)
        self.composition_activation_threshold = float(composition_activation_threshold)
        self.hard_routing_threshold = float(hard_routing_threshold)
        self.feature_repair_enabled = bool(feature_repair_enabled)
        self.feature_repair_scale = float(feature_repair_scale)
        self.feature_repair_hard_only = bool(feature_repair_hard_only)
        self.repair_probe_enabled = bool(repair_probe_enabled)
        self.runtime_composition_gate_max: float | None = None
        self.runtime_composition_residual_scale: float | None = None
        self.lang_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (lang_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.domain_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (domain_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.lang_to_idx = self._build_value_to_idx(langs, self.lang_alias_map)
        self.domain_to_idx = self._build_value_to_idx(domains, self.domain_alias_map)
        self.lang_embedding = nn.Embedding(len(self.lang_to_idx), embed_dim, padding_idx=0)
        self.domain_embedding = nn.Embedding(len(self.domain_to_idx), embed_dim, padding_idx=0)
        if self.condition_source == "soft_text":
            self.soft_condition_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.soft_condition_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.soft_condition_hidden, embed_dim * 2),
            )
        else:
            self.soft_condition_proj = None
        self.feature_repair = None
        if self.feature_repair_enabled:
            self.feature_repair = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, int(feature_repair_hidden)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(feature_repair_hidden), self.hidden_dim),
            )
            if feature_repair_zero_init:
                nn.init.zeros_(self.feature_repair[-1].weight)
                nn.init.zeros_(self.feature_repair[-1].bias)
        self.repair_probe = None
        if self.repair_probe_enabled:
            self.repair_probe = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, int(repair_probe_hidden)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(repair_probe_hidden), 2),
            )

        self.shared_trunk = None
        self.shared_out = None
        if self.use_shared:
            self.shared_trunk = nn.Sequential(
                nn.Linear(self.hidden_dim + 2, shared_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_hidden, shared_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.shared_out = nn.Linear(shared_hidden, 2)

        self.lang_trunk = None
        self.lang_out = None
        if self.use_lang_branch:
            self.lang_trunk = nn.Sequential(
                nn.Linear(self.hidden_dim + 2 + embed_dim, branch_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(branch_hidden, branch_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.lang_out = nn.Linear(branch_hidden, 2)
        self.domain_trunk = None
        self.domain_out = None
        if self.use_domain_branch:
            self.domain_trunk = nn.Sequential(
                nn.Linear(self.hidden_dim + 2 + embed_dim, branch_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(branch_hidden, branch_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.domain_out = nn.Linear(branch_hidden, 2)
        self.comp_trunk = None
        self.comp_out = None
        if self.use_composition:
            self.comp_trunk = nn.Sequential(
                nn.Linear(self.hidden_dim + 2 + 2 * embed_dim, composition_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(composition_hidden, composition_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.comp_out = nn.Linear(composition_hidden, 2)

        gate_dim = 3 if self.use_composition else 2
        self.gate_proj = nn.Linear(embed_dim * 2, gate_dim)

        for layer in [self.shared_out, self.lang_out, self.domain_out, self.comp_out]:
            if layer is not None:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_bias_init))
        if self.use_composition and self.gate_proj.bias is not None:
            with torch.no_grad():
                self.gate_proj.bias[-1].add_(float(composition_gate_bias))

        self._last_cache: dict[str, torch.Tensor | float | None] = {}

    def set_runtime_controls(
        self,
        *,
        composition_gate_max: float | None = None,
        composition_residual_scale: float | None = None,
    ) -> None:
        self.runtime_composition_gate_max = (
            None if composition_gate_max is None else float(composition_gate_max)
        )
        self.runtime_composition_residual_scale = (
            None if composition_residual_scale is None else float(composition_residual_scale)
        )

    def clear_runtime_controls(self) -> None:
        self.runtime_composition_gate_max = None
        self.runtime_composition_residual_scale = None

    def _base_forward(
        self,
        source_features: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(self.base_head, "forward_with_metadata"):
            return self.base_head.forward_with_metadata(
                source_features,
                batch_lang=batch_lang,
                batch_domain=batch_domain,
            )
        return self.base_head(source_features)

    def _bound_delta(self, delta_raw: torch.Tensor) -> torch.Tensor:
        if self.bound_mode == "clip":
            return torch.clamp(delta_raw, -self.clamp_delta, self.clamp_delta)
        scale = max(self.clamp_delta, 1e-9)
        return torch.tanh(delta_raw / scale) * scale

    def _resolve_condition(
        self,
        source_features: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None,
        batch_domain: list[str] | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.condition_source == "soft_text":
            soft_cond = self.soft_condition_proj(source_features.to(self.soft_condition_proj[1].weight.dtype))
            soft_cond = soft_cond.to(source_features.dtype)
            return torch.chunk(soft_cond, 2, dim=-1)
        if batch_lang is None:
            batch_lang = ["__unk__"] * source_features.size(0)
        if batch_domain is None:
            batch_domain = ["__unk__"] * source_features.size(0)
        lang_ids, domain_ids = self.encode_group_ids(
            batch_lang,
            batch_domain,
            device=source_features.device,
        )
        lang_emb = self.lang_embedding(lang_ids).to(source_features.dtype)
        domain_emb = self.domain_embedding(domain_ids).to(source_features.dtype)
        return lang_emb, domain_emb

    def _resolve_condition_from_routing_logits(
        self,
        routing_lang_logits: torch.Tensor,
        routing_domain_logits: torch.Tensor,
        source_features: torch.Tensor,
        *,
        routing_temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = max(float(routing_temperature or 1.0), 1e-6)
        lang_weight = self.lang_embedding.weight
        domain_weight = self.domain_embedding.weight
        if routing_lang_logits.size(-1) != lang_weight.size(0):
            raise ValueError(
                f"routing_lang_logits dim mismatch: {routing_lang_logits.size(-1)} != {lang_weight.size(0)}"
            )
        if routing_domain_logits.size(-1) != domain_weight.size(0):
            raise ValueError(
                f"routing_domain_logits dim mismatch: {routing_domain_logits.size(-1)} != {domain_weight.size(0)}"
            )
        lang_post = torch.softmax(routing_lang_logits.to(lang_weight.dtype) / temperature, dim=-1)
        domain_post = torch.softmax(routing_domain_logits.to(domain_weight.dtype) / temperature, dim=-1)
        lang_emb = lang_post @ lang_weight
        domain_emb = domain_post @ domain_weight
        return (
            lang_emb.to(source_features.dtype),
            domain_emb.to(source_features.dtype),
            lang_post.to(source_features.dtype),
            domain_post.to(source_features.dtype),
        )

    def _branch_delta(self, trunk: nn.Module | None, out_proj: nn.Linear | None, x: torch.Tensor) -> torch.Tensor:
        if trunk is None or out_proj is None:
            return torch.zeros(x.size(0), 2, device=x.device, dtype=x.dtype)
        hidden = trunk(x.to(next(trunk.parameters()).dtype))
        delta = out_proj(hidden).to(x.dtype)
        return self._bound_delta(delta)

    def _repair_features(
        self,
        source_features: torch.Tensor,
        hard_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_features = source_features.detach()
        if self.feature_repair is None or max(self.feature_repair_scale, 0.0) <= 0.0:
            zero_delta = torch.zeros_like(base_features)
            return base_features, zero_delta

        repair_hidden = self.feature_repair(
            base_features.to(next(self.feature_repair.parameters()).dtype)
        ).to(base_features.dtype)
        repair_delta = torch.tanh(repair_hidden) * max(self.feature_repair_scale, 0.0)
        if self.feature_repair_hard_only:
            if hard_mask is not None:
                repair_delta = repair_delta * hard_mask
            else:
                repair_delta = torch.zeros_like(base_features)
        return base_features + repair_delta, repair_delta

    def forward_with_metadata(
        self,
        source_features: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
        sample_hardness: torch.Tensor | None = None,
        sample_hard_mask: torch.Tensor | None = None,
        repaired_features: torch.Tensor | None = None,
        routing_lang_logits: torch.Tensor | None = None,
        routing_domain_logits: torch.Tensor | None = None,
        routing_temperature: float | None = None,
    ) -> torch.Tensor:
        base_pred = self._base_forward(
            source_features,
            batch_lang=batch_lang,
            batch_domain=batch_domain,
        )
        pred_input = base_pred.detach() if self.detach_base_prediction else base_pred

        hard_mask = None
        if sample_hard_mask is not None:
            hard_mask = sample_hard_mask.to(device=source_features.device, dtype=source_features.dtype).view(-1, 1)
        elif sample_hardness is not None and self.hard_routing_threshold > 0:
            hard_mask = (
                sample_hardness.to(device=source_features.device, dtype=source_features.dtype).view(-1, 1)
                >= float(self.hard_routing_threshold)
            ).to(source_features.dtype)

        base_features = source_features.detach()
        if repaired_features is None:
            repaired_features, repair_delta = self._repair_features(source_features, hard_mask=hard_mask)
        else:
            repaired_features = repaired_features.to(device=source_features.device, dtype=source_features.dtype)
            repair_delta = repaired_features - base_features
        routing_lang_post = None
        routing_domain_post = None
        if routing_lang_logits is not None and routing_domain_logits is not None:
            lang_emb, domain_emb, routing_lang_post, routing_domain_post = (
                self._resolve_condition_from_routing_logits(
                    routing_lang_logits,
                    routing_domain_logits,
                    repaired_features,
                    routing_temperature=routing_temperature,
                )
            )
        else:
            lang_emb, domain_emb = self._resolve_condition(repaired_features, batch_lang, batch_domain)
        cond = torch.cat([lang_emb, domain_emb], dim=-1)
        gate_logits = self.gate_proj(cond.to(self.gate_proj.weight.dtype)).to(source_features.dtype)
        gates = torch.sigmoid(gate_logits)
        lang_gate = gates[:, 0:1]
        domain_gate = gates[:, 1:2]
        comp_gate = gates[:, 2:3] if self.use_composition else None

        if hard_mask is not None:
            lang_gate = lang_gate * hard_mask
            domain_gate = domain_gate * hard_mask
            if comp_gate is not None:
                comp_gate = comp_gate * hard_mask

        if comp_gate is not None and self.composition_activation_threshold > 0:
            comp_gate = comp_gate * (comp_gate >= float(self.composition_activation_threshold)).to(comp_gate.dtype)

        composition_gate_max = (
            self.composition_gate_max
            if self.runtime_composition_gate_max is None
            else self.runtime_composition_gate_max
        )
        composition_residual_scale = (
            self.composition_residual_scale
            if self.runtime_composition_residual_scale is None
            else self.runtime_composition_residual_scale
        )
        if comp_gate is not None and composition_gate_max != 1.0:
            comp_gate = comp_gate * max(float(composition_gate_max), 0.0)

        shared_delta = self._branch_delta(
            self.shared_trunk,
            self.shared_out,
            torch.cat([repaired_features, pred_input], dim=-1),
        ) if self.use_shared else torch.zeros_like(base_pred)
        lang_delta = (
            lang_gate
            * self._branch_delta(
                self.lang_trunk,
                self.lang_out,
                torch.cat([repaired_features, pred_input, lang_emb], dim=-1),
            )
            if self.use_lang_branch
            else torch.zeros_like(base_pred)
        )
        domain_delta = (
            domain_gate
            * self._branch_delta(
                self.domain_trunk,
                self.domain_out,
                torch.cat([repaired_features, pred_input, domain_emb], dim=-1),
            )
            if self.use_domain_branch
            else torch.zeros_like(base_pred)
        )
        if self.use_composition and comp_gate is not None:
            comp_delta = comp_gate * self._branch_delta(
                self.comp_trunk,
                self.comp_out,
                torch.cat([repaired_features, pred_input, lang_emb, domain_emb], dim=-1),
            )
            if composition_residual_scale != 1.0:
                comp_delta = comp_delta * float(composition_residual_scale)
        else:
            comp_delta = torch.zeros_like(base_pred)

        total_delta = shared_delta + lang_delta + domain_delta + comp_delta
        final_pred = base_pred + total_delta
        repair_probe_pred = None
        if self.repair_probe is not None:
            repair_probe_pred = self.repair_probe(
                repaired_features.to(next(self.repair_probe.parameters()).dtype)
            ).to(base_pred.dtype)
        self._last_cache = {
            "base_features": base_features,
            "repaired_features": repaired_features,
            "feature_repair_delta": repair_delta,
            "base_pred": base_pred,
            "shared_delta": shared_delta,
            "lang_delta": lang_delta,
            "domain_delta": domain_delta,
            "comp_delta": comp_delta,
            "total_delta": total_delta,
            "lang_gate": lang_gate,
            "domain_gate": domain_gate,
            "comp_gate": comp_gate,
            "lang_emb": lang_emb,
            "domain_emb": domain_emb,
            "routing_lang_logits": None if routing_lang_logits is None else routing_lang_logits,
            "routing_domain_logits": None if routing_domain_logits is None else routing_domain_logits,
            "routing_lang_post": routing_lang_post,
            "routing_domain_post": routing_domain_post,
            "repair_probe_pred": repair_probe_pred,
            "hard_mask": hard_mask,
            "composition_gate_max": float(composition_gate_max),
            "composition_residual_scale": float(composition_residual_scale),
        }
        return final_pred

    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_metadata(source_features, None, None)

    def predict(self, source_features: torch.Tensor) -> torch.Tensor:
        out = self.forward_with_metadata(source_features, None, None).clamp(1.0, 9.0)
        return (out * 100).round() / 100

    def regularization_loss(self) -> torch.Tensor | None:
        if hasattr(self.base_head, "regularization_loss"):
            return self.base_head.regularization_loss()
        return None


class _VAMoEExpert(nn.Module):
    """Single expert MLP used by the MoE VA head."""

    def __init__(self, hidden_dim: int, expert_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class VAMixtureOfExpertsHead(nn.Module):
    """Minimal pooled-feature MoE head for VA regression."""

    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2,
                 expert_hidden: int = 256, dropout: float = 0.1,
                 router_aux_weight: float = 0.01, router_z_weight: float = 0.0):
        super().__init__()
        self.num_experts = max(int(num_experts), 1)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.router_aux_weight = float(router_aux_weight)
        self.router_z_weight = float(router_z_weight)
        self.router = nn.Linear(hidden_dim, self.num_experts)
        self.experts = nn.ModuleList(
            _VAMoEExpert(hidden_dim, expert_hidden=expert_hidden, dropout=dropout)
            for _ in range(self.num_experts)
        )
        self._last_router_logits = None
        self._last_router_probs = None
        self._last_dispatch_mask = None

    def _compute_mixture_weights(self, router_logits: torch.Tensor) -> torch.Tensor:
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        if self.top_k >= self.num_experts:
            dispatch_mask = torch.ones_like(router_probs)
            mixture_weights = router_probs
        else:
            topk_scores, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
            dispatch_mask = torch.zeros_like(router_probs)
            dispatch_mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_scores))
            mixture_weights = torch.zeros_like(router_probs)
            mixture_weights.scatter_(dim=-1, index=topk_indices, src=topk_scores)
            mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        self._last_router_logits = router_logits
        self._last_router_probs = router_probs
        self._last_dispatch_mask = dispatch_mask
        return mixture_weights.to(router_logits.dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(h)
        mixture_weights = self._compute_mixture_weights(router_logits)
        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * mixture_weights.unsqueeze(-1), dim=1)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        out = self.forward(h).clamp(1.0, 9.0)
        return (out * 100).round() / 100

    def load_balancing_loss(self) -> torch.Tensor | None:
        if self._last_router_probs is None or self._last_dispatch_mask is None:
            return None
        importance = self._last_router_probs.mean(dim=0)
        load = self._last_dispatch_mask.mean(dim=0)
        return self.num_experts * torch.sum(importance * load) - 1.0

    def router_z_loss(self) -> torch.Tensor | None:
        if self._last_router_logits is None:
            return None
        return torch.mean(torch.logsumexp(self._last_router_logits.float(), dim=-1).pow(2))

    def regularization_loss(self) -> torch.Tensor | None:
        total = None
        if self.router_aux_weight > 0:
            aux_loss = self.load_balancing_loss()
            if aux_loss is not None:
                total = self.router_aux_weight * aux_loss if total is None else total + self.router_aux_weight * aux_loss
        if self.router_z_weight > 0:
            z_loss = self.router_z_loss()
            if z_loss is not None:
                total = self.router_z_weight * z_loss if total is None else total + self.router_z_weight * z_loss
        return total


class VAUncertaintyHead(nn.Module):
    """Uncertainty-aware VA regression: outputs mu_v, mu_a, log_sigma_v, log_sigma_a."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, 2)
        self.fc_log_sigma = nn.Linear(hidden_dim, 2)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc_mu(h), self.fc_log_sigma(h)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        mu = self.fc_mu(h).clamp(1.0, 9.0)
        return (mu * 100).round() / 100

    def predict_with_sigma(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(h).clamp(1.0, 9.0)
        sigma = self.fc_log_sigma(h).exp()
        return (mu * 100).round() / 100, sigma


class PrototypeDistHead(nn.Module):
    """原型分布预测 head：输出 K 个 logits."""
    def __init__(self, hidden_dim: int, n_clusters: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_clusters)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            h = self.dropout(h)
        return self.fc(h)


class ArousalOrdinalHead(nn.Module):
    """Ordinal regression head for arousal."""
    def __init__(self, hidden_dim: int, n_thresholds: int = 4, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, n_thresholds),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class PolarityHead(nn.Module):
    """D39: Sentiment polarity classification head (neg/neu/pos from V)."""
    def __init__(self, hidden_dim: int, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class ArousalQuantileHead(nn.Module):
    """Multi-quantile head for arousal. Outputs Q quantile predictions."""
    def __init__(self, hidden_dim: int, quantiles: list[float] | None = None,
                 dropout: float = 0.1):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.register_buffer("tau", torch.tensor(quantiles, dtype=torch.float32))
        n_quantiles = len(quantiles)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, n_quantiles),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns (B, Q) quantile predictions."""
        return self.head(h)

    def predict_median(self, h: torch.Tensor) -> torch.Tensor:
        """Return median quantile prediction (B,). For future inference use."""
        preds = self.head(h)  # (B, Q)
        # Find index closest to 0.5
        median_idx = (self.tau - 0.5).abs().argmin().item()
        return preds[:, median_idx]


class PivotHardnessGateHead(nn.Module):
    """Small MLP that maps source/pivot pair features to a gate logit."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features).squeeze(-1)


class PivotAResidualHead(nn.Module):
    """Small MLP that predicts an arousal residual from source/pivot pair features."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features).squeeze(-1)


class PivotTrustResidualHead(nn.Module):
    """Shared pair encoder + trust gate + arousal residual head.

    This is the minimal integrated A3-v3 module:
    - one shared pair encoder over source/pivot pair features
    - one trust gate head that decides whether correction should apply
    - one residual head that predicts how much to correct arousal
    """

    def __init__(self, input_dim: int = 5, pair_hidden: int = 16, dropout: float = 0.0):
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.Linear(input_dim, pair_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, pair_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate_head = nn.Linear(pair_hidden, 1)
        self.residual_head = nn.Linear(pair_hidden, 1)

    def forward(self, pair_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pair = self.pair_encoder(pair_features)
        gate_logits = self.gate_head(z_pair).squeeze(-1)
        delta_a = self.residual_head(z_pair).squeeze(-1)
        return gate_logits, delta_a


class VAResidualCorrectorHead(nn.Module):
    """Bounded VA residual head with optional learned repair gate.

    This implements the explicit residual form:
      y_final = y_base + gate(x) * bounded_delta(x)

    The residual branch is zero-initialized so the module starts as an identity
    mapping on top of the frozen or semi-frozen W1 predictor.
    """

    def __init__(
        self,
        hidden_dim: int,
        corrector_hidden: int = 128,
        dropout: float = 0.1,
        *,
        output_dim: int = 2,
        learned_gate: bool = True,
        per_dim_gate: bool = False,
        zero_init: bool = True,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.learned_gate = bool(learned_gate)
        self.per_dim_gate = bool(per_dim_gate)
        gate_dim = self.output_dim if self.per_dim_gate else 1

        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + 2, corrector_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(corrector_hidden, corrector_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(corrector_hidden, self.output_dim)
        self.gate_head = nn.Linear(corrector_hidden, gate_dim) if self.learned_gate else None

        if zero_init:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
            if self.gate_head is not None:
                nn.init.zeros_(self.gate_head.weight)
                nn.init.constant_(self.gate_head.bias, float(gate_bias_init))

    def forward(self, source_features: torch.Tensor, source_pred_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.cat([source_features, source_pred_raw], dim=-1)
        z = self.trunk(x)
        delta_raw = self.delta_head(z)
        gate_logits = self.gate_head(z) if self.gate_head is not None else None
        return delta_raw, gate_logits


class _VAResidualExpert(nn.Module):
    """Single expert used by the group-routed VA residual head."""

    def __init__(self, input_dim: int, expert_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupRoutedVAResidualHead(nn.Module, _GroupEncodingMixin):
    """Group-conditioned residual corrector with optional pivot pair features.

    Router inputs are dominated by language/domain embeddings so the module can
    learn coarse correction regions, while experts see a compact source feature
    projection and optional pivot-derived features.
    """

    def __init__(
        self,
        hidden_dim: int,
        langs: list[str] | None = None,
        domains: list[str] | None = None,
        feature_hidden: int = 128,
        group_embed_dim: int = 16,
        router_hidden: int = 32,
        expert_hidden: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        pair_input_dim: int = 0,
        dropout: float = 0.1,
        router_aux_weight: float = 0.01,
        router_z_weight: float = 0.0,
        lang_alias_map: dict[str, str] | None = None,
        domain_alias_map: dict[str, str] | None = None,
    ):
        super().__init__()
        self.num_experts = max(int(num_experts), 1)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.pair_input_dim = max(int(pair_input_dim), 0)
        self.router_aux_weight = float(router_aux_weight)
        self.router_z_weight = float(router_z_weight)

        self.lang_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (lang_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.domain_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (domain_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }

        self.lang_to_idx = {"__unk__": 0}
        for value in langs or []:
            value = self._canonicalize_value(value, self.lang_alias_map)
            if value and value not in self.lang_to_idx:
                self.lang_to_idx[value] = len(self.lang_to_idx)

        self.domain_to_idx = {"__unk__": 0}
        for value in domains or []:
            value = self._canonicalize_value(value, self.domain_alias_map)
            if value and value not in self.domain_to_idx:
                self.domain_to_idx[value] = len(self.domain_to_idx)

        self.lang_embedding = nn.Embedding(len(self.lang_to_idx), group_embed_dim, padding_idx=0)
        self.domain_embedding = nn.Embedding(len(self.domain_to_idx), group_embed_dim, padding_idx=0)

        self.source_proj = nn.Sequential(
            nn.Linear(hidden_dim + 2, feature_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        router_input_dim = group_embed_dim * 2 + 2 + self.pair_input_dim
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, self.num_experts),
        )

        expert_input_dim = feature_hidden + group_embed_dim * 2 + self.pair_input_dim
        self.experts = nn.ModuleList(
            _VAResidualExpert(expert_input_dim, expert_hidden=expert_hidden, dropout=dropout)
            for _ in range(self.num_experts)
        )

        self._last_router_logits = None
        self._last_router_probs = None
        self._last_dispatch_mask = None

    def _compute_mixture_weights(self, router_logits: torch.Tensor) -> torch.Tensor:
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        if self.top_k >= self.num_experts:
            dispatch_mask = torch.ones_like(router_probs)
            mixture_weights = router_probs
        else:
            topk_scores, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
            dispatch_mask = torch.zeros_like(router_probs)
            dispatch_mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_scores))
            mixture_weights = torch.zeros_like(router_probs)
            mixture_weights.scatter_(dim=-1, index=topk_indices, src=topk_scores)
            mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        self._last_router_logits = router_logits
        self._last_router_probs = router_probs
        self._last_dispatch_mask = dispatch_mask
        return mixture_weights.to(router_logits.dtype)

    def forward(
        self,
        source_features: torch.Tensor,
        source_pred_raw: torch.Tensor,
        lang_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        pair_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pair_input_dim > 0:
            if pair_features is None:
                pair_features = torch.zeros(
                    source_features.size(0),
                    self.pair_input_dim,
                    device=source_features.device,
                    dtype=source_features.dtype,
                )
            elif pair_features.shape[-1] != self.pair_input_dim:
                raise ValueError(
                    f"pair_features has dim={pair_features.shape[-1]} but expected {self.pair_input_dim}"
                )

        source_repr = self.source_proj(torch.cat([source_features, source_pred_raw], dim=-1))
        lang_emb = self.lang_embedding(lang_ids)
        domain_emb = self.domain_embedding(domain_ids)
        group_repr = torch.cat([lang_emb, domain_emb], dim=-1).to(source_repr.dtype)

        if self.pair_input_dim > 0:
            router_input = torch.cat([group_repr, source_pred_raw, pair_features], dim=-1)
            expert_input = torch.cat([source_repr, group_repr, pair_features], dim=-1)
        else:
            router_input = torch.cat([group_repr, source_pred_raw], dim=-1)
            expert_input = torch.cat([source_repr, group_repr], dim=-1)

        router_logits = self.router(router_input)
        mixture_weights = self._compute_mixture_weights(router_logits)
        expert_outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * mixture_weights.unsqueeze(-1), dim=1)

    def load_balancing_loss(self) -> torch.Tensor | None:
        if self._last_router_probs is None or self._last_dispatch_mask is None:
            return None
        importance = self._last_router_probs.mean(dim=0)
        load = self._last_dispatch_mask.mean(dim=0)
        return self.num_experts * torch.sum(importance * load) - 1.0

    def router_z_loss(self) -> torch.Tensor | None:
        if self._last_router_logits is None:
            return None
        return torch.mean(torch.logsumexp(self._last_router_logits.float(), dim=-1).pow(2))

    def regularization_loss(self) -> torch.Tensor | None:
        total = None
        if self.router_aux_weight > 0:
            aux_loss = self.load_balancing_loss()
            if aux_loss is not None:
                total = self.router_aux_weight * aux_loss if total is None else total + self.router_aux_weight * aux_loss
        if self.router_z_weight > 0:
            z_loss = self.router_z_loss()
            if z_loss is not None:
                total = self.router_z_weight * z_loss if total is None else total + self.router_z_weight * z_loss
        return total


class SharedPlusRoutedVAResidualHead(nn.Module, _GroupEncodingMixin):
    """Shared residual branch plus group-routed residual experts."""

    def __init__(
        self,
        hidden_dim: int,
        langs: list[str] | None = None,
        domains: list[str] | None = None,
        shared_hidden: int = 128,
        feature_hidden: int = 128,
        group_embed_dim: int = 16,
        router_hidden: int = 32,
        expert_hidden: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        pair_input_dim: int = 0,
        dropout: float = 0.1,
        router_aux_weight: float = 0.01,
        router_z_weight: float = 0.0,
        shared_learned_gate: bool = True,
        shared_per_dim_gate: bool = False,
        shared_gate_bias_init: float = 0.0,
        shared_bound_mode: str = "tanh",
        shared_clamp_delta: float = 1.0,
        total_clamp_delta: float = 1.0,
        lang_alias_map: dict[str, str] | None = None,
        domain_alias_map: dict[str, str] | None = None,
    ):
        super().__init__()
        self.shared_bound_mode = str(shared_bound_mode).strip().lower()
        self.shared_clamp_delta = float(shared_clamp_delta)
        self.total_clamp_delta = float(total_clamp_delta)
        self.lang_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (lang_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.domain_alias_map = {
            str(k).strip(): str(v).strip()
            for k, v in (domain_alias_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.lang_to_idx = self._build_value_to_idx(langs, self.lang_alias_map)
        self.domain_to_idx = self._build_value_to_idx(domains, self.domain_alias_map)
        self.shared_head = VAResidualCorrectorHead(
            hidden_dim=hidden_dim,
            corrector_hidden=shared_hidden,
            dropout=dropout,
            output_dim=2,
            learned_gate=shared_learned_gate,
            per_dim_gate=shared_per_dim_gate,
            zero_init=True,
            gate_bias_init=shared_gate_bias_init,
        )
        self.routed_head = GroupRoutedVAResidualHead(
            hidden_dim=hidden_dim,
            langs=langs,
            domains=domains,
            feature_hidden=feature_hidden,
            group_embed_dim=group_embed_dim,
            router_hidden=router_hidden,
            expert_hidden=expert_hidden,
            num_experts=num_experts,
            top_k=top_k,
            pair_input_dim=pair_input_dim,
            dropout=dropout,
            router_aux_weight=router_aux_weight,
            router_z_weight=router_z_weight,
            lang_alias_map=self.lang_alias_map,
            domain_alias_map=self.domain_alias_map,
        )
        self._last_shared_delta = None
        self._last_routed_delta = None

    def _bound_shared_delta(self, delta_raw: torch.Tensor, gate_logits: torch.Tensor | None) -> torch.Tensor:
        if self.shared_bound_mode == "clip":
            delta = torch.clamp(delta_raw, -self.shared_clamp_delta, self.shared_clamp_delta)
        else:
            scale = max(self.shared_clamp_delta, 1e-9)
            delta = torch.tanh(delta_raw / scale) * scale
        if gate_logits is None:
            return delta
        gate = torch.sigmoid(gate_logits).to(delta.dtype)
        if gate.shape[-1] == 1 and delta.shape[-1] > 1:
            gate = gate.expand_as(delta)
        return gate * delta

    def forward(
        self,
        source_features: torch.Tensor,
        source_pred_raw: torch.Tensor,
        lang_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        pair_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shared_delta_raw, shared_gate_logits = self.shared_head(source_features, source_pred_raw)
        shared_delta = self._bound_shared_delta(shared_delta_raw, shared_gate_logits)
        routed_delta = self.routed_head(
            source_features=source_features,
            source_pred_raw=source_pred_raw,
            lang_ids=lang_ids,
            domain_ids=domain_ids,
            pair_features=pair_features,
        )
        total_delta = shared_delta + routed_delta
        if self.total_clamp_delta > 0:
            total_delta = torch.clamp(total_delta, -self.total_clamp_delta, self.total_clamp_delta)
        self._last_shared_delta = shared_delta
        self._last_routed_delta = routed_delta
        return total_delta

    def regularization_loss(self) -> torch.Tensor | None:
        return self.routed_head.regularization_loss()


class VARegionHead(nn.Module):
    """D40: VA region classification head for Two-Stage approach.

    Classifies samples into VA grid regions (e.g. 3x3=9 classes).
    Forces the model to learn decision boundaries in VA space,
    breaking the "predict mean" tendency that causes arousal collapse.
    """
    def __init__(self, hidden_dim: int, n_v_bins: int = 3, n_a_bins: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.n_v_bins = n_v_bins
        self.n_a_bins = n_a_bins
        n_classes = n_v_bins * n_a_bins
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter: down → act → dropout → up + residual."""
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 128,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up_proj(self.dropout(self.act(self.down_proj(h))))


class VADecoupledAdapterHead(nn.Module):
    """V/A decoupled adapter head: shared encoder → separate adapters → V and A."""
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 128,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.v_adapter = BottleneckAdapter(hidden_dim, bottleneck_dim, dropout, activation)
        self.a_adapter = BottleneckAdapter(hidden_dim, bottleneck_dim, dropout, activation)
        self.v_head = nn.Linear(hidden_dim, 1)
        self.a_head = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_v = self.v_adapter(h)
        h_a = self.a_adapter(h)
        v = self.v_head(h_v)
        a = self.a_head(h_a)
        return torch.cat([v, a], dim=-1)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        out = self.forward(h).clamp(1.0, 9.0)
        return (out * 100).round() / 100

    def get_adapter_features(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.v_adapter(h), self.a_adapter(h)



def _build_base_va_head(cfg: dict, hidden_dim: int):
    """Build the main VA prediction head from config."""
    use_uncertainty = cfg.get("uncertainty", False)
    head_hidden = cfg.get("head_hidden", 0)
    head_type = cfg.get("head_type", "linear")
    decoupled_cfg = cfg.get("decoupled_adapter", {})
    decoupled_enabled = decoupled_cfg.get("enabled", False)
    dual_head_cfg = cfg.get("dual_head_mlp", {})
    moe_cfg = cfg.get("moe_head", {})

    if use_uncertainty:
        return VAUncertaintyHead(hidden_dim)
    if decoupled_enabled or head_type == "decoupled_adapter":
        return VADecoupledAdapterHead(
            hidden_dim,
            bottleneck_dim=decoupled_cfg.get("bottleneck_dim", 128),
            dropout=decoupled_cfg.get("dropout", 0.1),
            activation=decoupled_cfg.get("activation", "relu"),
        )
    if head_type == "dual_mlp":
        return VADualHeadMLP(
            hidden_dim,
            shared_hidden=dual_head_cfg.get("shared_hidden", 512),
            branch_hidden=dual_head_cfg.get("branch_hidden", 256),
            head_dropout=dual_head_cfg.get("dropout", cfg.get("head_dropout", 0.1)),
        )
    if head_type == "moe":
        return VAMixtureOfExpertsHead(
            hidden_dim,
            num_experts=moe_cfg.get("num_experts", 4),
            top_k=moe_cfg.get("top_k", 2),
            expert_hidden=moe_cfg.get("expert_hidden", dual_head_cfg.get("branch_hidden", 256)),
            dropout=moe_cfg.get("dropout", cfg.get("head_dropout", 0.1)),
            router_aux_weight=moe_cfg.get("router_aux_weight", 0.01),
            router_z_weight=moe_cfg.get("router_z_weight", 0.0),
        )
    if head_type == "mlp" or head_hidden > 0:
        return VARegressionMLPHead(
            hidden_dim,
            head_hidden=head_hidden or dual_head_cfg.get("shared_hidden", 256),
            head_dropout=cfg.get("head_dropout", 0.1),
        )
    return VARegressionHead(hidden_dim)


def build_va_head(cfg: dict, hidden_dim: int):
    """Build the main VA prediction head from config."""
    base_head = _build_base_va_head(cfg, hidden_dim)
    meta_cfg = cfg.get("metadata_conditioning", {})
    if meta_cfg.get("enabled", False):
        langs = [str(v).strip() for v in meta_cfg.get("langs", []) if str(v).strip()]
        domains = [str(v).strip() for v in meta_cfg.get("domains", []) if str(v).strip()]
        if not langs:
            langs = infer_group_values_from_cfg(cfg, "lang")
        if not domains:
            domains = infer_group_values_from_cfg(cfg, "domain")

        base_head = MetadataConditionedHeadAdapter(
            base_head=base_head,
            hidden_dim=hidden_dim,
            langs=langs,
            domains=domains,
            embed_dim=int(meta_cfg.get("embed_dim", 16)),
            conditioner_hidden=int(meta_cfg.get("conditioner_hidden", 128)),
            dropout=float(meta_cfg.get("dropout", cfg.get("head_dropout", 0.1))),
            learned_gate=bool(meta_cfg.get("learned_gate", True)),
            lang_alias_map=meta_cfg.get("lang_alias_map", {}),
            domain_alias_map=meta_cfg.get("domain_alias_map", {}),
        )

    out_cfg = cfg.get("metadata_output_residual", {})
    factorized_out_cfg = cfg.get("factorized_output_residual", {})
    if factorized_out_cfg.get("enabled", False):
        langs = [str(v).strip() for v in factorized_out_cfg.get("langs", []) if str(v).strip()]
        domains = [str(v).strip() for v in factorized_out_cfg.get("domains", []) if str(v).strip()]
        if not langs:
            langs = infer_group_values_from_cfg(cfg, "lang")
        if not domains:
            domains = infer_group_values_from_cfg(cfg, "domain")
        return FactorizedOutputResidualHead(
            base_head=base_head,
            hidden_dim=hidden_dim,
            langs=langs,
            domains=domains,
            embed_dim=int(factorized_out_cfg.get("embed_dim", 16)),
            shared_hidden=int(factorized_out_cfg.get("shared_hidden", 128)),
            branch_hidden=int(factorized_out_cfg.get("branch_hidden", 128)),
            composition_hidden=int(factorized_out_cfg.get("composition_hidden", 128)),
            dropout=float(factorized_out_cfg.get("dropout", cfg.get("head_dropout", 0.1))),
            use_shared=bool(factorized_out_cfg.get("use_shared", True)),
            use_lang_branch=bool(factorized_out_cfg.get("use_lang_branch", True)),
            use_domain_branch=bool(factorized_out_cfg.get("use_domain_branch", True)),
            use_composition=bool(factorized_out_cfg.get("use_composition", True)),
            condition_source=str(factorized_out_cfg.get("condition_source", "metadata")),
            soft_condition_hidden=int(factorized_out_cfg.get("soft_condition_hidden", 128)),
            detach_base_prediction=bool(factorized_out_cfg.get("detach_base_prediction", True)),
            clamp_delta=float(factorized_out_cfg.get("clamp_delta", 1.0)),
            bound_mode=str(factorized_out_cfg.get("bound_mode", "tanh")),
            gate_bias_init=float(factorized_out_cfg.get("gate_bias_init", 0.0)),
            composition_gate_bias=float(factorized_out_cfg.get("composition_gate_bias", 0.0)),
            composition_gate_max=float(factorized_out_cfg.get("composition_gate_max", 1.0)),
            composition_residual_scale=float(factorized_out_cfg.get("composition_residual_scale", 1.0)),
            composition_activation_threshold=float(
                factorized_out_cfg.get("composition_activation_threshold", 0.0)
            ),
            hard_routing_threshold=float(factorized_out_cfg.get("hard_routing_threshold", 0.0)),
            feature_repair_enabled=bool(factorized_out_cfg.get("feature_repair_enabled", False)),
            feature_repair_hidden=int(factorized_out_cfg.get("feature_repair_hidden", 256)),
            feature_repair_scale=float(factorized_out_cfg.get("feature_repair_scale", 0.5)),
            feature_repair_zero_init=bool(factorized_out_cfg.get("feature_repair_zero_init", True)),
            feature_repair_hard_only=bool(factorized_out_cfg.get("feature_repair_hard_only", False)),
            repair_probe_enabled=bool(factorized_out_cfg.get("repair_probe_enabled", False)),
            repair_probe_hidden=int(factorized_out_cfg.get("repair_probe_hidden", 128)),
            lang_alias_map=factorized_out_cfg.get("lang_alias_map", {}),
            domain_alias_map=factorized_out_cfg.get("domain_alias_map", {}),
        )

    if not out_cfg.get("enabled", False):
        return base_head

    langs = [str(v).strip() for v in out_cfg.get("langs", []) if str(v).strip()]
    domains = [str(v).strip() for v in out_cfg.get("domains", []) if str(v).strip()]
    if not langs:
        langs = infer_group_values_from_cfg(cfg, "lang")
    if not domains:
        domains = infer_group_values_from_cfg(cfg, "domain")

    return MetadataConditionedOutputResidualHead(
        base_head=base_head,
        hidden_dim=hidden_dim,
        langs=langs,
        domains=domains,
        embed_dim=int(out_cfg.get("embed_dim", 16)),
        conditioner_hidden=int(out_cfg.get("conditioner_hidden", 128)),
        dropout=float(out_cfg.get("dropout", cfg.get("head_dropout", 0.1))),
        learned_gate=bool(out_cfg.get("learned_gate", True)),
        per_dim_gate=bool(out_cfg.get("per_dim_gate", True)),
        gate_bias_init=float(out_cfg.get("gate_bias_init", 0.0)),
        detach_base_prediction=bool(out_cfg.get("detach_base_prediction", True)),
        clamp_delta=float(out_cfg.get("clamp_delta", 0.5)),
        bound_mode=str(out_cfg.get("bound_mode", "tanh")),
        lang_alias_map=out_cfg.get("lang_alias_map", {}),
        domain_alias_map=out_cfg.get("domain_alias_map", {}),
    )
