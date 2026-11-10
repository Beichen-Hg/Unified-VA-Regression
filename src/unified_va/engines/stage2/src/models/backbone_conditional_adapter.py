"""Lightweight factorized conditional adapters for decoder backbones."""
from __future__ import annotations

import weakref
import math

import torch
import torch.nn as nn


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


class FactorizedConditionalResidualAdapter(nn.Module, _GroupEncodingMixin):
    """Shared + language + domain + optional composition residual adapter."""

    def __init__(
        self,
        hidden_dim: int,
        langs: list[str],
        domains: list[str],
        *,
        embed_dim: int = 16,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        use_shared: bool = True,
        use_composition: bool = False,
        composition_bottleneck_dim: int | None = None,
        composition_dropout: float | None = None,
        composition_gate_bias: float = 0.0,
        composition_gate_max: float = 1.0,
        composition_residual_scale: float = 1.0,
        composition_state_aware: bool = False,
        composition_state_hidden: int = 64,
        shared_residual_scale: float = 1.0,
        shared_branch_dropout: float = 0.0,
        confidence_gated_composition: bool = False,
        confidence_gate_scale: float = 1.0,
        confidence_gate_bias: float = 0.0,
        confidence_gate_min: float = 0.0,
        lang_alias_map: dict[str, str] | None = None,
        domain_alias_map: dict[str, str] | None = None,
        condition_source: str = "metadata",
        soft_condition_hidden: int = 128,
        soft_condition_pooling: str = "mean",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.use_shared = bool(use_shared)
        self.use_composition = bool(use_composition)
        self.composition_bottleneck_dim = int(composition_bottleneck_dim or bottleneck_dim)
        self.composition_gate_bias = float(composition_gate_bias)
        self.composition_gate_max = float(composition_gate_max)
        self.composition_residual_scale = float(composition_residual_scale)
        self.composition_state_aware = bool(composition_state_aware) and self.use_composition
        self.composition_state_hidden = int(composition_state_hidden)
        self.shared_residual_scale = float(shared_residual_scale)
        self.shared_branch_dropout = float(shared_branch_dropout)
        self.confidence_gated_composition = bool(confidence_gated_composition) and self.use_composition
        self.confidence_gate_scale = float(confidence_gate_scale)
        self.confidence_gate_bias = float(confidence_gate_bias)
        self.confidence_gate_min = float(confidence_gate_min)
        self.condition_source = str(condition_source or "metadata").strip().lower()
        if self.condition_source not in {"metadata", "soft_text"}:
            raise ValueError(f"unsupported condition_source: {self.condition_source}")
        self.soft_condition_hidden = int(soft_condition_hidden)
        self.soft_condition_pooling = str(soft_condition_pooling or "mean").strip().lower()
        if self.soft_condition_pooling not in {"mean", "masked_mean", "last_token"}:
            raise ValueError(f"unsupported soft_condition_pooling: {self.soft_condition_pooling}")
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
                nn.Linear(self.hidden_dim, self.soft_condition_hidden),
                nn.GELU(),
                nn.Linear(self.soft_condition_hidden, embed_dim * 2),
            )
        else:
            self.soft_condition_proj = None
        self.pre_norm = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.comp_dropout = nn.Dropout(composition_dropout if composition_dropout is not None else dropout)

        if self.use_shared:
            self.shared_down = nn.Linear(self.hidden_dim, self.bottleneck_dim)
            self.shared_up = nn.Linear(self.bottleneck_dim, self.hidden_dim)
        else:
            self.shared_down = None
            self.shared_up = None

        self.lang_down = nn.Linear(self.hidden_dim, self.bottleneck_dim)
        self.lang_up = nn.Linear(self.bottleneck_dim, self.hidden_dim)
        self.domain_down = nn.Linear(self.hidden_dim, self.bottleneck_dim)
        self.domain_up = nn.Linear(self.bottleneck_dim, self.hidden_dim)
        self.lang_scale = nn.Linear(embed_dim, self.bottleneck_dim)
        self.domain_scale = nn.Linear(embed_dim, self.bottleneck_dim)

        gate_dim = 2 if self.composition_state_aware else (3 if self.use_composition else 2)
        self.gate_proj = nn.Linear(embed_dim * 2, gate_dim)

        if self.use_composition:
            self.comp_down = nn.Linear(self.hidden_dim, self.composition_bottleneck_dim)
            self.comp_up = nn.Linear(self.composition_bottleneck_dim, self.hidden_dim)
            self.comp_scale = nn.Linear(embed_dim * 2, self.composition_bottleneck_dim)
            if self.composition_state_aware:
                self.comp_state_proj = nn.Sequential(
                    nn.Linear(self.hidden_dim + 3, self.composition_state_hidden),
                    nn.GELU(),
                    nn.Dropout(composition_dropout if composition_dropout is not None else dropout),
                )
                self.comp_gate_proj = nn.Linear(embed_dim * 2 + self.composition_state_hidden, 1)
            else:
                self.comp_state_proj = None
                self.comp_gate_proj = None
        else:
            self.comp_down = None
            self.comp_up = None
            self.comp_scale = None
            self.comp_state_proj = None
            self.comp_gate_proj = None

        for module in [self.shared_up, self.lang_up, self.domain_up, self.comp_up]:
            if module is not None:
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        if self.use_composition and self.composition_state_aware and self.comp_gate_proj is not None:
            nn.init.zeros_(self.comp_gate_proj.weight)
            nn.init.zeros_(self.comp_gate_proj.bias)
            if abs(self.composition_gate_bias) > 1e-12:
                with torch.no_grad():
                    self.comp_gate_proj.bias.add_(self.composition_gate_bias)
        elif self.use_composition and abs(self.composition_gate_bias) > 1e-12 and self.gate_proj.bias is not None:
            with torch.no_grad():
                self.gate_proj.bias[-1].add_(self.composition_gate_bias)

        self._last_gate_stats: dict[str, float] = {}
        self._last_forward_cache: dict[str, torch.Tensor | None] = {}
        self.runtime_shared_residual_scale: float | None = None
        self.runtime_shared_branch_dropout: float | None = None
        self.runtime_composition_gate_max: float | None = None
        self.runtime_composition_residual_scale: float | None = None
    def set_runtime_controls(
        self,
        *,
        shared_residual_scale: float | None = None,
        shared_branch_dropout: float | None = None,
        composition_gate_max: float | None = None,
        composition_residual_scale: float | None = None,
    ) -> None:
        self.runtime_shared_residual_scale = (
            None if shared_residual_scale is None else float(shared_residual_scale)
        )
        self.runtime_shared_branch_dropout = (
            None if shared_branch_dropout is None else float(shared_branch_dropout)
        )
        self.runtime_composition_gate_max = (
            None if composition_gate_max is None else float(composition_gate_max)
        )
        self.runtime_composition_residual_scale = (
            None if composition_residual_scale is None else float(composition_residual_scale)
        )

    def clear_runtime_controls(self) -> None:
        self.runtime_shared_residual_scale = None
        self.runtime_shared_branch_dropout = None
        self.runtime_composition_gate_max = None
        self.runtime_composition_residual_scale = None

    def _masked_mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def _last_token_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1]
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def _pool_for_soft_condition(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.soft_condition_pooling == "masked_mean":
            return self._masked_mean_pool(hidden_states, attention_mask)
        if self.soft_condition_pooling == "last_token":
            return self._last_token_pool(hidden_states, attention_mask)
        return hidden_states.mean(dim=1)

    def _pool_cache_tensor(
        self,
        tensor: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._masked_mean_pool(tensor, attention_mask)

    def _branch_delta(
        self,
        x: torch.Tensor,
        down: nn.Linear | None,
        up: nn.Linear | None,
        scale: torch.Tensor | None,
        dropout_module: nn.Module | None = None,
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        if down is None or up is None:
            return torch.zeros_like(x)
        x_in = x.to(down.weight.dtype)
        hidden = self.activation(down(x_in))
        if scale is not None:
            hidden = hidden * scale.to(hidden.dtype).unsqueeze(1)
        hidden = (dropout_module or self.dropout)(hidden)
        out = up(hidden).to(x.dtype)
        if residual_scale != 1.0:
            out = out * float(residual_scale)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_lang: list[str] | torch.Tensor | None = None,
        batch_domain: list[str] | torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.condition_source == "soft_text":
            pooled = self._pool_for_soft_condition(hidden_states, attention_mask).to(hidden_states.dtype)
            soft_cond = self.soft_condition_proj(pooled.to(self.soft_condition_proj[0].weight.dtype))
            soft_cond = soft_cond.to(hidden_states.dtype)
            lang_emb, domain_emb = torch.chunk(soft_cond, 2, dim=-1)
        else:
            lang_ids, domain_ids = self.encode_group_ids(
                batch_lang,
                batch_domain,
                device=hidden_states.device,
            )
            lang_emb = self.lang_embedding(lang_ids)
            domain_emb = self.domain_embedding(domain_ids)
        cond = torch.cat([lang_emb, domain_emb], dim=-1)
        route_logits = self.gate_proj(cond.to(self.gate_proj.weight.dtype)).to(hidden_states.dtype)
        gates = torch.sigmoid(route_logits)
        x = self.pre_norm(hidden_states.to(self.pre_norm.weight.dtype)).to(hidden_states.dtype)

        total_delta = torch.zeros_like(hidden_states)
        branch_names: list[str] = []

        if self.use_shared:
            shared_residual_scale = (
                self.shared_residual_scale
                if self.runtime_shared_residual_scale is None
                else self.runtime_shared_residual_scale
            )
            shared_delta = self._branch_delta(
                x,
                self.shared_down,
                self.shared_up,
                None,
                residual_scale=shared_residual_scale,
            )
            shared_branch_dropout = (
                self.shared_branch_dropout
                if self.runtime_shared_branch_dropout is None
                else self.runtime_shared_branch_dropout
            )
            shared_keep_mask = None
            if self.training and shared_branch_dropout > 0:
                keep_prob = max(1.0 - float(shared_branch_dropout), 1e-6)
                shared_keep_mask = (
                    torch.rand(shared_delta.size(0), 1, 1, device=shared_delta.device) < keep_prob
                ).to(shared_delta.dtype)
                shared_delta = shared_delta * shared_keep_mask / keep_prob
            total_delta = total_delta + shared_delta
            branch_names.append("shared")
        else:
            shared_delta = None
            shared_keep_mask = None

        lang_delta = self._branch_delta(
            x,
            self.lang_down,
            self.lang_up,
            torch.tanh(self.lang_scale(lang_emb.to(self.lang_scale.weight.dtype))),
        )
        lang_delta = gates[:, 0].unsqueeze(-1).unsqueeze(-1) * lang_delta
        total_delta = total_delta + lang_delta
        branch_names.append("lang")

        domain_delta = self._branch_delta(
            x,
            self.domain_down,
            self.domain_up,
            torch.tanh(self.domain_scale(domain_emb.to(self.domain_scale.weight.dtype))),
        )
        domain_delta = gates[:, 1].unsqueeze(-1).unsqueeze(-1) * domain_delta
        total_delta = total_delta + domain_delta
        branch_names.append("domain")

        if self.use_composition:
            if self.composition_state_aware and self.comp_gate_proj is not None and self.comp_state_proj is not None:
                pooled_mean = x.mean(dim=1)
                mean_abs = x.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
                token_std = x.float().std(dim=1, unbiased=False).mean(dim=-1, keepdim=True).to(x.dtype)
                pooled_norm = (pooled_mean.norm(dim=-1, keepdim=True) / math.sqrt(max(self.hidden_dim, 1))).to(x.dtype)
                state_input = torch.cat([pooled_mean, mean_abs, token_std, pooled_norm], dim=-1)
                state_feat = self.comp_state_proj(state_input.to(self.comp_state_proj[0].weight.dtype))
                comp_gate_input = torch.cat([cond.to(state_feat.dtype), state_feat], dim=-1)
                comp_gate_logit = self.comp_gate_proj(comp_gate_input).squeeze(-1).to(hidden_states.dtype)
                comp_gate = torch.sigmoid(comp_gate_logit)
            else:
                comp_gate_logit = route_logits[:, 2]
                comp_gate = gates[:, 2]
            comp_confidence = None
            if self.confidence_gated_composition:
                route_conf = 0.5 * (route_logits[:, 0].abs() + route_logits[:, 1].abs())
                comp_confidence = torch.sigmoid(
                    self.confidence_gate_scale * route_conf + self.confidence_gate_bias
                ).to(hidden_states.dtype)
                if self.confidence_gate_min != 0.0:
                    comp_confidence = self.confidence_gate_min + (1.0 - self.confidence_gate_min) * comp_confidence
                comp_gate = comp_gate * comp_confidence
            composition_gate_max = (
                self.composition_gate_max
                if self.runtime_composition_gate_max is None
                else self.runtime_composition_gate_max
            )
            if composition_gate_max != 1.0:
                comp_gate = comp_gate * max(composition_gate_max, 0.0)
            composition_residual_scale = (
                self.composition_residual_scale
                if self.runtime_composition_residual_scale is None
                else self.runtime_composition_residual_scale
            )
            comp_delta = self._branch_delta(
                x,
                self.comp_down,
                self.comp_up,
                torch.tanh(self.comp_scale(cond.to(self.comp_scale.weight.dtype))),
                dropout_module=self.comp_dropout,
                residual_scale=composition_residual_scale,
            )
            comp_delta = comp_gate.unsqueeze(-1).unsqueeze(-1) * comp_delta
            total_delta = total_delta + comp_delta
            branch_names.append("composition")
        else:
            comp_gate = None
            comp_delta = None
            comp_confidence = None

        self._last_gate_stats = {
            "lang": float(gates[:, 0].detach().mean().cpu()),
            "domain": float(gates[:, 1].detach().mean().cpu()),
        }
        if self.use_composition:
            self._last_gate_stats["composition"] = float(comp_gate.detach().mean().cpu())
        self._last_forward_cache = {
            "lang_emb": lang_emb,
            "domain_emb": domain_emb,
            "route_logit_lang": route_logits[:, 0],
            "route_logit_domain": route_logits[:, 1],
            "route_logit_composition": comp_gate_logit if self.use_composition else None,
            "gate_lang": gates[:, 0],
            "gate_domain": gates[:, 1],
            "gate_composition": comp_gate,
            "composition_confidence": comp_confidence,
            "shared_keep_mask": shared_keep_mask.squeeze(-1).squeeze(-1) if shared_keep_mask is not None else None,
            "shared_delta_pooled": (
                self._pool_cache_tensor(shared_delta, attention_mask) if shared_delta is not None else None
            ),
            "lang_delta_pooled": self._pool_cache_tensor(lang_delta, attention_mask),
            "domain_delta_pooled": self._pool_cache_tensor(domain_delta, attention_mask),
            "composition_delta_pooled": (
                self._pool_cache_tensor(comp_delta, attention_mask) if comp_delta is not None else None
            ),
            "total_delta_pooled": self._pool_cache_tensor(total_delta, attention_mask),
        }

        return hidden_states + total_delta

    def gate_stats(self) -> dict[str, float]:
        return dict(self._last_gate_stats)

    def forward_cache(self, detach: bool = False) -> dict[str, torch.Tensor | None]:
        if not detach:
            return dict(self._last_forward_cache)
        out: dict[str, torch.Tensor | None] = {}
        for key, value in self._last_forward_cache.items():
            if torch.is_tensor(value):
                out[key] = value.detach()
            else:
                out[key] = value
        return out


class PostCheckpointRepairResidualAdapter(nn.Module):
    """Tiny independent residual editor added only after the W1 checkpoint stage."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        bottleneck_dim: int = 128,
        gate_hidden: int = 64,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        zero_init: bool = True,
        gate_bias_init: float = -4.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.residual_scale = float(residual_scale)

        self.pre_norm = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.down = nn.Linear(self.hidden_dim, self.bottleneck_dim)
        self.mid = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.up = nn.Linear(self.bottleneck_dim, self.hidden_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )

        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
            final_gate = self.gate_mlp[-1]
            nn.init.zeros_(final_gate.weight)
            nn.init.constant_(final_gate.bias, float(gate_bias_init))

        self._last_gate_mean = 0.0
        self._last_forward_cache: dict[str, torch.Tensor | None] = {}

    def forward(self, hidden_states: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        scale = max(float(scale), 0.0)
        if scale <= 0.0:
            self._last_gate_mean = 0.0
            self._last_forward_cache = {
                "gate": hidden_states.new_zeros(hidden_states.size(0)),
            }
            return hidden_states

        x = self.pre_norm(hidden_states.to(self.pre_norm.weight.dtype)).to(hidden_states.dtype)
        pooled = x.mean(dim=1)
        gate = torch.sigmoid(
            self.gate_mlp(pooled.to(self.gate_mlp[0].weight.dtype))
        ).to(hidden_states.dtype)
        gate = gate * scale

        h = self.activation(self.down(x.to(self.down.weight.dtype)))
        h = self.dropout(h)
        h = self.activation(self.mid(h))
        h = self.dropout(h)
        delta = self.up(h).to(hidden_states.dtype)
        if self.residual_scale != 1.0:
            delta = delta * self.residual_scale

        self._last_gate_mean = float(gate.detach().mean().cpu())
        self._last_forward_cache = {
            "gate": gate.squeeze(-1),
        }
        return hidden_states + gate.unsqueeze(1) * delta

    def gate_stats(self) -> dict[str, float]:
        return {"repair": float(self._last_gate_mean)}

    def forward_cache(self, detach: bool = False) -> dict[str, torch.Tensor | None]:
        if not detach:
            return dict(self._last_forward_cache)
        out: dict[str, torch.Tensor | None] = {}
        for key, value in self._last_forward_cache.items():
            if torch.is_tensor(value):
                out[key] = value.detach()
            else:
                out[key] = value
        return out


class ConditionalDecoderLayerWrapper(nn.Module):
    """Wrap a decoder layer and apply a conditional residual adapter to its output."""

    def __init__(
        self,
        layer: nn.Module,
        adapter: FactorizedConditionalResidualAdapter | None,
        owner,
        repair_adapter: PostCheckpointRepairResidualAdapter | None = None,
        adapter_index: int | None = None,
    ):
        super().__init__()
        self.layer = layer
        self.adapter = adapter
        self.repair_adapter = repair_adapter
        self.adapter_index = adapter_index
        self._owner_ref = weakref.ref(owner)

    def __getattr__(self, name: str):
        if name in {"layer", "adapter", "repair_adapter", "adapter_index", "_owner_ref"}:
            return super().__getattr__(name)
        layer = super().__getattr__("layer")
        try:
            return getattr(layer, name)
        except AttributeError:
            raise

    def forward(self, *args, **kwargs):
        outputs = self.layer(*args, **kwargs)
        owner = self._owner_ref()
        batch_lang, batch_domain = (None, None) if owner is None else owner.get_backbone_conditioning()
        attention_mask = None if owner is None else owner.get_backbone_attention_mask()
        repair_scale = 1.0 if owner is None else owner.get_post_ckpt_repair_adapter_scale()

        def _apply_wrapped_adapters(hidden_states: torch.Tensor) -> torch.Tensor:
            if owner is not None and owner.training and self.adapter_index is not None:
                ema_adapter = owner.get_ema_teacher_adapter(self.adapter_index)
                if ema_adapter is not None:
                    with torch.no_grad():
                        ema_adapter(
                            hidden_states.detach(),
                            batch_lang=batch_lang,
                            batch_domain=batch_domain,
                            attention_mask=attention_mask,
                        )
            if self.adapter is not None:
                hidden_states = self.adapter(
                    hidden_states,
                    batch_lang=batch_lang,
                    batch_domain=batch_domain,
                    attention_mask=attention_mask,
                )
            if self.repair_adapter is not None:
                hidden_states = self.repair_adapter(hidden_states, scale=repair_scale)
            return hidden_states

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            hidden_states = _apply_wrapped_adapters(hidden_states)
            return (hidden_states, *outputs[1:])
        if torch.is_tensor(outputs):
            return _apply_wrapped_adapters(outputs)
        return outputs
