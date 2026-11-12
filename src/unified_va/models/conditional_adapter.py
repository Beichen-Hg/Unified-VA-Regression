"""The complete global/language/domain/compound residual transformation."""

import torch
from torch import nn


class ConditionalAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim,
        bottleneck_dim=64,
        embed_dim=16,
        dropout=0.1,
        soft_condition_hidden=128,
    ):
        super().__init__()
        self.soft_condition_proj = nn.Sequential(
            nn.Linear(hidden_dim, soft_condition_hidden),
            nn.GELU(),
            nn.Linear(soft_condition_hidden, 2 * embed_dim),
        )
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.activation, self.dropout, self.comp_dropout = (
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Dropout(dropout),
        )
        self.shared_down, self.shared_up = nn.Linear(
            hidden_dim, bottleneck_dim
        ), nn.Linear(bottleneck_dim, hidden_dim)
        self.lang_down, self.lang_up = nn.Linear(hidden_dim, bottleneck_dim), nn.Linear(
            bottleneck_dim, hidden_dim
        )
        self.domain_down, self.domain_up = nn.Linear(
            hidden_dim, bottleneck_dim
        ), nn.Linear(bottleneck_dim, hidden_dim)
        self.lang_scale, self.domain_scale = nn.Linear(
            embed_dim, bottleneck_dim
        ), nn.Linear(embed_dim, bottleneck_dim)
        self.gate_proj = nn.Linear(2 * embed_dim, 3)
        self.comp_down, self.comp_up = nn.Linear(hidden_dim, bottleneck_dim), nn.Linear(
            bottleneck_dim, hidden_dim
        )
        self.comp_scale = nn.Linear(2 * embed_dim, bottleneck_dim)
        for module in (self.shared_up, self.lang_up, self.domain_up, self.comp_up):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.condition_embeddings = None

    def branch(self, value, down, up, scale=None, dropout=None):
        hidden = self.activation(down(value.to(down.weight.dtype)))
        if scale is not None:
            hidden = hidden * scale.to(hidden.dtype).unsqueeze(1)
        return up((dropout or self.dropout)(hidden)).to(value.dtype)

    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        conditions = self.soft_condition_proj(
            pooled.to(self.soft_condition_proj[0].weight.dtype)
        ).to(hidden_states.dtype)
        language, domain = conditions.chunk(2, dim=-1)
        self.condition_embeddings = language, domain
        gates = torch.sigmoid(
            self.gate_proj(conditions.to(self.gate_proj.weight.dtype))
        ).to(hidden_states.dtype)
        value = self.pre_norm(hidden_states.to(self.pre_norm.weight.dtype)).to(
            hidden_states.dtype
        )
        global_delta = self.branch(value, self.shared_down, self.shared_up)
        lang_delta = self.branch(
            value,
            self.lang_down,
            self.lang_up,
            torch.tanh(self.lang_scale(language.to(self.lang_scale.weight.dtype))),
        )
        domain_delta = self.branch(
            value,
            self.domain_down,
            self.domain_up,
            torch.tanh(self.domain_scale(domain.to(self.domain_scale.weight.dtype))),
        )
        compound_delta = self.branch(
            value,
            self.comp_down,
            self.comp_up,
            torch.tanh(self.comp_scale(conditions.to(self.comp_scale.weight.dtype))),
            self.comp_dropout,
        )
        delta = torch.zeros_like(hidden_states) + global_delta
        delta = delta + gates[:, 0, None, None] * lang_delta
        delta = delta + gates[:, 1, None, None] * domain_delta
        delta = delta + gates[:, 2, None, None] * compound_delta
        return hidden_states + delta
