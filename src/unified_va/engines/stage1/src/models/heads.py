"""Model heads for DimABSA2026 VA regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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



def build_va_head(cfg: dict, hidden_dim: int):
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
