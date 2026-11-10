"""VA-region loss reweighting: upweight underperforming VA regions.

Given gold (V, A) targets, assigns per-sample weights based on which
VA region the sample falls into. Regions where the model typically
struggles (e.g. V_mid) get higher weights.

Usage in training loop:
    from src.models.va_region_weights import compute_va_region_weights
    sw = compute_va_region_weights(target, cfg)  # (B,) tensor
    loss = va_loss(pred, target, sample_weights=sw)
"""
import torch
from typing import Optional


# Default region weight map (from ZHO error analysis):
# V_mid regions have highest RMSE, V_extreme regions lowest.
DEFAULT_REGION_WEIGHTS = {
    "V_low_A_low": 1.0,
    "V_low_A_mid": 1.0,
    "V_low_A_high": 1.2,
    "V_mid_A_low": 1.5,
    "V_mid_A_mid": 1.5,
    "V_mid_A_high": 1.8,
    "V_high_A_low": 1.0,
    "V_high_A_mid": 0.8,
    "V_high_A_high": 0.8,
}


def _classify_region(v: float, a: float) -> str:
    vr = "low" if v < 4.0 else ("mid" if v < 6.0 else "high")
    ar = "low" if a < 4.0 else ("mid" if a < 6.0 else "high")
    return f"V_{vr}_A_{ar}"


def compute_va_region_weights(
    target: torch.Tensor,
    region_weights: Optional[dict[str, float]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-sample weights from gold VA region.

    Args:
        target: (B, 2) gold [V, A] values
        region_weights: {region_name: weight}, defaults to DEFAULT_REGION_WEIGHTS
        normalize: if True, normalize weights so mean=1.0

    Returns:
        (B,) weight tensor on same device as target
    """
    if region_weights is None:
        region_weights = DEFAULT_REGION_WEIGHTS

    B = target.shape[0]
    t_cpu = target.detach().cpu()
    weights = torch.ones(B)

    for i in range(B):
        v, a = t_cpu[i, 0].item(), t_cpu[i, 1].item()
        region = _classify_region(v, a)
        weights[i] = region_weights.get(region, 1.0)

    if normalize and B > 0:
        weights = weights * B / weights.sum().clamp(min=1e-9)

    return weights.to(target.device)
