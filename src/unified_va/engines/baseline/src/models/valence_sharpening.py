"""Valence Sharpening Loss - Optimize V extremes to indirectly help A.

Core idea: V and A are coupled in feature space. Better V extremes
may pull A extremes through shared representations.
"""
import torch
import torch.nn.functional as F


def valence_sharpening_loss(
    pred_v: torch.Tensor,
    true_v: torch.Tensor,
    pred_a: torch.Tensor,
    true_a: torch.Tensor,
    v_extreme_threshold: float = 1.5,
    weight: float = 0.3
) -> torch.Tensor:
    """Focus on extreme valence regions to indirectly help arousal.

    Args:
        pred_v: Predicted valence (B,)
        true_v: True valence (B,)
        pred_a: Predicted arousal (B,)
        true_a: True arousal (B,)
        v_extreme_threshold: Distance from center (5.0) to be "extreme"
        weight: Loss weight

    Returns:
        Weighted loss scalar
    """
    # Identify extreme V samples (very negative or very positive)
    v_center = 5.0
    v_distance = torch.abs(true_v - v_center)
    extreme_mask = v_distance > v_extreme_threshold

    if extreme_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_v.device)

    # Weighted MSE on extreme V samples
    v_loss = F.mse_loss(pred_v[extreme_mask], true_v[extreme_mask])

    # Bonus: also weight A loss on these samples
    # (coupling hypothesis: good V extremes help A extremes)
    a_loss = F.mse_loss(pred_a[extreme_mask], true_a[extreme_mask])

    total_loss = v_loss + 0.5 * a_loss

    return weight * total_loss
