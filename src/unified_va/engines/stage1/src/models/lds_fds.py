"""Label Distribution Smoothing (LDS) and Feature Distribution Smoothing (FDS).

Based on: Yang et al., "Delving into Deep Imbalanced Regression", ICML 2021.

LDS: Compute per-sample weights inversely proportional to smoothed label density.
FDS: Smooth feature statistics across nearby label bins (applied in forward pass).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d


def compute_lds_weights(
    targets: np.ndarray,
    num_bins: int = 50,
    kernel_size: int = 5,
    kernel_sigma: float = 2.0,
    reweight: str = "sqrt_inv",  # "inv" or "sqrt_inv"
    clip_max: float = 10.0,
) -> np.ndarray:
    """Compute LDS sample weights for a 1D target array.

    Args:
        targets: 1D array of continuous target values.
        num_bins: Number of bins for density estimation.
        kernel_size: Gaussian kernel size for smoothing (unused, sigma controls width).
        kernel_sigma: Gaussian kernel sigma for smoothing.
        reweight: "inv" for 1/density, "sqrt_inv" for 1/sqrt(density).
        clip_max: Maximum weight to prevent extreme values.

    Returns:
        weights: Per-sample weights, same length as targets.
    """
    # Bin targets
    t_min, t_max = targets.min(), targets.max()
    bin_edges = np.linspace(t_min - 1e-6, t_max + 1e-6, num_bins + 1)
    bin_indices = np.digitize(targets, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Count per bin
    bin_counts = np.zeros(num_bins, dtype=np.float64)
    for idx in bin_indices:
        bin_counts[idx] += 1

    # Smooth with Gaussian kernel
    smoothed_counts = gaussian_filter1d(bin_counts, sigma=kernel_sigma)
    smoothed_counts = np.maximum(smoothed_counts, 1e-6)  # avoid division by zero

    # Per-sample density
    sample_density = smoothed_counts[bin_indices]

    # Reweight
    if reweight == "inv":
        weights = 1.0 / sample_density
    elif reweight == "sqrt_inv":
        weights = 1.0 / np.sqrt(sample_density)
    else:
        raise ValueError(f"Unknown reweight: {reweight}")

    # Normalize to mean=1
    weights = weights / weights.mean()

    # Clip extreme weights
    weights = np.clip(weights, 0.0, clip_max)

    return weights.astype(np.float32)


def compute_lds_weights_va(
    targets_v: np.ndarray,
    targets_a: np.ndarray,
    focus_dim: str = "a",  # "v", "a", or "va"
    **kwargs,
) -> np.ndarray:
    """Compute LDS weights for VA regression.

    Args:
        targets_v: Valence targets.
        targets_a: Arousal targets.
        focus_dim: Which dimension to base weights on.
        **kwargs: Passed to compute_lds_weights.

    Returns:
        weights: Per-sample weights.
    """
    if focus_dim == "a":
        return compute_lds_weights(targets_a, **kwargs)
    elif focus_dim == "v":
        return compute_lds_weights(targets_v, **kwargs)
    elif focus_dim == "va":
        w_v = compute_lds_weights(targets_v, **kwargs)
        w_a = compute_lds_weights(targets_a, **kwargs)
        return (w_v * w_a) ** 0.5  # geometric mean
    else:
        raise ValueError(f"Unknown focus_dim: {focus_dim}")


class FDS(nn.Module):
    """Feature Distribution Smoothing.

    Smooths running feature statistics (mean, var) across nearby label bins
    using a Gaussian kernel, then calibrates features at inference time.
    """

    def __init__(
        self,
        feature_dim: int,
        num_bins: int = 50,
        start_update: int = 0,
        start_smooth: int = 1,
        kernel_sigma: float = 2.0,
        target_range: tuple = (1.0, 9.0),
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.start_update = start_update
        self.start_smooth = start_smooth
        self.kernel_sigma = kernel_sigma
        self.target_range = target_range

        # Running statistics per bin
        self.register_buffer("running_mean", torch.zeros(num_bins, feature_dim))
        self.register_buffer("running_var", torch.ones(num_bins, feature_dim))
        self.register_buffer("running_count", torch.zeros(num_bins))
        self.register_buffer("smoothed_mean", torch.zeros(num_bins, feature_dim))
        self.register_buffer("smoothed_var", torch.ones(num_bins, feature_dim))

    def _get_bin_idx(self, targets: torch.Tensor) -> torch.Tensor:
        t_min, t_max = self.target_range
        normalized = (targets - t_min) / (t_max - t_min + 1e-6)
        bin_idx = (normalized * self.num_bins).long()
        return bin_idx.clamp(0, self.num_bins - 1)

    def update_stats(self, features: torch.Tensor, targets: torch.Tensor):
        """Update running statistics with current batch."""
        bin_idx = self._get_bin_idx(targets)
        for i in range(self.num_bins):
            mask = bin_idx == i
            if mask.sum() > 0:
                batch_mean = features[mask].mean(dim=0)
                batch_var = features[mask].var(dim=0) if mask.sum() > 1 else torch.zeros_like(batch_mean)
                count = mask.sum().float()

                # Exponential moving average
                alpha = 0.9
                self.running_mean[i] = alpha * self.running_mean[i] + (1 - alpha) * batch_mean
                self.running_var[i] = alpha * self.running_var[i] + (1 - alpha) * batch_var
                self.running_count[i] += count

    def smooth_stats(self):
        """Apply Gaussian smoothing to running statistics."""
        # Convert to numpy for gaussian_filter1d
        mean_np = self.running_mean.cpu().numpy()
        var_np = self.running_var.cpu().numpy()

        smoothed_mean = gaussian_filter1d(mean_np, sigma=self.kernel_sigma, axis=0)
        smoothed_var = gaussian_filter1d(var_np, sigma=self.kernel_sigma, axis=0)

        self.smoothed_mean.copy_(torch.from_numpy(smoothed_mean))
        self.smoothed_var.copy_(torch.from_numpy(np.maximum(smoothed_var, 1e-6)))

    def forward(self, features: torch.Tensor, targets: torch.Tensor, epoch: int = 0):
        """Calibrate features using smoothed statistics."""
        if epoch < self.start_smooth:
            return features

        bin_idx = self._get_bin_idx(targets)

        # Get per-sample smoothed stats
        smoothed_mean = self.smoothed_mean[bin_idx]  # (B, D)
        smoothed_var = self.smoothed_var[bin_idx]     # (B, D)
        running_mean = self.running_mean[bin_idx]     # (B, D)
        running_var = self.running_var[bin_idx]       # (B, D)

        # Calibrate: shift and scale features
        calibrated = (features - running_mean) / (running_var.sqrt() + 1e-6)
        calibrated = calibrated * smoothed_var.sqrt() + smoothed_mean

        return calibrated
