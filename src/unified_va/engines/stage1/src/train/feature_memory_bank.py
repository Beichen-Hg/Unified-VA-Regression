"""Feature Memory Bank for batch-level losses with batch_size=1.

Stores recent features in a ring buffer to enable batch-dependent computations
(e.g., VICReg variance regularization) when physical batch size is 1.
"""
from __future__ import annotations

import torch


class FeatureMemoryBank:
    """Ring buffer storing recent (feature, target) pairs."""

    def __init__(self, capacity: int, feature_dim: int, device: torch.device):
        self.capacity = capacity
        self.features = torch.zeros(capacity, feature_dim, device=device)
        self.targets = torch.zeros(capacity, 2, device=device)
        self.pointer = 0
        self.filled = 0

    def push(self, features: torch.Tensor, targets: torch.Tensor):
        """Push detached features and targets into the bank."""
        batch_size = features.size(0)
        for i in range(batch_size):
            self.features[self.pointer] = features[i].detach()
            self.targets[self.pointer] = targets[i].detach()
            self.pointer = (self.pointer + 1) % self.capacity
            self.filled = min(self.filled + 1, self.capacity)

    def get(self, min_samples: int = 4) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return stored (features, targets) if enough samples."""
        if self.filled < min_samples:
            return None
        n = self.filled
        return self.features[:n], self.targets[:n]

    def clear(self):
        self.pointer = 0
        self.filled = 0
