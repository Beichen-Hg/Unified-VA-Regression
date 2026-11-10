"""Test-Time Augmentation (TTA) for VA regression.

Implements MC Dropout and ensemble prediction to potentially improve
prediction variance and reduce arousal collapse.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal

from src.utils.logging import get_logger

logger = get_logger(__name__)


def enable_dropout_for_inference(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC Dropout.

    Args:
        model: The model to modify
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def disable_dropout_for_inference(model: nn.Module) -> None:
    """Restore dropout layers to eval mode.

    Args:
        model: The model to modify
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def tta_predict(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_samples: int = 10,
    aggregation: Literal["mean", "median"] = "mean",
    mc_dropout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Perform test-time augmentation prediction.

    Args:
        model: The VA regression model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        n_samples: Number of forward passes for TTA
        aggregation: How to aggregate predictions ("mean" or "median")
        mc_dropout: Whether to use MC Dropout (keep dropout enabled)

    Returns:
        - Aggregated VA predictions [batch_size, 2]
        - Prediction uncertainty (std across samples) [batch_size, 2] or None
    """
    if n_samples <= 1:
        # No TTA, standard inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            if isinstance(outputs, tuple):
                return outputs[0], None  # uncertainty head
            return outputs, None

    # Enable MC Dropout if requested
    original_training_state = model.training
    if mc_dropout:
        model.eval()  # Set model to eval mode
        enable_dropout_for_inference(model)  # But keep dropout active
    else:
        model.eval()

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(input_ids, attention_mask)
            if isinstance(outputs, tuple):
                # Uncertainty head: (mu, log_sigma)
                mu = outputs[0]
            else:
                mu = outputs
            predictions.append(mu)

    # Restore original state
    if mc_dropout:
        disable_dropout_for_inference(model)
    model.train(original_training_state)

    # Stack predictions: [n_samples, batch_size, 2]
    predictions = torch.stack(predictions, dim=0)

    # Compute uncertainty (std across samples)
    uncertainty = predictions.std(dim=0)  # [batch_size, 2]

    # Aggregate predictions
    if aggregation == "mean":
        aggregated = predictions.mean(dim=0)
    elif aggregation == "median":
        aggregated = predictions.median(dim=0).values
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return aggregated, uncertainty


def tta_predict_batch(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    n_samples: int = 10,
    aggregation: Literal["mean", "median"] = "mean",
    mc_dropout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Convenience wrapper for batch prediction with TTA.

    Args:
        model: The VA regression model
        batch: Batch dict with "input_ids" and "attention_mask"
        device: Device to run on
        n_samples: Number of forward passes for TTA
        aggregation: How to aggregate predictions
        mc_dropout: Whether to use MC Dropout

    Returns:
        - Aggregated VA predictions [batch_size, 2]
        - Prediction uncertainty [batch_size, 2] or None
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    return tta_predict(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        n_samples=n_samples,
        aggregation=aggregation,
        mc_dropout=mc_dropout,
    )
