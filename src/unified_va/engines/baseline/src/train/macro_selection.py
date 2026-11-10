"""Helpers for macro-dev checkpoint selection."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_macro_dev_selection_cfg(selection_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(selection_cfg or {})
    mode = str(cfg.get("mode", cfg.get("metric", "macro_mean"))).strip().lower() or "macro_mean"
    aliases = {
        "macro": "macro_mean",
        "mean": "macro_mean",
        "macro_mean": "macro_mean",
        "macro+worst_k": "macro_plus_worst_k",
        "macro_plus_worst_k": "macro_plus_worst_k",
        "macro_worst_k": "macro_plus_worst_k",
        "robust": "macro_plus_worst_k",
        "robustselect": "macro_plus_worst_k",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"macro_mean", "macro_plus_worst_k"}:
        raise ValueError(f"unsupported macro-dev selection mode: {mode}")

    worst_k = max(int(cfg.get("worst_k", 0) or 0), 0)
    worst_k_weight = float(cfg.get("worst_k_weight", cfg.get("lambda", 0.0)))
    if mode == "macro_plus_worst_k":
        if worst_k <= 0:
            raise ValueError("macro_plus_worst_k selection requires worst_k > 0")
        if worst_k_weight < 0:
            raise ValueError("macro_plus_worst_k selection requires worst_k_weight >= 0")

    return {
        "mode": mode,
        "worst_k": worst_k,
        "worst_k_weight": worst_k_weight,
    }


def macro_dev_selection_metric_name(selection_cfg: dict[str, Any] | None) -> str:
    cfg = normalize_macro_dev_selection_cfg(selection_cfg)
    if cfg["mode"] == "macro_mean":
        return "macro_dev_rmse_va"
    return "macro_dev_selection_score"


def compute_macro_dev_selection(
    per_target: dict[str, float],
    selection_cfg: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    cfg = normalize_macro_dev_selection_cfg(selection_cfg)
    items = sorted(((str(name), float(value)) for name, value in per_target.items()), key=lambda x: x[1], reverse=True)
    if not items:
        return None, {
            "mode": cfg["mode"],
            "n_targets": 0,
            "macro_mean": None,
        }

    macro_mean = float(np.mean([value for _, value in items]))
    summary: dict[str, Any] = {
        "mode": cfg["mode"],
        "n_targets": len(items),
        "macro_mean": macro_mean,
    }
    if cfg["mode"] == "macro_mean":
        summary["selection_score"] = macro_mean
        return macro_mean, summary

    worst_k = min(cfg["worst_k"], len(items))
    worst_items = items[:worst_k]
    worst_k_mean = float(np.mean([value for _, value in worst_items]))
    score = macro_mean + cfg["worst_k_weight"] * worst_k_mean
    summary.update(
        {
            "worst_k": worst_k,
            "worst_k_weight": cfg["worst_k_weight"],
            "worst_k_mean": worst_k_mean,
            "worst_targets": [{"target": name, "rmse_va": value} for name, value in worst_items],
            "selection_score": score,
        }
    )
    return score, summary
