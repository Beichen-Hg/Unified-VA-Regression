"""Helpers for macro-dev checkpoint selection."""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_target_list(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        raw_value = [x.strip() for x in raw_value.split(",") if x.strip()]
    return [str(x).strip() for x in (raw_value or []) if str(x).strip()]


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

    pseudo_ood_targets = _normalize_target_list(cfg.get("pseudo_ood_targets", []))
    pseudo_ood_weight = float(cfg.get("pseudo_ood_weight", 0.0) or 0.0)
    pseudo_ood_worst_k = max(int(cfg.get("pseudo_ood_worst_k", 0) or 0), 0)
    pseudo_ood_worst_k_weight = float(cfg.get("pseudo_ood_worst_k_weight", 0.0) or 0.0)
    if pseudo_ood_weight < 0:
        raise ValueError("macro-dev selection requires pseudo_ood_weight >= 0")
    if pseudo_ood_worst_k_weight < 0:
        raise ValueError("macro-dev selection requires pseudo_ood_worst_k_weight >= 0")
    if pseudo_ood_worst_k > 0 and not pseudo_ood_targets:
        raise ValueError("pseudo_ood_worst_k requires non-empty pseudo_ood_targets")

    std_gap_v_weight = float(cfg.get("std_gap_v_weight", 0.0) or 0.0)
    std_gap_a_weight = float(cfg.get("std_gap_a_weight", 0.0) or 0.0)
    ccc_macro_floor = cfg.get("ccc_macro_floor")
    ccc_macro_floor_weight = float(cfg.get("ccc_macro_floor_weight", 0.0) or 0.0)
    pcc_macro_floor = cfg.get("pcc_macro_floor")
    pcc_macro_floor_weight = float(cfg.get("pcc_macro_floor_weight", 0.0) or 0.0)
    if std_gap_v_weight < 0 or std_gap_a_weight < 0:
        raise ValueError("macro-dev selection requires std_gap weights >= 0")
    if ccc_macro_floor_weight < 0 or pcc_macro_floor_weight < 0:
        raise ValueError("macro-dev selection requires floor weights >= 0")

    protected_targets = _normalize_target_list(cfg.get("protected_targets", []))
    protected_targets_weight = float(cfg.get("protected_targets_weight", 0.0) or 0.0)
    protected_targets_worst_k = max(int(cfg.get("protected_targets_worst_k", 0) or 0), 0)
    protected_targets_worst_k_weight = float(cfg.get("protected_targets_worst_k_weight", 0.0) or 0.0)
    protected_std_gap_v_weight = float(cfg.get("protected_std_gap_v_weight", 0.0) or 0.0)
    protected_std_gap_a_weight = float(cfg.get("protected_std_gap_a_weight", 0.0) or 0.0)
    protected_ccc_macro_floor = cfg.get("protected_ccc_macro_floor")
    protected_ccc_macro_floor_weight = float(cfg.get("protected_ccc_macro_floor_weight", 0.0) or 0.0)
    protected_pcc_macro_floor = cfg.get("protected_pcc_macro_floor")
    protected_pcc_macro_floor_weight = float(cfg.get("protected_pcc_macro_floor_weight", 0.0) or 0.0)
    if protected_targets_weight < 0 or protected_targets_worst_k_weight < 0:
        raise ValueError("macro-dev selection requires protected target weights >= 0")
    if protected_std_gap_v_weight < 0 or protected_std_gap_a_weight < 0:
        raise ValueError("macro-dev selection requires protected std_gap weights >= 0")
    if protected_ccc_macro_floor_weight < 0 or protected_pcc_macro_floor_weight < 0:
        raise ValueError("macro-dev selection requires protected floor weights >= 0")
    if protected_targets_worst_k > 0 and not protected_targets:
        raise ValueError("protected_targets_worst_k requires non-empty protected_targets")

    return {
        "mode": mode,
        "worst_k": worst_k,
        "worst_k_weight": worst_k_weight,
        "pseudo_ood_targets": pseudo_ood_targets,
        "pseudo_ood_weight": pseudo_ood_weight,
        "pseudo_ood_worst_k": pseudo_ood_worst_k,
        "pseudo_ood_worst_k_weight": pseudo_ood_worst_k_weight,
        "std_gap_v_weight": std_gap_v_weight,
        "std_gap_a_weight": std_gap_a_weight,
        "ccc_macro_floor": float(ccc_macro_floor) if ccc_macro_floor is not None else None,
        "ccc_macro_floor_weight": ccc_macro_floor_weight,
        "pcc_macro_floor": float(pcc_macro_floor) if pcc_macro_floor is not None else None,
        "pcc_macro_floor_weight": pcc_macro_floor_weight,
        "protected_targets": protected_targets,
        "protected_targets_weight": protected_targets_weight,
        "protected_targets_worst_k": protected_targets_worst_k,
        "protected_targets_worst_k_weight": protected_targets_worst_k_weight,
        "protected_std_gap_v_weight": protected_std_gap_v_weight,
        "protected_std_gap_a_weight": protected_std_gap_a_weight,
        "protected_ccc_macro_floor": (
            float(protected_ccc_macro_floor) if protected_ccc_macro_floor is not None else None
        ),
        "protected_ccc_macro_floor_weight": protected_ccc_macro_floor_weight,
        "protected_pcc_macro_floor": (
            float(protected_pcc_macro_floor) if protected_pcc_macro_floor is not None else None
        ),
        "protected_pcc_macro_floor_weight": protected_pcc_macro_floor_weight,
    }


def macro_dev_selection_metric_name(selection_cfg: dict[str, Any] | None) -> str:
    cfg = normalize_macro_dev_selection_cfg(selection_cfg)
    if (
        cfg["mode"] == "macro_mean"
        and cfg["pseudo_ood_weight"] <= 0
        and cfg["pseudo_ood_worst_k_weight"] <= 0
        and cfg["std_gap_v_weight"] <= 0
        and cfg["std_gap_a_weight"] <= 0
        and cfg["ccc_macro_floor_weight"] <= 0
        and cfg["pcc_macro_floor_weight"] <= 0
        and cfg["protected_targets_weight"] <= 0
        and cfg["protected_targets_worst_k_weight"] <= 0
        and cfg["protected_std_gap_v_weight"] <= 0
        and cfg["protected_std_gap_a_weight"] <= 0
        and cfg["protected_ccc_macro_floor_weight"] <= 0
        and cfg["protected_pcc_macro_floor_weight"] <= 0
    ):
        return "macro_dev_rmse_va"
    return "macro_dev_selection_score"


def _coerce_target_metrics(value: Any) -> dict[str, float | None]:
    if isinstance(value, dict):
        return {
            "rmse_va": float(value.get("rmse_va")) if value.get("rmse_va") is not None else None,
            "ccc_macro": float(value.get("ccc_macro")) if value.get("ccc_macro") is not None else None,
            "pcc_macro": float(value.get("pcc_macro")) if value.get("pcc_macro") is not None else None,
            "std_gap_v": float(value.get("std_gap_v")) if value.get("std_gap_v") is not None else None,
            "std_gap_a": float(value.get("std_gap_a")) if value.get("std_gap_a") is not None else None,
        }
    return {
        "rmse_va": float(value),
        "ccc_macro": None,
        "pcc_macro": None,
        "std_gap_v": None,
        "std_gap_a": None,
    }


def compute_macro_dev_selection(
    per_target: dict[str, Any],
    selection_cfg: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    cfg = normalize_macro_dev_selection_cfg(selection_cfg)
    metrics_by_target = {
        str(name): _coerce_target_metrics(value)
        for name, value in per_target.items()
    }
    items = sorted(
        (
            (name, metrics["rmse_va"])
            for name, metrics in metrics_by_target.items()
            if metrics["rmse_va"] is not None
        ),
        key=lambda x: x[1],
        reverse=True,
    )
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
    score = macro_mean
    if cfg["mode"] == "macro_mean":
        summary["selection_score"] = macro_mean
    else:
        worst_k = min(cfg["worst_k"], len(items))
        worst_items = items[:worst_k]
        worst_k_mean = float(np.mean([value for _, value in worst_items]))
        score += cfg["worst_k_weight"] * worst_k_mean
        summary.update(
            {
                "worst_k": worst_k,
                "worst_k_weight": cfg["worst_k_weight"],
                "worst_k_mean": worst_k_mean,
                "worst_targets": [{"target": name, "rmse_va": value} for name, value in worst_items],
            }
        )

    pseudo_ood_targets = set(cfg["pseudo_ood_targets"])
    if pseudo_ood_targets:
        pseudo_items = [(name, value) for name, value in items if name in pseudo_ood_targets]
        pseudo_missing = sorted(name for name in pseudo_ood_targets if name not in per_target)
        pseudo_summary: dict[str, Any] = {
            "pseudo_ood_targets": sorted(pseudo_ood_targets),
            "pseudo_ood_present_targets": [name for name, _ in pseudo_items],
            "pseudo_ood_missing_targets": pseudo_missing,
            "pseudo_ood_weight": cfg["pseudo_ood_weight"],
            "pseudo_ood_worst_k": cfg["pseudo_ood_worst_k"],
            "pseudo_ood_worst_k_weight": cfg["pseudo_ood_worst_k_weight"],
        }
        if pseudo_items:
            pseudo_mean = float(np.mean([value for _, value in pseudo_items]))
            pseudo_summary["pseudo_ood_mean"] = pseudo_mean
            score += cfg["pseudo_ood_weight"] * pseudo_mean
            if cfg["pseudo_ood_worst_k"] > 0:
                pseudo_worst_k = min(cfg["pseudo_ood_worst_k"], len(pseudo_items))
                pseudo_worst_items = pseudo_items[:pseudo_worst_k]
                pseudo_worst_k_mean = float(np.mean([value for _, value in pseudo_worst_items]))
                score += cfg["pseudo_ood_worst_k_weight"] * pseudo_worst_k_mean
                pseudo_summary["pseudo_ood_worst_k"] = pseudo_worst_k
                pseudo_summary["pseudo_ood_worst_k_mean"] = pseudo_worst_k_mean
                pseudo_summary["pseudo_ood_worst_targets"] = [
                    {"target": name, "rmse_va": value} for name, value in pseudo_worst_items
                ]
        summary.update(pseudo_summary)

    std_gap_v_values = [metrics_by_target[name]["std_gap_v"] for name, _ in items if metrics_by_target[name]["std_gap_v"] is not None]
    std_gap_a_values = [metrics_by_target[name]["std_gap_a"] for name, _ in items if metrics_by_target[name]["std_gap_a"] is not None]
    ccc_macro_values = [metrics_by_target[name]["ccc_macro"] for name, _ in items if metrics_by_target[name]["ccc_macro"] is not None]
    pcc_macro_values = [metrics_by_target[name]["pcc_macro"] for name, _ in items if metrics_by_target[name]["pcc_macro"] is not None]

    if cfg["std_gap_v_weight"] > 0 and std_gap_v_values:
        macro_std_gap_v = float(np.mean(std_gap_v_values))
        score += cfg["std_gap_v_weight"] * macro_std_gap_v
        summary["macro_std_gap_v"] = macro_std_gap_v
        summary["std_gap_v_weight"] = cfg["std_gap_v_weight"]
    if cfg["std_gap_a_weight"] > 0 and std_gap_a_values:
        macro_std_gap_a = float(np.mean(std_gap_a_values))
        score += cfg["std_gap_a_weight"] * macro_std_gap_a
        summary["macro_std_gap_a"] = macro_std_gap_a
        summary["std_gap_a_weight"] = cfg["std_gap_a_weight"]
    if cfg["ccc_macro_floor_weight"] > 0 and cfg["ccc_macro_floor"] is not None and ccc_macro_values:
        ccc_floor_deficit = float(np.mean([max(0.0, cfg["ccc_macro_floor"] - value) for value in ccc_macro_values]))
        score += cfg["ccc_macro_floor_weight"] * ccc_floor_deficit
        summary["ccc_macro_floor"] = cfg["ccc_macro_floor"]
        summary["ccc_macro_floor_weight"] = cfg["ccc_macro_floor_weight"]
        summary["ccc_macro_floor_deficit"] = ccc_floor_deficit
    if cfg["pcc_macro_floor_weight"] > 0 and cfg["pcc_macro_floor"] is not None and pcc_macro_values:
        pcc_floor_deficit = float(np.mean([max(0.0, cfg["pcc_macro_floor"] - value) for value in pcc_macro_values]))
        score += cfg["pcc_macro_floor_weight"] * pcc_floor_deficit
        summary["pcc_macro_floor"] = cfg["pcc_macro_floor"]
        summary["pcc_macro_floor_weight"] = cfg["pcc_macro_floor_weight"]
        summary["pcc_macro_floor_deficit"] = pcc_floor_deficit

    protected_targets = set(cfg["protected_targets"])
    if protected_targets:
        protected_items = [(name, value) for name, value in items if name in protected_targets]
        protected_missing = sorted(name for name in protected_targets if name not in per_target)
        protected_summary: dict[str, Any] = {
            "protected_targets": sorted(protected_targets),
            "protected_present_targets": [name for name, _ in protected_items],
            "protected_missing_targets": protected_missing,
            "protected_targets_weight": cfg["protected_targets_weight"],
            "protected_targets_worst_k": cfg["protected_targets_worst_k"],
            "protected_targets_worst_k_weight": cfg["protected_targets_worst_k_weight"],
            "protected_std_gap_v_weight": cfg["protected_std_gap_v_weight"],
            "protected_std_gap_a_weight": cfg["protected_std_gap_a_weight"],
            "protected_ccc_macro_floor": cfg["protected_ccc_macro_floor"],
            "protected_ccc_macro_floor_weight": cfg["protected_ccc_macro_floor_weight"],
            "protected_pcc_macro_floor": cfg["protected_pcc_macro_floor"],
            "protected_pcc_macro_floor_weight": cfg["protected_pcc_macro_floor_weight"],
        }
        if protected_items:
            protected_mean = float(np.mean([value for _, value in protected_items]))
            protected_summary["protected_mean"] = protected_mean
            score += cfg["protected_targets_weight"] * protected_mean

            if cfg["protected_targets_worst_k"] > 0:
                protected_worst_k = min(cfg["protected_targets_worst_k"], len(protected_items))
                protected_worst_items = protected_items[:protected_worst_k]
                protected_worst_k_mean = float(np.mean([value for _, value in protected_worst_items]))
                score += cfg["protected_targets_worst_k_weight"] * protected_worst_k_mean
                protected_summary["protected_targets_worst_k"] = protected_worst_k
                protected_summary["protected_worst_k_mean"] = protected_worst_k_mean
                protected_summary["protected_worst_targets"] = [
                    {"target": name, "rmse_va": value} for name, value in protected_worst_items
                ]

            protected_std_gap_v_values = [
                metrics_by_target[name]["std_gap_v"]
                for name, _ in protected_items
                if metrics_by_target[name]["std_gap_v"] is not None
            ]
            protected_std_gap_a_values = [
                metrics_by_target[name]["std_gap_a"]
                for name, _ in protected_items
                if metrics_by_target[name]["std_gap_a"] is not None
            ]
            protected_ccc_macro_values = [
                metrics_by_target[name]["ccc_macro"]
                for name, _ in protected_items
                if metrics_by_target[name]["ccc_macro"] is not None
            ]
            protected_pcc_macro_values = [
                metrics_by_target[name]["pcc_macro"]
                for name, _ in protected_items
                if metrics_by_target[name]["pcc_macro"] is not None
            ]

            if cfg["protected_std_gap_v_weight"] > 0 and protected_std_gap_v_values:
                protected_macro_std_gap_v = float(np.mean(protected_std_gap_v_values))
                score += cfg["protected_std_gap_v_weight"] * protected_macro_std_gap_v
                protected_summary["protected_macro_std_gap_v"] = protected_macro_std_gap_v
            if cfg["protected_std_gap_a_weight"] > 0 and protected_std_gap_a_values:
                protected_macro_std_gap_a = float(np.mean(protected_std_gap_a_values))
                score += cfg["protected_std_gap_a_weight"] * protected_macro_std_gap_a
                protected_summary["protected_macro_std_gap_a"] = protected_macro_std_gap_a
            if (
                cfg["protected_ccc_macro_floor_weight"] > 0
                and cfg["protected_ccc_macro_floor"] is not None
                and protected_ccc_macro_values
            ):
                protected_ccc_floor_deficit = float(
                    np.mean(
                        [max(0.0, cfg["protected_ccc_macro_floor"] - value) for value in protected_ccc_macro_values]
                    )
                )
                score += cfg["protected_ccc_macro_floor_weight"] * protected_ccc_floor_deficit
                protected_summary["protected_ccc_macro_floor_deficit"] = protected_ccc_floor_deficit
            if (
                cfg["protected_pcc_macro_floor_weight"] > 0
                and cfg["protected_pcc_macro_floor"] is not None
                and protected_pcc_macro_values
            ):
                protected_pcc_floor_deficit = float(
                    np.mean(
                        [max(0.0, cfg["protected_pcc_macro_floor"] - value) for value in protected_pcc_macro_values]
                    )
                )
                score += cfg["protected_pcc_macro_floor_weight"] * protected_pcc_floor_deficit
                protected_summary["protected_pcc_macro_floor_deficit"] = protected_pcc_floor_deficit
        summary.update(protected_summary)

    summary["selection_score"] = score
    return score, summary
