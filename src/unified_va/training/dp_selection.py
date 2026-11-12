"""Distribution-preserving development checkpoint ranking."""

import math
import statistics


def compute_macro_dev_selection(per_target, selection_cfg=None):
    cfg = selection_cfg or {"mode": "macro_mean"}
    if not per_target:
        raise ValueError("DP requires development targets")
    rows = {
        k: v if isinstance(v, dict) else {"rmse_va": float(v)}
        for k, v in per_target.items()
    }
    if any(not math.isfinite(float(v["rmse_va"])) for v in rows.values()):
        raise ValueError("Nonfinite development RMSE")
    ordered = sorted(rows, key=lambda k: rows[k]["rmse_va"], reverse=True)
    average = statistics.mean(rows[k]["rmse_va"] for k in ordered)
    score, detail = float(average), dict(macro_mean=float(average), n_targets=len(rows))
    if cfg.get("mode", "macro_mean") == "macro_plus_worst_k":
        worst = statistics.mean(rows[k]["rmse_va"] for k in ordered[: cfg["worst_k"]])
        score += cfg["worst_k_weight"] * worst
        detail["worst_k_mean"] = worst
    elif cfg.get("mode", "macro_mean") != "macro_mean":
        raise ValueError("Unknown development selector")
    pseudo = [k for k in ordered if k in cfg.get("pseudo_ood_targets", [])]
    if pseudo:
        value = statistics.mean(rows[k]["rmse_va"] for k in pseudo)
        score += cfg.get("pseudo_ood_weight", 0) * value
        detail["pseudo_ood_mean"] = value
        count = cfg.get("pseudo_ood_worst_k", 0)
        if count:
            value = statistics.mean(rows[k]["rmse_va"] for k in pseudo[:count])
            score += cfg.get("pseudo_ood_worst_k_weight", 0) * value
            detail["pseudo_ood_worst_k_mean"] = value
    for dimension in ("v", "a"):
        field = "std_gap_" + dimension
        weight = cfg.get(field + "_weight", 0)
        if weight:
            value = statistics.mean(v[field] for v in rows.values())
            score += weight * value
            detail["macro_" + field] = value
    for metric in ("ccc", "pcc"):
        field = metric + "_macro_floor"
        weight = cfg.get(field + "_weight", 0)
        if weight:
            value = statistics.mean(
                max(0, cfg[field] - v[metric + "_macro"]) for v in rows.values()
            )
            score += weight * value
            detail[field + "_deficit"] = value
    detail["selection_score"] = float(score)
    return float(score), detail
