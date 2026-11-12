"""VA metrics and the single native-Torch reporting path."""

import numpy as np


def report_values(value, stats):
    value = value.float()
    means = value.new_tensor([stats["v_mean"], stats["a_mean"]])
    scales = value.new_tensor([stats["v_std"], stats["a_std"]])
    value = (value * scales + means).clamp(1.0, 9.0)
    return (value * 100).round() / 100


def correlation(prediction, gold):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(gold, dtype=np.float64)
    vp, vy = np.var(p), np.var(y)
    covariance = np.mean((p - p.mean()) * (y - y.mean()))
    pcc = float(covariance / np.sqrt(vp * vy)) if vp > 1e-18 and vy > 1e-18 else 0.0
    denominator = vp + vy + (p.mean() - y.mean()) ** 2
    ccc = float(2 * covariance / denominator) if denominator > 1e-18 else 0.0
    return pcc, ccc


def metrics(rows):
    if not rows:
        raise ValueError("Metrics require nonempty predictions")
    values = np.asarray(
        [[r[k] for k in ("v_pred", "a_pred", "v_true", "a_true")] for r in rows],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite prediction or label")
    pred, gold = values[:, :2], values[:, 2:]
    mse = np.mean((pred - gold) ** 2, axis=0)
    correlations = [correlation(pred[:, i], gold[:, i]) for i in range(2)]
    return dict(
        rmse_v=float(np.sqrt(mse[0])),
        rmse_a=float(np.sqrt(mse[1])),
        rmse_va=float(np.sqrt(mse.sum())),
        pcc_macro=float(np.mean([v[0] for v in correlations])),
        ccc_macro=float(np.mean([v[1] for v in correlations])),
        std_gap_v=float(abs(np.std(pred[:, 0]) - np.std(gold[:, 0]))),
        std_gap_a=float(abs(np.std(pred[:, 1]) - np.std(gold[:, 1]))),
    )
