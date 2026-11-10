"""VA regression metrics: RMSE for valence, arousal, and combined."""
import numpy as np
from scipy.stats import pearsonr


def rmse_v(v_pred: np.ndarray, v_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((v_pred - v_true) ** 2)))


def rmse_a(a_pred: np.ndarray, a_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a_pred - a_true) ** 2)))


def rmse_va(v_pred: np.ndarray, a_pred: np.ndarray,
            v_true: np.ndarray, a_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((v_pred - v_true) ** 2 + (a_pred - a_true) ** 2)))


def pcc_v(v_pred: np.ndarray, v_true: np.ndarray) -> float:
    if np.std(v_pred) < 1e-9 or np.std(v_true) < 1e-9:
        return 0.0
    return float(pearsonr(v_pred, v_true)[0])


def pcc_a(a_pred: np.ndarray, a_true: np.ndarray) -> float:
    if np.std(a_pred) < 1e-9 or np.std(a_true) < 1e-9:
        return 0.0
    return float(pearsonr(a_pred, a_true)[0])


def fn_risk_error_correlation(fn_risk: np.ndarray,
                              v_pred: np.ndarray, a_pred: np.ndarray,
                              v_true: np.ndarray, a_true: np.ndarray) -> dict:
    """计算 fn_risk 与预测误差的相关性（诊断用）.

    Returns:
        {"pearson": float, "spearman": float}
    """
    from scipy.stats import pearsonr, spearmanr
    err = np.sqrt((v_pred - v_true) ** 2 + (a_pred - a_true) ** 2)
    pr, _ = pearsonr(fn_risk, err)
    sr, _ = spearmanr(fn_risk, err)
    return {"pearson": round(float(pr), 4), "spearman": round(float(sr), 4)}
