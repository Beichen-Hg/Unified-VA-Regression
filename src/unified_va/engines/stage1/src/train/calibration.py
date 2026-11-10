"""Post-hoc calibration strategies for VA regression.

Implements multiple calibration approaches:
- OLS: standard least-squares (current default)
- MeanShift: only adjust intercept, preserve variance
- Constrained: OLS with slope lower bound
- RidgeToIdentity: Ridge regression shrinking toward slope=1
  (closed-form, continuously adjustable via λ)
- TriAnchor: reliability and variance-aware closed-form shrinkage

All calibrators follow the same interface:
  fit(pred, gold) -> self
  transform(pred) -> calibrated
  get_params() -> dict
"""

import numpy as np
from typing import Optional


class OLSCalibrator:
    """Standard OLS: cal = intercept + slope * raw."""

    def __init__(self):
        self.intercept = 0.0
        self.slope = 1.0

    def fit(self, pred: np.ndarray, gold: np.ndarray) -> "OLSCalibrator":
        from numpy.polynomial.polynomial import polyfit
        coefs = polyfit(pred, gold, 1)
        self.intercept, self.slope = float(coefs[0]), float(coefs[1])
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * pred

    def get_params(self) -> dict:
        return {"method": "ols", "intercept": self.intercept, "slope": self.slope}


class MeanShiftCalibrator:
    """Only shift mean, keep slope=1: cal = raw + (mean_gold - mean_pred)."""

    def __init__(self):
        self.shift = 0.0

    def fit(self, pred: np.ndarray, gold: np.ndarray) -> "MeanShiftCalibrator":
        self.shift = float(gold.mean() - pred.mean())
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return pred + self.shift

    def get_params(self) -> dict:
        return {"method": "mean_shift", "intercept": self.shift, "slope": 1.0}


class ConstrainedCalibrator:
    """OLS with slope lower bound: slope = max(s_min, ols_slope)."""

    def __init__(self, s_min: float = 0.8):
        self.s_min = s_min
        self.intercept = 0.0
        self.slope = 1.0

    def fit(self, pred: np.ndarray, gold: np.ndarray) -> "ConstrainedCalibrator":
        from numpy.polynomial.polynomial import polyfit
        coefs = polyfit(pred, gold, 1)
        ols_slope = float(coefs[1])
        self.slope = max(self.s_min, ols_slope)
        # Re-fit intercept with constrained slope
        self.intercept = float(gold.mean() - self.slope * pred.mean())
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * pred

    def get_params(self) -> dict:
        return {"method": "constrained", "intercept": self.intercept,
                "slope": self.slope, "s_min": self.s_min}


class RidgeToIdentityCalibrator:
    """Ridge regression shrinking slope toward 1 (identity).

    Demeaned formulation:
        min_α  Σ(y' - α·ŷ')² + λ·(α - 1)²

    Closed-form solution:
        α_λ = (Σŷ'y' + λ) / (Σŷ'² + λ)

    When λ=0: reduces to OLS slope.
    When λ→∞: α→1 (mean-shift only).
    """

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.slope = 1.0
        self.intercept = 0.0

    def fit(self, pred: np.ndarray, gold: np.ndarray) -> "RidgeToIdentityCalibrator":
        pred_mean = pred.mean()
        gold_mean = gold.mean()
        p_centered = pred - pred_mean
        g_centered = gold - gold_mean

        numerator = float(np.dot(p_centered, g_centered) + self.lam)
        denominator = float(np.dot(p_centered, p_centered) + self.lam)
        self.slope = numerator / denominator
        self.intercept = float(gold_mean - self.slope * pred_mean)
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * pred

    def get_params(self) -> dict:
        return {"method": "ridge_to_identity", "intercept": self.intercept,
                "slope": self.slope, "lambda": self.lam}


class ReliabilityGatedCalibrator:
    """Reliability-gated Ridge-to-Identity calibration.

    Automatically determines λ from dev-set calibration reliability:
        λ = SS_x · (1 - r²) / r²

    This yields the James-Stein shrinkage estimator:
        α_shrunk = r² · α_OLS + (1 - r²) · 1

    Interpretation: weight r² on OLS estimate, weight (1-r²) on identity prior.
    - r → 1: trust OLS fully (λ → 0)
    - r → 0: fall back to mean-shift (λ → ∞)
    - No hyperparameters to tune.
    """

    def __init__(self, r_floor: float = 0.05):
        self.r_floor = r_floor  # prevent division by zero
        self.slope = 1.0
        self.intercept = 0.0
        self.r = 0.0
        self.r_sq = 0.0
        self.lam = 0.0
        self.ols_slope = 1.0
        self.se_slope = 0.0

    def fit(self, pred: np.ndarray, gold: np.ndarray) -> "ReliabilityGatedCalibrator":
        n = len(pred)
        pred_mean = pred.mean()
        gold_mean = gold.mean()
        p_centered = pred - pred_mean
        g_centered = gold - gold_mean

        ss_x = float(np.dot(p_centered, p_centered))
        ss_xy = float(np.dot(p_centered, g_centered))

        # OLS slope and correlation
        self.ols_slope = ss_xy / max(ss_x, 1e-12)
        r = float(np.corrcoef(pred, gold)[0, 1]) if pred.std() > 1e-8 else 0.0
        self.r = r
        self.r_sq = max(r ** 2, self.r_floor ** 2)

        # OLS slope standard error
        residuals = g_centered - self.ols_slope * p_centered
        sigma_sq = float(np.dot(residuals, residuals)) / max(n - 2, 1)
        self.se_slope = float(np.sqrt(sigma_sq / max(ss_x, 1e-12)))

        # Reliability-gated λ: James-Stein optimal
        self.lam = ss_x * (1.0 - self.r_sq) / self.r_sq

        # Ridge slope
        self.slope = (ss_xy + self.lam) / (ss_x + self.lam)
        self.intercept = float(gold_mean - self.slope * pred_mean)
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * pred

    def get_params(self) -> dict:
        return {
            "method": "reliability_gated",
            "intercept": self.intercept,
            "slope": self.slope,
            "lambda": self.lam,
            "r": self.r,
            "r_sq": self.r_sq,
            "ols_slope": self.ols_slope,
            "se_slope": self.se_slope,
        }


class TriAnchorCalibrator:
    """Tri-anchor reliability & distribution-aware calibration.

    Objective on slope α:
      (α - α_OLS)^2 + w_r (α - 1)^2 + w_v (α - ρ)^2

    where:
      ρ = σ_y / σ_x
      q = max(r^2, q_min)
      w_r = (1 - q) / q
      w_v = (1 - q) * |log(σ_x / σ_y)|

    Closed-form:
      α* = (α_OLS + w_r * 1 + w_v * ρ) / (1 + w_r + w_v)
    """

    def __init__(self, q_min: float = 0.0025, eps: float = 1e-8,
                 enable_clip: bool = True, clip_eps: float = 0.25):
        self.q_min = q_min
        self.eps = eps
        self.enable_clip = enable_clip
        self.clip_eps = clip_eps
        self.intercept = 0.0
        self.slope = 1.0
        self.alpha_before_clip = 1.0
        self.alpha_ols = 1.0
        self.alpha_var = 1.0
        self.r = 0.0
        self.q = q_min
        self.w_r = 0.0
        self.w_v = 0.0

    def fit(self, pred: np.ndarray, gold: np.ndarray) -> "TriAnchorCalibrator":
        pred_mean = pred.mean()
        gold_mean = gold.mean()
        p_centered = pred - pred_mean
        g_centered = gold - gold_mean

        ss_x = float(np.dot(p_centered, p_centered))
        ss_xy = float(np.dot(p_centered, g_centered))

        sigma_x = float(pred.std())
        sigma_y = float(gold.std())
        if ss_x <= self.eps:
            self.alpha_ols = 1.0
        else:
            self.alpha_ols = ss_xy / ss_x

        # For near-constant predictions, keep variance anchor conservative.
        self.alpha_var = (
            sigma_y / (sigma_x + self.eps) if sigma_x > self.eps else 1.0
        )

        if sigma_x <= self.eps or sigma_y <= self.eps:
            self.r = 0.0
        else:
            self.r = float(np.corrcoef(pred, gold)[0, 1])
            if not np.isfinite(self.r):
                self.r = 0.0

        r_sq = float(self.r ** 2)
        q_floor = min(max(self.q_min, self.eps), 1.0)
        self.q = min(max(r_sq, q_floor), 1.0)
        self.w_r = (1.0 - self.q) / self.q
        self.w_v = (1.0 - self.q) * abs(
            float(np.log((sigma_x + self.eps) / (sigma_y + self.eps)))
        )

        denom = 1.0 + self.w_r + self.w_v
        self.alpha_before_clip = (
            self.alpha_ols + self.w_r * 1.0 + self.w_v * self.alpha_var
        ) / denom

        self.slope = self.alpha_before_clip
        if self.enable_clip:
            clip_span = abs(float(self.clip_eps))
            lower = self.alpha_var * float(np.exp(-clip_span))
            upper = self.alpha_var * float(np.exp(clip_span))
            self.slope = float(np.clip(self.slope, lower, upper))

        self.intercept = float(gold_mean - self.slope * pred_mean)
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * pred

    def get_params(self) -> dict:
        return {
            "method": "tri_anchor",
            "intercept": self.intercept,
            "slope": self.slope,
            "alpha_ols": self.alpha_ols,
            "alpha_var": self.alpha_var,
            "r": self.r,
            "q": self.q,
            "w_r": self.w_r,
            "w_v": self.w_v,
            "enable_clip": self.enable_clip,
            "clip_eps": self.clip_eps,
            "alpha_before_clip": self.alpha_before_clip,
            "q_min": self.q_min,
            "eps": self.eps,
        }


def _loo_rmse_ridge(pred: np.ndarray, gold: np.ndarray, lam: float) -> float:
    """LOO-CV RMSE for RidgeToIdentity at given λ.

    Uses the closed-form LOO formula for ridge regression:
        e_i = (y_i - ŷ_cal_i) / (1 - h_ii)
    where h_ii is the leverage of sample i.

    For our 1D demeaned case with intercept:
        Design matrix X = [1, ŷ_centered]
        H = X (X'X + λI')⁻¹ X'  (with λ only on slope, not intercept)
    """
    n = len(pred)
    pm, gm = pred.mean(), gold.mean()
    pc = pred - pm

    # Fit ridge slope
    ss = float(np.dot(pc, pc))
    sg = float(np.dot(pc, gold - gm))
    slope = (sg + lam) / (ss + lam)
    intercept = gm - slope * pm

    # Predictions
    cal = intercept + slope * pred

    # Leverage: h_ii for [1, pc_i] design with ridge penalty on slope only
    # H = X (X'X + Λ)⁻¹ X' where Λ = diag(0, λ)
    # X'X + Λ = [[n, 0], [0, ss + λ]]  (since Σpc = 0)
    # (X'X + Λ)⁻¹ = [[1/n, 0], [0, 1/(ss+λ)]]
    # h_ii = 1/n + pc_i² / (ss + λ)
    h = 1.0 / n + pc ** 2 / (ss + lam)

    residuals = gold - cal
    loo_residuals = residuals / (1.0 - h)
    return float(np.sqrt(np.mean(loo_residuals ** 2)))


def select_lambda_loo(pred: np.ndarray, gold: np.ndarray,
                      candidates: Optional[list] = None) -> float:
    """Select optimal λ for RidgeToIdentity via LOO-CV on dev set.

    Returns the λ that minimizes LOO-RMSE.
    """
    if candidates is None:
        candidates = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5,
                      1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    best_lam, best_rmse = 0.0, float("inf")
    for lam in candidates:
        r = _loo_rmse_ridge(pred, gold, lam)
        if r < best_rmse:
            best_rmse = r
            best_lam = lam
    return best_lam


def make_calibrator(method: str = "ols", **kwargs):
    """Factory: create a calibrator by name.

    Args:
        method: one of "ols", "mean_shift", "constrained",
                "ridge_to_identity", "ridge_to_identity_cv",
                "reliability_gated", "tri_anchor"
        **kwargs: passed to calibrator constructor.
            - constrained: s_min (default 0.8)
            - ridge_to_identity: lam (default 1.0)
            - ridge_to_identity_cv: candidates (optional list of λ)
            - reliability_gated: r_floor (default 0.05)
            - tri_anchor: q_min, eps, enable_clip, clip_eps
    """
    if method == "ols":
        return OLSCalibrator()
    elif method == "mean_shift":
        return MeanShiftCalibrator()
    elif method == "constrained":
        return ConstrainedCalibrator(s_min=kwargs.get("s_min", 0.8))
    elif method == "ridge_to_identity":
        return RidgeToIdentityCalibrator(lam=kwargs.get("lam", 1.0))
    elif method == "ridge_to_identity_cv":
        # Placeholder — actual λ selected at fit time
        return _RidgeToIdentityCVCalibrator(
            candidates=kwargs.get("candidates", None))
    elif method == "adaptive":
        return AdaptiveCalibrator(margin=kwargs.get("margin", 0.01))
    elif method == "reliability_gated":
        return ReliabilityGatedCalibrator(
            r_floor=kwargs.get("r_floor", 0.05))
    elif method == "tri_anchor":
        return TriAnchorCalibrator(
            q_min=kwargs.get("q_min", 0.0025),
            eps=kwargs.get("eps", 1e-8),
            enable_clip=kwargs.get("enable_clip", True),
            clip_eps=kwargs.get("clip_eps", 0.25),
        )
    else:
        raise ValueError(f"Unknown calibration method: {method}")


class _RidgeToIdentityCVCalibrator:
    """RidgeToIdentity with automatic λ selection via LOO-CV."""

    def __init__(self, candidates=None):
        self.candidates = candidates
        self.selected_lam = None
        self._inner = None

    def fit(self, pred: np.ndarray, gold: np.ndarray) -> "_RidgeToIdentityCVCalibrator":
        self.selected_lam = select_lambda_loo(pred, gold, self.candidates)
        self._inner = RidgeToIdentityCalibrator(lam=self.selected_lam)
        self._inner.fit(pred, gold)
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return self._inner.transform(pred)

    def get_params(self) -> dict:
        p = self._inner.get_params()
        p["method"] = "ridge_to_identity_cv"
        p["selected_lambda"] = self.selected_lam
        return p


class _IdentityCalibrator:
    """No-op calibrator: returns predictions unchanged."""

    def fit(self, pred: np.ndarray, gold: np.ndarray) -> "_IdentityCalibrator":
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return pred

    def get_params(self) -> dict:
        return {"method": "identity", "intercept": 0.0, "slope": 1.0}


def _loo_rmse_calibrator(pred: np.ndarray, gold: np.ndarray, make_cal) -> float:
    """LOO-CV RMSE for an arbitrary calibrator factory.

    Args:
        pred: (N,) predictions
        gold: (N,) gold values
        make_cal: callable() -> calibrator with fit/transform interface

    Returns:
        LOO-CV RMSE estimate
    """
    n = len(pred)
    loo_errors = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        cal = make_cal()
        cal.fit(pred[mask], gold[mask])
        cal_pred_i = cal.transform(pred[i:i+1])[0]
        loo_errors[i] = gold[i] - cal_pred_i
    return float(np.sqrt(np.mean(loo_errors ** 2)))


class AdaptiveCalibrator:
    """Automatically selects the best calibration strategy via LOO-CV.

    Candidates: identity (no calibration), MeanShift, ReliabilityGated.
    Picks whichever achieves lowest LOO-CV RMSE on the dev set.

    A conservative margin ensures we only calibrate when the LOO improvement
    is substantial enough to generalize (default 1%).
    """

    def __init__(self, margin: float = 0.02):
        self._selected = None
        self._selected_name = "identity"
        self._margin = margin

    def fit(self, pred: np.ndarray, gold: np.ndarray) -> "AdaptiveCalibrator":
        n = len(pred)

        # Adaptive margin: larger for small dev sets to avoid overfitting
        # Base margin + 0.5/sqrt(n) correction
        # n=100: 0.02+0.05=0.07, n=200: 0.02+0.035=0.055, n=500: 0.02+0.022=0.042
        effective_margin = self._margin + 0.5 / max(np.sqrt(n), 1.0)

        # Candidate 1: identity (no calibration) — LOO-RMSE = raw RMSE
        loo_identity = float(np.sqrt(np.mean((gold - pred) ** 2)))

        # Candidate 2: MeanShift — closed-form LOO
        # LOO mean shift: shift_i = mean(gold[-i]) - mean(pred[-i])
        # cal_i = pred_i + shift_i
        # shift_i = (sum(gold) - gold_i)/(n-1) - (sum(pred) - pred_i)/(n-1)
        # cal_i = pred_i + (S_g - gold_i - S_p + pred_i) / (n-1)
        #       where S_g = sum(gold), S_p = sum(pred)
        s_g, s_p = gold.sum(), pred.sum()
        loo_cal_ms = pred + (s_g - gold - s_p + pred) / max(n - 1, 1)
        loo_meanshift = float(np.sqrt(np.mean((gold - loo_cal_ms) ** 2)))

        # Candidate 3: RidgeToIdentity with LOO-CV λ selection
        # This covers OLS (λ=0) through identity (λ→∞)
        candidates_lam = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5,
                          1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0]
        best_ridge_rmse = float("inf")
        best_ridge_lam = 1000.0
        for lam in candidates_lam:
            r = _loo_rmse_ridge(pred, gold, lam)
            if r < best_ridge_rmse:
                best_ridge_rmse = r
                best_ridge_lam = lam

        # Pick best, but prefer identity unless calibration improves by > margin
        results = [
            ("identity", loo_identity, _IdentityCalibrator()),
            ("mean_shift", loo_meanshift, MeanShiftCalibrator()),
            ("ridge_cv", best_ridge_rmse, RidgeToIdentityCalibrator(lam=best_ridge_lam)),
        ]
        threshold = loo_identity * (1.0 - effective_margin)
        # Find best non-identity candidate
        best_cal_name, best_cal_score, best_cal = "identity", loo_identity, results[0][2]
        for name, score, cal in results[1:]:
            if score < best_cal_score:
                best_cal_name, best_cal_score, best_cal = name, score, cal
        # Only use calibration if it beats identity by margin
        if best_cal_score < threshold:
            self._selected_name = best_cal_name
            self._selected = best_cal
        else:
            self._selected_name = "identity"
            self._selected = results[0][2]

        self._selected.fit(pred, gold)

        self._loo_scores = {name: score for name, score, _ in results}
        self._ridge_lam = best_ridge_lam
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        return self._selected.transform(pred)

    def get_params(self) -> dict:
        p = self._selected.get_params()
        p["method"] = f"adaptive({self._selected_name})"
        p["loo_scores"] = self._loo_scores
        if self._selected_name == "ridge_cv":
            p["ridge_lambda"] = self._ridge_lam
        return p
