"""Loss functions: BCE, MSE, KL, JS, InfoNCE, distill KL, optional Sinkhorn OT."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1e-9)


def combine_primary_aux_losses(
    per_sample_loss: torch.Tensor,
    is_auxiliary: torch.Tensor | None,
    auxiliary_loss_weight: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine primary and auxiliary sample losses with explicit group weighting.

    Semantics follow the project docs:
        loss = mean(primary) + auxiliary_loss_weight * mean(auxiliary)

    Optional ``sample_weights`` are applied within each group before taking the mean.
    """
    if is_auxiliary is None:
        if sample_weights is not None:
            return (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp(min=1e-9)
        return per_sample_loss.mean()

    aux_mask = (is_auxiliary > 0.5).float()
    primary_mask = 1.0 - aux_mask

    has_primary = primary_mask.sum().item() > 0
    has_aux = aux_mask.sum().item() > 0
    if not has_primary and not has_aux:
        return torch.tensor(0.0, device=per_sample_loss.device, requires_grad=True)

    loss = torch.tensor(0.0, device=per_sample_loss.device)
    if has_primary:
        loss = loss + masked_weighted_mean(per_sample_loss, primary_mask, sample_weights)
    if has_aux:
        loss = loss + auxiliary_loss_weight * masked_weighted_mean(
            per_sample_loss, aux_mask, sample_weights
        )
    return loss


def va_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = "mse",
            weight_v: float = 1.0, weight_a: float = 1.0,
            sample_weights: torch.Tensor | None = None,
            is_auxiliary: torch.Tensor | None = None,
            auxiliary_loss_weight: float = 1.0,
            loss_a_mode: str | None = None,
            a_tail_weight: float = 1.0,
            a_tail_lo: float | None = None,
            a_tail_hi: float | None = None) -> torch.Tensor:
    """Loss for VA regression. pred/target: (B, 2).
    sample_weights: (B,) optional per-sample weights (e.g. VA-region reweighting).
    loss_a_mode: override loss function for A (e.g. "mse" when mode="huber").
    a_tail_weight/a_tail_lo/a_tail_hi: upweight A samples in tail regions."""
    loss_fn_v = F.smooth_l1_loss if mode == "huber" else F.mse_loss
    a_mode = loss_a_mode or mode
    loss_fn_a = F.smooth_l1_loss if a_mode == "huber" else F.mse_loss
    # Fast path: same loss, equal weights, no tail reweighting → mean(dim=-1)
    use_fast = (loss_a_mode is None and weight_v == 1.0 and weight_a == 1.0
                and a_tail_weight <= 1.0)
    if use_fast:
        per_sample = loss_fn_v(pred, target, reduction="none").mean(dim=-1)
    else:
        loss_v = loss_fn_v(pred[:, 0], target[:, 0], reduction="none")  # (B,)
        loss_a = loss_fn_a(pred[:, 1], target[:, 1], reduction="none")  # (B,)
        # Tail reweighting for A
        if a_tail_weight > 1.0 and a_tail_lo is not None and a_tail_hi is not None:
            a_true = target[:, 1]
            tail_mask = (a_true <= a_tail_lo) | (a_true >= a_tail_hi)
            a_weights = torch.ones_like(loss_a)
            a_weights[tail_mask] = a_tail_weight
            loss_a = loss_a * a_weights
        per_sample = weight_v * loss_v + weight_a * loss_a
    return combine_primary_aux_losses(
        per_sample,
        is_auxiliary=is_auxiliary,
        auxiliary_loss_weight=auxiliary_loss_weight,
        sample_weights=sample_weights,
    )


def gaussian_nll_va(mu: torch.Tensor, log_sigma: torch.Tensor,
                    target: torch.Tensor,
                    sample_weights: torch.Tensor | None = None,
                    is_auxiliary: torch.Tensor | None = None,
                    auxiliary_loss_weight: float = 1.0) -> torch.Tensor:
    """Gaussian NLL for VA regression. mu/log_sigma/target: (B, 2).
    sample_weights: (B,) optional per-sample weights."""
    per_sample = (log_sigma + 0.5 * ((target - mu) ** 2) * (-2 * log_sigma).exp()).mean(dim=1)
    return combine_primary_aux_losses(
        per_sample,
        is_auxiliary=is_auxiliary,
        auxiliary_loss_weight=auxiliary_loss_weight,
        sample_weights=sample_weights,
    )


def sigma_weights(log_sigma: torch.Tensor, mode: str = "downweight") -> torch.Tensor:
    """Compute per-sample weights from predicted log_sigma. Returns (B,) weights.
    mode='downweight': w = 1 / (1 + mean_sigma), high sigma → low weight
    mode='upweight': w = mean_sigma, high sigma → high weight (hard example mining)
    """
    sigma = log_sigma.detach().exp().mean(dim=1)  # (B,)
    if mode == "upweight":
        return sigma / sigma.mean().clamp(min=1e-9)
    # downweight (default)
    return 1.0 / (1.0 + sigma)


def format_va(v: float, a: float) -> str:
    """Format V,A to official 'V#A' string."""
    v = max(1.0, min(9.0, round(v, 2)))
    a = max(1.0, min(9.0, round(a, 2)))
    return f"{v:.2f}#{a:.2f}"


def masked_mse(pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(pred, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1e-9)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """KL(p || q), both are probability distributions."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def distill_kl(teacher: torch.Tensor, student_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """KL distillation loss."""
    t_soft = F.softmax(teacher / temperature, dim=-1) if teacher.requires_grad or teacher.dim() > 1 else teacher
    s_soft = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(s_soft, t_soft, reduction="batchmean") * (temperature ** 2)


def info_nce(anchors: torch.Tensor, positives: torch.Tensor, temperature: float = 0.07,
             neg_mask: torch.Tensor | None = None) -> torch.Tensor:
    """InfoNCE contrastive loss with optional negative masking."""
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    all_feats = torch.cat([positives, anchors], dim=0)
    sim = anchors @ all_feats.T / temperature
    n = anchors.size(0)
    labels = torch.arange(n, device=anchors.device)
    if neg_mask is not None:
        # neg_mask: (n, 2n) — 0 means exclude from negatives
        sim = sim + (1 - neg_mask.float()) * (-1e9)
    return F.cross_entropy(sim, labels)


def sup_con_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss."""
    features = F.normalize(features, dim=-1)
    n = features.size(0)
    sim = features @ features.T / temperature
    # Mask out self-similarity
    mask_self = torch.eye(n, device=features.device).bool()
    sim.masked_fill_(mask_self, -1e9)
    # Positive mask: same label
    label_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    label_eq.masked_fill_(mask_self, 0)
    n_pos = label_eq.sum(dim=1).clamp(min=1)
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-9))
    loss = -(label_eq * log_prob).sum(dim=1) / n_pos
    return loss.mean()


def arousal_rank_loss(pred: torch.Tensor, target: torch.Tensor,
                      margin: float = 0.0) -> torch.Tensor:
    """Pairwise ranking loss for arousal dimension.

    For each pair (i, j) where target_a[i] > target_a[j],
    enforce pred_a[i] > pred_a[j] - margin.

    This forces the model to learn relative ordering of arousal,
    which is easier than predicting absolute values.
    """
    pred_a = pred[:, 1]    # (B,)
    target_a = target[:, 1]  # (B,)
    n = pred_a.size(0)
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    # Pairwise differences
    pred_diff = pred_a.unsqueeze(0) - pred_a.unsqueeze(1)    # (B, B)
    target_diff = target_a.unsqueeze(0) - target_a.unsqueeze(1)  # (B, B)

    # Only consider pairs where target differs (avoid ties)
    valid = (target_diff.abs() > 0.01)

    # Sign agreement: pred should have same sign as target
    # margin_ranking_loss: loss = max(0, -sign * (pred_i - pred_j) + margin)
    sign = target_diff.sign()
    pair_loss = F.relu(-sign * pred_diff + margin)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    return pair_loss[valid].mean()


def ordinal_arousal_loss(logits: torch.Tensor, target_a: torch.Tensor,
                         boundaries: list[float] = None) -> torch.Tensor:
    """Ordinal regression loss for arousal.

    Treats arousal as ordinal: predict P(A > threshold) for each threshold.
    This forces the model to learn clear decision boundaries.

    Args:
        logits: (B, K) raw logits for K ordinal thresholds
        target_a: (B,) arousal values
        boundaries: K threshold values, default [4.8, 5.2, 5.5, 6.0]
    """
    if boundaries is None:
        boundaries = [4.8, 5.2, 5.5, 6.0]

    device = logits.device
    boundaries_t = torch.tensor(boundaries, device=device)

    # Binary targets: 1 if target_a > boundary, 0 otherwise
    # Shape: (B, K)
    binary_targets = (target_a.unsqueeze(1) > boundaries_t.unsqueeze(0)).float()

    # Binary cross-entropy for each threshold
    loss = F.binary_cross_entropy_with_logits(logits, binary_targets)
    return loss


def sinkhorn_ot(cost: torch.Tensor, reg: float = 0.1, n_iter: int = 50) -> torch.Tensor:
    """Sinkhorn OT assignment (torch-native). cost: (n, k)."""
    try:
        import ot as pot_lib
        n, k = cost.shape
        a = np.ones(n) / n
        b = np.ones(k) / k
        T = pot_lib.sinkhorn(a, b, cost.detach().cpu().numpy(), reg=reg, numItermax=n_iter)
        return torch.from_numpy(T).float().to(cost.device)
    except ImportError:
        # Pure torch fallback
        n, k = cost.shape
        log_K = -cost / reg
        log_u = torch.zeros(n, device=cost.device)
        log_v = torch.zeros(k, device=cost.device)
        for _ in range(n_iter):
            log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1) - np.log(n)
            log_v = -torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0) - np.log(k)
        T = torch.exp(log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0))
        return T / T.sum()


# ============================================================
# D4_proto losses
# ============================================================

def masked_weighted_mean(per_sample_loss: torch.Tensor, mask: torch.Tensor,
                         weight: torch.Tensor | None = None) -> torch.Tensor:
    """带 mask 和可选权重的样本级损失均值. 全 mask=0 时返回 0."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=per_sample_loss.device, requires_grad=True)
    w = mask
    if weight is not None:
        w = w * weight
    return (per_sample_loss * w).sum() / w.sum().clamp(min=1e-9)


def proto_distill_loss(proto_logits: torch.Tensor, q_target: torch.Tensor,
                       mask: torch.Tensor, weight: torch.Tensor | None = None,
                       loss_type: str = "kl") -> torch.Tensor:
    """原型分布蒸馏损失.

    Args:
        proto_logits: (B, K) 模型预测的原型 logits
        q_target: (B, K) 离线构建的目标分布
        mask: (B,) proto_mask
        weight: (B,) 可选样本权重（如 stability weight）
        loss_type: "kl" 或 "soft_ce"

    Returns:
        scalar loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=proto_logits.device, requires_grad=True)

    if loss_type == "soft_ce":
        # soft cross-entropy: -sum(q * log_softmax(logits))
        log_p = F.log_softmax(proto_logits, dim=-1)
        per_sample = -(q_target * log_p).sum(dim=-1)  # (B,)
    else:
        # KL(q_target || softmax(logits))
        log_p = F.log_softmax(proto_logits, dim=-1)
        q_safe = q_target.clamp(min=1e-9)
        per_sample = (q_safe * (q_safe.log() - log_p)).sum(dim=-1)  # (B,)

    return masked_weighted_mean(per_sample, mask, weight)


def proto_va_consistency_loss(va_pred: torch.Tensor, proto_probs: torch.Tensor,
                              proto_va_anchors: torch.Tensor, mask: torch.Tensor,
                              weight: torch.Tensor | None = None) -> torch.Tensor:
    """原型 VA 一致性损失: VA 预测应与原型加权 VA anchor 一致.

    Args:
        va_pred: (B, 2) 模型 VA 预测
        proto_probs: (B, K) softmax(proto_logits)
        proto_va_anchors: (K, 2) 每个原型的 VA 均值
        mask: (B,) proto_mask
        weight: (B,) 可选样本权重

    Returns:
        scalar loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=va_pred.device, requires_grad=True)

    # 原型加权 VA: (B, K) @ (K, 2) → (B, 2)
    va_proto = proto_probs @ proto_va_anchors
    per_sample = ((va_pred - va_proto) ** 2).mean(dim=-1)  # (B,)
    return masked_weighted_mean(per_sample, mask, weight)


def teacher_va_distill_loss(
    student_va: torch.Tensor,
    teacher_va: torch.Tensor,
    mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    mode: str = "mse",
    channel_weights: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Teacher-aware VA distillation loss.

    Args:
        student_va: (B, 2) 学生模型 VA 预测
        teacher_va: (B, 2) 教师软标签
        mask: (B,) teacher 有效标记（0/1），None 时全1
        weight: (B,) 样本级权重（confidence * lang_weight），None 时全1
        mode: "mse" 或 "smoothl1"
        channel_weights: (weight_v, weight_a) per-channel weights, None=equal

    Returns:
        scalar loss（mask 全0 时返回 0，不产生 NaN）
    """
    B = student_va.shape[0]
    if mask is None:
        mask = torch.ones(B, device=student_va.device)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_va.device, requires_grad=True)

    if mode == "smoothl1":
        per_dim = F.smooth_l1_loss(
            student_va, teacher_va, reduction="none")  # (B, 2)
    else:
        per_dim = (student_va - teacher_va) ** 2  # (B, 2)

    if channel_weights is not None:
        w_v, w_a = channel_weights
        per_sample = w_v * per_dim[:, 0] + w_a * per_dim[:, 1]
    else:
        per_sample = per_dim.mean(dim=-1)

    return masked_weighted_mean(per_sample, mask, weight)


# ============================================================
# D19: Calibration-aware training losses
# ============================================================

def variance_preservation_loss(pred: torch.Tensor, target: torch.Tensor,
                               dim_weight_v: float = 1.0,
                               dim_weight_a: float = 1.0) -> torch.Tensor:
    """Penalize |std(pred)/std(target) - 1| per dimension.

    Encourages model predictions to preserve the variance of the gold distribution,
    reducing reliance on post-hoc calibration.

    Args:
        pred: (B, 2) predicted VA
        target: (B, 2) gold VA
        dim_weight_v, dim_weight_a: per-dimension weights

    Returns:
        scalar loss (0 if batch too small)
    """
    if pred.shape[0] < 4:
        return torch.tensor(0.0, device=pred.device)
    std_pred = pred.std(dim=0).clamp(min=1e-6)   # (2,)
    std_tgt = target.std(dim=0).clamp(min=1e-6)   # (2,)
    ratio = std_pred / std_tgt
    penalty = (ratio - 1.0).abs()  # (2,)
    return dim_weight_v * penalty[0] + dim_weight_a * penalty[1]


def slope_alignment_loss(pred: torch.Tensor, target: torch.Tensor,
                         dim_weight_v: float = 1.0,
                         dim_weight_a: float = 1.0) -> torch.Tensor:
    """Penalize (slope - 1)^2 where slope = cov(pred, target) / var(pred).

    A perfectly calibrated predictor has slope=1 when regressing target on pred.

    Args:
        pred: (B, 2) predicted VA
        target: (B, 2) gold VA
        dim_weight_v, dim_weight_a: per-dimension weights

    Returns:
        scalar loss (0 if batch too small)
    """
    if pred.shape[0] < 4:
        return torch.tensor(0.0, device=pred.device)
    penalties = []
    weights = [dim_weight_v, dim_weight_a]
    for d in range(2):
        p = pred[:, d]
        t = target[:, d]
        p_mean = p.mean()
        t_mean = t.mean()
        cov = ((p - p_mean) * (t - t_mean)).mean()
        var_p = ((p - p_mean) ** 2).mean().clamp(min=1e-6)
        slope = cov / var_p
        penalties.append(weights[d] * (slope - 1.0) ** 2)
    return penalties[0] + penalties[1]

def synthetic_uncertainty_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    uncertainty_va: torch.Tensor,
    mode: str = "mse",
    weight_v: float = 1.0,
    weight_a: float = 1.0,
) -> torch.Tensor:
    """Uncertainty-weighted loss for synthetic samples.

    Lower uncertainty → higher weight (more reliable synthetic label).

    Args:
        pred: (B, 2) student VA predictions
        target: (B, 2) synthetic VA labels
        uncertainty_va: (B, 2) per-sample uncertainty (std from multi-sampling)
        mode: "mse" or "huber"
        weight_v, weight_a: channel weights

    Returns:
        scalar loss
    """
    if pred.shape[0] == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    loss_fn = F.smooth_l1_loss if mode == "huber" else F.mse_loss
    per_dim = loss_fn(pred, target, reduction="none")  # (B, 2)

    # Uncertainty weighting: w = 1 / (1 + sigma)
    sample_w = 1.0 / (1.0 + uncertainty_va)  # (B, 2)

    if weight_v != 1.0 or weight_a != 1.0:
        weighted = weight_v * per_dim[:, 0] * sample_w[:, 0] + \
                   weight_a * per_dim[:, 1] * sample_w[:, 1]
    else:
        weighted = (per_dim * sample_w).mean(dim=-1)  # (B,)

    return weighted.mean()


# ============================================================
# D7-v2: soft bin distribution distillation
# ============================================================

def soft_bin_distill_loss(
    student_logits: torch.Tensor,
    teacher_soft_targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    mode: str = "kl",
) -> torch.Tensor:
    """Soft bin distribution distillation loss (D7-v2).

    Args:
        student_logits: (B, num_bins) raw logits from student bin head
        teacher_soft_targets: (B, num_bins) soft probability targets
        mask: (B,) validity mask
        weight: (B,) per-sample weights
        mode: "kl" or "ce" (cross-entropy with soft labels)

    Returns:
        scalar loss
    """
    B = student_logits.shape[0]
    if mask is None:
        mask = torch.ones(B, device=student_logits.device)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    log_p = F.log_softmax(student_logits, dim=-1)

    if mode == "kl":
        q = teacher_soft_targets.clamp(min=1e-9)
        per_sample = (q * (q.log() - log_p)).sum(dim=-1)
    else:  # ce
        per_sample = -(teacher_soft_targets * log_p).sum(dim=-1)

    return masked_weighted_mean(per_sample, mask, weight)


# ============================================================
# D23: R-Drop (Dropout Consistency Regularization)
# ============================================================

def rdrop_loss(pred1: torch.Tensor, pred2: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    """R-Drop consistency loss between two predictions with different dropout masks.

    Forces the model to produce consistent predictions under different dropout masks,
    improving robustness and reducing overfitting.

    Args:
        pred1: (B, 2) first VA prediction
        pred2: (B, 2) second VA prediction (same input, different dropout)
        mode: "mse" or "kl" (for distribution matching)

    Returns:
        scalar consistency loss
    """
    if mode == "kl":
        # Treat predictions as distributions (requires softmax)
        # For VA regression, KL mode is less common
        p1 = F.softmax(pred1, dim=-1)
        p2 = F.softmax(pred2, dim=-1)
        kl1 = F.kl_div(p1.log(), p2, reduction="batchmean")
        kl2 = F.kl_div(p2.log(), p1, reduction="batchmean")
        return (kl1 + kl2) / 2
    else:
        # MSE mode (default for regression)
        return F.mse_loss(pred1, pred2)


# ============================================================
# D24: Mixup (Feature Space Data Augmentation)
# ============================================================

def sample_mixup_lambda(alpha: float, device: torch.device) -> torch.Tensor:
    """Sample mixing coefficient from Beta distribution.

    Args:
        alpha: Beta distribution parameter (typical: 0.2-0.4)
        device: torch device

    Returns:
        scalar lambda in [0, 1]
    """
    if alpha > 0:
        # Sample from Beta(alpha, alpha)
        lam = torch.distributions.Beta(alpha, alpha).sample().to(device)
    else:
        # No mixup, return 1.0
        lam = torch.tensor(1.0, device=device)
    return lam


def mixup_data(h: torch.Tensor, target: torch.Tensor,
               lam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup to features and targets.

    Args:
        h: (B, D) feature vectors
        target: (B, 2) VA targets
        lam: scalar mixing coefficient

    Returns:
        h_mixed: (B, D) mixed features
        target_mixed: (B, 2) mixed targets
    """
    batch_size = h.size(0)
    if batch_size < 2:
        return h, target

    # Random permutation
    index = torch.randperm(batch_size, device=h.device)

    # Mix features and targets
    h_mixed = lam * h + (1 - lam) * h[index]
    target_mixed = lam * target + (1 - lam) * target[index]

    return h_mixed, target_mixed


# ============================================================
# D28: Contrastive Regression + Distribution Matching
# ============================================================

def contrastive_regression_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.07,
    threshold: float = 1.0,
    focus_dim: int | None = 1,  # 0=V, 1=A, None=both
    pair_mask: torch.Tensor | None = None,  # (B, B) bool: only consider masked pairs
) -> torch.Tensor:
    """Contrastive regression loss for continuous values.

    Unlike traditional contrastive learning (for classification), this loss
    treats samples with similar target values as positives in a continuous manner.

    Core idea: Samples with similar Arousal values should have similar representations.
    This helps prevent the model from collapsing predictions to the mean.

    Args:
        features: (B, D) normalized feature vectors
        targets: (B, 2) VA targets
        temperature: temperature parameter for softmax
        threshold: similarity threshold (samples within this distance are positives)
        focus_dim: which dimension to focus on (0=V, 1=A, None=both dimensions)

    Returns:
        scalar contrastive loss
    """
    if features.size(0) < 2:
        return torch.tensor(0.0, device=features.device)

    # Normalize features
    features = F.normalize(features, dim=-1)

    # Compute pairwise similarities in feature space
    sim_matrix = features @ features.T / temperature  # (B, B)

    # Compute pairwise similarities in target space
    if focus_dim is None:
        # Use both dimensions: Euclidean distance in VA space
        target_diff = torch.cdist(targets, targets, p=2)  # (B, B)
    else:
        # Use single dimension
        target_vals = targets[:, focus_dim]  # (B,)
        target_diff = (target_vals.unsqueeze(0) - target_vals.unsqueeze(1)).abs()  # (B, B)

    # Define positive pairs: samples with similar target values
    # Use Gaussian kernel: weight = exp(-diff^2 / (2 * threshold^2))
    target_sim = torch.exp(-target_diff ** 2 / (2 * threshold ** 2))  # (B, B)

    # Mask out self-similarity
    mask_self = torch.eye(features.size(0), device=features.device).bool()
    target_sim = target_sim.masked_fill(mask_self, 0)

    # Apply language-aware mask: only consider pairs within same language
    if pair_mask is not None:
        target_sim = target_sim * pair_mask.float()

    # Normalize target similarity to get weights
    target_weights = target_sim / (target_sim.sum(dim=1, keepdim=True).clamp(min=1e-9))

    # Compute contrastive loss
    # For each anchor, pull positives closer and push negatives away
    exp_sim = torch.exp(sim_matrix)
    exp_sim = exp_sim.masked_fill(mask_self, 0)

    # Weighted log probability
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-9))

    # Weight by target similarity
    loss = -(target_weights * log_prob).sum(dim=1).mean()

    return loss


def distribution_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "wasserstein",
    focus_dim: int | None = 1,  # 0=V, 1=A, None=both
    num_bins: int = 20,
) -> torch.Tensor:
    """Distribution matching loss to prevent prediction collapse.

    Forces the distribution of predictions to match the distribution of targets.
    This directly addresses the Arousal collapse problem.

    Args:
        pred: (B, 2) predicted VA values
        target: (B, 2) target VA values
        mode: "wasserstein" or "kl"
        focus_dim: which dimension to focus on (0=V, 1=A, None=both dimensions)
        num_bins: number of bins for histogram (for KL mode)

    Returns:
        scalar distribution matching loss
    """
    if pred.size(0) < 4:
        return torch.tensor(0.0, device=pred.device)

    if focus_dim is None:
        # Apply to both dimensions and average
        loss_v = distribution_matching_loss(pred, target, mode, focus_dim=0, num_bins=num_bins)
        loss_a = distribution_matching_loss(pred, target, mode, focus_dim=1, num_bins=num_bins)
        return (loss_v + loss_a) / 2

    pred_vals = pred[:, focus_dim]
    target_vals = target[:, focus_dim]

    if mode == "wasserstein":
        # 1D Wasserstein distance (Earth Mover's Distance)
        # Sort both distributions
        # Note: Use detach for sorting to avoid gradient issues
        # The gradient flows through the mean absolute difference
        pred_sorted, _ = torch.sort(pred_vals)
        target_sorted, _ = torch.sort(target_vals.detach())

        # Wasserstein-1 distance: mean absolute difference of sorted values
        w_dist = (pred_sorted - target_sorted).abs().mean()
        return w_dist

    elif mode == "kl":
        # KL divergence between histograms
        # Create histograms
        min_val = min(pred_vals.min().item(), target_vals.min().item(), 1.0)
        max_val = max(pred_vals.max().item(), target_vals.max().item(), 9.0)

        pred_hist = torch.histc(pred_vals, bins=num_bins, min=min_val, max=max_val)
        target_hist = torch.histc(target_vals, bins=num_bins, min=min_val, max=max_val)

        # Normalize to probabilities
        pred_prob = (pred_hist + 1e-9) / (pred_hist.sum() + num_bins * 1e-9)
        target_prob = (target_hist + 1e-9) / (target_hist.sum() + num_bins * 1e-9)

        # KL divergence
        kl = (target_prob * (target_prob.log() - pred_prob.log())).sum()
        return kl

    else:
        raise ValueError(f"Unknown mode: {mode}")


def arousal_anti_collapse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "ordinal",
    loss_weight: float = 1.0,
    boundaries: list[float] | None = None,
    ordinal_temperature: float = 0.5,
    distribution_mode: str = "wasserstein",
    distribution_num_bins: int = 20,
    sample_weights: torch.Tensor | None = None,
    is_auxiliary: torch.Tensor | None = None,
    auxiliary_loss_weight: float = 1.0,
    primary_only: bool = False,
) -> torch.Tensor:
    """A-specific anti-collapse loss for small-LLM VA regression.

    This helper is intentionally minimal and supports two modes:
      - ``ordinal``: works with batch_size=1 and is the recommended default.
      - ``distribution``: uses batch-level distribution matching on A, which is only
        effective when the effective in-batch sample count is sufficiently large.
    """
    if loss_weight <= 0:
        return torch.tensor(0.0, device=pred.device)

    pred_work = pred
    target_work = target
    aux_mask_work = is_auxiliary
    weight_work = sample_weights
    if primary_only and is_auxiliary is not None:
        primary_mask = is_auxiliary <= 0.5
        if primary_mask.sum().item() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred_work = pred_work[primary_mask]
        target_work = target_work[primary_mask]
        aux_mask_work = None
        if weight_work is not None:
            weight_work = weight_work[primary_mask]

    if mode == "distribution":
        return loss_weight * distribution_matching_loss(
            pred_work,
            target_work,
            mode=distribution_mode,
            focus_dim=1,
            num_bins=distribution_num_bins,
        )

    if mode == "ccc":
        # 1 - CCC_A: directly optimise concordance correlation coefficient
        pred_a = pred_work[:, 1]
        target_a = target_work[:, 1]
        if pred_a.size(0) < 4:
            return torch.tensor(0.0, device=pred.device)
        pred_mean = pred_a.mean()
        target_mean = target_a.mean().detach()
        pred_var = pred_a.var()
        target_var = target_a.var().detach()
        cov = ((pred_a - pred_mean) * (target_a - target_mean).detach()).mean()
        ccc = 2 * cov / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
        return loss_weight * (1.0 - ccc)

    if mode == "variance_penalty":
        # Penalise only when predicted std is below target std (collapse)
        pred_a = pred_work[:, 1]
        target_a = target_work[:, 1]
        if pred_a.size(0) < 4:
            return torch.tensor(0.0, device=pred.device)
        pred_std = pred_a.std()
        target_std = target_a.std().detach()
        gap = F.relu(target_std - pred_std)
        return loss_weight * (gap ** 2) / (target_std ** 2 + 1e-8)

    if mode != "ordinal":
        raise ValueError(f"Unknown arousal anti-collapse mode: {mode}")

    boundaries = boundaries or [4.8, 5.2, 5.8, 6.2]
    temperature = max(float(ordinal_temperature), 1e-6)
    pred_a = pred_work[:, 1]
    target_a = target_work[:, 1]
    th = pred_a.new_tensor(boundaries).view(1, -1)
    logits = (pred_a.unsqueeze(1) - th) / temperature
    labels = (target_a.unsqueeze(1) > th).float()
    per_sample = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)
    return loss_weight * combine_primary_aux_losses(
        per_sample,
        is_auxiliary=aux_mask_work,
        auxiliary_loss_weight=auxiliary_loss_weight,
        sample_weights=weight_work,
    )


def quantile_pinball_loss(
    quantile_preds: torch.Tensor,
    target_a: torch.Tensor,
    quantiles: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    is_auxiliary: torch.Tensor | None = None,
    auxiliary_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Pinball (quantile) loss for multi-quantile arousal prediction.

    Args:
        quantile_preds: (B, Q) predicted quantile values
        target_a: (B,) arousal targets
        quantiles: (Q,) quantile levels (e.g. [0.1, 0.25, 0.5, 0.75, 0.9])
        sample_weights: (B,) optional per-sample weights
        is_auxiliary: (B,) optional auxiliary data mask
        auxiliary_loss_weight: weight for auxiliary samples

    Returns:
        scalar pinball loss averaged over quantiles
    """
    # e = target - pred, shape (B, Q)
    errors = target_a.unsqueeze(1) - quantile_preds  # (B, Q)
    tau = quantiles.unsqueeze(0)  # (1, Q)
    # pinball: tau * max(e, 0) + (1 - tau) * max(-e, 0)
    per_sample_q = tau * F.relu(errors) + (1 - tau) * F.relu(-errors)  # (B, Q)
    per_sample = per_sample_q.mean(dim=1)  # (B,) average over quantiles
    return combine_primary_aux_losses(
        per_sample,
        is_auxiliary=is_auxiliary,
        auxiliary_loss_weight=auxiliary_loss_weight,
        sample_weights=sample_weights,
    )


def moment_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    focus_dim: int = 1,  # 0=V, 1=A
    moments: list[int] = [1, 2],  # which moments to match
) -> torch.Tensor:
    """Match statistical moments between prediction and target distributions.

    A simpler alternative to full distribution matching.
    Matches mean, variance, skewness, kurtosis, etc.

    Args:
        pred: (B, 2) predicted VA values
        target: (B, 2) target VA values
        focus_dim: which dimension to focus on (0=V, 1=A)
        moments: list of moments to match (1=mean, 2=variance, 3=skewness, 4=kurtosis)

    Returns:
        scalar moment matching loss
    """
    if pred.size(0) < 4:
        return torch.tensor(0.0, device=pred.device)

    pred_vals = pred[:, focus_dim]
    target_vals = target[:, focus_dim]

    loss = 0.0

    for k in moments:
        if k == 1:
            # Mean
            pred_moment = pred_vals.mean()
            target_moment = target_vals.mean()
        elif k == 2:
            # Variance
            pred_moment = pred_vals.var()
            target_moment = target_vals.var()
        elif k == 3:
            # Skewness
            pred_centered = pred_vals - pred_vals.mean()
            target_centered = target_vals - target_vals.mean()
            pred_moment = (pred_centered ** 3).mean() / (pred_vals.var() ** 1.5 + 1e-9)
            target_moment = (target_centered ** 3).mean() / (target_vals.var() ** 1.5 + 1e-9)
        elif k == 4:
            # Kurtosis
            pred_centered = pred_vals - pred_vals.mean()
            target_centered = target_vals - target_vals.mean()
            pred_moment = (pred_centered ** 4).mean() / (pred_vals.var() ** 2 + 1e-9)
            target_moment = (target_centered ** 4).mean() / (target_vals.var() ** 2 + 1e-9)
        else:
            continue

        loss += (pred_moment - target_moment) ** 2

    return loss


# ---------------------------------------------------------------------------
# W3: V-A decorrelation loss — penalise excess correlation between V/A errors
# ---------------------------------------------------------------------------

def va_decorrelation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gold_corr: float = 0.45,
) -> torch.Tensor:
    """Penalise V-A prediction correlation that exceeds the gold V-A correlation.

    The model tends to over-couple V and A predictions (e.g. gold corr=0.42,
    pred corr=0.75).  This loss pushes the *error* correlation toward zero,
    which indirectly reduces the excess V-A coupling.

    Args:
        pred: (B, 2) predicted [V, A]
        target: (B, 2) gold [V, A]
        gold_corr: expected V-A correlation in the gold data (used as target)

    Returns:
        scalar loss  (0 when pred corr == gold corr)
    """
    v_err = pred[:, 0] - target[:, 0]
    a_err = pred[:, 1] - target[:, 1]
    # Pearson correlation of errors
    v_err_c = v_err - v_err.mean()
    a_err_c = a_err - a_err.mean()
    numer = (v_err_c * a_err_c).mean()
    denom = (v_err_c.var() * a_err_c.var()).sqrt().clamp(min=1e-8)
    err_corr = numer / denom
    # Penalise: error correlation should be near zero
    return err_corr ** 2


# ---------------------------------------------------------------------------
# D39: Polarity classification loss (derive labels from V)
# ---------------------------------------------------------------------------

def valence_to_polarity(v: torch.Tensor, boundaries: tuple[float, float] = (4.0, 6.0)) -> torch.Tensor:
    """Convert valence values to polarity labels: 0=neg, 1=neu, 2=pos."""
    labels = torch.ones_like(v, dtype=torch.long)
    labels[v < boundaries[0]] = 0
    labels[v > boundaries[1]] = 2
    return labels


def polarity_classification_loss(
    logits: torch.Tensor,
    target_v: torch.Tensor,
    boundaries: tuple[float, float] = (4.0, 6.0),
) -> torch.Tensor:
    """Cross-entropy loss for polarity classification derived from V."""
    labels = valence_to_polarity(target_v, boundaries)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# D40: VA region classification loss (Two-Stage approach)
# ---------------------------------------------------------------------------

def va_to_region_label(
    v: torch.Tensor,
    a: torch.Tensor,
    v_boundaries: list[float],
    a_boundaries: list[float],
) -> torch.Tensor:
    """Convert VA values to region labels for grid classification.

    For n_v_bins=3, v_boundaries=[b1, b2]:
      v < b1 → bin 0, b1 <= v < b2 → bin 1, v >= b2 → bin 2
    Region label = v_bin * n_a_bins + a_bin

    Args:
        v: Valence values [B]
        a: Arousal values [B]
        v_boundaries: Sorted list of V boundaries (len = n_v_bins - 1)
        a_boundaries: Sorted list of A boundaries (len = n_a_bins - 1)

    Returns:
        Region labels [B] in range [0, n_v_bins * n_a_bins)
    """
    n_a_bins = len(a_boundaries) + 1

    # Compute V bin
    v_bin = torch.zeros_like(v, dtype=torch.long)
    for b in v_boundaries:
        v_bin += (v >= b).long()

    # Compute A bin
    a_bin = torch.zeros_like(a, dtype=torch.long)
    for b in a_boundaries:
        a_bin += (a >= b).long()

    return v_bin * n_a_bins + a_bin


def va_region_classification_loss(
    logits: torch.Tensor,
    target_va: torch.Tensor,
    v_boundaries: list[float],
    a_boundaries: list[float],
) -> torch.Tensor:
    """Cross-entropy loss for VA region classification.

    Args:
        logits: Region classification logits [B, n_regions]
        target_va: Gold VA values [B, 2] (col 0 = V, col 1 = A)
        v_boundaries: V bin boundaries
        a_boundaries: A bin boundaries

    Returns:
        Scalar loss
    """
    labels = va_to_region_label(
        target_va[:, 0], target_va[:, 1],
        v_boundaries, a_boundaries,
    )
    return F.cross_entropy(logits, labels)
