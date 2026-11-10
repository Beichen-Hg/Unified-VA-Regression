"""D37a: Variance Regularization Loss

简单的方差正则化，约束预测方差接近真值方差。
理论基础：直接优化 σ_pred/σ_gold 指标。
"""
import torch
import torch.nn.functional as F


def variance_regularization_loss(pred, target, focus_dim=1, mode="ratio"):
    """
    方差正则化损失

    Args:
        pred: 预测值 (batch, 2) - [V, A]
        target: 真值 (batch, 2) - [V, A]
        focus_dim: 0=Valence, 1=Arousal, None=both
        mode: "ratio" | "diff" | "std"

    Returns:
        loss: scalar
    """
    if focus_dim is not None:
        pred = pred[:, focus_dim]
        target = target[:, focus_dim]

    # 计算方差
    pred_var = pred.var(dim=0, unbiased=True)
    target_var = target.var(dim=0, unbiased=True)

    if mode == "ratio":
        # 方差比损失: (σ_pred/σ_gold - 1)²
        var_ratio = pred_var / (target_var + 1e-8)
        loss = (var_ratio - 1.0) ** 2
    elif mode == "diff":
        # 方差差异损失: (σ_pred - σ_gold)²
        loss = (pred_var - target_var) ** 2
    elif mode == "std":
        # 标准差损失: (√σ_pred - √σ_gold)²
        pred_std = torch.sqrt(pred_var + 1e-8)
        target_std = torch.sqrt(target_var + 1e-8)
        loss = (pred_std - target_std) ** 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return loss.mean() if loss.dim() > 0 else loss


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    batch_size = 32
    pred = torch.randn(batch_size, 2) * 0.5 + 5.0  # 预测值，方差较小
    target = torch.randn(batch_size, 2) * 1.0 + 5.0  # 真值，方差较大

    # 计算损失
    loss_ratio = variance_regularization_loss(pred, target, focus_dim=1, mode="ratio")
    loss_std = variance_regularization_loss(pred, target, focus_dim=1, mode="std")

    print(f"Pred std (A): {pred[:, 1].std():.4f}")
    print(f"Target std (A): {target[:, 1].std():.4f}")
    print(f"Variance ratio loss: {loss_ratio:.4f}")
    print(f"Std loss: {loss_std:.4f}")
