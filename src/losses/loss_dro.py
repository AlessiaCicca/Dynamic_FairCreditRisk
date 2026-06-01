
"""
DRO (Distributionally Robust Optimization) loss for fairness in survival analysis.

Reference:
    Hu & Chen (2024). "Fairness in Survival Analysis with Distributionally Robust Optimization."
    JMLR, 25(246), 1-85. https://arxiv.org/abs/2409.10538

Key idea:
    Instead of minimizing the average BCE loss (mean over all samples), DRO minimizes
    the worst-case expected loss over all subpopulations that are "large enough"
    (occurring with at least probability eps).

    This is equivalent to upweighting samples with high loss — which tends to
    improve fairness because disadvantaged groups typically have higher loss.

    Crucially, DRO does NOT use the sensitive attribute explicitly.
    Fairness emerges implicitly as a by-product of robustness.

Chi-square DRO loss (closed form):
    L_DRO(θ) = min_{η ∈ R} { η + sqrt((1 + 1/(2*eps)) * (1/n) * Σ_i max(0, l_i - η)^2) }

    where:
        l_i = per-sample BCE loss for sample i
        eps = size of the chi-square uncertainty set (controls robustness strength)
        η   = dual variable (optimized in closed form via bisection or analytical solution)

    In practice we use the CVaR (Conditional Value at Risk) approximation:
        L_DRO ≈ (1/k) * Σ_{i in top-k} l_i
    where k = ceil(n * eps), i.e. average loss over the top-eps fraction of samples.
    This is the standard CVaR/superquantile DRO approximation used in the DRO survival code.

Integration in train_mlp.py:
    Replace:
        loss = L_bce
    With:
        per_sample_losses = F.binary_cross_entropy_with_logits(
            logits, y_train, pos_weight=pos_w, reduction="none"
        )
        loss = dro_loss(per_sample_losses, eps=eps)

Note: DRO replaces the entire BCE loss — it is NOT added as a penalty on top of BCE.
      Therefore alpha/beta are not used with DRO; instead eps controls robustness.
"""

import torch
import torch.nn.functional as F


def dro_loss(per_sample_losses: torch.Tensor, eps: float = 0.2) -> torch.Tensor:
    """
    Chi-square DRO loss via CVaR approximation.

    This is the standard approach used in Hu & Chen (2024):
    compute the average loss over the top-eps fraction of per-sample losses.

    Args:
        per_sample_losses: tensor of shape (n,) — one BCE loss value per sample.
                           Must be computed with reduction="none".
        eps: float in (0, 1] — size of the uncertainty set.
             eps=1.0 → standard mean (no robustness)
             eps=0.2 → average over worst 20% of samples (default, from paper)
             eps=0.1 → more aggressive robustness

    Returns:
        Scalar DRO loss (differentiable, usable for backprop).

    Example:
        per_sample = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = dro_loss(per_sample, eps=0.2)
        loss.backward()
    """
    n = per_sample_losses.shape[0]

    # Number of samples in the worst-case tail
    k = max(1, int(torch.ceil(torch.tensor(n * eps)).item()))

    # Sort losses descending and take the top-k
    sorted_losses, _ = torch.sort(per_sample_losses, descending=True)
    top_k_losses = sorted_losses[:k]

    return top_k_losses.mean()
