"""
CMIA (Conditional Mutual Information Augmentation) fairness loss.

Reference:
    Xie & Ge (2025). "Fairness in Survival Analysis: A Novel Conditional
    Mutual Information Augmentation Approach."
    arXiv:2502.02567. https://arxiv.org/abs/2502.02567

Fairness notion: EQUALIZED ODDS via Conditional Mutual Information
    EO condition: Ŷ_t ⊥ Z | Y_t,  ∀t ∈ Q   (Eq. 6 in the paper)

    This is the same EO notion as eo_dynamic.py, but enforced via
    information-theoretic regularization instead of direct FPR/FNR differences.

CMI regularization term (Eq. 9 in the paper):
    CMI_hat_{m,t} = (1/n) Σ_i (1/m) Σ_j log [
        (1/n_{Y_i,t,Z_i}) Σ_{l: Y_l=Y_i, Z_l=Z_i} φ_τ(ε_j + g(X_i) - g(X_l))
        ──────────────────────────────────────────────────────────────────────
        (1/n_{Y_i,t}) Σ_{k: Y_k=Y_i} φ_τ(ε_j + g(X_i) - g(X_k))
    ]

    where:
        g(X_i)  = MLP logit for sample i
        φ_τ     = Gaussian density N(0, τ) evaluated at the argument
        ε_j     = noise drawn m times from N(0, τ)
        Y_i,t   = true binary label at landmark t (0 or 1)
        Z_i     = sensitive attribute (0 or 1)
        n_{Y,Z} = number of samples with same Y and Z as sample i
        n_{Y}   = number of samples with same Y as sample i

    Total regularization: R_EO = Σ_{t ∈ Q} CMI_hat_{m,t}   (Eq. 11)

Key difference from eo_dynamic.py:
    - CMIA measures conditional dependence via information theory (CMI)
    - eo_dynamic directly measures FPR/FNR gaps
    - CMIA is model-agnostic and theoretically grounded
    - eo_dynamic is more interpretable and landmark-aware with trend penalty

Computational note:
    The naive implementation is O(N²) per landmark. We use random subsampling
    (max_samples_per_t) to make it tractable for large datasets.
    The paper uses m=5 noise samples by default.

Integration in train_mlp.py:
    L_fair = cmia_loss(logits, sens_train, y_train, time_train)
    loss   = (1 - lambda1) * L_bce + lambda1 * L_fair
"""

import torch
import torch.nn.functional as F


def cmia_loss(
    label_pred: torch.Tensor,
    sensitive: torch.Tensor,
    label_true: torch.Tensor,
    time_vals: torch.Tensor,
    tau: float = 0.5,
    m: int = 5,
    max_samples_per_t: int = 500,
    min_group_size: int = 5,
) -> torch.Tensor:
    """
    CMIA fairness regularization term R_EO = Σ_t CMI_hat_{m,t}.

    Args:
        label_pred      : tensor (n,) — raw MLP logits g(X_i).
        sensitive       : tensor (n,) — sensitive attribute Z_i ∈ {0, 1, NaN}.
        label_true      : tensor (n,) — binary label Y_t ∈ {0, 1}.
        time_vals       : tensor (n,) — landmark time point for each sample.
        tau             : float — bandwidth of the Gaussian kernel φ_τ.
                          Controls smoothness of the CMI approximation.
                          Default 0.5 (logit scale).
        m               : int — number of noise samples ε_j per observation.
                          Higher m → better approximation, slower.
                          Default 5 (as in paper).
        max_samples_per_t: int — max samples per landmark for efficiency.
                          Subsampled randomly if n_t > max_samples_per_t.
                          Default 500.
        min_group_size  : int — minimum samples per (Y, Z) cell to include.
                          Default 5.

    Returns:
        Scalar CMIA loss R_EO (differentiable, usable for backprop).
        Returns 0.0 if no valid landmarks found.
    """
    eps    = 1e-10
    device = label_pred.device
    unique_times = torch.unique(time_vals)

    total_cmi = []

    for t in unique_times:
        mask_t = time_vals == t
        if mask_t.sum() == 0:
            continue

        lp = label_pred[mask_t]
        s  = sensitive[mask_t]
        yt = label_true[mask_t].float()

        # Remove NaN sensitive values
        valid = ~torch.isnan(s)
        if valid.sum() < min_group_size * 2:
            continue
        lp = lp[valid]
        s  = s[valid]
        yt = yt[valid]

        # Need both groups and both labels
        if torch.unique(s).shape[0] < 2:
            continue
        if torch.unique(yt).shape[0] < 2:
            continue

        # Subsample for efficiency
        n_t = lp.shape[0]
        if n_t > max_samples_per_t:
            idx = torch.randperm(n_t, device=device)[:max_samples_per_t]
            lp = lp[idx]
            s  = s[idx]
            yt = yt[idx]
            n_t = max_samples_per_t

        # Compute CMI_hat for this landmark
        cmi_per_sample = []

        for i in range(n_t):
            # Masks for same (Y, Z) and same Y
            same_yz = (yt == yt[i]) & (s == s[i])
            same_y  = (yt == yt[i])

            n_yz = same_yz.sum().item()
            n_y  = same_y.sum().item()

            if n_yz < min_group_size or n_y < min_group_size:
                continue

            lp_yz = lp[same_yz]  # g(X_l) with Y_l=Y_i, Z_l=Z_i
            lp_y  = lp[same_y]   # g(X_k) with Y_k=Y_i

            # Sample m noise values ε_j ~ N(0, τ)
            eps_j = torch.randn(m, device=device) * tau  # (m,)

            # For each ε_j: compute φ_τ(ε_j + g(X_i) - g(X_l)) as Gaussian kernel
            # Numerator: average over same (Y, Z) group
            # diff_yz[j, l] = ε_j + g(X_i) - g(X_l)  for l in same_yz
            diff_yz = eps_j.unsqueeze(1) + lp[i] - lp_yz.unsqueeze(0)  # (m, n_yz)
            phi_yz  = torch.exp(-0.5 * (diff_yz / tau) ** 2)             # (m, n_yz)
            num     = phi_yz.mean(dim=1) + eps                            # (m,)

            # Denominator: average over same Y group
            diff_y = eps_j.unsqueeze(1) + lp[i] - lp_y.unsqueeze(0)    # (m, n_y)
            phi_y  = torch.exp(-0.5 * (diff_y / tau) ** 2)              # (m, n_y)
            den    = phi_y.mean(dim=1) + eps                             # (m,)

            # CMI contribution for sample i: (1/m) Σ_j log(num_j / den_j)
            log_ratio = torch.log(num / den)                             # (m,)
            cmi_i     = log_ratio.mean()                                 # scalar

            if torch.isfinite(cmi_i):
                cmi_per_sample.append(cmi_i)

        if len(cmi_per_sample) == 0:
            continue

        # CMI_hat_{m,t} = (1/n) Σ_i CMI_i
        cmi_t = torch.stack(cmi_per_sample).mean()

        if torch.isfinite(cmi_t):
            total_cmi.append(cmi_t)

    if len(total_cmi) == 0:
        return torch.tensor(0.0, device=device)

    # R_EO = Σ_t CMI_hat_{m,t}
    return torch.stack(total_cmi).sum()
