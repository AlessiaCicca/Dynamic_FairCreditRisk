"""
Fair-DSP Group Fairness loss for dynamic survival prediction.

Reference:
    Huang et al. (2023). "Fair-DSP: Fair Dynamic Survival Prediction on
    Longitudinal Electronic Health Record."
    DaWaK 2023, LNCS 14148, pp. 149-157.
    https://doi.org/10.1007/978-3-031-39831-5_15

Fairness notion: GROUP FAIRNESS (demographic parity over time)
    F_G = max_{a ∈ A} |E[O_{k,t}(a)] - E[O_{k,t}(x)]|   (Eq. 4 in the paper)

    where:
        O_{k,t}(a)  = predicted probability for group a at time t
        O_{k,t}(x)  = global mean predicted probability at time t
        A           = set of sensitive attribute values {0, 1}

    In words: for each landmark t, measure the maximum deviation of the
    group-conditional mean prediction from the global mean prediction.
    Average this deviation across all landmarks.

Key difference from EO (eo_dynamic.py):
    - Fair-DSP group fairness does NOT condition on the true label Y
    - It measures disparity in predicted outcomes, not in TPR/FPR
    - This is demographic parity, not equalized odds
    - Simpler to compute: O(n) per landmark, no Y conditioning

Integration in train_mlp.py:
    L_fair = fairdsp_group_loss(logits, sens_train, time_train)
    loss   = (1 - gamma) * L_bce + gamma * L_fair

    Use a separate coefficient (gamma) to distinguish from alpha (eo_dynamic).
"""

import torch


def fairdsp_group_loss(
    label_pred: torch.Tensor,
    sensitive: torch.Tensor,
    time_vals: torch.Tensor,
    min_group_size: int = 10,
) -> torch.Tensor:
    """
    Fair-DSP Group Fairness loss — demographic parity over landmark time points.

    For each landmark t, computes the maximum absolute deviation of the
    group-conditional mean prediction from the global mean prediction,
    then averages across all landmarks.

    Args:
        label_pred : tensor of shape (n,) — raw logits from MLP (NOT sigmoid-ed).
        sensitive  : tensor of shape (n,) — sensitive attribute values (0 or 1).
                     NaN values are ignored.
        time_vals  : tensor of shape (n,) — landmark time point for each sample.
        min_group_size: minimum number of samples per group to include a landmark.

    Returns:
        Scalar group fairness loss (differentiable, usable for backprop).
        Returns 0.0 if no valid landmarks found.
    """
    eps = 1e-10
    device = label_pred.device

    # Convert logits to probabilities
    probs = torch.sigmoid(label_pred)

    unique_times = torch.unique(time_vals)
    deviations = []

    for t in unique_times:
        mask_t = time_vals == t
        if mask_t.sum() == 0:
            continue

        lp = probs[mask_t]
        s  = sensitive[mask_t]

        # Remove NaN sensitive values
        valid = ~torch.isnan(s)
        if valid.sum() == 0:
            continue
        lp = lp[valid]
        s  = s[valid]

        # Need both groups present
        if torch.unique(s).shape[0] < 2:
            continue

        # Global mean prediction at this landmark
        mean_global = lp.mean()

        # Max deviation across groups
        max_dev = torch.tensor(0.0, device=device)
        for g in [0.0, 1.0]:
            mask_g = s == g
            if mask_g.sum() < min_group_size:
                continue
            mean_g = lp[mask_g].mean()
            dev = torch.abs(mean_g - mean_global)
            # Use torch.max via relu trick to keep gradient
            max_dev = max_dev + torch.relu(dev - max_dev)

        if torch.isfinite(max_dev) and max_dev > 0:
            deviations.append(max_dev)

    if len(deviations) == 0:
        return torch.tensor(0.0, device=device)

    # Average deviation across all valid landmarks
    return torch.stack(deviations).mean()
