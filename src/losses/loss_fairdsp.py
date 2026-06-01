"""
Equitable Allocation of Healthcare Resources with Fair Cox Models
https://github.com/kkeya1/FairSurv

"""

import torch


def fairsurv_group_loss(label_pred, sensitive, time_vals, min_group_size=10):
    probs = torch.sigmoid(label_pred)
    unique_times = torch.unique(time_vals)
    losses = []

    for t in unique_times:
        mask_t = time_vals == t
        lp = probs[mask_t]
        s  = sensitive[mask_t]

        valid = ~torch.isnan(s)
        if valid.sum() == 0:
            continue
        lp = lp[valid]; s = s[valid]

        if torch.unique(s).shape[0] < 2:
            continue

        mask_0 = s == 0
        mask_1 = s == 1
        if mask_0.sum() < min_group_size or mask_1.sum() < min_group_size:
            continue

        # g-difference: differenza assoluta delle medie per gruppo
        diff = torch.abs(lp[mask_0].mean() - lp[mask_1].mean())
        losses.append(diff)

    if len(losses) == 0:
        return torch.tensor(0.0, device=label_pred.device)

    return torch.stack(losses).mean()

    if len(deviations) == 0:
        return torch.tensor(0.0, device=device)

    # Average deviation across all valid landmarks
    return torch.stack(deviations).mean()
