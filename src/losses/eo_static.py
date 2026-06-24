import torch
def equalized_odds_loss(label_pred, sensitive, label_true):
    eps   = 1e-10
    valid = ~torch.isnan(sensitive)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=label_pred.device)
    label_pred = torch.sigmoid(label_pred[valid])
    sensitive  = sensitive[valid]
    label_true = label_true[valid]
    s_bar = sensitive;        s   = 1.0 - sensitive
    pos   = label_true;       neg = 1.0 - label_true
    n_sbar_neg = torch.sum(s_bar * neg) + eps
    n_s_neg    = torch.sum(s     * neg) + eps
    n_sbar_pos = torch.sum(s_bar * pos) + eps
    n_s_pos    = torch.sum(s     * pos) + eps
    fpr_sbar   = torch.sum(label_pred * s_bar * neg) / n_sbar_neg
    fpr_s      = torch.sum(label_pred * s     * neg) / n_s_neg
    fnr_sbar   = torch.sum((1 - label_pred) * s_bar * pos) / n_sbar_pos
    fnr_s      = torch.sum((1 - label_pred) * s     * pos) / n_s_pos
    #eq_odds    = torch.abs(fpr_sbar - fpr_s) + torch.abs(fnr_sbar - fnr_s)
    eq_odds    = torch.abs((fpr_sbar - fpr_s) +(fnr_sbar - fnr_s))
    return eq_odds if torch.isfinite(eq_odds) else torch.tensor(0.0, device=label_pred.device)
