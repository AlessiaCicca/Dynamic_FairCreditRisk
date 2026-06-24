import torch

def alpha_schedule(epoch, time_val, max_epoch=200, warmup=50,
                   t_min=0, t_max=48, mode="u_shaped"):
    if epoch < warmup:
        f = 0.0
    else:
        f = min(1.0, (epoch - warmup) / (max_epoch - warmup))

    t_norm = (time_val - t_min) / (t_max - t_min + 1e-9)
    if mode == "decay":
        g = 10.0 - 9.0 * t_norm
    elif mode == "growth":
        g = 1.0 + 9.0 * t_norm  # va da 1 a 10
    elif mode == "flat":
        g = 1.0
    elif mode == "u_shaped":
        g = 0.5 + 0.5 * abs(2*t_norm - 1)
    elif mode == "n_shaped":
        g = 1.0 + 9.0 * (1 - abs(2*t_norm - 1))
    else:
        raise ValueError(mode)

    result = f * g
    return result


def equalized_odds_loss_dynamic(
    label_pred, sensitive, label_true, time_vals,
    mode="trend_aware",
    min_group_frac=0.01,
    trend_weight=0.4,
    current_epoch=0,
    time_schedule_mode="flat",
    group_idx=None,        # NEW: per-row (subject, landmark) group id -> collapse to PD-12
    n_pos_min=5,          # NEW: min eventi/non-eventi nel gruppo minoritario per landmark
):
    """
    Se group_idx è fornito, gli hazard per-bin vengono collassati in
    PD(L, L+h) = 1 - prod_j (1 - sigmoid(logit_j)) a livello (soggetto, landmark),
    e la penalità EO viene calcolata SULLA PD-12 (stesso oggetto della metrica),
    non sugli hazard di bin. Questo riallinea leva (penalità) e righello (separation).
    """
    eps = 1e-10
    device = label_pred.device

    # ================== COLLAPSE PER-BIN -> PD-12 (Opzione A) ==================
    if group_idx is not None:
        p = torch.sigmoid(label_pred)                       # hazard di bin
        n_groups = int(group_idx.max().item()) + 1

        # S = prod(1 - h)  via somma dei log  ->  PD = 1 - S  (differenziabile)
        log_surv = torch.zeros(n_groups, device=device).index_add_(
            0, group_idx, torch.log(1.0 - p + eps))
        pd = 1.0 - torch.exp(log_surv)                      # PD per (soggetto, L)

        # y12 = 1 se default in QUALCHE bin della finestra
        y12 = torch.zeros(n_groups, device=device).index_add_(
            0, group_idx, label_true.float()).clamp(max=1.0)

        # sensitive e landmark per gruppo (costanti dentro il gruppo)
        sens_g = torch.full((n_groups,), float("nan"), device=device)
        sens_g[group_idx] = sensitive.float()
        lmk_g = torch.zeros(n_groups, device=device)
        lmk_g[group_idx] = time_vals.float()

        # da qui in poi si lavora a livello PD, NON sui bin
        pred_use  = pd          # GIÀ probabilità in [0,1]  -> NON ri-sigmoidare
        true_use  = y12
        sens_use  = sens_g
        time_use  = lmk_g
        already_prob = True
    else:
        pred_use  = label_pred
        true_use  = label_true
        sens_use  = sensitive
        time_use  = time_vals
        already_prob = False
    # ==========================================================================

    if mode == "trend_aware":
        unique_times = torch.sort(torch.unique(time_use)).values

        eo_per_t  = []
        fpr_gap_t = []
        fnr_gap_t = []

        for t in unique_times:
            mask  = time_use == t
            if mask.sum().item() == 0:
                continue

            lp = pred_use[mask]
            s  = sens_use[mask]
            lt = true_use[mask]

            valid = ~torch.isnan(s)
            if valid.sum() == 0:
                continue

            lp = lp[valid] if already_prob else torch.sigmoid(lp[valid])
            s  = s[valid]; lt = lt[valid]

            pos = lt; neg = 1.0 - lt
            s1  = s;  s0  = 1.0 - s

            # --- GATE SUGLI EVENTI, non sulle righe ---
            npos_s1 = torch.sum(s1 * pos).item()
            npos_s0 = torch.sum(s0 * pos).item()
            nneg_s1 = torch.sum(s1 * neg).item()
            nneg_s0 = torch.sum(s0 * neg).item()

            if min(npos_s0, npos_s1) < n_pos_min or min(nneg_s0, nneg_s1) < n_pos_min:
                continue
            if torch.unique(s).shape[0] < 2 or torch.unique(lt).shape[0] < 2:
                continue

            fpr_s1 = torch.sum(lp * s1 * neg) / (torch.sum(s1 * neg) + eps)
            fpr_s0 = torch.sum(lp * s0 * neg) / (torch.sum(s0 * neg) + eps)
            fnr_s1 = torch.sum((1 - lp) * s1 * pos) / (torch.sum(s1 * pos) + eps)
            fnr_s0 = torch.sum((1 - lp) * s0 * pos) / (torch.sum(s0 * pos) + eps)

            fpr_gap = torch.abs(fpr_s1 - fpr_s0)
            fnr_gap = torch.abs(fnr_s1 - fnr_s0)
            eo_t    = fpr_gap + fnr_gap

            if torch.isfinite(eo_t):
                a_t = alpha_schedule(
                    epoch=current_epoch,
                    time_val=t.item(),
                    mode=time_schedule_mode,
                )
                eo_per_t.append(a_t * eo_t)
                fpr_gap_t.append(fpr_gap)
                fnr_gap_t.append(fnr_gap)

        if len(eo_per_t) == 0:
            return torch.tensor(0.0, device=device)

        eo_stack  = torch.stack(eo_per_t)
        fpr_stack = torch.stack(fpr_gap_t)
        fnr_stack = torch.stack(fnr_gap_t)

        if len(eo_stack) >= 2:
            delta_fpr  = torch.relu(fpr_stack[1:] - fpr_stack[:-1])
            delta_fnr  = torch.relu(fnr_stack[1:] - fnr_stack[:-1])
            trend_loss = (delta_fpr + delta_fnr).mean()
        else:
            trend_loss = torch.tensor(0.0, device=device)

        unweighted_mean = eo_stack.detach().mean().item()  # diagnostica

        gap_w    = eo_stack.detach() + eps
        gap_w    = gap_w / gap_w.sum()

        loss_gap = (eo_stack * gap_w).sum()

        result = (1 - trend_weight) * loss_gap + trend_weight * trend_loss
        return result

    else:
        unique_times = torch.unique(time_use)

        eo_per_t = []
        weights  = []

        for t in unique_times:
            mask = time_use == t
            if mask.sum() == 0:
                continue

            lp = pred_use[mask]
            s  = sens_use[mask]
            lt = true_use[mask]

            valid = ~torch.isnan(s)
            if valid.sum() == 0:
                continue
            lp = lp[valid] if already_prob else torch.sigmoid(lp[valid])
            s  = s[valid]; lt = lt[valid]

            pos = lt; neg = 1.0 - lt
            s1  = s;  s0  = 1.0 - s

            # gate sugli eventi (coerente col ramo trend_aware)
            npos_s1 = torch.sum(s1 * pos).item()
            npos_s0 = torch.sum(s0 * pos).item()
            nneg_s1 = torch.sum(s1 * neg).item()
            nneg_s0 = torch.sum(s0 * neg).item()
            if min(npos_s0, npos_s1) < n_pos_min or min(nneg_s0, nneg_s1) < n_pos_min:
                continue
            if torch.unique(s).shape[0] < 2 or torch.unique(lt).shape[0] < 2:
                continue

            s_bar = s;  s_ = 1.0 - s
            n_sbar_neg = torch.sum(s_bar * neg) + eps
            n_s_neg    = torch.sum(s_    * neg) + eps
            n_sbar_pos = torch.sum(s_bar * pos) + eps
            n_s_pos    = torch.sum(s_    * pos) + eps

            fpr_sbar = torch.sum(lp * s_bar * neg) / n_sbar_neg
            fpr_s    = torch.sum(lp * s_    * neg) / n_s_neg
            fnr_sbar = torch.sum((1 - lp) * s_bar * pos) / n_sbar_pos
            fnr_s    = torch.sum((1 - lp) * s_    * pos) / n_s_pos

            eo_t = torch.abs((fpr_sbar - fpr_s) + (fnr_sbar - fnr_s))

            if torch.isfinite(eo_t):
                a_t = alpha_schedule(
                    epoch=current_epoch,
                    time_val=t.item(),
                    mode=time_schedule_mode,
                )
                eo_per_t.append(a_t * eo_t)
                weights.append(mask.sum().float())

        if len(eo_per_t) == 0:
            return torch.tensor(0.0, device=device)

        eo_stack = torch.stack(eo_per_t)

        if mode == "mean":
            return eo_stack.mean()
        elif mode == "weighted":
            w = torch.stack(weights)
            w = w / w.sum()
            return (eo_stack * w).sum()
        else:
            raise ValueError(f"mode={mode} non riconosciuto")