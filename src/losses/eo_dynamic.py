import torch


def alpha_schedule(epoch, time_val, max_epoch=120, warmup=20,
                   t_min=0, t_max=48, mode="u_shaped"):
    # Null penalty in firsts epochs
    if epoch < warmup:
        f = 0.0                        
    else:
        f = min(1.0, (epoch - warmup) / (max_epoch - warmup))

    t_norm = (time_val - t_min) / (t_max - t_min + 1e-9)   # 0 = primo landmark, 1 = ultimo
    if mode == "decay":
        g = 10.0 - 9.0 * t_norm                   
    elif mode == "growth":
        g = 1.0 + 9.0 * t_norm               
    elif mode == "flat":
        g = 1.0                               
    elif mode == "u_shaped":
        g = 0.5 + 0.5 * abs(2 * t_norm - 1)    
    elif mode == "n_shaped":
        g = 1.0 + 9.0 * (1 - abs(2 * t_norm - 1)) 
    else:
        raise ValueError(mode)

    return f * g 


def _per_landmark_gaps(pred, sens, true, time_use, already_prob,
                       n_pos_min, current_epoch, time_schedule_mode,
                       t_min, t_max, eps):
    gaps = []
    # Loop on ordered landmakr 
    for t in torch.sort(torch.unique(time_use)).values:
        mask = time_use == t                      
        if mask.sum().item() == 0:
            continue

        lp = pred[mask]                            # predictions
        s  = sens[mask]                            # sensAttribute
        lt = true[mask]                            # true value

        # Consider only subject with valid sensitive attribute
        valid = ~torch.isnan(s)                    
        if valid.sum() == 0:
            continue
        lp = lp[valid] if already_prob else torch.sigmoid(lp[valid])
        s  = s[valid]; lt = lt[valid]

        # Soft prediction
        pos = lt; neg = 1.0 - lt            
        s1  = s;  s0  = 1.0 - s            

        # Counting process
        npos_s1 = torch.sum(s1 * pos).item()
        npos_s0 = torch.sum(s0 * pos).item()
        nneg_s1 = torch.sum(s1 * neg).item()
        nneg_s0 = torch.sum(s0 * neg).item()
        if min(npos_s0, npos_s1) < n_pos_min or min(nneg_s0, nneg_s1) < n_pos_min:
            continue
       
        if torch.unique(s).shape[0] < 2 or torch.unique(lt).shape[0] < 2:
            continue

        # FPR e FNR soft for each groups
        fpr_s1 = torch.sum(lp * s1 * neg) / (torch.sum(s1 * neg) + eps)
        fpr_s0 = torch.sum(lp * s0 * neg) / (torch.sum(s0 * neg) + eps)
        fnr_s1 = torch.sum((1 - lp) * s1 * pos) / (torch.sum(s1 * pos) + eps)
        fnr_s0 = torch.sum((1 - lp) * s0 * pos) / (torch.sum(s0 * pos) + eps)

        fpr_gap = torch.abs(fpr_s1 - fpr_s0)
        fnr_gap = torch.abs(fnr_s1 - fnr_s0)
        if not torch.isfinite(fpr_gap + fnr_gap):
            continue

        a_t = alpha_schedule(epoch=current_epoch, time_val=t.item(),
                             mode=time_schedule_mode,
                             t_min=t_min, t_max=t_max)
        gaps.append({
            "fpr": fpr_gap, "fnr": fnr_gap,
            "alpha": a_t, "n": mask.sum().float(),
        })
    return gaps


def equalized_odds_loss_dynamic(
    label_pred, sensitive, label_true, time_vals,
    mode="trend_aware",
    min_group_frac=0.01,
    trend_weight=0.4,
    current_epoch=0,
    time_schedule_mode="flat",
    group_idx=None,
    n_pos_min=5,
    t_min=0, t_max=48,   
):

    eps = 1e-10
    device = label_pred.device

    if group_idx is not None:
        p = torch.sigmoid(label_pred)                       # hazard per bin
        n_groups = int(group_idx.max().item()) + 1       

        log_surv = torch.zeros(n_groups, device=device).index_add_(
            0, group_idx, torch.log(1.0 - p + eps))
        pred_use = 1.0 - torch.exp(log_surv)           

        true_use = torch.zeros(n_groups, device=device).index_add_(
            0, group_idx, label_true.float()).clamp(max=1.0)

        sens_use = torch.full((n_groups,), float("nan"), device=device)
        vm = ~torch.isnan(sensitive)
        if vm.any():
            sens_use[group_idx[vm]] = sensitive[vm].float()

        time_use = torch.zeros(n_groups, device=device)
        time_use[group_idx] = time_vals.float()
        already_prob = True                                
    else:
         pred_use, sens_use, true_use, time_use = (
                label_pred, sensitive, label_true, time_vals)
         already_prob = False

    gaps = _per_landmark_gaps(
      pred_use, sens_use, true_use, time_use, already_prob,
      n_pos_min, current_epoch, time_schedule_mode,
      t_min, t_max, eps)

    if len(gaps) == 0:
        return torch.tensor(0.0, device=device)            

    # stack dei gap e del termine EO pesato (a_t * (|fpr|+|fnr|))
    fpr_stack = torch.stack([g["fpr"] for g in gaps])
    fnr_stack = torch.stack([g["fnr"] for g in gaps])
    eo_stack  = torch.stack([g["alpha"] * (g["fpr"] + g["fnr"]) for g in gaps])


    if mode == "mean":
        return eo_stack.mean()                            
    if mode == "weighted":
        w = torch.stack([g["n"] for g in gaps])             
        return (eo_stack * (w / w.sum())).sum()

    if mode == "trend_aware":
        if len(eo_stack) >= 2:
            d_fpr = torch.relu(fpr_stack[1:] - fpr_stack[:-1])
            d_fnr = torch.relu(fnr_stack[1:] - fnr_stack[:-1])
            trend_loss = (d_fpr + d_fnr).mean()
        else:
            trend_loss = torch.tensor(0.0, device=device)


        gap_w    = eo_stack.detach() + eps
        gap_w    = gap_w / gap_w.sum()
        loss_gap = (eo_stack * gap_w).sum()

        return (1 - trend_weight) * loss_gap + trend_weight * trend_loss

    raise ValueError(f"mode={mode} non riconosciuto")
