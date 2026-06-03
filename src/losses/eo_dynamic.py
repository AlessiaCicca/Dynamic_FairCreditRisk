import torch

def alpha_schedule(epoch, time_val, max_epoch=200, warmup=50,
                   t_min=1, t_max=12, mode="u_shaped"):
    if epoch < warmup:
        f = 0.0
    else:
        f = min(1.0, (epoch - warmup) / (max_epoch - warmup))

    t_norm = (time_val - t_min) / (t_max - t_min + 1e-9)
    if mode == "decay":
        g = 1.0 - 0.6 * t_norm
    elif mode == "growth":
        g = 0.4 + 0.6 * t_norm
    elif mode == "flat":
        g = 1.0
    elif mode == "u_shaped":
        g = 0.4 + 0.6 * (2 * abs(t_norm - 0.5))
    else:
        raise ValueError(mode)

    return f * g



def equalized_odds_loss_dynamic(
    label_pred, sensitive, label_true, time_vals,
    mode="trend_aware",
    min_group_frac=0.03,
    trend_weight=0.4,
    current_epoch=0,
    time_schedule_mode="flat", 
):

  if mode== "trend_aware":
        eps = 1e-10
        device = label_pred.device
        unique_times = torch.sort(torch.unique(time_vals)).values

        eo_per_t  = []
        fpr_gap_t = []
        fnr_gap_t = []

        for t in unique_times:
            mask  = time_vals == t
            n_tot = mask.sum().item()
            if n_tot == 0:
                continue

            lp = label_pred[mask]
            s  = sensitive[mask]
            lt = label_true[mask]

            valid = ~torch.isnan(s)
            if valid.sum() == 0:
                continue

            

            lp = torch.sigmoid(lp[valid])
            s  = s[valid]; lt = lt[valid]

            n_s0 = (s == 0).sum().item()
            n_s1 = (s == 1).sum().item()
            min_n = max(5, int(min_group_frac * (n_s0 + n_s1)))
            if min(n_s0, n_s1) < min_n:
                continue
            if torch.unique(s).shape[0] < 2 or torch.unique(lt).shape[0] < 2:
                continue

            pos = lt; neg = 1.0 - lt
            s1  = s;  s0  = 1.0 - s

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

        gap_w    = eo_stack.detach() + eps
        gap_w    = gap_w / gap_w.sum()

        loss_gap = (eo_stack * gap_w).sum()

        
        result=(1 - trend_weight) * loss_gap + trend_weight * trend_loss
        return result
  else: 
          eps = 1e-10
          unique_times = torch.unique(time_vals)

          eo_per_t    = []
          weights     = []

          for t in unique_times:
              mask = time_vals == t
              if mask.sum() == 0:
                  continue

              lp  = label_pred[mask]
              s   = sensitive[mask]
              lt  = label_true[mask]

         
              valid = ~torch.isnan(s)
              if valid.sum() == 0:
                  continue
              lp = torch.sigmoid(lp[valid])
              s  = s[valid]; lt = lt[valid]



              n_s0 = (s == 0).sum().item()
              n_s1 = (s == 1).sum().item()


              min_frac = 0.05 
              if min(n_s0, n_s1) < max(10, int(min_frac * (n_s0 + n_s1))):
                 continue


              if torch.unique(s).shape[0] < 2:
                  continue
              if torch.unique(lt).shape[0] < 2:
                  continue

              s_bar = s;           s_   = 1.0 - s
              pos   = lt;          neg  = 1.0 - lt

              n_sbar_neg = torch.sum(s_bar * neg) + eps
              n_s_neg    = torch.sum(s_    * neg) + eps
              n_sbar_pos = torch.sum(s_bar * pos) + eps
              n_s_pos    = torch.sum(s_    * pos) + eps

              fpr_sbar = torch.sum(lp * s_bar * neg) / n_sbar_neg
              fpr_s    = torch.sum(lp * s_    * neg) / n_s_neg
              fnr_sbar = torch.sum((1 - lp) * s_bar * pos) / n_sbar_pos
              fnr_s    = torch.sum((1 - lp) * s_    * pos) / n_s_pos

              eo_t = torch.abs(fpr_sbar - fpr_s) + torch.abs(fnr_sbar - fnr_s)

              if torch.isfinite(eo_t):
                  eo_per_t.append(eo_t)
                  weights.append(mask.sum().float())

          if len(eo_per_t) == 0:
              return torch.tensor(0.0, device=label_pred.device)

          eo_stack = torch.stack(eo_per_t)

          if mode == "mean":
              return eo_stack.mean()
          elif mode == "weighted":
              w = torch.stack(weights)
              w = w / w.sum()
              return (eo_stack * w).sum()
          else:
              raise ValueError(f"mode={mode} non riconosciuto")
