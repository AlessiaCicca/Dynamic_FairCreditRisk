"""
src/training/train_mlp.py

Training loop for M_STATIC, M_DYNAMIC, and M_PP models.
Imports MLP from src.models.mlp and loss functions from src.losses.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.models.mlp import MLP
from src.losses.eo_static import equalized_odds_loss
from src.losses.eo_dynamic import equalized_odds_loss_dynamic


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    sensitive_tr: np.ndarray = None,
    time_tr: np.ndarray = None,
    subj_ids_tr: np.ndarray = None,
    model_name: str = "",
    # architecture
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.3,
    # optimiser
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 200,
    patience: int = 30,
    min_lr: float = 1e-5,
    pw_clip: float = 20.0,
    # fairness coefficients
    beta: float = 0.0,    # M_STATIC
    alpha: float = 0.0,   # M_DYNAMIC
    gamma: float = 0.0,   # M_PP
    # EO loss options
    eo_mode_d: str = "mean",
    eo_mode_p: str = "mean",
    schedule_mode_d: str = "flat",
    schedule_mode_p: str = "flat",
    # misc
    verbose: bool = False,
) -> tuple:
    """
    Train an MLP with optional fairness regularisation (FairNN / dynamic EO).

    Parameters
    ----------
    Xtr, ytr        : training features and labels (numpy arrays)
    Xte, yte        : test features and labels (numpy arrays)
    sensitive_tr    : sensitive attribute for training rows (0/1 or NaN)
    time_tr         : landmark / loan_age for training rows
    subj_ids_tr     : subject IDs for training rows
    model_name      : one of "static" | "dynamic" | "person_period"
    hidden1/2       : hidden layer sizes
    dropout         : dropout rate
    lr              : initial learning rate
    weight_decay    : Adam L2 regularisation
    n_epochs        : maximum training epochs
    patience        : ReduceLROnPlateau patience
    min_lr          : minimum learning rate
    pw_clip         : cap on pos_weight to avoid gradient explosion
    beta            : EO penalty weight for M_STATIC
    alpha           : EO penalty weight for M_DYNAMIC
    gamma           : EO penalty weight for M_PP
    eo_mode_d       : EO mode for dynamic  ("mean"|"weighted"|"trend_aware"|"weighted+trend")
    eo_mode_p       : EO mode for PP       ("mean"|"weighted"|"trend_aware"|"weighted+trend")
    schedule_mode_d : time schedule for dynamic penalty ("flat"|"decay"|"growth"|"u_shaped")
    schedule_mode_p : time schedule for PP penalty
    verbose         : print metrics every 20 epochs (fold 0 only)

    Returns
    -------
    p_te    : predicted probabilities on test set  (numpy, shape N_te)
    p_tr    : predicted probabilities on train set (numpy, shape N_tr)
    model   : trained MLP instance
    scaler  : fitted StandardScaler
    """

    # ── Preprocessing ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    Xtr_s = np.nan_to_num(
        scaler.fit_transform(Xtr).astype(np.float32),
        nan=0., posinf=5., neginf=-5.,
    )
    Xte_s = np.nan_to_num(
        scaler.transform(Xte).astype(np.float32),
        nan=0., posinf=5., neginf=-5.,
    )

    X_train = torch.tensor(Xtr_s, device=DEVICE)
    y_train = torch.tensor(ytr.astype(np.float32), device=DEVICE)
    X_test  = torch.tensor(Xte_s, device=DEVICE)

    sens_train = (
        torch.tensor(sensitive_tr.astype(np.float32), device=DEVICE)
        if sensitive_tr is not None else None
    )
    time_train = (
        torch.tensor(time_tr.astype(np.float32), device=DEVICE)
        if time_tr is not None else None
    )

    # ── pos_weight ────────────────────────────────────────────────────────────
    n_pos = max((ytr == 1).sum(), 1)
    n_neg = max((ytr == 0).sum(), 1)
    pw    = float(np.clip(n_neg / n_pos, 1.0, pw_clip))
    pos_w = torch.tensor([pw], dtype=torch.float32, device=DEVICE)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = MLP(X_train.shape[1], hidden1, hidden2, dropout).to(DEVICE)
    model.init_bias(prev=float(ytr.mean()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=0.5, min_lr=min_lr
    )

    # ── Which fairness branch applies ─────────────────────────────────────────
    apply_fair_static  = (model_name == "static")        and (sens_train is not None)
    apply_fair_dynamic = (model_name == "dynamic")       and (sens_train is not None) and (time_train is not None)
    apply_fair_pp      = (model_name == "person_period") and (sens_train is not None) and (time_train is not None)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits  = model(X_train)
        L_bce   = criterion(logits, y_train)
        L_eo    = torch.tensor(0.0, device=DEVICE)
        apply_eo = epoch > 20   # warmup: let BCE learn first

        if apply_fair_static and apply_eo:
            L_eo = equalized_odds_loss(logits, sens_train, y_train)
            loss = (1 - beta) * L_bce + beta * L_eo

        elif apply_fair_dynamic and apply_eo:
            L_eo = equalized_odds_loss_dynamic(
                logits, sens_train, y_train, time_train,
                mode=eo_mode_d,
                current_epoch=epoch,
                time_schedule_mode=schedule_mode_d,
            )
            loss = (1 - alpha) * L_bce + alpha * L_eo

        elif apply_fair_pp and apply_eo:
            L_eo = equalized_odds_loss_dynamic(
                logits, sens_train, y_train, time_train,
                mode=eo_mode_p,
                current_epoch=epoch,
                time_schedule_mode=schedule_mode_p,
            )
            loss = (1 - gamma) * L_bce + gamma * L_eo

        else:
            loss = L_bce

        if not torch.isfinite(loss):
            print(f"  [WARN] Non-finite loss at epoch {epoch} — stopping.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(loss.detach())

        # ── Verbose logging ───────────────────────────────────────────────────
        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            with torch.no_grad():
                p_tr_v = torch.sigmoid(logits).cpu().numpy()
            try:
                auc_tr = roc_auc_score(ytr, p_tr_v)
            except Exception:
                auc_tr = float("nan")

            print(
                f"  [{model_name}] epoch={epoch:3d}  "
                f"L_bce={L_bce.item():.4f}  L_eo={L_eo.item():.4f}  "
                f"loss={loss.item():.4f}  AUC_train={auc_tr:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            if sens_train is not None:
                s = sensitive_tr
                for g, gname in [(0, "S0"), (1, "S1")]:
                    mask = s == g
                    if mask.sum() > 0 and ytr[mask].sum() > 0:
                        tpr = ((p_tr_v[mask] >= 0.5) & (ytr[mask] == 1)).sum() / (ytr[mask] == 1).sum()
                        fpr = ((p_tr_v[mask] >= 0.5) & (ytr[mask] == 0)).sum() / (ytr[mask] == 0).sum()
                        print(f"    {gname}: TPR={tpr:.3f}  FPR={fpr:.3f}")
                sep = abs(
                    ((p_tr_v[s == 1] >= 0.5) & (ytr[s == 1] == 1)).sum() / max((ytr[s == 1] == 1).sum(), 1) -
                    ((p_tr_v[s == 0] >= 0.5) & (ytr[s == 0] == 1)).sum() / max((ytr[s == 0] == 1).sum(), 1)
                )
                print(f"    δTPR={sep:.3f}  Eq.Odds={L_eo.item():.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(X_test)).cpu().numpy()
        p_tr = torch.sigmoid(model(X_train)).cpu().numpy()

    fair_type = (
        "static"   if apply_fair_static  else
        eo_mode_d  if apply_fair_dynamic else
        eo_mode_p  if apply_fair_pp      else
        "none"
    )
    coeff = beta if apply_fair_static else alpha if apply_fair_dynamic else gamma if apply_fair_pp else 0.0
    print(
        f"  [{model_name}|eo={fair_type}]  "
        f"pred_mean_train={p_tr.mean():.4f}  "
        f"pred_mean_test={p_te.mean():.4f}  "
        f"pos_weight={pw:.2f}  coeff={coeff:.2f}"
    )
    return p_te, p_tr, model, scaler
