"""
Training loop for M_STATIC and M_DYNAMIC models.
Imports MLP from src.models.mlp and loss functions from src.losses.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.models.mlp import MLP
from src.losses.eo_static import equalized_odds_loss
from src.losses.eo_dynamic import equalized_odds_loss_dynamic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#   Train an MLP with fairness regularisation
def train_mlp(
    Xtr,
    ytr,
    Xte,
    yte,
    sensitive_tr=None,
    time_tr=None,
    subj_ids_tr=None,
    model_name="",
    hidden1=64,
    hidden2=32,
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    n_epochs=200,
    patience=30,
    min_lr=1e-5,
    pw_clip=10.0,
    beta=0.0,    # M_STATIC
    alpha=0.0,   # M_DYNAMIC
    eo_mode_d="mean",
    schedule_mode_d="flat",
    t_min=0.0, t_max=48.0,   
    verbose=False,
):


    # Preprocessing -> StandardScaler (mean=0 and std=1) due to the different scale of the features
    # Fit on the training set and transform on the test set
    scaler = StandardScaler()
    Xtr_s = np.nan_to_num(scaler.fit_transform(Xtr).astype(np.float32),
        nan=0., posinf=5., neginf=-5.,)
    Xte_s = np.nan_to_num(scaler.transform(Xte).astype(np.float32),
        nan=0., posinf=5., neginf=-5.,)


    # PyTorch tensor conversion
    X_train = torch.tensor(Xtr_s, device=DEVICE)
    y_train = torch.tensor(ytr.astype(np.float32), device=DEVICE)
    X_test  = torch.tensor(Xte_s, device=DEVICE)
    sens_train = ( torch.tensor(sensitive_tr.astype(np.float32), device=DEVICE)
        if sensitive_tr is not None else None)
    time_train = (torch.tensor(time_tr.astype(np.float32), device=DEVICE)
        if time_tr is not None else None)

    # Class-weight: the pos_weight gives more weight to the minority class (default) 
    # during training, and the clip avoids exaggerating this weight when the imbalance is extreme.
    n_pos = max((ytr == 1).sum(), 1)
    n_neg = max((ytr == 0).sum(), 1)
    pw    = float(np.clip(n_neg / n_pos, 1.0, pw_clip))   
    pos_w = torch.tensor([pw], dtype=torch.float32, device=DEVICE)

    # Model Initialization
    model     = MLP(X_train.shape[1], hidden1, hidden2, dropout).to(DEVICE)
    model.init_bias(prev=float(ytr.mean()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=0.5, min_lr=min_lr)

    apply_fair_static  = (model_name == "static")        and (sens_train is not None)
    apply_fair_dynamic = (model_name == "dynamic")       and (sens_train is not None) and (time_train is not None)

    # Training loop
    model.train()  
    # Mappa (soggetto, landmark) -> id gruppo, per il collapse PD-12 nella loss
    group_idx = None
    if apply_fair_dynamic and (subj_ids_tr is not None) and (time_tr is not None):
        gkey = (
            pd.DataFrame({"s": np.asarray(subj_ids_tr), "t": np.asarray(time_tr)})
            .groupby(["s", "t"], sort=False)
            .ngroup()
            .to_numpy()
        )
        group_idx = torch.tensor(gkey, dtype=torch.long, device=DEVICE)
    for epoch in range(n_epochs):
        # Resets the accumulated gradients from the previous step
        optimizer.zero_grad()
        # Forward pass — passes all data through the network and gets the logits
        logits  = model(X_train)
        # Binary Cross Entropy
        L_bce   = criterion(logits, y_train)
        # Initialization of the fairness loss
        L_eo    = torch.tensor(0.0, device=DEVICE)
        # Warmup let BCE learn first
        apply_eo = epoch > 20   

        if apply_fair_static and apply_eo:
            L_eo = equalized_odds_loss(logits, sens_train, y_train)
            loss = (1 - beta) * L_bce + beta * L_eo

        elif apply_fair_dynamic and apply_eo:
            L_eo = equalized_odds_loss_dynamic(
                logits, sens_train, y_train, time_train,
                mode=eo_mode_d, current_epoch=epoch,
                time_schedule_mode=schedule_mode_d,group_idx=group_idx,
                t_min=t_min, t_max=t_max) 
            loss = (1 - alpha) * L_bce + alpha * L_eo

        else:
            loss = L_bce

        if not torch.isfinite(loss):
            break

        # Backpropagation 
        loss.backward()
        # Gradient clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # Weight update with Adam
        optimizer.step()
        # Learning rate update
        scheduler.step(loss.detach())

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            with torch.no_grad():
                # Converts logits to probabilities
                p_tr_v = torch.sigmoid(logits).cpu().numpy()
            # To manage situation with single classes for fold
            try:
                auc_tr = roc_auc_score(ytr, p_tr_v)
            except Exception:
                auc_tr = float("nan")


            print(
                f" epoch={epoch:3d}  ->  "
                f"L_bce={L_bce.item():.4f} | L_eo={L_eo.item():.4f}  |  "
                f"loss={loss.item():.4f} "
            )

            if sens_train is not None:
                s = sensitive_tr
                for g, gname in [(0, "S0"), (1, "S1")]:
                    mask = s == g
                    if mask.sum() > 0 and ytr[mask].sum() > 0:
                        tpr = ((p_tr_v[mask] >= 0.5) & (ytr[mask] == 1)).sum() / (ytr[mask] == 1).sum()
                        fpr = ((p_tr_v[mask] >= 0.5) & (ytr[mask] == 0)).sum() / (ytr[mask] == 0).sum()
                        print( f" {gname}: TPR={tpr:.3f}  FPR={fpr:.3f}")

    # Inference
    model.eval()
    with torch.no_grad():
        # Forward Pass + Converts logits to probabilities
        p_te = torch.sigmoid(model(X_test)).cpu().numpy()
        p_tr = torch.sigmoid(model(X_train)).cpu().numpy()
    return p_te, p_tr, model, scaler
