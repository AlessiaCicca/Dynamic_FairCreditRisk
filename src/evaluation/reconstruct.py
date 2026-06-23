import numpy as np

def reconstruct_pd(model, X_L, bin_times_L, enc_lmk, all_bin_times,
                   lmk_oh_slice, scaler=None, device=None):
    """
    PD(L, L+horizon) per i soggetti a rischio al landmark L.
    X_L: feature NON temporali (num+cat) dei soggetti a L, x(L) congelate.
         NON deve contenere le colonne di bin_time one-hot.
    bin_times_L: lista dei bin_time da moltiplicare per questo L,
                 es. [L, L+delta, L+2delta, L+3delta].
    """
    surv = np.ones(X_L.shape[0], dtype=np.float64)

    for bt in bin_times_L:
        # one-hot del bin corrente, uguale per tutti i soggetti
        oh = enc_lmk.transform(np.array([[bt]]))          # (1, n_bin_times)
        oh = np.repeat(oh, X_L.shape[0], axis=0)          # (N, n_bin_times)

        X_bin = np.hstack([X_L, oh]).astype(np.float32)   # x(L) + indicatore bin
        if scaler is not None:
            X_bin = scaler.transform(X_bin)

        h = _predict_hazard(model, X_bin, device)         # hazard del bin bt
        surv *= (1.0 - h)

    return 1.0 - surv                                     # PD(L, L+horizon)
