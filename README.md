# An Equitable Framework for 
# Dynamic Credit Risk

**Fairness Penalization in Discrete-Time Survival Models**

This repository contains the code developed for a doctoral thesis comparing a **static** classification model (`M_STATIC`) and a **landmark-based, discrete-time survival model** (`M_DYNAMIC`) for mortgage default prediction, under an equalized-odds fairness constraint. Both models are evaluated on simulated data and on real mortgage data (Freddie Mac Single-Family Loan-Level Dataset matched with HMDA demographic records).

## Overview

- **M_STATIC** — a single-timepoint binary classifier predicting default risk from loan-level features at origination/observation time.
- **M_DYNAMIC** — a landmark supermodel for discrete-time survival analysis: the same underlying model is trained on a stacked, landmark-augmented dataset, allowing default-risk predictions to be updated at multiple points along the life of the loan while sharing information across landmarks.

Both models are trained with a **fairness penalty** based on equalized odds (EO), added to the binary cross-entropy loss, and evaluated with:
- **Predictive performance**: AUC (M_STATIC) / integrated AUC (M_DYNAMIC), via `GroupKFold` cross-validation with out-of-fold predictions.
- **Fairness**: `separation_auc` (equalized-odds violation, computed via trapezoidal integration) and adTPR/adFPR-style group metrics.

## Repository structure

```
.
├── config.py                     # Central configuration: model, training, fairness, and data parameters
├── run_thesis.ipynb              # End-to-end pipeline (Colab-oriented): simulation, real data, training, grid search
│
├── data_generation/
│   ├── simulation/                # R scripts generating synthetic survival data under 4 discrimination scenarios
│   │   ├── genvar.R                  # VAR-based generation of time-varying covariates (X3, X4, X6)
│   │   ├── timevarying_gnrt.R        # Time-varying feature generation
│   │   ├── traindtv_autocorr_gnrt.R  # Autocorrelated training data generation
│   │   ├── utils.R
│   │   └── run.R                     # Entry point for the simulation pipeline
│   │
│   └── realData/                  # Real-data pipeline (Freddie Mac SFLLD + HMDA)
│       ├── match_FreddieMac_HMDA.py  # Matches loan-level records to HMDA demographic data
│       ├── build_panel.py            # Builds the longitudinal (panel) dataset
│       ├── sampling.py               # Weighted undersampling for class imbalance
│       ├── preprocessing.py          # Cleaning and preprocessing of the panel dataset
│       ├── EDA.ipynb                 # Exploratory data analysis
│       └── EDA__finalData.ipynb      # EDA on the final processed dataset
│
├── src/
│   ├── models/
│   │   └── mlp.py                 # Two-hidden-layer MLP (Linear → ReLU → BatchNorm → Dropout) x2 → logit
│   ├── losses/
│   │   ├── eo_static.py           # Equalized-odds penalty for M_STATIC
│   │   └── eo_dynamic.py          # Equalized-odds penalty for M_DYNAMIC, incl. alpha_schedule (decay/growth/flat/u_shaped/n_shaped)
│   ├── data/
│   │   ├── build_static.py        # Builds the static feature/label dataset
│   │   └── build_dynamic.py       # Builds the landmark-stacked dynamic dataset
│   ├── training/
│   │   ├── train_mlp.py           # Core training loop
│   │   └── run_train.py           # Training orchestration (CV, grid search, logging)
│   └── evaluation/
│       ├── fairness_metrics.py    # Separation / adTPR / adFPR computation
│       ├── fairness_plots.py      # Fairness visualization utilities
│       └── fold_evaluation.py     # Per-fold performance and fairness aggregation
│
└── experiments/
    ├── run_simulation.py          # Runs M_STATIC / M_DYNAMIC on simulated scenarios (single run or grid search)
    └── run_realData.py            # Runs M_STATIC / M_DYNAMIC on the processed real dataset (single run or grid search)
```

## Simulation scenarios

The R-based simulation generates data under four discrimination scenarios, controlling how the sensitive attribute `S` influences outcomes:

| Scenario   | Description |
|------------|-------------|
| `fair`     | No discrimination — baseline |
| `direct`   | Direct effect of the sensitive attribute on the outcome |
| `proxy`    | Discrimination mediated through a proxy variable |
| `temporal` | Discrimination that evolves/grows across landmarks |

Each scenario can be run at Low/High intensity, with time-varying covariates (`X3`, `X4`, `X6`) generated via a VAR process (`genvar.R`).

## Getting started

The pipeline is designed to run end-to-end from `run_thesis.ipynb` (Colab-oriented), which covers:

1. **Simulation** — generate synthetic data (`data_generation/simulation/run.R`), then train/evaluate (`experiments/run_simulation.py`), optionally as a grid search over the fairness penalty coefficients (`ALPHA`, `BETA`).
2. **Real data** — match Freddie Mac SFLLD with HMDA (`match_FreddieMac_HMDA.py`), build the longitudinal panel (`build_panel.py`), apply weighted undersampling (`sampling.py`) and preprocessing (`preprocessing.py`), then train/evaluate (`experiments/run_realData.py`).

Example (simulation):
```bash
cd data_generation/simulation && Rscript run.R
python experiments/run_simulation.py \
    --data_dir data_generation/simulation/<run_folder> \
    --scenario fair
```

Example (real data):
```bash
python data_generation/realData/match_FreddieMac_HMDA.py --drive_root <path> --year 2024
python data_generation/realData/build_panel.py --drive_root <path> --year 2019
python data_generation/realData/sampling.py --path <path>/output/
python data_generation/realData/preprocessing.py --path_in <path>/output/panel_sampled.csv --path_out <path>
python experiments/run_realData.py --data_path <path>/output/panel_cleaned.csv --config experiments/configs/real_data_config.yaml
```

## Configuration

All key parameters (fairness penalty weights `ALPHA`/`BETA`, EO aggregation mode, MLP architecture, training hyperparameters, landmark schedules, sensitive attribute selection, grid search ranges, W&B logging) are centralized in `config.py`.

## Fairness metric

The primary fairness criterion is **separation (equalized odds)**, computed per landmark for `M_DYNAMIC` and aggregated (mean / weighted / trend-aware) across the prediction horizon, in addition to group-level TPR/FPR breakdowns (`fairness_metrics.py`).

## Experiment tracking

Training runs can be logged to [Weights & Biases](https://wandb.ai/) (`USE_WANDB` in `config.py`).

## License

No license has been specified for this repository.


#

Code for a Master's thesis in Data Science & Engineering: *An Equitable Framework for Dynamic Credit Risk — Fairness Penalization in Discrete-Time Survival Models*.

Compares a **static** credit-scoring model (`M_STATIC`) against a **dynamic, landmark-based discrete-time survival model** (`M_DYNAMIC`), each trainable with an equalized-odds fairness penalty, on both simulated and real (Freddie Mac + HMDA) mortgage data.

## Models

| | M_STATIC | M_DYNAMIC |
|---|---|---|
| Architecture | shared MLP (64→32, ReLU + BatchNorm + Dropout) | same MLP |
| Prediction | single default probability at a fixed horizon `v` | discrete hazard per sub-interval at each landmark `L`, combined into a default probability over `(L, L+v]` |
| Data format | one row per loan | landmark supermodel: stacked rows per loan per landmark, covariates frozen at `L` |

## Fairness penalty

Equalized-odds penalty (`|δFPR| + |δFNR|`, continuous relaxation), added to BCE loss:
- **Static** (`src/losses/eo_static.py`) — single gap over the whole dataset, weight `β`.
- **Dynamic** (`src/losses/eo_dynamic.py`) — per-landmark gaps combined via `mean` / `weighted` / `trend` aggregation, weight `α`, with an optional temporal schedule (`flat`, `decay`, `growth`, `u_shaped`, `n_shaped`).



## Simulation scenarios

| Scenario | Mechanism |
|---|---|
| `fair` | `S` has no effect — control case |
| `direct` | `S` enters the hazard directly + added noise for the disadvantaged group |
| `proxy` | `S` shifts correlated covariates that act as proxies |
| `temporal` | Same as `proxy`, but the shift grows over time |

## Real data

Freddie Mac SFLLD (2018–2024) exact-matched to HMDA on 8 loan variables (~3.35% match rate), reshaped to person-period format, with weighted sampling (100k loans) favoring defaults, protected groups, and time-varying loans. Sensitive attributes: `sex`, `race`, `age`.

## Evaluation

- **Performance**: iAUC / IBS (trapezoidal-integrated, landmark-normalized AUC / Brier score), comparable to `M_STATIC`'s single AUC/Brier.
- **Fairness**: adTPR / adFPR, and **SEP-AUC** (integrated separation curve across landmarks).
- **CV**: loan-level `GroupKFold` (K=3) + `GroupShuffleSplit` (50/50) for val/test.
- **Protocol**: grid search over `α`/`β ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7}` (seed=42), then 10 repetitions at the selected value with different seeds; results reported as mean ± std.
