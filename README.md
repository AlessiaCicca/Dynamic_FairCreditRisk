# FairSurvival_CreditRisk


## Repository Structure

```
FairSurvival_CreditRisk/
│
├── config.py                          # global parameters
├── requirements.txt
│
├── data_generation/
│   ├── simulation/                    # R scripts for synthetic data
│   │   ├── simulate_timevarying.R
│   │   └── simulate_test.R
│   └── fnma/                          # FNMA preprocessing
│       ├── build_panel.py
│       ├── build_static.py
│       ├── match_hmda.py
│       └── notebooks/
│           └── CheckDistributionMatch.ipynb
│
├── src/
│   ├── models/
│   │   └── mlp.py                     # MLP architecture
│   ├── losses/
│   │   ├── eo_static.py               # static EO penalty
│   │   └── eo_dynamic.py              # dynamic EO penalty (all modes)
│   ├── training/
│   │   ├── train_mlp.py               # training loop
│   │   └── cross_validation.py        # GroupKFold CV
│   ├── data/
│   │   ├── build_person_period.py
│   │   ├── build_dynamic.py
│   │   └── build_static.py
│   └── evaluation/
│       ├── fairness_metrics.py
│       ├── fairness_plots.py
│       └── auc_fairness.py
│
├── experiments/
│   ├── run_simulation.py
│   ├── run_fnma.py
│   └── configs/
│       ├── simulation_fair.yaml
│       ├── simulation_unfair.yaml
│       └── fnma.yaml
│
├── notebooks/
│   ├── InitialModel.ipynb
│   ├── ModelFairness.ipynb

```
│   └── Evaluation.ipynb
│
└── outputs/
    ├── simulation/
    └── fnma/
