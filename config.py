"""
Global configuration for the project.
All parameters are defined here and imported by experiments and src modules.

"""

import torch

# Device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
SEED = 42

# Fairness penalty coefficients 
BETA  = 0.5   # M_STATIC  EO penalty weight
ALPHA = 0.5    # M_DYNAMIC EO penalty weight

# EO penalty mode
# Options: "mean" | "weighted" | "trend_aware" | "weighted+trend"
EO_MODE_D = "trend_aware"   # dynamic model

# Time schedule mode (alpha_schedule)
# Options: "flat" | "decay" | "growth" | "u_shaped" | "early_focus"
SCHEDULE_MODE_D = "decay"

# MLP architecture 
HIDDEN1  = 64
HIDDEN2  = 32
DROPOUT  = 0.3

# Training 
LR           = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS     = 200
PATIENCE     = 30
MIN_LR       = 1e-5
PW_CLIP      = 20.0    # cap on pos_weight

# Cross-validation 
N_FOLDS = 5

# Simulation dataset
SCENARIO         = "fair"          # "fair" | "direct" | "proxy" | "temporal"
HORIZON          = 3               # prediction horizon (periods)
LANDMARKS_SIM        = list(range(1, 13))
N_TEST_LANDMARKS = 12

# Column names in simulation dataset
ID_COL    = "ID"
TIME_COL  = "Time"
EVENT_COL = "Event"
SENS_COL  = "S"

STATIC_COLS_SIM  = ["X1", "X2"]
TVC_COLS_SIM     = ["X3", "X4", "X6"]
CAT_COLS_SIM     = ["X5"]
ALL_NUM_COLS = STATIC_COLS_SIM + TVC_COLS_SIM

ATTR_NAME   = "S"
GROUP_NAMES_SIM = {0: "S0", 1: "S1"}

# Real dataset
HORIZON_MONTHS = 12
LANDMARKS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]

STATIC_COLS = [
    "credit_score", "original_dti", "original_ltv", 
    "interest_rate", "loan_term", "num_borrowers", 
]
TVC_COLS = [
    "current_upb", "current_interest_rate", "estimated_ltv", "bd_pct",
]
CAT_COLS = ["occupancy_status_orig", "loan_purpose_orig", ]
'''

STATIC_COLS = [
    "credit_score", "original_dti", "original_ltv", "first_time_homebuyer",
    "interest_rate", "loan_term", "num_borrowers", "loan_amount"
]
TVC_COLS = [
    "current_upb", "current_interest_rate", "estimated_ltv", "bd_pct", "current_upb_delta"
]
CAT_COLS = ["occupancy_status_orig", "loan_purpose_orig", "borrower_assistance_status"]
'''

FAIR_ATTR   = "SEX"    # "SEX" | "RACE" | "AGE"
GROUP_NAMES = {
    "SEX":  {0: "Male/Joint",        1: "Female"},
    "RACE": {0: "White/Asian", 1: "Black/Indian"},
    "AGE":  {0: "Old",         1: "Young"},
}

# Sampling
SAMPLING = {
    "random_seed":   42,
    "target_loans":  100_000,
    "w_default":     1.8,
    "w_disc":        1.4,
    "w_tvc":         1.5,
    "w_base":        1.0,
}

RACE_MINORITY = {
    "black or african american",
    "american indian or alaska native",
    "native hawaiian or other pacific islander",
    "2 or more races"
}

# Grid search
GRID_BETAS  = [0.0, 0.3, 0.5, 0.7, 1.0]
GRID_ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]

# W&B
USE_WANDB    = True
WANDB_ENTITY = "alessia-ciccaglione02-"
WANDB_PROJECT = "ThesisFairness"
