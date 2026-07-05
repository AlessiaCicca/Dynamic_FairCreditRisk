"""
Global configuration for the project.
All parameters are defined here and imported by experiments and src modules.

"""

import torch

# Device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility 42 | 123 | 456 | 789 | 1234 | 2024 | 314 | 99 | 7 | 2025
SEED = 42

# Fairness penalty coefficients 
BETA  = 0   # M_STATIC  EO penalty weight
ALPHA = 0 # M_DYNAMIC EO penalty weight

# EO penalty mode
# Options: "mean" | "weighted" | "trend_aware"
EO_MODE_D = "mean"   # dynamic model

# Time schedule mode (alpha_schedule)
# Options: "flat" | "decay" | "growth" | "u_shaped"  | "n_shaped" 
SCHEDULE_MODE_D = "flat"

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
PW_CLIP      = 10.0    # cap on pos_weight

# Cross-validation 
N_FOLDS = 5

# Simulation dataset
SCENARIO         = "fair"          # "fair" | "direct" | "proxy" | "temporal"
HORIZON          = 3               # prediction horizon (periods)
LANDMARKS_SIM        = list(range(1, 13))
N_TEST_LANDMARKS = 12
DELTA_SIM=1

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
#LANDMARKS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
#LANDMARKS = [0, 6,  12,  18,  24,  30,  36,  42, 48]
LANDMARKS = [0, 4, 8, 12, 16,  20,  24, 28,  32,  36, 40,  44, 48]
DELTA=4

STATIC_COLS = [
    "credit_score", "original_dti", "original_ltv", 
    "interest_rate", "loan_term", "num_borrowers", "loan_amount"
]
TVC_COLS = [
    "current_upb", "current_interest_rate", "estimated_ltv", "bd_pct","current_upb_delta"
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


# Grid search SIMULATION
#GRID_BETAS  = [0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
#GRID_ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Grid search REAL_DATA
GRID_BETAS  = [0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
GRID_ALPHAS = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]



# W&B
USE_WANDB    = False
WANDB_ENTITY = "alessia-ciccaglione02-"
WANDB_PROJECT = "ThesisFairness"
