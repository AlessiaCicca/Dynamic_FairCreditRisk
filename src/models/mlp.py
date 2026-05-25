"""
MLP architecture for fair survival prediction.

"""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Two-hidden-layer MLP with BatchNorm and Dropout.

    Architecture:
        input → Linear(hidden1) → ReLU → BN → Dropout
               → Linear(hidden2) → ReLU → BN → Dropout
               → Linear(1) → scalar logit → nn.BCEWithLogitsLoss()
    """

    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout),

            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)

    # It converts the observed prevalence in the training set into log-odds and uses 
    # it as the initial bias for the final layer. This way, the model is already calibrated 
    # to the actual frequency of the event.
    def init_bias(self, prev):
        prev = float(np.clip(prev, 1e-6, 1 - 1e-6))
        with torch.no_grad():
            self.net[-1].bias.fill_(float(np.log(prev / (1 - prev))))
