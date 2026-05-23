"""
src/models/mlp.py

MLP architecture for fair survival prediction.
Used by M_STATIC, M_DYNAMIC, and M_PP models.
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
               → Linear(1) → scalar logit

    The output is a raw logit (no sigmoid). Apply torch.sigmoid()
    for probabilities or use nn.BCEWithLogitsLoss() during training.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden1 : int, default 64
        Size of first hidden layer.
    hidden2 : int, default 32
        Size of second hidden layer.
    dropout : float, default 0.3
        Dropout rate applied after each hidden layer.
    """

    def __init__(self, input_dim: int, hidden1: int = 64,
                 hidden2: int = 32, dropout: float = 0.3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (N, input_dim)

        Returns
        -------
        torch.Tensor of shape (N,) — raw logits
        """
        return self.net(x).view(-1)

    def init_bias(self, prev: float) -> None:
        """
        Initialise the output layer bias with the log-odds of the
        training prevalence. This speeds up convergence on imbalanced
        datasets by starting predictions close to the base rate.

        Parameters
        ----------
        prev : float
            Positive class prevalence in the training set (e.g. 0.05).
        """
        prev = float(np.clip(prev, 1e-6, 1 - 1e-6))
        with torch.no_grad():
            self.net[-1].bias.fill_(float(np.log(prev / (1 - prev))))
