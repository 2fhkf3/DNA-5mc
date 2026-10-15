"""Classifier architectures copied from the original Step-5 implementation."""

import torch
from torch import nn


class CenterBaseLinearRaw768(nn.Module):
    """Rockfish-style linear probe on the center base representation."""

    def __init__(self, hidden_dim: int, window_size: int) -> None:
        super().__init__()
        self.center_idx = window_size // 2
        self.base_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center = self.base_norm(x[:, self.center_idx])
        return self.head(center).squeeze(-1)


class Center5merLinearRaw768(nn.Module):
    """Flatten the center 5 x 768 base vectors and apply a linear probe."""

    def __init__(self, hidden_dim: int, window_size: int) -> None:
        super().__init__()
        self.center_idx = window_size // 2
        self.base_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(5 * hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5 = self.base_norm(x[:, self.center_idx - 2 : self.center_idx + 3])
        return self.head(x5.flatten(1)).squeeze(-1)


class Center5merGaussianPoolLinearRaw768(nn.Module):
    """Use a fixed Gaussian-weighted center 5mer representation with a linear probe."""

    def __init__(self, hidden_dim: int, window_size: int, sigma: float) -> None:
        super().__init__()
        self.center_idx = window_size // 2
        self.base_norm = nn.LayerNorm(hidden_dim)
        offsets = torch.arange(-2, 3, dtype=torch.float32)
        weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
        weights = weights / weights.sum()
        self.register_buffer("weights", weights.view(1, 5, 1))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5 = self.base_norm(x[:, self.center_idx - 2 : self.center_idx + 3])
        pooled = (x5 * self.weights).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class Center5merLearnedPoolLinearRaw768(nn.Module):
    """Learn one global set of center-5mer weights and apply a linear probe."""

    def __init__(self, hidden_dim: int, window_size: int) -> None:
        super().__init__()
        self.center_idx = window_size // 2
        self.base_norm = nn.LayerNorm(hidden_dim)
        self.weight_logits = nn.Parameter(torch.zeros(5))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5 = self.base_norm(x[:, self.center_idx - 2 : self.center_idx + 3])
        weights = torch.softmax(self.weight_logits, dim=0).view(1, 5, 1)
        pooled = (x5 * weights).sum(dim=1)
        return self.head(pooled).squeeze(-1)
