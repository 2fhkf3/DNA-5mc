"""Classifier architectures copied from the original Step-5 implementation."""

import torch
from torch import nn


def center_5mer(x: torch.Tensor, center_idx: int) -> torch.Tensor:
    """Return the five base embeddings centred on ``center_idx``."""
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, window, hidden], got {tuple(x.shape)}")
    if center_idx < 2 or center_idx + 2 >= x.shape[1]:
        raise ValueError(
            f"Window length {x.shape[1]} is too short for a center 5-mer"
        )
    return x[:, center_idx - 2 : center_idx + 3]


class Center5merMLPRaw768(nn.Module):
    """Two-hidden-layer MLP over the position-preserving center 5-mer."""

    def __init__(
        self,
        hidden_dim: int,
        window_size: int,
        hidden_dims: tuple[int, int] = (512, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        first_dim, second_dim = hidden_dims
        if first_dim <= 0 or second_dim <= 0:
            raise ValueError("MLP hidden dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.center_idx = window_size // 2
        self.base_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(5 * hidden_dim, first_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(first_dim, second_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(second_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5 = self.base_norm(center_5mer(x, self.center_idx))
        return self.head(x5.flatten(start_dim=1)).squeeze(-1)


class ResidualDepthwiseConvBlock(nn.Module):
    """Residual depthwise-separable Conv1D block over base positions."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + residual

        # Conv1D uses [B, C, K], whereas LayerNorm normalizes the last axis.
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class Center5merResidualDepthwiseConvRaw768(nn.Module):
    """Two residual depthwise Conv1D blocks on the center five positions."""

    def __init__(
        self,
        hidden_dim: int,
        window_size: int,
        channels: int = 128,
        head_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channels <= 0 or head_dim <= 0:
            raise ValueError("Conv channels and head dimension must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.center_idx = window_size // 2
        self.base_norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, channels)
        self.conv_blocks = nn.Sequential(
            ResidualDepthwiseConvBlock(channels, dropout),
            ResidualDepthwiseConvBlock(channels, dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(5 * channels, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5 = self.base_norm(center_5mer(x, self.center_idx))
        x5 = self.projection(x5)          # [B, 5, channels]
        x5 = x5.transpose(1, 2)           # [B, channels, 5]
        x5 = self.conv_blocks(x5)
        return self.head(x5.flatten(start_dim=1)).squeeze(-1)
