"""Signal/base fusion with an 8-D base embedding and no signal statistics."""

import torch
from torch import nn


class DeepPlantSignalBaseBiLSTM(nn.Module):
    """DeepPlant fusion backbone with a 768-D-to-128-D signal MLP."""

    def __init__(
        self,
        hidden_dim: int,
        signal_mlp_hidden: int = 256,
        base_hidden: int = 128,
        fusion_hidden: int = 256,
        base_layers: int = 2,
        fusion_layers: int = 3,
        dropout: float = 0.5,
        signal_mlp_dropout: float = 0.1,
        vocab_size: int = 16,
    ) -> None:
        super().__init__()
        self.selected_stats = ()
        self.embedding_dim = 8
        # Retain the original empty buffer for checkpoint compatibility.
        self.register_buffer("selected_stat_indices", torch.empty(0, dtype=torch.long))

        self.base_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.base_lstm = nn.LSTM(
            input_size=8,
            hidden_size=base_hidden,
            num_layers=base_layers,
            dropout=dropout if base_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.base_projection = nn.Linear(base_hidden * 2, 128)

        self.signal_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, signal_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(signal_mlp_dropout),
            nn.Linear(signal_mlp_hidden, 128),
            nn.ReLU(),
        )

        self.fusion_lstm = nn.LSTM(
            input_size=256,
            hidden_size=fusion_hidden,
            num_layers=fusion_layers,
            dropout=dropout if fusion_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.fusion_hidden = fusion_hidden
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_hidden * 2, 2)

    def forward(
        self,
        signal_hidden: torch.Tensor,
        base_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.base_lstm.flatten_parameters()
        self.fusion_lstm.flatten_parameters()

        base_embedding = self.base_embedding(base_ids)
        base_input = base_embedding
        if base_input.shape[-1] != 8:
            raise RuntimeError(f"Expected base input dim 8, got {base_input.shape[-1]}")

        base_output, _ = self.base_lstm(base_input)
        base_output = self.activation(self.base_projection(base_output))
        signal_output = self.signal_mlp(signal_hidden)

        fused = torch.cat((base_output, signal_output), dim=2)
        fused, _ = self.fusion_lstm(fused)
        forward_last = fused[:, -1, : self.fusion_hidden]
        backward_first = fused[:, 0, self.fusion_hidden :]
        representation = torch.cat((forward_last, backward_first), dim=1)
        return self.classifier(self.dropout(representation))
