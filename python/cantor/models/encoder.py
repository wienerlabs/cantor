"""State encoding layers for the prediction model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class StateEncoder(nn.Module):
    """Encodes raw transaction features into model hidden dimension."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] raw features
        Returns:
            [batch, seq_len, hidden_dim] encoded features
        """
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Transpose for positional encoding: [seq_len, batch, hidden_dim]
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        
        # Back to [batch, seq_len, hidden_dim]
        return x.transpose(0, 1)


class StateDecoder(nn.Module):
    """Decodes hidden representations back to state vectors."""

    def __init__(
        self,
        hidden_dim: int = 512,
        output_dim: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )
        
        # Uncertainty head for confidence estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, hidden_dim] final hidden state
        Returns:
            prediction: [batch, output_dim] predicted state
            uncertainty: [batch, 1] prediction uncertainty
        """
        prediction = self.output_projection(x)
        uncertainty = self.uncertainty_head(x)
        return prediction, uncertainty


class GatedResidualBlock(nn.Module):
    """Gated residual connection for feature refinement."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        
        gate = torch.sigmoid(self.gate(x))
        x = self.fc2(x)
        
        return self.norm(residual + gate * x)

