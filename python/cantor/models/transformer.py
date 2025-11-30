"""Transformer model for state transition prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from cantor.models.encoder import StateEncoder, StateDecoder, GatedResidualBlock
from cantor.core.config import ModelConfig


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        
        feedforward_dim = feedforward_dim or hidden_dim * 4
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x


class StatePredictor(nn.Module):
    """Transformer model for predicting next blockchain state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        
        self.encoder = StateEncoder(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        self.refinement = GatedResidualBlock(config.hidden_dim, config.dropout)
        
        self.decoder = StateDecoder(
            hidden_dim=config.hidden_dim,
            output_dim=config.state_dim,
            dropout=config.dropout,
        )
        
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, state_dim] transaction sequence
            mask: optional attention mask
        Returns:
            prediction: [batch, state_dim] predicted next state
            uncertainty: [batch, 1] prediction confidence
        """
        # Encode input sequence
        hidden = self.encoder(x)
        
        # Generate causal mask if not provided
        if mask is None:
            seq_len = x.size(1)
            mask = self._generate_causal_mask(seq_len, x.device)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden = block(hidden, mask)
        
        # Use last position for prediction
        last_hidden = hidden[:, -1, :]
        
        # Refine and decode
        last_hidden = self.refinement(last_hidden)
        prediction, uncertainty = self.decoder(last_hidden)
        
        return prediction, uncertainty

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def predict_with_confidence(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with normalized confidence score."""
        self.eval()
        with torch.no_grad():
            prediction, uncertainty = self.forward(x)
            confidence = 1.0 / (1.0 + uncertainty)
        return prediction, confidence

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

