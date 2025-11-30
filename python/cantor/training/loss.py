"""Loss functions for state prediction training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLoss(nn.Module):
    """Combined loss for state prediction with uncertainty weighting."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        smoothness_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.uncertainty_weight = uncertainty_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            prediction: [batch, dim] predicted state vectors
            target: [batch, dim] actual state vectors
            uncertainty: [batch, 1] predicted uncertainty
        Returns:
            total_loss: scalar loss value
            metrics: dict of individual loss components
        """
        # MSE loss weighted by inverse uncertainty
        mse = F.mse_loss(prediction, target, reduction="none").mean(dim=-1, keepdim=True)
        
        # Uncertainty-weighted loss (Gaussian NLL approximation)
        precision = 1.0 / (uncertainty + 1e-6)
        weighted_mse = (precision * mse).mean()
        
        # Regularize uncertainty to prevent collapse
        uncertainty_reg = uncertainty.mean()
        
        # Smoothness regularization on predictions
        smoothness = self._smoothness_loss(prediction)
        
        total_loss = (
            self.mse_weight * weighted_mse
            + self.uncertainty_weight * uncertainty_reg
            + self.smoothness_weight * smoothness
        )
        
        metrics = {
            "mse": mse.mean().item(),
            "weighted_mse": weighted_mse.item(),
            "uncertainty": uncertainty.mean().item(),
            "smoothness": smoothness.item(),
            "total": total_loss.item(),
        }
        
        return total_loss, metrics

    def _smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Encourage smooth predictions across feature dimensions."""
        if x.size(-1) < 2:
            return torch.tensor(0.0, device=x.device)
        diff = x[:, 1:] - x[:, :-1]
        return (diff ** 2).mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning discriminative state representations."""

    def __init__(self, temperature: float = 0.07, margin: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: [batch, dim] anchor embeddings
            positive: [batch, dim] similar state embeddings
            negative: [batch, dim] dissimilar state embeddings
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        
        # Compute similarities
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature
        neg_sim = (anchor * negative).sum(dim=-1) / self.temperature
        
        # Triplet margin loss
        loss = F.relu(self.margin - pos_sim + neg_sim).mean()
        
        return loss


class CompressionAwareLoss(nn.Module):
    """Loss that encourages predictions amenable to compression."""

    def __init__(
        self,
        base_weight: float = 1.0,
        sparsity_weight: float = 0.1,
        delta_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.base_weight = base_weight
        self.sparsity_weight = sparsity_weight
        self.delta_threshold = delta_threshold

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Base prediction loss
        mse = F.mse_loss(prediction, target)
        
        # Encourage sparse deltas (many near-zero differences)
        delta = prediction - target
        delta_magnitude = delta.abs()
        
        # Soft threshold penalty - encourage deltas below threshold
        threshold_penalty = F.relu(delta_magnitude - self.delta_threshold).mean()
        
        # Compression ratio estimate
        compressible = (delta_magnitude < self.delta_threshold).float().mean()
        
        total_loss = self.base_weight * mse + self.sparsity_weight * threshold_penalty
        
        metrics = {
            "mse": mse.item(),
            "threshold_penalty": threshold_penalty.item(),
            "compressible_ratio": compressible.item(),
        }
        
        return total_loss, metrics

