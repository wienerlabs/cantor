"""Training loop for state prediction models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import structlog

from cantor.models.transformer import StatePredictor
from cantor.training.loss import PredictionLoss
from cantor.core.config import TrainingConfig

logger = structlog.get_logger()


class Trainer:
    """Training loop with learning rate scheduling and checkpointing."""

    def __init__(
        self,
        model: StatePredictor,
        config: TrainingConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2,
        )
        
        self.loss_fn = PredictionLoss()
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0

    def train_epoch(
        self, dataloader: DataLoader
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_metrics: dict[str, float] = {}
        num_batches = 0
        
        for batch in dataloader:
            sequences, targets = batch
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions, uncertainty = self.model(sequences)
            loss, metrics = self.loss_fn(predictions, targets, uncertainty)
            
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate metrics
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0) + value
            
            num_batches += 1
            self.global_step += 1
        
        # Average metrics
        return {k: v / num_batches for k, v in total_metrics.items()}

    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_metrics: dict[str, float] = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                sequences, targets = batch
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                predictions, uncertainty = self.model(sequences)
                _, metrics = self.loss_fn(predictions, targets, uncertainty)
                
                for key, value in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0) + value
                
                num_batches += 1
        
        return {k: v / num_batches for k, v in total_metrics.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop with early stopping."""
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        
        for epoch in range(self.config.max_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            
            logger.info(
                "epoch_complete",
                epoch=epoch + 1,
                train_loss=train_metrics["total"],
                val_loss=val_metrics["total"],
                lr=self.optimizer.param_groups[0]["lr"],
            )
            
            # Early stopping check
            if val_metrics["total"] < self.best_loss:
                self.best_loss = val_metrics["total"]
                self.patience_counter = 0
                
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir / "best_model.pt")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info("early_stopping", epoch=epoch + 1)
                    break
        
        return history

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }, path)
        logger.info("checkpoint_saved", path=str(path))

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        logger.info("checkpoint_loaded", path=str(path))

