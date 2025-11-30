"""Configuration management for CANTOR system."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Neural network architecture configuration."""
    
    state_dim: int = Field(default=4096, description="State vector dimension")
    hidden_dim: int = Field(default=512, description="Transformer hidden dimension")
    num_layers: int = Field(default=8, description="Number of transformer layers")
    num_heads: int = Field(default=8, description="Number of attention heads")
    context_length: int = Field(default=128, description="Transaction context window")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    

class CompressionConfig(BaseModel):
    """Compression protocol configuration."""
    
    delta_threshold: float = Field(default=0.01, description="Max normalized delta for compression")
    min_confidence: float = Field(default=0.7, description="Minimum prediction confidence")
    adaptive_threshold: bool = Field(default=True, description="Use confidence-based thresholds")
    encoding: Literal["varint", "huffman", "lz4"] = Field(default="lz4")


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=1000, ge=0)
    max_epochs: int = Field(default=100, ge=1)
    gradient_clip: float = Field(default=1.0, gt=0)
    early_stopping_patience: int = Field(default=10, ge=1)
    

class DataConfig(BaseModel):
    """Data pipeline configuration."""
    
    rpc_url: str = Field(default="http://localhost:8545")
    start_block: int = Field(default=0, ge=0)
    end_block: int | None = Field(default=None)
    batch_size: int = Field(default=100, ge=1)
    cache_dir: Path = Field(default=Path("data/cache"))
    

class CantorConfig(BaseSettings):
    """Root configuration for CANTOR system."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")
    seed: int = Field(default=42)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    
    model_config = {"env_prefix": "CANTOR_", "env_nested_delimiter": "__"}
    
    @classmethod
    def from_yaml(cls, path: Path) -> CantorConfig:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

