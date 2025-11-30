"""Delta computation and compression logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch
import structlog

from cantor.core.types import StateDelta, StateVector, CompressionResult, Bytes32
from cantor.core.config import CompressionConfig
from cantor.models.transformer import StatePredictor
from cantor.compression.encoder import DeltaEncoder

logger = structlog.get_logger()


@dataclass(slots=True)
class PredictionResult:
    """Result of state prediction for a transaction."""
    
    predicted: NDArray[np.float32]
    actual: NDArray[np.float32]
    confidence: float
    delta: NDArray[np.float32]
    
    @property
    def delta_norm(self) -> float:
        return float(np.linalg.norm(self.delta))
    
    @property
    def is_compressible(self) -> bool:
        return self.confidence > 0.5 and self.delta_norm < 1.0


class DeltaCompressor:
    """Computes and compresses state transition deltas."""

    def __init__(
        self,
        model: StatePredictor,
        config: CompressionConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.encoder = DeltaEncoder(config.encoding)

    def compute_delta(
        self,
        sequence: NDArray[np.float32],
        actual_state: NDArray[np.float32],
    ) -> PredictionResult:
        """Compute prediction delta for a single state transition."""
        with torch.no_grad():
            seq_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            prediction, uncertainty = self.model(seq_tensor)
            
            predicted = prediction.cpu().numpy()[0]
            confidence = 1.0 / (1.0 + uncertainty.cpu().numpy()[0, 0])
        
        delta = actual_state - predicted
        
        return PredictionResult(
            predicted=predicted,
            actual=actual_state,
            confidence=float(confidence),
            delta=delta,
        )

    def compress_block(
        self,
        sequences: Sequence[NDArray[np.float32]],
        actual_states: Sequence[NDArray[np.float32]],
        tx_hashes: Sequence[Bytes32],
        block_number: int,
    ) -> CompressionResult:
        """Compress all state transitions in a block."""
        deltas: list[StateDelta] = []
        original_size = 0
        compressed_size = 0
        
        for seq, actual, tx_hash in zip(sequences, actual_states, tx_hashes):
            result = self.compute_delta(seq, actual)
            original_size += actual.nbytes
            
            # Decide compression strategy based on confidence
            threshold = self._adaptive_threshold(result.confidence)
            
            if result.delta_norm < threshold:
                delta_bytes = self.encoder.encode(result.delta)
                compressed_size += len(delta_bytes)
            else:
                delta_bytes = self.encoder.encode_full(actual)
                compressed_size += len(delta_bytes)
            
            deltas.append(StateDelta(
                tx_hash=tx_hash,
                predicted_root=self._compute_hash(result.predicted),
                actual_root=self._compute_hash(actual),
                delta_bytes=delta_bytes,
                confidence=result.confidence,
            ))
        
        # Build merkle tree
        from cantor.compression.merkle import MerkleDeltaTree
        merkle_tree = MerkleDeltaTree()
        delta_tree_root = merkle_tree.build([d.delta_bytes for d in deltas])
        
        proofs = tuple(
            merkle_tree.generate_proof(i, d, "v1.0")
            for i, d in enumerate(deltas)
        )
        
        logger.info(
            "block_compressed",
            block=block_number,
            ratio=original_size / max(compressed_size, 1),
            deltas=len(deltas),
        )
        
        return CompressionResult(
            block_number=block_number,
            original_size=original_size,
            compressed_size=compressed_size,
            delta_tree_root=delta_tree_root,
            deltas=tuple(deltas),
            proofs=proofs,
        )

    def _adaptive_threshold(self, confidence: float) -> float:
        """Compute adaptive compression threshold based on confidence."""
        if not self.config.adaptive_threshold:
            return self.config.delta_threshold
        
        # Higher confidence = more aggressive compression
        base = self.config.delta_threshold
        if confidence > 0.9:
            return base * 2.0
        elif confidence > 0.7:
            return base * 1.5
        elif confidence > 0.5:
            return base
        else:
            return base * 0.5

    def _compute_hash(self, data: NDArray[np.float32]) -> Bytes32:
        import hashlib
        return hashlib.sha256(data.tobytes()).digest()

