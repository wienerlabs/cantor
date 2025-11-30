"""State verification for compressed proofs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
import torch
import structlog

from cantor.core.types import (
    VerificationProof,
    CompressionResult,
    StateDelta,
    Bytes32,
)
from cantor.models.transformer import StatePredictor
from cantor.compression.merkle import MerkleDeltaTree
from cantor.compression.encoder import DeltaEncoder

logger = structlog.get_logger()


class VerificationStatus(Enum):
    VALID = "valid"
    INVALID_MERKLE = "invalid_merkle"
    INVALID_PREDICTION = "invalid_prediction"
    INVALID_DELTA = "invalid_delta"
    MODEL_MISMATCH = "model_mismatch"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: VerificationStatus
    message: str
    tx_hash: Bytes32 | None = None
    expected_state: Bytes32 | None = None
    actual_state: Bytes32 | None = None


class StateVerifier:
    """Verifies compressed state proofs."""

    def __init__(
        self,
        model: StatePredictor,
        model_version: str,
        device: str = "cpu",
        tolerance: float = 1e-5,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.model_version = model_version
        self.device = device
        self.tolerance = tolerance
        self.encoder = DeltaEncoder()

    def verify_proof(
        self,
        proof: VerificationProof,
        sequence: NDArray[np.float32],
        expected_root: Bytes32,
    ) -> VerificationResult:
        """Verify a single transaction proof."""
        # Check model version
        if proof.model_version != self.model_version:
            return VerificationResult(
                status=VerificationStatus.MODEL_MISMATCH,
                message=f"Model version mismatch: {proof.model_version} != {self.model_version}",
                tx_hash=proof.tx_hash,
            )
        
        # Verify merkle proof
        merkle_tree = MerkleDeltaTree()
        if not merkle_tree.verify_proof(proof.merkle_proof, expected_root):
            return VerificationResult(
                status=VerificationStatus.INVALID_MERKLE,
                message="Merkle proof verification failed",
                tx_hash=proof.tx_hash,
            )
        
        # Recompute prediction
        with torch.no_grad():
            seq_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            prediction, _ = self.model(seq_tensor)
            predicted = prediction.cpu().numpy()[0]
        
        # Verify predicted state matches
        predicted_hash = self._compute_hash(predicted)
        if predicted_hash != proof.predicted_state:
            return VerificationResult(
                status=VerificationStatus.INVALID_PREDICTION,
                message="Predicted state does not match",
                tx_hash=proof.tx_hash,
                expected_state=proof.predicted_state,
                actual_state=predicted_hash,
            )
        
        # Apply delta and verify
        delta = self.encoder.decode(proof.delta.delta_bytes)
        reconstructed = predicted + delta
        reconstructed_hash = self._compute_hash(reconstructed)
        
        if reconstructed_hash != proof.delta.actual_root:
            return VerificationResult(
                status=VerificationStatus.INVALID_DELTA,
                message="Reconstructed state does not match actual",
                tx_hash=proof.tx_hash,
                expected_state=proof.delta.actual_root,
                actual_state=reconstructed_hash,
            )
        
        return VerificationResult(
            status=VerificationStatus.VALID,
            message="Proof verified successfully",
            tx_hash=proof.tx_hash,
        )

    def verify_block(
        self,
        result: CompressionResult,
        sequences: list[NDArray[np.float32]],
    ) -> list[VerificationResult]:
        """Verify all proofs in a compressed block."""
        results: list[VerificationResult] = []
        
        for proof, sequence in zip(result.proofs, sequences):
            verification = self.verify_proof(
                proof,
                sequence,
                result.delta_tree_root,
            )
            results.append(verification)
            
            if verification.status != VerificationStatus.VALID:
                logger.warning(
                    "verification_failed",
                    tx_hash=proof.tx_hash.hex() if proof.tx_hash else None,
                    status=verification.status.value,
                    message=verification.message,
                )
        
        valid_count = sum(1 for r in results if r.status == VerificationStatus.VALID)
        logger.info(
            "block_verified",
            block=result.block_number,
            valid=valid_count,
            total=len(results),
        )
        
        return results

    def _compute_hash(self, data: NDArray[np.float32]) -> Bytes32:
        import hashlib
        return hashlib.sha256(data.tobytes()).digest()

