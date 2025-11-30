"""Fraud proof generation for dispute resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import structlog

from cantor.core.types import (
    VerificationProof,
    CompressionResult,
    StateDelta,
    Bytes32,
)
from cantor.verification.verifier import StateVerifier, VerificationStatus

logger = structlog.get_logger()


class FraudType(Enum):
    INVALID_MERKLE_ROOT = "invalid_merkle_root"
    PREDICTION_MISMATCH = "prediction_mismatch"
    DELTA_MANIPULATION = "delta_manipulation"
    STATE_RECONSTRUCTION_FAILURE = "state_reconstruction_failure"


@dataclass(frozen=True, slots=True)
class FraudProof:
    """Proof of fraudulent compression."""
    
    fraud_type: FraudType
    block_number: int
    tx_index: int
    tx_hash: Bytes32
    claimed_root: Bytes32
    actual_root: Bytes32
    evidence: bytes
    description: str


@dataclass(frozen=True, slots=True)
class ChallengeRequest:
    """Request to challenge a specific transaction."""
    
    block_number: int
    tx_index: int
    challenger: bytes


@dataclass(frozen=True, slots=True)
class ChallengeResponse:
    """Response to a challenge with proof data."""
    
    proof: VerificationProof
    sequence: bytes
    is_valid: bool


class FraudProofGenerator:
    """Generates fraud proofs for invalid compressions."""

    def __init__(self, verifier: StateVerifier) -> None:
        self.verifier = verifier

    def detect_fraud(
        self,
        result: CompressionResult,
        sequences: Sequence[NDArray[np.float32]],
    ) -> list[FraudProof]:
        """Detect and generate fraud proofs for a block."""
        frauds: list[FraudProof] = []
        
        verification_results = self.verifier.verify_block(
            result,
            list(sequences),
        )
        
        for i, (vr, proof) in enumerate(zip(verification_results, result.proofs)):
            if vr.status == VerificationStatus.VALID:
                continue
            
            fraud = self._generate_fraud_proof(
                result.block_number,
                i,
                proof,
                vr,
                result.delta_tree_root,
            )
            if fraud:
                frauds.append(fraud)
        
        if frauds:
            logger.warning(
                "fraud_detected",
                block=result.block_number,
                count=len(frauds),
            )
        
        return frauds

    def _generate_fraud_proof(
        self,
        block_number: int,
        tx_index: int,
        proof: VerificationProof,
        vr: "VerificationStatus",
        claimed_root: Bytes32,
    ) -> FraudProof | None:
        """Generate a fraud proof for a specific verification failure."""
        from cantor.verification.verifier import VerificationResult
        
        if not isinstance(vr, VerificationResult):
            return None
        
        fraud_type = self._map_status_to_fraud_type(vr.status)
        if fraud_type is None:
            return None
        
        evidence = self._serialize_evidence(proof, vr)
        
        return FraudProof(
            fraud_type=fraud_type,
            block_number=block_number,
            tx_index=tx_index,
            tx_hash=proof.tx_hash,
            claimed_root=claimed_root,
            actual_root=vr.actual_state or b"",
            evidence=evidence,
            description=vr.message,
        )

    def _map_status_to_fraud_type(
        self, status: VerificationStatus
    ) -> FraudType | None:
        mapping = {
            VerificationStatus.INVALID_MERKLE: FraudType.INVALID_MERKLE_ROOT,
            VerificationStatus.INVALID_PREDICTION: FraudType.PREDICTION_MISMATCH,
            VerificationStatus.INVALID_DELTA: FraudType.DELTA_MANIPULATION,
        }
        return mapping.get(status)

    def _serialize_evidence(
        self,
        proof: VerificationProof,
        vr: "VerificationResult",
    ) -> bytes:
        """Serialize evidence for on-chain submission."""
        import struct
        
        parts = [
            proof.tx_hash,
            proof.predicted_state,
            proof.delta.delta_bytes,
        ]
        
        for hash_val in proof.merkle_proof.path:
            parts.append(hash_val)
        
        indices_bytes = struct.pack(
            f"{len(proof.merkle_proof.indices)}B",
            *proof.merkle_proof.indices,
        )
        parts.append(indices_bytes)
        
        return b"".join(parts)

    def respond_to_challenge(
        self,
        challenge: ChallengeRequest,
        result: CompressionResult,
        sequences: Sequence[NDArray[np.float32]],
    ) -> ChallengeResponse:
        """Respond to a challenge with proof data."""
        if challenge.tx_index >= len(result.proofs):
            raise IndexError("Transaction index out of range")
        
        proof = result.proofs[challenge.tx_index]
        sequence = sequences[challenge.tx_index]
        
        vr = self.verifier.verify_proof(
            proof,
            sequence,
            result.delta_tree_root,
        )
        
        return ChallengeResponse(
            proof=proof,
            sequence=sequence.tobytes(),
            is_valid=vr.status == VerificationStatus.VALID,
        )

