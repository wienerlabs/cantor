//! High-performance verification for CANTOR proofs.

use cantor_core::{
    Hash32, VerificationProof, CompressionResult, MerkleProof, CantorError, Result,
};
use cantor_merkle::MerkleDeltaTree;
use cantor_compress::{DeltaEncoder, CompressionMethod};
use sha2::{Sha256, Digest};

/// Verification status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerificationStatus {
    Valid,
    InvalidMerkle,
    InvalidPrediction,
    InvalidDelta,
    ModelMismatch,
}

/// Result of verification.
#[derive(Clone, Debug)]
pub struct VerificationResult {
    pub status: VerificationStatus,
    pub tx_hash: Option<Hash32>,
    pub message: String,
}

impl VerificationResult {
    pub fn valid(tx_hash: Hash32) -> Self {
        Self {
            status: VerificationStatus::Valid,
            tx_hash: Some(tx_hash),
            message: "Proof verified successfully".to_string(),
        }
    }

    pub fn invalid(status: VerificationStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            tx_hash: None,
            message: message.into(),
        }
    }
}

/// High-performance state verifier.
pub struct StateVerifier {
    model_version: String,
    encoder: DeltaEncoder,
}

impl StateVerifier {
    pub fn new(model_version: impl Into<String>) -> Self {
        Self {
            model_version: model_version.into(),
            encoder: DeltaEncoder::new(CompressionMethod::Lz4),
        }
    }

    /// Verify a single proof.
    pub fn verify_proof(
        &self,
        proof: &VerificationProof,
        predicted_state: &[f32],
        expected_root: &Hash32,
    ) -> VerificationResult {
        // Check model version
        if proof.model_version != self.model_version {
            return VerificationResult::invalid(
                VerificationStatus::ModelMismatch,
                format!(
                    "Model version mismatch: {} != {}",
                    proof.model_version, self.model_version
                ),
            );
        }

        // Verify merkle proof
        if !MerkleDeltaTree::verify_proof(&proof.merkle_proof, expected_root) {
            return VerificationResult::invalid(
                VerificationStatus::InvalidMerkle,
                "Merkle proof verification failed",
            );
        }

        // Verify predicted state hash
        let predicted_hash = Self::compute_hash(predicted_state);
        if predicted_hash != proof.predicted_state {
            return VerificationResult::invalid(
                VerificationStatus::InvalidPrediction,
                "Predicted state hash mismatch",
            );
        }

        // Decode delta and reconstruct
        let delta = match self.encoder.decode(&proof.delta.delta_bytes) {
            Ok(d) => d,
            Err(_) => {
                return VerificationResult::invalid(
                    VerificationStatus::InvalidDelta,
                    "Failed to decode delta",
                );
            }
        };

        // Reconstruct actual state
        if delta.len() != predicted_state.len() {
            return VerificationResult::invalid(
                VerificationStatus::InvalidDelta,
                "Delta dimension mismatch",
            );
        }

        let reconstructed: Vec<f32> = predicted_state
            .iter()
            .zip(delta.iter())
            .map(|(p, d)| p + d)
            .collect();

        let reconstructed_hash = Self::compute_hash(&reconstructed);
        if reconstructed_hash != proof.delta.actual_root {
            return VerificationResult::invalid(
                VerificationStatus::InvalidDelta,
                "Reconstructed state hash mismatch",
            );
        }

        VerificationResult::valid(proof.tx_hash)
    }

    /// Batch verify multiple proofs.
    pub fn verify_batch(
        &self,
        result: &CompressionResult,
        predicted_states: &[Vec<f32>],
    ) -> Vec<VerificationResult> {
        result
            .proofs
            .iter()
            .zip(predicted_states.iter())
            .map(|(proof, predicted)| {
                self.verify_proof(proof, predicted, &result.delta_tree_root)
            })
            .collect()
    }

    fn compute_hash(data: &[f32]) -> Hash32 {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let result = Sha256::digest(&bytes);
        Hash32::from_slice(&result).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verifier_creation() {
        let verifier = StateVerifier::new("v1.0.0");
        assert_eq!(verifier.model_version, "v1.0.0");
    }
}

