//! Core type definitions for CANTOR.

use serde::{Deserialize, Serialize};
use std::fmt;

/// 32-byte hash type used throughout the system.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hash32(pub [u8; 32]);

impl Hash32 {
    pub const ZERO: Self = Self([0u8; 32]);

    pub fn from_slice(slice: &[u8]) -> Option<Self> {
        if slice.len() != 32 {
            return None;
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(slice);
        Some(Self(arr))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for Hash32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hash32({})", hex::encode(&self.0[..8]))
    }
}

impl fmt::Display for Hash32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", hex::encode(&self.0))
    }
}

impl AsRef<[u8]> for Hash32 {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// State vector representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateVector {
    pub data: Vec<f32>,
    pub dimension: usize,
}

impl StateVector {
    pub fn new(data: Vec<f32>) -> Self {
        let dimension = data.len();
        Self { data, dimension }
    }

    pub fn zeros(dimension: usize) -> Self {
        Self {
            data: vec![0.0; dimension],
            dimension,
        }
    }

    pub fn compute_hash(&self) -> Hash32 {
        use sha2::{Sha256, Digest};
        let bytes: Vec<u8> = self.data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let result = Sha256::digest(&bytes);
        Hash32::from_slice(&result).unwrap()
    }
}

/// Delta between predicted and actual state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateDelta {
    pub tx_hash: Hash32,
    pub predicted_root: Hash32,
    pub actual_root: Hash32,
    pub delta_bytes: Vec<u8>,
    pub confidence: f32,
}

/// Merkle proof for a delta.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_hash: Hash32,
    pub path: Vec<Hash32>,
    pub indices: Vec<u8>,
}

impl MerkleProof {
    pub fn verify(&self, root: &Hash32) -> bool {
        use sha2::{Sha256, Digest};
        
        let mut current = self.leaf_hash;
        
        for (sibling, &index) in self.path.iter().zip(self.indices.iter()) {
            let combined = if index == 0 {
                [current.as_ref(), sibling.as_ref()].concat()
            } else {
                [sibling.as_ref(), current.as_ref()].concat()
            };
            let result = Sha256::digest(&combined);
            current = Hash32::from_slice(&result).unwrap();
        }
        
        current == *root
    }
}

/// Verification proof for a transaction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationProof {
    pub tx_hash: Hash32,
    pub predicted_state: Hash32,
    pub delta: StateDelta,
    pub merkle_proof: MerkleProof,
    pub model_version: String,
}

/// Compression result for a block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionResult {
    pub block_number: u64,
    pub original_size: usize,
    pub compressed_size: usize,
    pub delta_tree_root: Hash32,
    pub deltas: Vec<StateDelta>,
    pub proofs: Vec<VerificationProof>,
}

impl CompressionResult {
    pub fn compression_ratio(&self) -> f64 {
        self.original_size as f64 / self.compressed_size.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash32_from_slice() {
        let bytes = [1u8; 32];
        let hash = Hash32::from_slice(&bytes).unwrap();
        assert_eq!(hash.0, bytes);
    }

    #[test]
    fn test_state_vector_hash() {
        let sv = StateVector::new(vec![1.0, 2.0, 3.0]);
        let hash = sv.compute_hash();
        assert_ne!(hash, Hash32::ZERO);
    }
}

