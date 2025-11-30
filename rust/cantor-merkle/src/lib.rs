//! High-performance Merkle tree for CANTOR delta commitments.

use cantor_core::{Hash32, MerkleProof, CantorError, Result};
use sha2::{Sha256, Digest};

/// Merkle tree for delta commitments.
pub struct MerkleDeltaTree {
    leaves: Vec<Hash32>,
    tree: Vec<Vec<Hash32>>,
    root: Hash32,
}

impl MerkleDeltaTree {
    /// Build a new Merkle tree from delta data.
    pub fn build(deltas: &[&[u8]]) -> Self {
        if deltas.is_empty() {
            return Self {
                leaves: vec![],
                tree: vec![],
                root: Self::hash(b"empty"),
            };
        }

        let leaves: Vec<Hash32> = deltas.iter().map(|d| Self::hash(d)).collect();
        
        // Pad to power of 2
        let mut padded = leaves.clone();
        let target_size = padded.len().next_power_of_two();
        while padded.len() < target_size {
            padded.push(Self::hash(b"padding"));
        }

        let mut tree = vec![padded.clone()];
        let mut current = padded;

        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len() / 2);
            for chunk in current.chunks(2) {
                let left = &chunk[0];
                let right = chunk.get(1).unwrap_or(left);
                let combined = [left.as_ref(), right.as_ref()].concat();
                next.push(Self::hash(&combined));
            }
            tree.push(next.clone());
            current = next;
        }

        let root = tree.last().map(|l| l[0]).unwrap_or(Self::hash(b"empty"));

        Self { leaves, tree, root }
    }

    /// Get the root hash.
    pub fn root(&self) -> Hash32 {
        self.root
    }

    /// Generate a proof for a specific leaf index.
    pub fn generate_proof(&self, index: usize) -> Result<MerkleProof> {
        if index >= self.leaves.len() {
            return Err(CantorError::TransactionNotFound(index.to_string()));
        }

        let mut path = Vec::new();
        let mut indices = Vec::new();
        let mut current_index = index;

        for level in &self.tree[..self.tree.len().saturating_sub(1)] {
            let sibling_index = current_index ^ 1;
            if sibling_index < level.len() {
                path.push(level[sibling_index]);
                indices.push((sibling_index % 2) as u8);
            }
            current_index /= 2;
        }

        Ok(MerkleProof {
            leaf_hash: self.tree[0][index],
            path,
            indices,
        })
    }

    /// Verify a proof against the root.
    pub fn verify_proof(proof: &MerkleProof, root: &Hash32) -> bool {
        proof.verify(root)
    }

    fn hash(data: &[u8]) -> Hash32 {
        let result = Sha256::digest(data);
        Hash32::from_slice(&result).unwrap()
    }
}

/// Incremental Merkle tree for streaming updates.
pub struct IncrementalMerkleTree {
    depth: usize,
    zeros: Vec<Hash32>,
    filled: Vec<Vec<Hash32>>,
    next_index: usize,
}

impl IncrementalMerkleTree {
    pub fn new(depth: usize) -> Self {
        let zeros = Self::compute_zeros(depth);
        Self {
            depth,
            zeros,
            filled: vec![vec![]; depth],
            next_index: 0,
        }
    }

    pub fn insert(&mut self, leaf: Hash32) -> Hash32 {
        let mut current = leaf;
        let mut index = self.next_index;

        for i in 0..self.depth {
            if index % 2 == 0 {
                self.filled[i].push(current);
                current = Self::hash_pair(&current, &self.zeros[i]);
            } else {
                let sibling = self.filled[i].last().copied().unwrap_or(self.zeros[i]);
                current = Self::hash_pair(&sibling, &current);
            }
            index /= 2;
        }

        self.next_index += 1;
        current
    }

    pub fn root(&self) -> Hash32 {
        if self.next_index == 0 {
            return self.zeros[self.depth - 1];
        }

        let mut current = self.filled[0].last().copied().unwrap_or(self.zeros[0]);
        for i in 1..self.depth {
            let sibling = self.filled[i].last().copied().unwrap_or(self.zeros[i]);
            current = Self::hash_pair(&sibling, &current);
        }
        current
    }

    fn compute_zeros(depth: usize) -> Vec<Hash32> {
        let mut zeros = vec![Self::hash_single(b"zero")];
        for _ in 1..depth {
            let last = zeros.last().unwrap();
            zeros.push(Self::hash_pair(last, last));
        }
        zeros
    }

    fn hash_single(data: &[u8]) -> Hash32 {
        let result = Sha256::digest(data);
        Hash32::from_slice(&result).unwrap()
    }

    fn hash_pair(left: &Hash32, right: &Hash32) -> Hash32 {
        let combined = [left.as_ref(), right.as_ref()].concat();
        Self::hash_single(&combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_build() {
        let deltas: Vec<&[u8]> = vec![b"delta1", b"delta2", b"delta3"];
        let tree = MerkleDeltaTree::build(&deltas);
        assert_ne!(tree.root(), Hash32::ZERO);
    }

    #[test]
    fn test_merkle_proof_verification() {
        let deltas: Vec<&[u8]> = vec![b"delta1", b"delta2", b"delta3", b"delta4"];
        let tree = MerkleDeltaTree::build(&deltas);
        
        for i in 0..deltas.len() {
            let proof = tree.generate_proof(i).unwrap();
            assert!(MerkleDeltaTree::verify_proof(&proof, &tree.root()));
        }
    }

    #[test]
    fn test_incremental_tree() {
        let mut tree = IncrementalMerkleTree::new(10);
        let leaf = Hash32::from_slice(&[1u8; 32]).unwrap();
        let root = tree.insert(leaf);
        assert_ne!(root, Hash32::ZERO);
    }
}

