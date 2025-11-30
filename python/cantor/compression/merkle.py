"""Merkle delta tree for cryptographic verification."""

from __future__ import annotations

import hashlib
from typing import Sequence

from cantor.core.types import StateDelta, MerkleProof, VerificationProof, Bytes32


class MerkleDeltaTree:
    """Merkle tree over prediction deltas for cryptographic commitment."""

    def __init__(self, hash_function: str = "sha256") -> None:
        self.hash_function = hash_function
        self._leaves: list[Bytes32] = []
        self._tree: list[list[Bytes32]] = []
        self._root: Bytes32 = b""

    def build(self, deltas: Sequence[bytes]) -> Bytes32:
        """Build merkle tree from delta data."""
        if not deltas:
            self._root = self._hash(b"empty")
            return self._root
        
        # Hash leaves
        self._leaves = [self._hash(d) for d in deltas]
        
        # Pad to power of 2
        target_size = 1
        while target_size < len(self._leaves):
            target_size *= 2
        
        while len(self._leaves) < target_size:
            self._leaves.append(self._hash(b"padding"))
        
        # Build tree bottom-up
        self._tree = [self._leaves]
        current_level = self._leaves
        
        while len(current_level) > 1:
            next_level: list[Bytes32] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent = self._hash(left + right)
                next_level.append(parent)
            self._tree.append(next_level)
            current_level = next_level
        
        self._root = self._tree[-1][0] if self._tree else self._hash(b"empty")
        return self._root

    def get_root(self) -> Bytes32:
        return self._root

    def generate_proof(
        self,
        index: int,
        delta: StateDelta,
        model_version: str,
    ) -> VerificationProof:
        """Generate merkle proof for a specific delta."""
        if index >= len(self._leaves):
            raise IndexError(f"Index {index} out of range")
        
        path: list[Bytes32] = []
        indices: list[int] = []
        
        current_index = index
        for level in self._tree[:-1]:
            sibling_index = current_index ^ 1
            if sibling_index < len(level):
                path.append(level[sibling_index])
                indices.append(sibling_index % 2)
            current_index //= 2
        
        merkle_proof = MerkleProof(
            leaf_hash=self._leaves[index],
            path=tuple(path),
            indices=tuple(indices),
        )
        
        return VerificationProof(
            tx_hash=delta.tx_hash,
            predicted_state=delta.predicted_root,
            delta=delta,
            merkle_proof=merkle_proof,
            model_version=model_version,
        )

    def verify_proof(self, proof: MerkleProof, expected_root: Bytes32) -> bool:
        """Verify a merkle proof against expected root."""
        current = proof.leaf_hash
        
        for sibling, index in zip(proof.path, proof.indices):
            if index == 0:
                current = self._hash(current + sibling)
            else:
                current = self._hash(sibling + current)
        
        return current == expected_root

    def _hash(self, data: bytes) -> Bytes32:
        """Hash data using configured hash function."""
        if self.hash_function == "sha256":
            return hashlib.sha256(data).digest()
        elif self.hash_function == "blake2b":
            return hashlib.blake2b(data, digest_size=32).digest()
        else:
            return hashlib.sha256(data).digest()


class IncrementalMerkleTree:
    """Merkle tree supporting incremental updates."""

    def __init__(self, depth: int = 20) -> None:
        self.depth = depth
        self._zeros = self._compute_zeros(depth)
        self._leaves: list[Bytes32] = []
        self._filled: list[list[Bytes32]] = [[] for _ in range(depth)]

    def _compute_zeros(self, depth: int) -> list[Bytes32]:
        """Compute zero hashes for each level."""
        zeros = [hashlib.sha256(b"zero").digest()]
        for _ in range(depth - 1):
            zeros.append(hashlib.sha256(zeros[-1] + zeros[-1]).digest())
        return zeros

    def insert(self, leaf: Bytes32) -> Bytes32:
        """Insert a leaf and return new root."""
        self._leaves.append(leaf)
        current = leaf
        
        for i in range(self.depth):
            if len(self._filled[i]) % 2 == 0:
                self._filled[i].append(current)
                break
            else:
                sibling = self._filled[i][-1]
                current = hashlib.sha256(sibling + current).digest()
                if i == self.depth - 1:
                    return current
        
        return self.get_root()

    def get_root(self) -> Bytes32:
        """Compute current root."""
        if not self._leaves:
            return self._zeros[-1]
        
        current = self._leaves[-1]
        for i in range(self.depth):
            if i < len(self._filled) and self._filled[i]:
                current = hashlib.sha256(self._filled[i][-1] + current).digest()
            else:
                current = hashlib.sha256(current + self._zeros[i]).digest()
        
        return current

