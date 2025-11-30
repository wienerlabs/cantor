"""Feature engineering for transaction data."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from cantor.core.types import Transaction, BlockData, StateVector, Bytes32
from cantor.core.config import ModelConfig


class FeatureEncoder:
    """Encodes blockchain transactions into fixed-dimension feature vectors."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.state_dim = config.state_dim

    def encode_transaction(self, tx: Transaction) -> NDArray[np.float32]:
        """Encode a single transaction into a feature vector."""
        features = []

        # Transaction type (one-hot, 4 dims)
        tx_type_onehot = np.zeros(4, dtype=np.float32)
        tx_type_onehot[tx.tx_type] = 1.0
        features.append(tx_type_onehot)

        # Value (log-normalized, 1 dim)
        value_log = np.log1p(tx.value / 1e18).astype(np.float32)
        features.append(np.array([value_log], dtype=np.float32))

        # Gas metrics (3 dims)
        gas_features = np.array([
            np.log1p(tx.gas_limit),
            np.log1p(tx.gas_used),
            tx.gas_used / max(tx.gas_limit, 1),
        ], dtype=np.float32)
        features.append(gas_features)

        # Nonce (log-normalized, 1 dim)
        features.append(np.array([np.log1p(tx.nonce)], dtype=np.float32))

        # Success flag (1 dim)
        features.append(np.array([float(tx.success)], dtype=np.float32))

        # Address hashes (64 dims each for from/to)
        from_hash = self._hash_to_vector(tx.from_addr, 64)
        to_hash = self._hash_to_vector(tx.to_addr, 64) if tx.to_addr else np.zeros(64, dtype=np.float32)
        features.extend([from_hash, to_hash])

        # Input data signature (32 dims)
        input_sig = self._encode_input_signature(tx.input_data)
        features.append(input_sig)

        base_features = np.concatenate(features)

        # Pad or truncate to target dimension
        if len(base_features) < self.state_dim:
            padding = np.zeros(self.state_dim - len(base_features), dtype=np.float32)
            return np.concatenate([base_features, padding])
        return base_features[:self.state_dim]

    def encode_block(self, block: BlockData) -> NDArray[np.float32]:
        """Encode a block's aggregate features."""
        features = []

        # Block metadata (4 dims)
        features.append(np.array([
            np.log1p(block.number),
            np.log1p(block.timestamp),
            np.log1p(block.gas_used),
            block.gas_used / max(block.gas_limit, 1),
        ], dtype=np.float32))

        # Transaction count (1 dim)
        features.append(np.array([np.log1p(len(block.transactions))], dtype=np.float32))

        # Base fee (1 dim)
        base_fee = np.log1p(block.base_fee / 1e9) if block.base_fee else 0.0
        features.append(np.array([base_fee], dtype=np.float32))

        # State root hash (64 dims)
        features.append(self._hash_to_vector(block.state_root, 64))

        return np.concatenate(features)

    def encode_transaction_sequence(
        self, transactions: Sequence[Transaction]
    ) -> NDArray[np.float32]:
        """Encode a sequence of transactions into a 2D array."""
        if not transactions:
            return np.zeros((0, self.state_dim), dtype=np.float32)
        return np.stack([self.encode_transaction(tx) for tx in transactions])

    def create_state_vector(
        self, transactions: Sequence[Transaction], block: BlockData
    ) -> StateVector:
        """Create a complete state vector from transactions and block data."""
        tx_features = self.encode_transaction_sequence(transactions)
        block_features = self.encode_block(block)

        if len(tx_features) > 0:
            tx_aggregate = tx_features.mean(axis=0)
        else:
            tx_aggregate = np.zeros(self.state_dim, dtype=np.float32)

        # Combine block and transaction features
        combined = tx_aggregate.copy()
        combined[:len(block_features)] += block_features * 0.1

        return StateVector(
            vector=combined,
            block_number=block.number,
            state_root=block.state_root,
        )

    def _hash_to_vector(self, data: bytes | None, dim: int) -> NDArray[np.float32]:
        """Convert bytes to a fixed-dimension vector via hashing."""
        if data is None:
            return np.zeros(dim, dtype=np.float32)
        h = hashlib.sha256(data).digest()
        # Expand hash to target dimension
        expanded = (h * ((dim // 32) + 1))[:dim]
        return np.frombuffer(expanded, dtype=np.uint8).astype(np.float32) / 255.0

    def _encode_input_signature(self, input_data: bytes) -> NDArray[np.float32]:
        """Encode the function signature from input data."""
        if len(input_data) < 4:
            return np.zeros(32, dtype=np.float32)
        signature = input_data[:4]
        return self._hash_to_vector(signature, 32)

