"""Core type definitions for CANTOR state compression system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Bytes32: TypeAlias = bytes
Address: TypeAlias = bytes
Wei: TypeAlias = int


class TransactionType(IntEnum):
    LEGACY = 0
    ACCESS_LIST = 1
    EIP1559 = 2
    BLOB = 3


@dataclass(frozen=True, slots=True)
class StorageSlot:
    key: Bytes32
    value: Bytes32


@dataclass(frozen=True, slots=True)
class AccountState:
    address: Address
    nonce: int
    balance: Wei
    code_hash: Bytes32
    storage_root: Bytes32
    storage_slots: tuple[StorageSlot, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Transaction:
    hash: Bytes32
    block_number: int
    tx_index: int
    tx_type: TransactionType
    from_addr: Address
    to_addr: Address | None
    value: Wei
    gas_limit: int
    gas_used: int
    gas_price: Wei
    input_data: bytes
    nonce: int
    success: bool


@dataclass(frozen=True, slots=True)
class BlockData:
    number: int
    hash: Bytes32
    parent_hash: Bytes32
    timestamp: int
    gas_used: int
    gas_limit: int
    base_fee: Wei | None
    transactions: tuple[Transaction, ...]
    state_root: Bytes32


@dataclass(slots=True)
class StateVector:
    """Fixed-dimension vector representation of blockchain state (4096-dim)."""
    
    vector: NDArray[np.float32]
    block_number: int
    state_root: Bytes32
    
    @property
    def dim(self) -> int:
        return self.vector.shape[0]
    
    def distance(self, other: StateVector) -> float:
        return float(np.linalg.norm(self.vector - other.vector))


@dataclass(frozen=True, slots=True)
class StateDelta:
    """Difference between predicted and actual state."""
    
    tx_hash: Bytes32
    predicted_root: Bytes32
    actual_root: Bytes32
    delta_bytes: bytes
    confidence: float
    
    @property
    def size(self) -> int:
        return len(self.delta_bytes)


@dataclass(frozen=True, slots=True)
class MerkleProof:
    leaf_hash: Bytes32
    path: tuple[Bytes32, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VerificationProof:
    """Cryptographic proof of correct compression."""
    
    tx_hash: Bytes32
    predicted_state: Bytes32
    delta: StateDelta
    merkle_proof: MerkleProof
    model_version: str


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """Result of compressing a block's state transitions."""
    
    block_number: int
    original_size: int
    compressed_size: int
    delta_tree_root: Bytes32
    deltas: tuple[StateDelta, ...]
    proofs: tuple[VerificationProof, ...]
    
    @property
    def compression_ratio(self) -> float:
        if self.compressed_size == 0:
            return 0.0
        return self.original_size / self.compressed_size
    
    @property
    def savings_percent(self) -> float:
        if self.original_size == 0:
            return 0.0
        return (1 - self.compressed_size / self.original_size) * 100

