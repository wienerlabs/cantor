"""PyTorch dataset for transaction sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from numpy.typing import NDArray

from cantor.core.types import Transaction, BlockData, StateVector
from cantor.core.config import ModelConfig
from cantor.data.features import FeatureEncoder


class TransactionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory dataset of transaction sequences for training."""

    def __init__(
        self,
        sequences: list[NDArray[np.float32]],
        targets: list[NDArray[np.float32]],
        context_length: int = 128,
    ) -> None:
        self.sequences = sequences
        self.targets = targets
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        target = self.targets[idx]

        # Pad or truncate sequence to context_length
        if len(seq) < self.context_length:
            padding = np.zeros(
                (self.context_length - len(seq), seq.shape[1]),
                dtype=np.float32
            )
            seq = np.concatenate([padding, seq], axis=0)
        else:
            seq = seq[-self.context_length:]

        return torch.from_numpy(seq), torch.from_numpy(target)


class StreamingTransactionDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Streaming dataset for large-scale training from disk."""

    def __init__(
        self,
        data_dir: Path,
        encoder: FeatureEncoder,
        context_length: int = 128,
        shuffle_buffer: int = 10000,
    ) -> None:
        self.data_dir = data_dir
        self.encoder = encoder
        self.context_length = context_length
        self.shuffle_buffer = shuffle_buffer
        self._files = sorted(data_dir.glob("*.npz"))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buffer: list[tuple[NDArray, NDArray]] = []

        for file_path in self._files:
            data = np.load(file_path)
            sequences = data["sequences"]
            targets = data["targets"]

            for seq, target in zip(sequences, targets):
                buffer.append((seq, target))

                if len(buffer) >= self.shuffle_buffer:
                    yield from self._yield_shuffled(buffer)
                    buffer = []

        if buffer:
            yield from self._yield_shuffled(buffer)

    def _yield_shuffled(
        self, buffer: list[tuple[NDArray, NDArray]]
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        indices = np.random.permutation(len(buffer))
        for idx in indices:
            seq, target = buffer[idx]
            seq = self._pad_sequence(seq)
            yield torch.from_numpy(seq), torch.from_numpy(target)

    def _pad_sequence(self, seq: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(seq) < self.context_length:
            padding = np.zeros(
                (self.context_length - len(seq), seq.shape[1]),
                dtype=np.float32
            )
            return np.concatenate([padding, seq], axis=0)
        return seq[-self.context_length:]


def create_training_data(
    blocks: list[BlockData],
    encoder: FeatureEncoder,
    context_length: int = 128,
) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    """Create training sequences from a list of blocks."""
    all_transactions: list[Transaction] = []
    all_features: list[NDArray[np.float32]] = []

    for block in blocks:
        for tx in block.transactions:
            all_transactions.append(tx)
            all_features.append(encoder.encode_transaction(tx))

    if not all_features:
        return [], []

    features_array = np.stack(all_features)
    sequences: list[NDArray[np.float32]] = []
    targets: list[NDArray[np.float32]] = []

    for i in range(context_length, len(features_array)):
        seq = features_array[i - context_length:i]
        target = features_array[i]
        sequences.append(seq)
        targets.append(target)

    return sequences, targets

