"""Storage backend for compressed state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import structlog

from cantor.core.types import CompressionResult, StateDelta, VerificationProof

logger = structlog.get_logger()


class CompressedStateStore:
    """Persistent storage for compressed blockchain state."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.blocks_path = base_path / "blocks"
        self.blocks_path.mkdir(exist_ok=True)
        
        self.index_path = base_path / "index.json"
        self._index: dict[int, str] = {}
        self._load_index()

    def store(self, result: CompressionResult) -> None:
        """Store a compression result."""
        block_file = self.blocks_path / f"{result.block_number}.bin"
        
        with open(block_file, "wb") as f:
            f.write(self._serialize_result(result))
        
        self._index[result.block_number] = str(block_file)
        self._save_index()
        
        logger.debug("block_stored", block=result.block_number)

    def load(self, block_number: int) -> CompressionResult | None:
        """Load a compression result by block number."""
        if block_number not in self._index:
            return None
        
        block_file = Path(self._index[block_number])
        if not block_file.exists():
            return None
        
        with open(block_file, "rb") as f:
            return self._deserialize_result(f.read())

    def has_block(self, block_number: int) -> bool:
        return block_number in self._index

    def list_blocks(self) -> list[int]:
        return sorted(self._index.keys())

    def iter_blocks(
        self,
        start: int | None = None,
        end: int | None = None,
    ) -> Iterator[CompressionResult]:
        """Iterate over stored blocks in range."""
        blocks = self.list_blocks()
        
        for block_num in blocks:
            if start is not None and block_num < start:
                continue
            if end is not None and block_num > end:
                break
            
            result = self.load(block_num)
            if result:
                yield result

    def get_stats(self) -> dict[str, int | float]:
        """Get storage statistics."""
        total_original = 0
        total_compressed = 0
        
        for result in self.iter_blocks():
            total_original += result.original_size
            total_compressed += result.compressed_size
        
        return {
            "block_count": len(self._index),
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "compression_ratio": total_original / max(total_compressed, 1),
        }

    def _serialize_result(self, result: CompressionResult) -> bytes:
        import struct
        
        parts = [
            struct.pack("<Q", result.block_number),
            struct.pack("<Q", result.original_size),
            struct.pack("<Q", result.compressed_size),
            result.delta_tree_root,
            struct.pack("<I", len(result.deltas)),
        ]
        
        for delta in result.deltas:
            parts.append(delta.tx_hash)
            parts.append(delta.predicted_root)
            parts.append(delta.actual_root)
            parts.append(struct.pack("<I", len(delta.delta_bytes)))
            parts.append(delta.delta_bytes)
            parts.append(struct.pack("<f", delta.confidence))
        
        return b"".join(parts)

    def _deserialize_result(self, data: bytes) -> CompressionResult:
        import struct
        
        pos = 0
        block_number = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        original_size = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        compressed_size = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        delta_tree_root = data[pos:pos+32]
        pos += 32
        num_deltas = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        
        deltas: list[StateDelta] = []
        for _ in range(num_deltas):
            tx_hash = data[pos:pos+32]
            pos += 32
            predicted_root = data[pos:pos+32]
            pos += 32
            actual_root = data[pos:pos+32]
            pos += 32
            delta_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            delta_bytes = data[pos:pos+delta_len]
            pos += delta_len
            confidence = struct.unpack_from("<f", data, pos)[0]
            pos += 4
            
            deltas.append(StateDelta(
                tx_hash=tx_hash,
                predicted_root=predicted_root,
                actual_root=actual_root,
                delta_bytes=delta_bytes,
                confidence=confidence,
            ))
        
        return CompressionResult(
            block_number=block_number,
            original_size=original_size,
            compressed_size=compressed_size,
            delta_tree_root=delta_tree_root,
            deltas=tuple(deltas),
            proofs=(),
        )

    def _load_index(self) -> None:
        if self.index_path.exists():
            with open(self.index_path) as f:
                raw = json.load(f)
                self._index = {int(k): v for k, v in raw.items()}

    def _save_index(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump({str(k): v for k, v in self._index.items()}, f)

