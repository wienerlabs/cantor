"""Delta encoding implementations."""

from __future__ import annotations

from typing import Literal
import struct

import numpy as np
from numpy.typing import NDArray
import lz4.frame


class DeltaEncoder:
    """Encodes deltas using various compression schemes."""

    def __init__(self, method: Literal["varint", "huffman", "lz4"] = "lz4") -> None:
        self.method = method

    def encode(self, delta: NDArray[np.float32]) -> bytes:
        """Encode a delta vector."""
        if self.method == "varint":
            return self._encode_varint(delta)
        elif self.method == "lz4":
            return self._encode_lz4(delta)
        else:
            return self._encode_lz4(delta)

    def decode(self, data: bytes) -> NDArray[np.float32]:
        """Decode a delta vector."""
        if self.method == "varint":
            return self._decode_varint(data)
        elif self.method == "lz4":
            return self._decode_lz4(data)
        else:
            return self._decode_lz4(data)

    def encode_full(self, state: NDArray[np.float32]) -> bytes:
        """Encode full state (fallback for large deltas)."""
        header = b"\x01"  # Flag indicating full state
        return header + self._encode_lz4(state)

    def decode_full(self, data: bytes) -> NDArray[np.float32]:
        """Decode full state."""
        if data[0] == 1:
            return self._decode_lz4(data[1:])
        return self.decode(data)

    def _encode_varint(self, delta: NDArray[np.float32]) -> bytes:
        """Variable-length integer encoding for sparse deltas."""
        quantized = np.round(delta * 1000).astype(np.int32)
        
        # Run-length encoding for zeros
        result = bytearray()
        i = 0
        while i < len(quantized):
            if quantized[i] == 0:
                # Count consecutive zeros
                zero_count = 0
                while i < len(quantized) and quantized[i] == 0 and zero_count < 255:
                    zero_count += 1
                    i += 1
                result.append(0)  # Zero marker
                result.append(zero_count)
            else:
                # Encode non-zero value
                val = quantized[i]
                encoded = self._encode_zigzag(val)
                result.extend(self._encode_varint_single(encoded))
                i += 1
        
        return bytes(result)

    def _decode_varint(self, data: bytes) -> NDArray[np.float32]:
        """Decode variable-length integers."""
        values: list[int] = []
        i = 0
        
        while i < len(data):
            if data[i] == 0:
                # Zero run
                i += 1
                if i < len(data):
                    zero_count = data[i]
                    values.extend([0] * zero_count)
                    i += 1
            else:
                # Decode varint
                val, consumed = self._decode_varint_single(data[i:])
                values.append(self._decode_zigzag(val))
                i += consumed
        
        return np.array(values, dtype=np.float32) / 1000.0

    def _encode_lz4(self, data: NDArray[np.float32]) -> bytes:
        """LZ4 compression for general case."""
        raw = data.astype(np.float16).tobytes()
        return lz4.frame.compress(raw, compression_level=9)

    def _decode_lz4(self, data: bytes) -> NDArray[np.float32]:
        """LZ4 decompression."""
        raw = lz4.frame.decompress(data)
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)

    def _encode_zigzag(self, n: int) -> int:
        """Zigzag encoding for signed integers."""
        return (n << 1) ^ (n >> 31)

    def _decode_zigzag(self, n: int) -> int:
        """Zigzag decoding."""
        return (n >> 1) ^ -(n & 1)

    def _encode_varint_single(self, n: int) -> bytes:
        """Encode single unsigned integer as varint."""
        result = bytearray()
        while n >= 0x80:
            result.append((n & 0x7F) | 0x80)
            n >>= 7
        result.append(n)
        return bytes(result)

    def _decode_varint_single(self, data: bytes) -> tuple[int, int]:
        """Decode single varint, return (value, bytes_consumed)."""
        result = 0
        shift = 0
        consumed = 0
        
        for byte in data:
            consumed += 1
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        
        return result, consumed


def estimate_compressed_size(delta: NDArray[np.float32]) -> int:
    """Estimate compressed size without full compression."""
    non_zero = np.count_nonzero(delta)
    sparsity = 1.0 - (non_zero / len(delta))
    
    # Rough estimate based on sparsity
    if sparsity > 0.9:
        return int(non_zero * 4 + len(delta) // 100)
    elif sparsity > 0.5:
        return int(len(delta) * 2 * (1 - sparsity * 0.5))
    else:
        return int(len(delta) * 2)

