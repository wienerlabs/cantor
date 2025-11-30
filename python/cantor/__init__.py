"""
CANTOR: Predictive State Compression Through Transaction Pattern Recognition

ML-driven blockchain state compression achieving sublinear storage growth through
pattern mining, predictive modeling, and differential encoding with cryptographic
verification.
"""

from cantor.core.types import (
    StateVector,
    Transaction,
    StateDelta,
    CompressionResult,
    VerificationProof,
)

__version__ = "0.1.0"
__all__ = [
    "StateVector",
    "Transaction",
    "StateDelta",
    "CompressionResult",
    "VerificationProof",
]

