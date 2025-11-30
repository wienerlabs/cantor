//! Error types for CANTOR.

use thiserror::Error;

/// Core error type for CANTOR operations.
#[derive(Error, Debug)]
pub enum CantorError {
    #[error("Invalid hash length: expected 32, got {0}")]
    InvalidHashLength(usize),

    #[error("Merkle proof verification failed")]
    MerkleVerificationFailed,

    #[error("State reconstruction failed: {0}")]
    StateReconstructionFailed(String),

    #[error("Model version mismatch: expected {expected}, got {actual}")]
    ModelVersionMismatch { expected: String, actual: String },

    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),

    #[error("Invalid delta encoding")]
    InvalidDeltaEncoding,

    #[error("Block not found: {0}")]
    BlockNotFound(u64),

    #[error("Transaction not found: {0}")]
    TransactionNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type alias for CANTOR operations.
pub type Result<T> = std::result::Result<T, CantorError>;

