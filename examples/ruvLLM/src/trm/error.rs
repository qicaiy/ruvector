//! TRM Error Types
//!
//! Defines error types for TRM operations.

use std::fmt;

/// TRM-specific errors
#[derive(Debug, Clone)]
pub enum TrmError {
    /// Dimension mismatch between tensors
    DimensionMismatch {
        expected: usize,
        got: usize,
        context: String,
    },

    /// Invalid K value
    InvalidK {
        value: usize,
        max: usize,
    },

    /// Invalid configuration
    InvalidConfig(String),

    /// Latent update operation failed
    LatentUpdateFailed(String),

    /// Answer refinement failed
    RefinementFailed(String),

    /// SIMD operation failed
    SimdError(String),

    /// Memory allocation failed
    AllocationFailed(String),

    /// Confidence scoring failed
    ConfidenceFailed(String),

    /// Initialization error
    InitializationError(String),
}

impl fmt::Display for TrmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrmError::DimensionMismatch { expected, got, context } => {
                write!(f, "Dimension mismatch in {}: expected {}, got {}", context, expected, got)
            }
            TrmError::InvalidK { value, max } => {
                write!(f, "Invalid K value: {} (max: {})", value, max)
            }
            TrmError::InvalidConfig(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            TrmError::LatentUpdateFailed(msg) => {
                write!(f, "Latent update failed: {}", msg)
            }
            TrmError::RefinementFailed(msg) => {
                write!(f, "Answer refinement failed: {}", msg)
            }
            TrmError::SimdError(msg) => {
                write!(f, "SIMD operation failed: {}", msg)
            }
            TrmError::AllocationFailed(msg) => {
                write!(f, "Memory allocation failed: {}", msg)
            }
            TrmError::ConfidenceFailed(msg) => {
                write!(f, "Confidence scoring failed: {}", msg)
            }
            TrmError::InitializationError(msg) => {
                write!(f, "Initialization error: {}", msg)
            }
        }
    }
}

impl std::error::Error for TrmError {}

/// Result type for TRM operations
pub type TrmResult<T> = Result<T, TrmError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TrmError::DimensionMismatch {
            expected: 256,
            got: 128,
            context: "latent update".to_string(),
        };
        assert!(err.to_string().contains("256"));
        assert!(err.to_string().contains("128"));

        let err = TrmError::InvalidK { value: 50, max: 20 };
        assert!(err.to_string().contains("50"));
        assert!(err.to_string().contains("20"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TrmError>();
    }
}
