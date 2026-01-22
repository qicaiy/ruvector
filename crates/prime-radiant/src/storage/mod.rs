//! # Storage Layer Module
//!
//! Hybrid storage with PostgreSQL for transactional authority and ruvector for
//! high-performance vector and graph queries.
//!
//! ## Architecture
//!
//! ```text
//! +----------------------------------------------+
//! |                Storage Layer                  |
//! +----------------------------------------------+
//! |                                              |
//! |  +------------------+  +------------------+  |
//! |  |   PostgreSQL     |  |    ruvector      |  |
//! |  |   (Authority)    |  |  (Graph/Vector)  |  |
//! |  |                  |  |                  |  |
//! |  | - Policy bundles |  | - Node states    |  |
//! |  | - Witnesses      |  | - Edge data      |  |
//! |  | - Lineage        |  | - HNSW index     |  |
//! |  | - Event log      |  | - Residual cache |  |
//! |  +------------------+  +------------------+  |
//! |                                              |
//! +----------------------------------------------+
//! ```

// TODO: Implement storage backends
// This is a placeholder for the storage bounded context

use serde::{Deserialize, Serialize};

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// PostgreSQL connection string (optional).
    pub postgres_url: Option<String>,
    /// Path for local graph storage.
    pub graph_path: String,
    /// Path for event log.
    pub event_log_path: String,
    /// Enable write-ahead logging.
    pub enable_wal: bool,
    /// Cache size in MB.
    pub cache_size_mb: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            postgres_url: None,
            graph_path: "./data/graph".to_string(),
            event_log_path: "./data/events".to_string(),
            enable_wal: true,
            cache_size_mb: 256,
        }
    }
}

/// Storage backend trait for graph operations.
pub trait GraphStorage: Send + Sync {
    /// Store a node state.
    fn store_node(&self, node_id: &str, state: &[f32]) -> Result<(), StorageError>;

    /// Retrieve a node state.
    fn get_node(&self, node_id: &str) -> Result<Option<Vec<f32>>, StorageError>;

    /// Store an edge.
    fn store_edge(&self, source: &str, target: &str, weight: f32) -> Result<(), StorageError>;

    /// Delete an edge.
    fn delete_edge(&self, source: &str, target: &str) -> Result<(), StorageError>;

    /// Find nodes similar to a query.
    fn find_similar(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, StorageError>;
}

/// Storage backend trait for governance data.
pub trait GovernanceStorage: Send + Sync {
    /// Store a policy bundle.
    fn store_policy(&self, bundle: &[u8]) -> Result<String, StorageError>;

    /// Retrieve a policy bundle.
    fn get_policy(&self, id: &str) -> Result<Option<Vec<u8>>, StorageError>;

    /// Store a witness record.
    fn store_witness(&self, witness: &[u8]) -> Result<String, StorageError>;

    /// Retrieve witness records for an action.
    fn get_witnesses_for_action(&self, action_id: &str) -> Result<Vec<Vec<u8>>, StorageError>;

    /// Store a lineage record.
    fn store_lineage(&self, lineage: &[u8]) -> Result<String, StorageError>;
}

/// Storage error type.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Transaction failed: {0}")]
    Transaction(String),
}

/// In-memory storage implementation for testing.
#[derive(Debug, Default)]
pub struct InMemoryStorage {
    nodes: parking_lot::RwLock<std::collections::HashMap<String, Vec<f32>>>,
    edges: parking_lot::RwLock<std::collections::HashMap<(String, String), f32>>,
}

impl InMemoryStorage {
    /// Create a new in-memory storage.
    pub fn new() -> Self {
        Self::default()
    }
}

impl GraphStorage for InMemoryStorage {
    fn store_node(&self, node_id: &str, state: &[f32]) -> Result<(), StorageError> {
        self.nodes.write().insert(node_id.to_string(), state.to_vec());
        Ok(())
    }

    fn get_node(&self, node_id: &str) -> Result<Option<Vec<f32>>, StorageError> {
        Ok(self.nodes.read().get(node_id).cloned())
    }

    fn store_edge(&self, source: &str, target: &str, weight: f32) -> Result<(), StorageError> {
        self.edges
            .write()
            .insert((source.to_string(), target.to_string()), weight);
        Ok(())
    }

    fn delete_edge(&self, source: &str, target: &str) -> Result<(), StorageError> {
        self.edges
            .write()
            .remove(&(source.to_string(), target.to_string()));
        Ok(())
    }

    fn find_similar(&self, _query: &[f32], _k: usize) -> Result<Vec<(String, f32)>, StorageError> {
        // Simplified: return empty for in-memory impl
        Ok(Vec::new())
    }
}
