//! Witness Log Index
//!
//! Audit logging with semantic indexing for postmortem analysis.
//! Every request generates a witness entry that is indexed in Ruvector
//! for semantic search over execution history.
//!
//! ## Use Cases
//!
//! - Debug failed requests by finding similar queries
//! - Analyze routing decision patterns
//! - Track quality metrics over time
//! - Identify latency bottlenecks

use crate::error::{Result, RuvLLMError};
use crate::types::{ErrorInfo, ModelSize, QualityMetrics};
use chrono::{DateTime, Utc};
use ruvector_core::{AgenticDB, SearchQuery, VectorEntry};
use ruvector_core::types::DbOptions;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;
use uuid::Uuid;

/// Latency breakdown for profiling
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    /// Embedding generation time (ms)
    pub embedding_ms: f32,
    /// HNSW retrieval time (ms)
    pub retrieval_ms: f32,
    /// Router decision time (ms)
    pub routing_ms: f32,
    /// Graph attention time (ms)
    pub attention_ms: f32,
    /// LLM generation time (ms)
    pub generation_ms: f32,
    /// Total end-to-end time (ms)
    pub total_ms: f32,
}

impl LatencyBreakdown {
    /// Create a new latency breakdown
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute total from components
    pub fn compute_total(&mut self) {
        self.total_ms = self.embedding_ms + self.retrieval_ms + self.routing_ms
            + self.attention_ms + self.generation_ms;
    }

    /// Check if any component exceeds threshold
    pub fn exceeds_threshold(&self, threshold_ms: f32) -> bool {
        self.total_ms > threshold_ms
    }

    /// Get the slowest component
    pub fn slowest_component(&self) -> (&'static str, f32) {
        let components = [
            ("embedding", self.embedding_ms),
            ("retrieval", self.retrieval_ms),
            ("routing", self.routing_ms),
            ("attention", self.attention_ms),
            ("generation", self.generation_ms),
        ];

        components
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("unknown", 0.0))
    }
}

/// Routing decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Selected model
    pub model: ModelSize,
    /// Context size bucket
    pub context_size: usize,
    /// Temperature used
    pub temperature: f32,
    /// Top-p used
    pub top_p: f32,
    /// Router confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Model probability distribution [tiny, small, medium, large]
    pub model_probs: [f32; 4],
}

impl Default for RoutingDecision {
    fn default() -> Self {
        Self {
            model: ModelSize::Small,
            context_size: 0,
            temperature: 0.7,
            top_p: 0.9,
            confidence: 0.5,
            model_probs: [0.25, 0.25, 0.25, 0.25],
        }
    }
}

/// Execution witness log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessEntry {
    /// Unique request identifier
    pub request_id: Uuid,
    /// Associated session ID
    pub session_id: String,
    /// Query embedding for semantic search (768-D)
    pub query_embedding: Vec<f32>,
    /// Routing decision made
    pub routing_decision: RoutingDecision,
    /// Model used for generation
    pub model_used: ModelSize,
    /// Quality score (0.0 - 1.0) from evaluation
    pub quality_score: f32,
    /// End-to-end latency breakdown
    pub latency: LatencyBreakdown,
    /// Context documents retrieved
    pub context_doc_ids: Vec<Uuid>,
    /// Response embedding for clustering
    pub response_embedding: Vec<f32>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Error details if failed
    pub error: Option<ErrorInfo>,
    /// Quality metrics breakdown
    pub quality_metrics: Option<QualityMetrics>,
    /// Custom tags for filtering
    pub tags: Vec<String>,
}

impl WitnessEntry {
    /// Create a new witness entry
    pub fn new(
        session_id: String,
        query_embedding: Vec<f32>,
        routing_decision: RoutingDecision,
    ) -> Self {
        Self {
            request_id: Uuid::new_v4(),
            session_id,
            query_embedding,
            routing_decision: routing_decision.clone(),
            model_used: routing_decision.model,
            quality_score: 0.0,
            latency: LatencyBreakdown::default(),
            context_doc_ids: Vec::new(),
            response_embedding: Vec::new(),
            timestamp: Utc::now(),
            error: None,
            quality_metrics: None,
            tags: Vec::new(),
        }
    }

    /// Set quality score
    pub fn with_quality(mut self, score: f32) -> Self {
        self.quality_score = score;
        self
    }

    /// Set latency breakdown
    pub fn with_latency(mut self, latency: LatencyBreakdown) -> Self {
        self.latency = latency;
        self
    }

    /// Set error
    pub fn with_error(mut self, error: ErrorInfo) -> Self {
        self.error = Some(error);
        self
    }

    /// Check if this was a successful request
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }

    /// Check if quality score meets threshold
    pub fn meets_quality_threshold(&self, threshold: f32) -> bool {
        self.quality_score >= threshold
    }
}

/// Write-back queue for batching writes
struct WritebackQueue {
    /// Pending entries
    entries: Vec<WitnessEntry>,
    /// Maximum batch size
    max_batch: usize,
    /// Maximum wait time (ms)
    max_wait_ms: u64,
    /// Last flush timestamp
    last_flush: DateTime<Utc>,
}

impl WritebackQueue {
    fn new(max_batch: usize, max_wait_ms: u64) -> Self {
        Self {
            entries: Vec::with_capacity(max_batch),
            max_batch,
            max_wait_ms,
            last_flush: Utc::now(),
        }
    }

    fn should_flush(&self) -> bool {
        if self.entries.len() >= self.max_batch {
            return true;
        }

        let elapsed = (Utc::now() - self.last_flush).num_milliseconds() as u64;
        elapsed >= self.max_wait_ms && !self.entries.is_empty()
    }

    fn push(&mut self, entry: WitnessEntry) {
        self.entries.push(entry);
    }

    fn drain(&mut self) -> Vec<WitnessEntry> {
        self.last_flush = Utc::now();
        std::mem::take(&mut self.entries)
    }
}

/// Witness log backed by Ruvector
pub struct WitnessLog {
    /// Ruvector database
    db: AgenticDB,
    /// Embedding dimension
    embedding_dim: usize,
    /// Write-back queue for batching
    writeback_queue: Arc<Mutex<WritebackQueue>>,
    /// Total entries recorded
    total_entries: AtomicUsize,
    /// Success count
    success_count: AtomicUsize,
    /// Error count
    error_count: AtomicUsize,
}

impl WitnessLog {
    /// Create a new witness log
    pub fn new(storage_path: &str, embedding_dim: usize) -> Result<Self> {
        let mut options = DbOptions::default();
        options.storage_path = storage_path.to_string();
        options.dimensions = embedding_dim;

        let db = AgenticDB::new(options)
            .map_err(|e| RuvLLMError::Storage(e.to_string()))?;

        Ok(Self {
            db,
            embedding_dim,
            writeback_queue: Arc::new(Mutex::new(WritebackQueue::new(100, 1000))),
            total_entries: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            error_count: AtomicUsize::new(0),
        })
    }

    /// Record a witness entry (async, non-blocking)
    pub fn record(&self, entry: WitnessEntry) -> Result<()> {
        // Update counters
        self.total_entries.fetch_add(1, Ordering::SeqCst);
        if entry.is_success() {
            self.success_count.fetch_add(1, Ordering::SeqCst);
        } else {
            self.error_count.fetch_add(1, Ordering::SeqCst);
        }

        // Add to writeback queue
        let mut queue = self.writeback_queue.lock();
        queue.push(entry);

        // Flush if needed
        if queue.should_flush() {
            let entries = queue.drain();
            drop(queue); // Release lock before writing
            self.flush_entries(entries)?;
        }

        Ok(())
    }

    /// Flush pending entries to storage
    fn flush_entries(&self, entries: Vec<WitnessEntry>) -> Result<()> {
        for entry in entries {
            let mut metadata = HashMap::new();
            metadata.insert("request_id".to_string(), serde_json::json!(entry.request_id.to_string()));
            metadata.insert("session_id".to_string(), serde_json::json!(entry.session_id));
            metadata.insert("model_used".to_string(), serde_json::to_value(&entry.model_used).unwrap_or_default());
            metadata.insert("quality_score".to_string(), serde_json::json!(entry.quality_score));
            metadata.insert("routing_decision".to_string(), serde_json::to_value(&entry.routing_decision).unwrap_or_default());
            metadata.insert("latency".to_string(), serde_json::to_value(&entry.latency).unwrap_or_default());
            metadata.insert("timestamp".to_string(), serde_json::json!(entry.timestamp.to_rfc3339()));
            metadata.insert("is_success".to_string(), serde_json::json!(entry.is_success()));
            metadata.insert("tags".to_string(), serde_json::json!(entry.tags));

            if let Some(error) = &entry.error {
                metadata.insert("error".to_string(), serde_json::to_value(error).unwrap_or_default());
            }

            if let Some(qm) = &entry.quality_metrics {
                metadata.insert("quality_metrics".to_string(), serde_json::to_value(qm).unwrap_or_default());
            }

            let vector_entry = VectorEntry {
                id: Some(entry.request_id.to_string()),
                vector: entry.query_embedding,
                metadata: Some(metadata),
            };

            self.db.insert(vector_entry)
                .map_err(|e| RuvLLMError::Storage(e.to_string()))?;
        }

        Ok(())
    }

    /// Force flush all pending entries
    pub fn flush(&self) -> Result<()> {
        let mut queue = self.writeback_queue.lock();
        if !queue.entries.is_empty() {
            let entries = queue.drain();
            drop(queue);
            self.flush_entries(entries)?;
        }
        Ok(())
    }

    /// Search witness logs by semantic similarity
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<WitnessEntry>> {
        let query = SearchQuery {
            vector: query_embedding.to_vec(),
            k: limit,
            filter: None,
            ef_search: None,
        };

        let results = self.db.search(query)
            .map_err(|e| RuvLLMError::Storage(e.to_string()))?;

        let mut entries = Vec::with_capacity(results.len());
        for result in results {
            if let Some(metadata) = &result.metadata {
                if let Some(entry) = self.entry_from_metadata(&result.id, query_embedding, metadata) {
                    entries.push(entry);
                }
            }
        }

        Ok(entries)
    }

    /// Get statistics
    pub fn stats(&self) -> WitnessLogStats {
        let total = self.total_entries.load(Ordering::SeqCst);
        let success = self.success_count.load(Ordering::SeqCst);
        let errors = self.error_count.load(Ordering::SeqCst);

        WitnessLogStats {
            total_entries: total,
            success_count: success,
            error_count: errors,
            success_rate: if total > 0 { success as f32 / total as f32 } else { 0.0 },
            pending_writes: self.writeback_queue.lock().entries.len(),
        }
    }

    /// Reconstruct WitnessEntry from metadata
    fn entry_from_metadata(
        &self,
        _id: &str,
        embedding: &[f32],
        metadata: &HashMap<String, serde_json::Value>,
    ) -> Option<WitnessEntry> {
        let request_id = metadata.get("request_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())?;

        let session_id = metadata.get("session_id")
            .and_then(|v| v.as_str())?
            .to_string();

        let model_used: ModelSize = metadata.get("model_used")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let quality_score = metadata.get("quality_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let routing_decision: RoutingDecision = metadata.get("routing_decision")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let latency: LatencyBreakdown = metadata.get("latency")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let timestamp = metadata.get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let error: Option<ErrorInfo> = metadata.get("error")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let quality_metrics: Option<QualityMetrics> = metadata.get("quality_metrics")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let tags: Vec<String> = metadata.get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        Some(WitnessEntry {
            request_id,
            session_id,
            query_embedding: embedding.to_vec(),
            routing_decision,
            model_used,
            quality_score,
            latency,
            context_doc_ids: Vec::new(),
            response_embedding: Vec::new(),
            timestamp,
            error,
            quality_metrics,
            tags,
        })
    }
}

/// Witness log statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WitnessLogStats {
    /// Total entries recorded
    pub total_entries: usize,
    /// Successful requests
    pub success_count: usize,
    /// Failed requests
    pub error_count: usize,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f32,
    /// Pending writes in queue
    pub pending_writes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_breakdown() {
        let mut latency = LatencyBreakdown {
            embedding_ms: 10.0,
            retrieval_ms: 5.0,
            routing_ms: 2.0,
            attention_ms: 50.0,
            generation_ms: 100.0,
            total_ms: 0.0,
        };

        latency.compute_total();
        assert_eq!(latency.total_ms, 167.0);

        let (name, _) = latency.slowest_component();
        assert_eq!(name, "generation");
    }

    #[test]
    fn test_witness_entry() {
        let entry = WitnessEntry::new(
            "session-1".to_string(),
            vec![0.1; 768],
            RoutingDecision::default(),
        );

        assert!(entry.is_success());
        assert!(!entry.meets_quality_threshold(0.5));

        let entry = entry.with_quality(0.8);
        assert!(entry.meets_quality_threshold(0.5));
    }

    #[test]
    fn test_routing_decision() {
        let decision = RoutingDecision::default();
        assert_eq!(decision.model, ModelSize::Small);
        assert_eq!(decision.temperature, 0.7);
    }
}
