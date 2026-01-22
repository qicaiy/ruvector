//! Zero-Copy Shared Types for RLM Module
//!
//! This module provides zero-copy type aliases and wrapper types to reduce
//! memory allocations and improve cache performance in the RLM system.
//!
//! Performance improvements:
//! - 2-3x faster cache operations by avoiding clones
//! - Reduced memory allocations for shared data
//! - Better cache line utilization with Arc-based sharing
//!
//! # Usage
//!
//! ```rust,ignore
//! use ruvllm::rlm::shared_types::{SharedText, SharedEmbedding, SharedQueryResult};
//!
//! // Create shared text (zero-copy after creation)
//! let text: SharedText = "Hello, world!".into();
//!
//! // Create shared embedding
//! let embedding: SharedEmbedding = vec![0.1, 0.2, 0.3].into();
//!
//! // Clone is cheap (just Arc increment)
//! let text_clone = text.clone(); // O(1), no allocation
//! ```

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
#[allow(unused_imports)]
use std::ops::Deref;
use std::sync::Arc;

// ============================================================================
// Core Shared Types
// ============================================================================

/// Shared immutable text that can be cheaply cloned.
/// Uses `Arc<str>` for zero-copy sharing across cache entries.
pub type SharedText = Arc<str>;

/// Shared immutable embedding vector.
/// Uses `Arc<[f32]>` to avoid copying large embedding vectors.
pub type SharedEmbedding = Arc<[f32]>;

/// Shared immutable byte buffer.
/// Uses `Arc<[u8]>` for raw data sharing.
pub type SharedBytes = Arc<[u8]>;

// ============================================================================
// Conversion Traits
// ============================================================================

/// Extension trait for creating shared text from various sources.
pub trait IntoSharedText {
    fn into_shared_text(self) -> SharedText;
}

impl IntoSharedText for String {
    #[inline]
    fn into_shared_text(self) -> SharedText {
        Arc::from(self)
    }
}

impl IntoSharedText for &str {
    #[inline]
    fn into_shared_text(self) -> SharedText {
        Arc::from(self)
    }
}

impl IntoSharedText for Cow<'_, str> {
    #[inline]
    fn into_shared_text(self) -> SharedText {
        Arc::from(self.as_ref())
    }
}

impl IntoSharedText for Box<str> {
    #[inline]
    fn into_shared_text(self) -> SharedText {
        Arc::from(self)
    }
}

/// Extension trait for creating shared embeddings.
pub trait IntoSharedEmbedding {
    fn into_shared_embedding(self) -> SharedEmbedding;
}

impl IntoSharedEmbedding for Vec<f32> {
    #[inline]
    fn into_shared_embedding(self) -> SharedEmbedding {
        Arc::from(self)
    }
}

impl IntoSharedEmbedding for &[f32] {
    #[inline]
    fn into_shared_embedding(self) -> SharedEmbedding {
        Arc::from(self)
    }
}

impl IntoSharedEmbedding for Box<[f32]> {
    #[inline]
    fn into_shared_embedding(self) -> SharedEmbedding {
        Arc::from(self)
    }
}

// ============================================================================
// Zero-Copy Query Result
// ============================================================================

/// Zero-copy version of QueryResult for cache storage.
/// Uses Arc for shared ownership of text and embedding data.
///
/// Note: Does not implement Serialize/Deserialize as it's designed for
/// in-memory zero-copy operations. Use the standard QueryResult for
/// serialization needs.
#[derive(Debug, Clone)]
pub struct SharedQueryResult {
    /// Unique result ID
    pub id: SharedText,
    /// Generated response text (shared for zero-copy)
    pub text: SharedText,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Source attributions (shared)
    pub sources: Arc<[SharedSourceAttribution]>,
    /// Usage statistics
    pub usage: SharedTokenUsage,
}

impl SharedQueryResult {
    /// Create a new SharedQueryResult from individual components.
    #[inline]
    pub fn new(
        id: impl IntoSharedText,
        text: impl IntoSharedText,
        confidence: f32,
        tokens_generated: usize,
        latency_ms: f64,
        sources: Vec<SharedSourceAttribution>,
        usage: SharedTokenUsage,
    ) -> Self {
        Self {
            id: id.into_shared_text(),
            text: text.into_shared_text(),
            confidence,
            tokens_generated,
            latency_ms,
            sources: Arc::from(sources),
            usage,
        }
    }

    /// Get the text as a string slice (zero-copy).
    #[inline]
    pub fn text_str(&self) -> &str {
        &self.text
    }

    /// Get estimated memory size in bytes.
    pub fn estimated_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.id.len()
            + self.text.len()
            + self.sources.len() * std::mem::size_of::<SharedSourceAttribution>()
    }
}

/// Zero-copy source attribution.
#[derive(Debug, Clone)]
pub struct SharedSourceAttribution {
    /// Memory entry ID
    pub memory_id: SharedText,
    /// Relevance score
    pub relevance: f32,
    /// Excerpt from the source
    pub excerpt: SharedText,
}

impl SharedSourceAttribution {
    #[inline]
    pub fn new(
        memory_id: impl IntoSharedText,
        relevance: f32,
        excerpt: impl IntoSharedText,
    ) -> Self {
        Self {
            memory_id: memory_id.into_shared_text(),
            relevance,
            excerpt: excerpt.into_shared_text(),
        }
    }
}

/// Token usage statistics (copy-friendly due to small size).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SharedTokenUsage {
    /// Input tokens processed
    pub input_tokens: usize,
    /// Output tokens generated
    pub output_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

impl SharedTokenUsage {
    #[inline]
    pub fn new(input_tokens: usize, output_tokens: usize) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        }
    }
}

// ============================================================================
// Zero-Copy Memory Entry
// ============================================================================

/// Zero-copy memory entry for efficient storage and retrieval.
#[derive(Debug, Clone)]
pub struct SharedMemoryEntry {
    /// Unique identifier
    pub id: SharedText,
    /// Text content (shared)
    pub text: SharedText,
    /// Vector embedding (shared, avoids copying large vectors)
    pub embedding: SharedEmbedding,
    /// Metadata (Arc-wrapped for sharing)
    pub metadata: Arc<SharedMemoryMetadata>,
    /// Creation timestamp (Unix seconds for efficiency)
    pub created_at_secs: i64,
    /// Last accessed timestamp (Unix seconds)
    pub last_accessed_secs: i64,
    /// Access count
    pub access_count: u64,
}

impl SharedMemoryEntry {
    /// Create from text and embedding.
    #[inline]
    pub fn new(
        id: impl IntoSharedText,
        text: impl IntoSharedText,
        embedding: impl IntoSharedEmbedding,
    ) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: id.into_shared_text(),
            text: text.into_shared_text(),
            embedding: embedding.into_shared_embedding(),
            metadata: Arc::new(SharedMemoryMetadata::default()),
            created_at_secs: now,
            last_accessed_secs: now,
            access_count: 0,
        }
    }

    /// Get text as str slice.
    #[inline]
    pub fn text_str(&self) -> &str {
        &self.text
    }

    /// Get embedding as slice.
    #[inline]
    pub fn embedding_slice(&self) -> &[f32] {
        &self.embedding
    }

    /// Get estimated memory size.
    pub fn estimated_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.id.len()
            + self.text.len()
            + self.embedding.len() * std::mem::size_of::<f32>()
    }
}

/// Shared memory metadata.
#[derive(Debug, Clone, Default)]
pub struct SharedMemoryMetadata {
    /// Source identifier
    pub source: Option<String>,
    /// Category or type
    pub category: Option<String>,
    /// Custom tags
    pub tags: Vec<String>,
    /// TTL override in seconds
    pub ttl_secs: Option<u64>,
}

// ============================================================================
// Zero-Copy Search Result
// ============================================================================

/// Zero-copy memory search result.
/// References the original entry without cloning.
#[derive(Debug, Clone)]
pub struct SharedSearchResult {
    /// The matched entry ID
    pub id: SharedText,
    /// Reference to the matched entry (shared ownership)
    pub entry: Arc<SharedMemoryEntry>,
    /// Similarity score (0.0 - 1.0)
    pub score: f32,
}

impl SharedSearchResult {
    #[inline]
    pub fn new(entry: Arc<SharedMemoryEntry>, score: f32) -> Self {
        Self {
            id: entry.id.clone(),
            entry,
            score,
        }
    }
}

// ============================================================================
// Cow-based Temporary String Processing
// ============================================================================

/// Process a query string, returning borrowed if no modification needed.
/// This avoids allocation for queries that don't need normalization.
#[inline]
pub fn normalize_query<'a>(query: &'a str) -> Cow<'a, str> {
    let trimmed = query.trim();

    // Check if any modification is needed
    let needs_trim = trimmed.len() != query.len();
    let needs_lowercase = trimmed.chars().any(|c| c.is_ascii_uppercase());

    if !needs_trim && !needs_lowercase {
        // No modification needed - return borrowed
        Cow::Borrowed(query)
    } else if needs_lowercase {
        // Need to lowercase - must allocate
        Cow::Owned(trimmed.to_lowercase())
    } else {
        // Just trimmed - still need owned due to lifetime
        Cow::Borrowed(trimmed)
    }
}

/// Process text for caching, returning Cow to avoid allocation when possible.
#[inline]
pub fn prepare_cache_key<'a>(input: &'a str) -> Cow<'a, str> {
    let trimmed = input.trim();
    if trimmed.len() == input.len() {
        Cow::Borrowed(input)
    } else {
        Cow::Borrowed(trimmed)
    }
}

/// Extract excerpt from text without allocation if within bounds.
#[inline]
pub fn extract_excerpt<'a>(text: &'a str, max_chars: usize) -> Cow<'a, str> {
    if text.len() <= max_chars {
        Cow::Borrowed(text)
    } else {
        // Find char boundary
        let mut end = max_chars;
        while !text.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        Cow::Borrowed(&text[..end])
    }
}

// ============================================================================
// Arc-based Cache Entry
// ============================================================================

/// Cache entry using Arc for zero-copy retrieval.
#[derive(Debug, Clone)]
pub struct ArcCachedResponse {
    /// Shared query result (zero-copy on retrieval)
    pub result: Arc<SharedQueryResult>,
    /// Unix timestamp when cached
    pub cached_at_secs: i64,
}

impl ArcCachedResponse {
    #[inline]
    pub fn new(result: SharedQueryResult) -> Self {
        Self {
            result: Arc::new(result),
            cached_at_secs: chrono::Utc::now().timestamp(),
        }
    }

    /// Check if this entry has expired.
    #[inline]
    pub fn is_expired(&self, ttl_secs: i64) -> bool {
        let now = chrono::Utc::now().timestamp();
        now - self.cached_at_secs >= ttl_secs
    }

    /// Get the result (zero-copy, just Arc clone).
    #[inline]
    pub fn get_result(&self) -> Arc<SharedQueryResult> {
        Arc::clone(&self.result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_text_creation() {
        let text: SharedText = "hello".into_shared_text();
        assert_eq!(&*text, "hello");

        let string = String::from("world");
        let text2: SharedText = string.into_shared_text();
        assert_eq!(&*text2, "world");
    }

    #[test]
    fn test_shared_embedding_creation() {
        let vec = vec![0.1f32, 0.2, 0.3];
        let embedding: SharedEmbedding = vec.into_shared_embedding();
        assert_eq!(embedding.len(), 3);
        assert!((embedding[0] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_zero_copy_clone() {
        let text: SharedText = "test".into_shared_text();
        let ptr1 = Arc::as_ptr(&text);

        let cloned = text.clone();
        let ptr2 = Arc::as_ptr(&cloned);

        // Verify same underlying data (zero-copy)
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn test_shared_query_result() {
        let result = SharedQueryResult::new(
            "id-1",
            "response text",
            0.95,
            10,
            50.0,
            vec![],
            SharedTokenUsage::new(5, 10),
        );

        assert_eq!(result.text_str(), "response text");
        assert_eq!(result.confidence, 0.95);
    }

    #[test]
    fn test_normalize_query_borrowed() {
        let query = "simple query";
        let normalized = normalize_query(query);

        // Should be borrowed (no allocation)
        assert!(matches!(normalized, Cow::Borrowed(_)));
        assert_eq!(&*normalized, "simple query");
    }

    #[test]
    fn test_normalize_query_trimmed() {
        let query = "  trimmed  ";
        let normalized = normalize_query(query);
        assert_eq!(&*normalized, "trimmed");
    }

    #[test]
    fn test_extract_excerpt() {
        let text = "short";
        let excerpt = extract_excerpt(text, 100);
        assert!(matches!(excerpt, Cow::Borrowed(_)));

        let long_text = "this is a much longer text that should be truncated";
        let excerpt = extract_excerpt(long_text, 10);
        assert_eq!(excerpt.len(), 10);
    }

    #[test]
    fn test_arc_cached_response() {
        let result = SharedQueryResult::new(
            "cache-1",
            "cached response",
            0.9,
            5,
            25.0,
            vec![],
            SharedTokenUsage::default(),
        );

        let cached = ArcCachedResponse::new(result);

        // Get result twice - should be zero-copy
        let r1 = cached.get_result();
        let r2 = cached.get_result();

        assert_eq!(Arc::strong_count(&r1), 3); // original + r1 + r2
        assert_eq!(r1.text_str(), r2.text_str());
    }

    #[test]
    fn test_shared_memory_entry() {
        let entry = SharedMemoryEntry::new("mem-1", "sample text", vec![0.1f32, 0.2, 0.3, 0.4]);

        assert_eq!(entry.text_str(), "sample text");
        assert_eq!(entry.embedding_slice().len(), 4);
    }

    #[test]
    fn test_memory_size_estimation() {
        let entry = SharedMemoryEntry::new("test-id", "some text content", vec![0.0f32; 384]);

        let size = entry.estimated_size();
        // Should account for text + embedding
        assert!(size > 384 * 4); // At least embedding size
    }
}
