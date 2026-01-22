//! Query Decomposition
//!
//! Pattern-based decomposition of complex queries into simpler sub-queries.
//! Optimized for <1ms query decomposition latency.

use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

use super::config::DecompositionConfig;

// Pre-computed lowercase keywords for fast matching (avoids repeated to_lowercase())
static QUESTION_WORDS: &[&str] = &["what", "why", "how", "when", "where", "who"];
static CLAUSE_INDICATORS: &[&str] = &[",", ";", "and", "but", "or", "because", "although"];

/// Query decomposer for breaking down complex queries.
#[derive(Debug, Clone)]
pub struct QueryDecomposer {
    /// Configuration for decomposition patterns.
    config: DecompositionConfig,
}

impl QueryDecomposer {
    /// Create a new query decomposer with default configuration.
    pub fn new() -> Self {
        Self {
            config: DecompositionConfig::default(),
        }
    }

    /// Create a new query decomposer with custom configuration.
    pub fn with_config(config: DecompositionConfig) -> Self {
        Self { config }
    }

    /// Decompose a query into sub-queries.
    /// Optimized for <1ms latency with early returns and minimal allocations.
    #[inline]
    pub fn decompose(&self, query: &str) -> DecompositionResult {
        // Fast path: short queries don't need decomposition
        let query_len = query.len();
        if query_len < self.config.min_decomposition_length {
            return DecompositionResult {
                original_query: query.to_string(),
                strategy: DecompositionStrategy::Direct,
                sub_queries: vec![SubQuery::new(query, QueryType::Atomic, 0)],
                complexity_score: self.compute_complexity_fast(query, query_len),
            };
        }

        // Allocate lowercase once for all checks
        let query_lower = query.to_ascii_lowercase();
        let query_lower_ref = query_lower.as_str();

        // Try comparison decomposition first (most specific)
        if let Some(result) = self.try_comparison_decomposition_fast(query, query_lower_ref) {
            return result;
        }

        // Try conjunction decomposition
        if let Some(result) = self.try_conjunction_decomposition_fast(query, query_lower_ref) {
            return result;
        }

        // Try sequential decomposition
        if let Some(result) = self.try_sequential_decomposition_fast(query, query_lower_ref) {
            return result;
        }

        // Fall back to direct processing
        DecompositionResult {
            original_query: query.to_string(),
            strategy: DecompositionStrategy::Direct,
            sub_queries: vec![SubQuery::new(query, QueryType::Atomic, 0)],
            complexity_score: self.compute_complexity_fast(query, query_len),
        }
    }

    /// Determine the type of a query.
    /// Optimized with early returns and minimal string operations.
    #[inline]
    pub fn classify_query(&self, query: &str) -> QueryType {
        // Use bytes for fast prefix checking (ASCII only for keywords)
        let bytes = query.as_bytes();
        let len = bytes.len();

        if len == 0 {
            return QueryType::Atomic;
        }

        // Check first character for fast prefix filtering
        let first_lower = bytes[0].to_ascii_lowercase();

        match first_lower {
            b'w' => {
                // Check for "what" or "why" or "when" or "where" or "who"
                if len >= 4 {
                    let prefix = &bytes[..4];
                    if prefix.eq_ignore_ascii_case(b"what") {
                        return QueryType::Definition;
                    }
                    if prefix.eq_ignore_ascii_case(b"when") || prefix.eq_ignore_ascii_case(b"wher")
                    {
                        return QueryType::Atomic;
                    }
                }
                if len >= 3 && bytes[..3].eq_ignore_ascii_case(b"why") {
                    return QueryType::Causal;
                }
                if len >= 3 && bytes[..3].eq_ignore_ascii_case(b"who") {
                    return QueryType::Atomic;
                }
            }
            b'h' => {
                if len >= 3 && bytes[..3].eq_ignore_ascii_case(b"how") {
                    return QueryType::Procedural;
                }
            }
            b'c' => {
                if len >= 7 && bytes[..7].eq_ignore_ascii_case(b"compare") {
                    return QueryType::Comparison;
                }
            }
            b'l' => {
                if len >= 4 && bytes[..4].eq_ignore_ascii_case(b"list") {
                    return QueryType::Enumeration;
                }
            }
            _ => {}
        }

        // Fall back to contains checks for non-prefix patterns
        let query_lower = query.to_ascii_lowercase();
        if query_lower.contains("what is") {
            return QueryType::Definition;
        }
        if query_lower.contains("reason") {
            return QueryType::Causal;
        }
        if query_lower.contains("steps") {
            return QueryType::Procedural;
        }
        if query_lower.contains("difference") {
            return QueryType::Comparison;
        }
        if query_lower.contains("enumerate") {
            return QueryType::Enumeration;
        }

        QueryType::Atomic
    }

    /// Compute complexity score for a query (0.0 = simple, 1.0 = complex).
    /// Original method kept for API compatibility.
    #[inline]
    pub fn compute_complexity(&self, query: &str) -> f32 {
        self.compute_complexity_fast(query, query.len())
    }

    /// Optimized complexity computation with pre-computed length.
    /// Avoids multiple to_lowercase() calls by using ASCII lowercase.
    #[inline]
    fn compute_complexity_fast(&self, query: &str, query_len: usize) -> f32 {
        // Length factor (fast - no string operations)
        let mut score = (query_len as f32 * 0.0015).min(0.3); // Simplified: len/200 * 0.3 => len * 0.0015

        // Only do lowercase if we need to check keywords
        if query_len > 10 {
            let query_lower = query.to_ascii_lowercase();

            // Conjunction factor - use count instead of filter
            let conjunction_count = self
                .config
                .conjunction_keywords
                .iter()
                .filter(|kw| query_lower.contains(kw.as_str()))
                .count();
            score += (conjunction_count as f32 * 0.15).min(0.3);

            // Question word factor - use static array
            let question_count = QUESTION_WORDS
                .iter()
                .filter(|w| query_lower.contains(*w))
                .count();
            score += (question_count as f32 * 0.1).min(0.2);

            // Clause count - use static array
            let clause_count = CLAUSE_INDICATORS
                .iter()
                .filter(|c| query_lower.contains(*c))
                .count();
            score += (clause_count as f32 * 0.1).min(0.2);
        }

        score.min(1.0)
    }

    /// Get decomposer statistics.
    pub fn stats(&self) -> DecomposerStatsSnapshot {
        DECOMPOSER_STATS.snapshot()
    }

    // Private helper methods - optimized versions

    /// Optimized comparison decomposition with fast keyword search.
    #[inline]
    fn try_comparison_decomposition_fast(
        &self,
        query: &str,
        query_lower: &str,
    ) -> Option<DecompositionResult> {
        // Find first matching keyword position for early termination
        for keyword in &self.config.comparison_keywords {
            if let Some(pos) = query_lower.find(keyword.as_str()) {
                // Try to extract comparison entities using found position
                let parts = self.extract_comparison_entities_at(query, keyword, pos);
                if parts.len() >= 2 {
                    DECOMPOSER_STATS
                        .record_decomposition(DecompositionStrategy::Comparison(parts.clone()));

                    // Pre-allocate sub_queries with exact capacity
                    let mut sub_queries = Vec::with_capacity(parts.len());
                    for (i, part) in parts.iter().enumerate() {
                        let trimmed = part.trim();
                        // Use Cow to avoid allocation for small strings
                        let sub_query = format!("What is {}?", trimmed);
                        sub_queries.push(SubQuery::new(&sub_query, QueryType::Definition, i));
                    }

                    return Some(DecompositionResult {
                        original_query: query.to_string(),
                        strategy: DecompositionStrategy::Comparison(parts),
                        sub_queries,
                        complexity_score: self.compute_complexity_fast(query, query.len()),
                    });
                }
            }
        }
        None
    }

    /// Optimized conjunction decomposition.
    #[inline]
    fn try_conjunction_decomposition_fast(
        &self,
        query: &str,
        query_lower: &str,
    ) -> Option<DecompositionResult> {
        for keyword in &self.config.conjunction_keywords {
            if let Some(_) = query_lower.find(keyword.as_str()) {
                let parts = self.split_by_keyword_fast(query, query_lower, keyword);
                if parts.len() >= 2 {
                    DECOMPOSER_STATS
                        .record_decomposition(DecompositionStrategy::Conjunction(parts.clone()));

                    let mut sub_queries = Vec::with_capacity(parts.len());
                    for (i, part) in parts.iter().enumerate() {
                        sub_queries.push(SubQuery::new(part.trim(), self.classify_query(part), i));
                    }

                    return Some(DecompositionResult {
                        original_query: query.to_string(),
                        strategy: DecompositionStrategy::Conjunction(parts),
                        sub_queries,
                        complexity_score: self.compute_complexity_fast(query, query.len()),
                    });
                }
            }
        }
        None
    }

    /// Optimized sequential decomposition.
    #[inline]
    fn try_sequential_decomposition_fast(
        &self,
        query: &str,
        query_lower: &str,
    ) -> Option<DecompositionResult> {
        for keyword in &self.config.sequential_keywords {
            if let Some(_) = query_lower.find(keyword.as_str()) {
                let parts = self.split_by_keyword_fast(query, query_lower, keyword);
                if parts.len() >= 2 {
                    DECOMPOSER_STATS
                        .record_decomposition(DecompositionStrategy::Sequential(parts.clone()));

                    let mut sub_queries = Vec::with_capacity(parts.len());
                    for (i, part) in parts.iter().enumerate() {
                        sub_queries.push(SubQuery::new(part.trim(), QueryType::Procedural, i));
                    }

                    return Some(DecompositionResult {
                        original_query: query.to_string(),
                        strategy: DecompositionStrategy::Sequential(parts),
                        sub_queries,
                        complexity_score: self.compute_complexity_fast(query, query.len()),
                    });
                }
            }
        }
        None
    }

    // Legacy methods for API compatibility
    fn try_comparison_decomposition(
        &self,
        query: &str,
        query_lower: &str,
    ) -> Option<DecompositionResult> {
        self.try_comparison_decomposition_fast(query, query_lower)
    }

    fn try_conjunction_decomposition(
        &self,
        query: &str,
        query_lower: &str,
    ) -> Option<DecompositionResult> {
        self.try_conjunction_decomposition_fast(query, query_lower)
    }

    fn try_sequential_decomposition(
        &self,
        query: &str,
        query_lower: &str,
    ) -> Option<DecompositionResult> {
        self.try_sequential_decomposition_fast(query, query_lower)
    }

    /// Extract comparison entities at a known position.
    #[inline]
    fn extract_comparison_entities_at(
        &self,
        query: &str,
        keyword: &str,
        pos: usize,
    ) -> Vec<String> {
        // Split around the comparison keyword at known position
        let before = query[..pos].trim();
        let after = query[pos + keyword.len()..].trim();

        // Clean up the parts
        let clean_before = self.clean_entity_fast(before);
        let clean_after = self.clean_entity_fast(after);

        if !clean_before.is_empty() && !clean_after.is_empty() {
            return vec![clean_before, clean_after];
        }
        Vec::new()
    }

    fn extract_comparison_entities(&self, query: &str, keyword: &str) -> Vec<String> {
        let query_lower = query.to_ascii_lowercase();
        if let Some(pos) = query_lower.find(keyword) {
            return self.extract_comparison_entities_at(query, keyword, pos);
        }
        Vec::new()
    }

    /// Optimized split using pre-computed lowercase query.
    #[inline]
    fn split_by_keyword_fast(&self, query: &str, query_lower: &str, keyword: &str) -> Vec<String> {
        // Pre-count occurrences for capacity hint
        let count = query_lower.matches(keyword).count();
        let mut result = Vec::with_capacity(count + 1);
        let mut last_end = 0;

        for (idx, _) in query_lower.match_indices(keyword) {
            if idx > last_end {
                let part = query[last_end..idx].trim();
                if !part.is_empty() {
                    result.push(part.to_string());
                }
            }
            last_end = idx + keyword.len();
        }

        if last_end < query.len() {
            let part = query[last_end..].trim();
            if !part.is_empty() {
                result.push(part.to_string());
            }
        }

        result
    }

    fn split_by_conjunction(&self, query: &str, keyword: &str) -> Vec<String> {
        let query_lower = query.to_ascii_lowercase();
        self.split_by_keyword_fast(query, &query_lower, keyword)
    }

    fn split_by_sequential(&self, query: &str, keyword: &str) -> Vec<String> {
        self.split_by_conjunction(query, keyword)
    }

    /// Optimized entity cleaning with reduced allocations.
    #[inline]
    fn clean_entity_fast(&self, entity: &str) -> String {
        let mut s = entity.trim();

        // Static prefix/suffix arrays
        const PREFIXES: &[&str] = &["the ", "a ", "an ", "what is ", "what are "];
        const SUFFIXES: &[&str] = &["?", ".", "!"];

        // Remove prefixes (check lowercase)
        for prefix in PREFIXES {
            let prefix_len = prefix.len();
            if s.len() >= prefix_len {
                let start_lower = s[..prefix_len].to_ascii_lowercase();
                if start_lower == *prefix {
                    s = &s[prefix_len..];
                    break; // Only remove one prefix
                }
            }
        }

        // Remove suffixes
        for suffix in SUFFIXES {
            if s.ends_with(suffix) {
                s = &s[..s.len() - suffix.len()];
                break; // Only remove one suffix
            }
        }

        s.trim().to_string()
    }

    fn clean_entity(&self, entity: &str) -> String {
        self.clean_entity_fast(entity)
    }
}

impl Default for QueryDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of query decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    /// Original query text.
    pub original_query: String,

    /// Strategy used for decomposition.
    pub strategy: DecompositionStrategy,

    /// Generated sub-queries.
    pub sub_queries: Vec<SubQuery>,

    /// Complexity score of the original query.
    pub complexity_score: f32,
}

impl DecompositionResult {
    /// Returns true if the query was decomposed into multiple sub-queries.
    pub fn was_decomposed(&self) -> bool {
        self.sub_queries.len() > 1 || !matches!(self.strategy, DecompositionStrategy::Direct)
    }

    /// Get the number of sub-queries.
    pub fn sub_query_count(&self) -> usize {
        self.sub_queries.len()
    }
}

/// Strategy used for decomposition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// No decomposition needed - direct answering.
    Direct,

    /// Decomposed by conjunction (e.g., "X and Y").
    Conjunction(Vec<String>),

    /// Decomposed for comparison (e.g., "compare X and Y").
    Comparison(Vec<String>),

    /// Decomposed by sequential steps (e.g., "first X, then Y").
    Sequential(Vec<String>),

    /// Decomposed by aspect/facet analysis.
    Faceted(Vec<String>),

    /// Decomposed by causal chain (causes -> effects).
    Causal(Vec<String>),
}

impl DecompositionStrategy {
    /// Returns the name of the strategy.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Conjunction(_) => "conjunction",
            Self::Comparison(_) => "comparison",
            Self::Sequential(_) => "sequential",
            Self::Faceted(_) => "faceted",
            Self::Causal(_) => "causal",
        }
    }

    /// Returns the decomposed parts, if any.
    pub fn parts(&self) -> Option<&[String]> {
        match self {
            Self::Direct => None,
            Self::Conjunction(parts) => Some(parts),
            Self::Comparison(parts) => Some(parts),
            Self::Sequential(parts) => Some(parts),
            Self::Faceted(parts) => Some(parts),
            Self::Causal(parts) => Some(parts),
        }
    }
}

/// A sub-query generated from decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQuery {
    /// The sub-query text.
    pub text: String,

    /// Type of the sub-query.
    pub query_type: QueryType,

    /// Order in the sequence (for sequential strategies).
    pub order: usize,

    /// Unique identifier for caching.
    pub id: u64,

    /// Dependencies on other sub-queries (by ID).
    pub depends_on: Vec<u64>,
}

impl SubQuery {
    /// Create a new sub-query.
    pub fn new(text: &str, query_type: QueryType, order: usize) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self {
            text: text.to_string(),
            query_type,
            order,
            id: COUNTER.fetch_add(1, Ordering::SeqCst),
            depends_on: Vec::new(),
        }
    }

    /// Add a dependency on another sub-query.
    pub fn add_dependency(&mut self, dependency_id: u64) {
        if !self.depends_on.contains(&dependency_id) {
            self.depends_on.push(dependency_id);
        }
    }

    /// Check if this sub-query has dependencies.
    pub fn has_dependencies(&self) -> bool {
        !self.depends_on.is_empty()
    }
}

/// Type of query for routing and processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    /// Simple factual query.
    Atomic,

    /// Definition or explanation.
    Definition,

    /// Causal reasoning (why questions).
    Causal,

    /// Procedural/how-to questions.
    Procedural,

    /// Comparison between entities.
    Comparison,

    /// List or enumeration.
    Enumeration,

    /// Opinion or subjective.
    Opinion,
}

impl QueryType {
    /// Suggested token budget multiplier for this query type.
    pub fn token_budget_multiplier(&self) -> f32 {
        match self {
            Self::Atomic => 0.5,
            Self::Definition => 0.8,
            Self::Causal => 1.2,
            Self::Procedural => 1.5,
            Self::Comparison => 1.3,
            Self::Enumeration => 1.0,
            Self::Opinion => 0.7,
        }
    }
}

/// Statistics for decomposer operations.
#[derive(Debug, Default)]
pub struct DecomposerStats {
    pub total_decompositions: AtomicU64,
    pub direct_count: AtomicU64,
    pub conjunction_count: AtomicU64,
    pub comparison_count: AtomicU64,
    pub sequential_count: AtomicU64,
    pub total_sub_queries: AtomicU64,
}

impl DecomposerStats {
    /// Record a decomposition operation.
    pub fn record_decomposition(&self, strategy: DecompositionStrategy) {
        self.total_decompositions.fetch_add(1, Ordering::SeqCst);

        match &strategy {
            DecompositionStrategy::Direct => {
                self.direct_count.fetch_add(1, Ordering::SeqCst);
                self.total_sub_queries.fetch_add(1, Ordering::SeqCst);
            }
            DecompositionStrategy::Conjunction(parts) => {
                self.conjunction_count.fetch_add(1, Ordering::SeqCst);
                self.total_sub_queries
                    .fetch_add(parts.len() as u64, Ordering::SeqCst);
            }
            DecompositionStrategy::Comparison(parts) => {
                self.comparison_count.fetch_add(1, Ordering::SeqCst);
                self.total_sub_queries
                    .fetch_add(parts.len() as u64, Ordering::SeqCst);
            }
            DecompositionStrategy::Sequential(parts) => {
                self.sequential_count.fetch_add(1, Ordering::SeqCst);
                self.total_sub_queries
                    .fetch_add(parts.len() as u64, Ordering::SeqCst);
            }
            _ => {}
        }
    }

    /// Get a snapshot of current statistics.
    pub fn snapshot(&self) -> DecomposerStatsSnapshot {
        DecomposerStatsSnapshot {
            total_decompositions: self.total_decompositions.load(Ordering::SeqCst),
            direct_count: self.direct_count.load(Ordering::SeqCst),
            conjunction_count: self.conjunction_count.load(Ordering::SeqCst),
            comparison_count: self.comparison_count.load(Ordering::SeqCst),
            sequential_count: self.sequential_count.load(Ordering::SeqCst),
            total_sub_queries: self.total_sub_queries.load(Ordering::SeqCst),
        }
    }
}

/// Snapshot of decomposer statistics.
#[derive(Debug, Clone, Default)]
pub struct DecomposerStatsSnapshot {
    pub total_decompositions: u64,
    pub direct_count: u64,
    pub conjunction_count: u64,
    pub comparison_count: u64,
    pub sequential_count: u64,
    pub total_sub_queries: u64,
}

// Global stats instance
static DECOMPOSER_STATS: DecomposerStats = DecomposerStats {
    total_decompositions: AtomicU64::new(0),
    direct_count: AtomicU64::new(0),
    conjunction_count: AtomicU64::new(0),
    comparison_count: AtomicU64::new(0),
    sequential_count: AtomicU64::new(0),
    total_sub_queries: AtomicU64::new(0),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposer_creation() {
        let decomposer = QueryDecomposer::new();
        assert!(!decomposer.config.conjunction_keywords.is_empty());
    }

    #[test]
    fn test_direct_decomposition_short_query() {
        let decomposer = QueryDecomposer::new();
        let result = decomposer.decompose("What is AI?");

        assert!(matches!(result.strategy, DecompositionStrategy::Direct));
        assert_eq!(result.sub_queries.len(), 1);
    }

    #[test]
    fn test_conjunction_decomposition() {
        let decomposer = QueryDecomposer::new();
        let result = decomposer.decompose("What are the causes and effects of climate change?");

        assert!(matches!(
            result.strategy,
            DecompositionStrategy::Conjunction(_)
        ));
        assert!(result.sub_queries.len() >= 2);
    }

    #[test]
    fn test_comparison_decomposition() {
        let decomposer = QueryDecomposer::new();
        let result = decomposer.decompose("Compare machine learning versus deep learning");

        assert!(matches!(
            result.strategy,
            DecompositionStrategy::Comparison(_)
        ));
    }

    #[test]
    fn test_complexity_scoring() {
        let decomposer = QueryDecomposer::new();

        let simple = decomposer.compute_complexity("What is AI?");
        let complex = decomposer.compute_complexity(
            "What are the primary causes of climate change, and how do they contribute to global warming, affecting both weather patterns and sea levels?"
        );

        assert!(simple < complex);
        assert!(simple >= 0.0 && simple <= 1.0);
        assert!(complex >= 0.0 && complex <= 1.0);
    }

    #[test]
    fn test_query_classification() {
        let decomposer = QueryDecomposer::new();

        assert_eq!(
            decomposer.classify_query("What is machine learning?"),
            QueryType::Definition
        );
        assert_eq!(
            decomposer.classify_query("Why does gravity exist?"),
            QueryType::Causal
        );
        assert_eq!(
            decomposer.classify_query("How do I learn programming?"),
            QueryType::Procedural
        );
        assert_eq!(
            decomposer.classify_query("Compare Python and JavaScript"),
            QueryType::Comparison
        );
    }

    #[test]
    fn test_sub_query_dependencies() {
        let mut sq1 = SubQuery::new("Query 1", QueryType::Atomic, 0);
        let sq2 = SubQuery::new("Query 2", QueryType::Atomic, 1);

        assert!(!sq1.has_dependencies());

        sq1.add_dependency(sq2.id);
        assert!(sq1.has_dependencies());
        assert!(sq1.depends_on.contains(&sq2.id));
    }

    #[test]
    fn test_decomposition_result_methods() {
        let decomposer = QueryDecomposer::new();

        let simple_result = decomposer.decompose("What is AI?");
        assert!(!simple_result.was_decomposed());

        let complex_result =
            decomposer.decompose("What are the causes and effects of climate change?");
        assert!(complex_result.was_decomposed());
    }
}
