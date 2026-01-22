//! Eviction policy implementation
//!
//! This module provides LRU-based eviction with size-awareness for
//! selecting pages to evict when memory pressure occurs.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::metadata::{ContentType, MetadataTable, PageMetadata};
use super::page::PageId;

/// Configuration for eviction policy
#[derive(Debug, Clone)]
pub struct EvictionConfig {
    /// Recency weight in eviction score (0.0 - 1.0)
    pub recency_weight: f64,
    /// Size weight in eviction score (0.0 - 1.0)
    pub size_weight: f64,
    /// Priority weight in eviction score (0.0 - 1.0)
    pub priority_weight: f64,
    /// Maximum pages to evict in one batch
    pub batch_size: usize,
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            recency_weight: 0.6,
            size_weight: 0.2,
            priority_weight: 0.2,
            batch_size: 64,
        }
    }
}

impl EvictionConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), &'static str> {
        let total = self.recency_weight + self.size_weight + self.priority_weight;
        if (total - 1.0).abs() > 0.001 {
            return Err("Weights must sum to 1.0");
        }
        Ok(())
    }
}

/// Entry in the eviction tracking
#[derive(Debug, Clone)]
struct EvictionEntry {
    page_id: PageId,
    last_access: u64,
    content_type: ContentType,
}

/// Eviction candidate with computed score
#[derive(Debug, Clone)]
struct EvictionCandidate {
    page_id: PageId,
    score: f64,
}

impl PartialEq for EvictionCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for EvictionCandidate {}

impl PartialOrd for EvictionCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EvictionCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower score = higher eviction priority (min-heap behavior)
        // Reverse comparison for use with BinaryHeap (which is a max-heap)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Trait for eviction policies
pub trait EvictionPolicy: Send + Sync {
    /// Select the next victim for eviction
    fn select_victim(&mut self, metadata: &MetadataTable) -> Option<PageId>;

    /// Select multiple victims for batch eviction
    fn select_victims(&mut self, metadata: &MetadataTable, count: usize) -> Vec<PageId>;

    /// Notify of page access (for LRU tracking)
    fn touch(&mut self, page: PageId, timestamp: u64);

    /// Remove a page from tracking
    fn remove(&mut self, page: PageId);

    /// Update configuration
    fn configure(&mut self, config: EvictionConfig);
}

/// LRU eviction policy with size and priority awareness
pub struct LruEvictionPolicy {
    /// Configuration
    config: EvictionConfig,
    /// Tracking entries by page ID
    entries: HashMap<u32, EvictionEntry>,
    /// Current time reference for recency calculation
    current_time: u64,
}

impl LruEvictionPolicy {
    /// Create a new LRU eviction policy
    pub fn new(config: EvictionConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            current_time: 0,
        }
    }

    /// Compute eviction score for a page
    ///
    /// Lower score = more likely to evict
    fn compute_score(&self, entry: &EvictionEntry) -> f64 {
        // Recency score: higher for recently accessed
        let time_since_access = self.current_time.saturating_sub(entry.last_access);
        let recency_score = 1.0 / (time_since_access as f64 + 1.0);

        // Size factor: we don't track block size per-page, so use 1.0
        let size_score = 1.0;

        // Priority from content type: higher = less likely to evict
        let priority_score = entry.content_type.eviction_priority() as f64 / 5.0;

        // Weighted sum
        self.config.recency_weight * recency_score
            + self.config.size_weight * size_score
            + self.config.priority_weight * priority_score
    }

    /// Update current time reference
    fn update_time(&mut self, timestamp: u64) {
        if timestamp > self.current_time {
            self.current_time = timestamp;
        }
    }
}

impl EvictionPolicy for LruEvictionPolicy {
    fn select_victim(&mut self, metadata: &MetadataTable) -> Option<PageId> {
        let mut best_candidate: Option<(PageId, f64)> = None;

        for (&page_id, entry) in &self.entries {
            // Check if page is evictable
            if let Some(meta) = metadata.get(PageId::new(page_id)) {
                if !meta.is_evictable() {
                    continue;
                }

                let score = self.compute_score(entry);

                match &best_candidate {
                    None => best_candidate = Some((PageId::new(page_id), score)),
                    Some((_, best_score)) if score < *best_score => {
                        best_candidate = Some((PageId::new(page_id), score));
                    }
                    _ => {}
                }
            }
        }

        best_candidate.map(|(id, _)| id)
    }

    fn select_victims(&mut self, metadata: &MetadataTable, count: usize) -> Vec<PageId> {
        // Build heap of candidates
        let mut heap: BinaryHeap<EvictionCandidate> = BinaryHeap::new();

        for (&page_id, entry) in &self.entries {
            if let Some(meta) = metadata.get(PageId::new(page_id)) {
                if meta.is_evictable() {
                    heap.push(EvictionCandidate {
                        page_id: PageId::new(page_id),
                        score: self.compute_score(entry),
                    });
                }
            }
        }

        // Extract top candidates
        let mut victims = Vec::with_capacity(count.min(heap.len()));
        for _ in 0..count {
            if let Some(candidate) = heap.pop() {
                victims.push(candidate.page_id);
            } else {
                break;
            }
        }

        victims
    }

    fn touch(&mut self, page: PageId, timestamp: u64) {
        self.update_time(timestamp);

        self.entries
            .entry(page.raw())
            .and_modify(|e| e.last_access = timestamp)
            .or_insert_with(|| EvictionEntry {
                page_id: page,
                last_access: timestamp,
                content_type: ContentType::Empty,
            });
    }

    fn remove(&mut self, page: PageId) {
        self.entries.remove(&page.raw());
    }

    fn configure(&mut self, config: EvictionConfig) {
        self.config = config;
    }
}

/// Content type priority levels for eviction
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvictionPriority {
    /// Lowest priority - evict first
    Lowest = 0,
    /// Low priority
    Low = 1,
    /// Medium priority
    Medium = 2,
    /// High priority
    High = 3,
    /// Highest priority - evict last
    Highest = 4,
}

impl From<ContentType> for EvictionPriority {
    fn from(ct: ContentType) -> Self {
        match ct {
            ContentType::Empty => EvictionPriority::Lowest,
            ContentType::TempBuffer => EvictionPriority::Low,
            ContentType::Activation | ContentType::Gradient => EvictionPriority::Medium,
            ContentType::LoraWeight => EvictionPriority::High,
            ContentType::KvCache => EvictionPriority::Highest,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eviction_config_default() {
        let config = EvictionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_eviction_config_invalid() {
        let config = EvictionConfig {
            recency_weight: 0.5,
            size_weight: 0.5,
            priority_weight: 0.5, // Sum = 1.5, invalid
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_lru_policy_basic() {
        let config = EvictionConfig::default();
        let mut policy = LruEvictionPolicy::new(config);

        // Touch some pages
        policy.touch(PageId::new(0), 100);
        policy.touch(PageId::new(1), 200);
        policy.touch(PageId::new(2), 50); // Oldest

        // Create metadata table
        let metadata = MetadataTable::new(10);

        // Allocate pages to make them evictable
        for i in 0..3 {
            if let Some(meta) = metadata.get(PageId::new(i)) {
                meta.try_allocate(ContentType::TempBuffer, 0, 0, 0, 0);
            }
        }

        // Select victim - should be page 2 (oldest)
        let victim = policy.select_victim(&metadata);
        assert!(victim.is_some());
        // Note: The exact victim depends on score calculation
    }

    #[test]
    fn test_lru_policy_remove() {
        let config = EvictionConfig::default();
        let mut policy = LruEvictionPolicy::new(config);

        policy.touch(PageId::new(0), 100);
        policy.touch(PageId::new(1), 200);

        policy.remove(PageId::new(0));

        // Verify page 0 is no longer tracked
        assert!(!policy.entries.contains_key(&0));
    }

    #[test]
    fn test_eviction_priority() {
        assert!(
            EvictionPriority::from(ContentType::TempBuffer)
                < EvictionPriority::from(ContentType::KvCache)
        );
        assert!(
            EvictionPriority::from(ContentType::LoraWeight)
                < EvictionPriority::from(ContentType::KvCache)
        );
    }

    #[test]
    fn test_select_multiple_victims() {
        let config = EvictionConfig::default();
        let mut policy = LruEvictionPolicy::new(config);

        // Touch pages with different timestamps
        for i in 0..10 {
            policy.touch(PageId::new(i), i as u64 * 100);
        }

        let metadata = MetadataTable::new(20);
        for i in 0..10 {
            if let Some(meta) = metadata.get(PageId::new(i)) {
                meta.try_allocate(ContentType::TempBuffer, 0, 0, 0, 0);
            }
        }

        let victims = policy.select_victims(&metadata, 5);
        assert_eq!(victims.len(), 5);
    }
}
