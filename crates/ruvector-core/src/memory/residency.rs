//! Adapter residency tier management
//!
//! This module implements Hot/Warm/Cold tiered storage for LoRA adapters,
//! following the S-LoRA paper's approach to efficient multi-tenant serving.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Residency tier for adapters
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidencyTier {
    /// Cold tier: Compressed on disk/NVMe, ~10ms load time
    Cold = 0,
    /// Warm tier: INT8 weights in CPU memory, ~1ms load time
    Warm = 1,
    /// Hot tier: FP16 weights in GPU memory, instant access
    Hot = 2,
}

impl ResidencyTier {
    /// Get the expected load latency for this tier
    pub fn load_latency(&self) -> Duration {
        match self {
            ResidencyTier::Cold => Duration::from_millis(10),
            ResidencyTier::Warm => Duration::from_millis(1),
            ResidencyTier::Hot => Duration::from_micros(100),
        }
    }

    /// Get the priority for eviction (lower = evict first)
    pub fn eviction_priority(&self) -> u8 {
        *self as u8
    }
}

impl std::fmt::Display for ResidencyTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidencyTier::Cold => write!(f, "Cold"),
            ResidencyTier::Warm => write!(f, "Warm"),
            ResidencyTier::Hot => write!(f, "Hot"),
        }
    }
}

/// Configuration for residency management
#[derive(Debug, Clone)]
pub struct ResidencyConfig {
    /// Maximum adapters in hot tier
    pub hot_tier_budget: usize,
    /// Maximum adapters in warm tier
    pub warm_tier_budget: usize,
    /// Access window for tier decisions (seconds)
    pub access_window_secs: u64,
    /// Accesses needed for promotion to hot
    pub hot_promotion_threshold: u32,
    /// Accesses needed for promotion to warm
    pub warm_promotion_threshold: u32,
    /// Idle time before demotion (seconds)
    pub demotion_idle_secs: u64,
}

impl Default for ResidencyConfig {
    fn default() -> Self {
        Self {
            hot_tier_budget: 100,
            warm_tier_budget: 1000,
            access_window_secs: 60,
            hot_promotion_threshold: 10,
            warm_promotion_threshold: 1,
            demotion_idle_secs: 300, // 5 minutes
        }
    }
}

/// Adapter identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdapterId(pub String);

impl AdapterId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for AdapterId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for AdapterId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl std::fmt::Display for AdapterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Tracking information for an adapter
#[derive(Debug)]
struct AdapterInfo {
    id: AdapterId,
    tier: ResidencyTier,
    accesses_in_window: AtomicU64,
    last_access: RwLock<Instant>,
    total_accesses: AtomicU64,
    size_pages: usize,
}

impl AdapterInfo {
    fn new(id: AdapterId, tier: ResidencyTier, size_pages: usize) -> Self {
        Self {
            id,
            tier,
            accesses_in_window: AtomicU64::new(0),
            last_access: RwLock::new(Instant::now()),
            total_accesses: AtomicU64::new(0),
            size_pages,
        }
    }

    fn record_access(&self) {
        self.accesses_in_window.fetch_add(1, Ordering::Relaxed);
        self.total_accesses.fetch_add(1, Ordering::Relaxed);
        *self.last_access.write() = Instant::now();
    }

    fn accesses(&self) -> u64 {
        self.accesses_in_window.load(Ordering::Relaxed)
    }

    fn last_access(&self) -> Instant {
        *self.last_access.read()
    }

    fn reset_window(&self) {
        self.accesses_in_window.store(0, Ordering::Relaxed);
    }
}

/// Manager for adapter residency tiers
pub struct ResidencyManager {
    config: ResidencyConfig,
    adapters: RwLock<HashMap<AdapterId, AdapterInfo>>,
    hot_count: AtomicU64,
    warm_count: AtomicU64,
    cold_count: AtomicU64,
}

impl ResidencyManager {
    /// Create a new residency manager
    pub fn new(config: ResidencyConfig) -> Self {
        Self {
            config,
            adapters: RwLock::new(HashMap::new()),
            hot_count: AtomicU64::new(0),
            warm_count: AtomicU64::new(0),
            cold_count: AtomicU64::new(0),
        }
    }

    /// Register a new adapter
    pub fn register_adapter(&self, id: AdapterId, size_pages: usize) -> ResidencyTier {
        let tier = self.compute_initial_tier();
        let info = AdapterInfo::new(id.clone(), tier, size_pages);

        let mut adapters = self.adapters.write();
        adapters.insert(id, info);

        self.increment_tier_count(tier);
        tier
    }

    /// Unregister an adapter
    pub fn unregister_adapter(&self, id: &AdapterId) -> Option<ResidencyTier> {
        let mut adapters = self.adapters.write();
        adapters.remove(id).map(|info| {
            self.decrement_tier_count(info.tier);
            info.tier
        })
    }

    /// Record an access to an adapter
    pub fn record_access(&self, id: &AdapterId) {
        let adapters = self.adapters.read();
        if let Some(info) = adapters.get(id) {
            info.record_access();
        }
    }

    /// Get the current tier for an adapter
    pub fn get_tier(&self, id: &AdapterId) -> Option<ResidencyTier> {
        let adapters = self.adapters.read();
        adapters.get(id).map(|info| info.tier)
    }

    /// Compute the optimal tier for an adapter based on access patterns
    pub fn compute_residency(&self, id: &AdapterId) -> Option<ResidencyTier> {
        let adapters = self.adapters.read();
        let info = adapters.get(id)?;

        let accesses = info.accesses();

        if accesses >= self.config.hot_promotion_threshold as u64 {
            Some(ResidencyTier::Hot)
        } else if accesses >= self.config.warm_promotion_threshold as u64 {
            Some(ResidencyTier::Warm)
        } else {
            Some(ResidencyTier::Cold)
        }
    }

    /// Rebalance adapters across tiers
    pub fn rebalance(&self) -> RebalanceResult {
        let mut adapters = self.adapters.write();
        let mut result = RebalanceResult::default();

        // Sort adapters by access frequency
        let mut sorted: Vec<_> = adapters.values().collect();
        sorted.sort_by(|a, b| b.accesses().cmp(&a.accesses()));

        // Assign to tiers based on budget
        let mut hot_assigned = 0;
        let mut warm_assigned = 0;

        for (i, info) in sorted.iter().enumerate() {
            let new_tier = if i < self.config.hot_tier_budget {
                hot_assigned += 1;
                ResidencyTier::Hot
            } else if i < self.config.hot_tier_budget + self.config.warm_tier_budget {
                warm_assigned += 1;
                ResidencyTier::Warm
            } else {
                ResidencyTier::Cold
            };

            if new_tier != info.tier {
                // Would need to update tier here
                match (info.tier, new_tier) {
                    (old, new) if old < new => result.promotions += 1,
                    (old, new) if old > new => result.demotions += 1,
                    _ => {}
                }
            }
        }

        // Update counts
        self.hot_count.store(hot_assigned as u64, Ordering::Release);
        self.warm_count
            .store(warm_assigned as u64, Ordering::Release);
        self.cold_count.store(
            (adapters.len() - hot_assigned - warm_assigned) as u64,
            Ordering::Release,
        );

        result
    }

    /// Get adapters that should be promoted
    pub fn get_promotion_candidates(&self) -> Vec<(AdapterId, ResidencyTier)> {
        let adapters = self.adapters.read();
        let mut candidates = Vec::new();

        for (id, info) in adapters.iter() {
            if let Some(optimal) = self.compute_residency_for_info(info) {
                if optimal > info.tier && self.can_promote_to(optimal) {
                    candidates.push((id.clone(), optimal));
                }
            }
        }

        candidates
    }

    /// Get adapters that should be demoted
    pub fn get_demotion_candidates(&self) -> Vec<(AdapterId, ResidencyTier)> {
        let adapters = self.adapters.read();
        let mut candidates = Vec::new();
        let now = Instant::now();

        for (id, info) in adapters.iter() {
            let idle_time = now.duration_since(info.last_access());
            if idle_time.as_secs() > self.config.demotion_idle_secs {
                let new_tier = match info.tier {
                    ResidencyTier::Hot => ResidencyTier::Warm,
                    ResidencyTier::Warm => ResidencyTier::Cold,
                    ResidencyTier::Cold => ResidencyTier::Cold,
                };

                if new_tier < info.tier {
                    candidates.push((id.clone(), new_tier));
                }
            }
        }

        candidates
    }

    /// Reset access windows for all adapters
    pub fn reset_access_windows(&self) {
        let adapters = self.adapters.read();
        for info in adapters.values() {
            info.reset_window();
        }
    }

    /// Get statistics
    pub fn stats(&self) -> ResidencyStats {
        let adapters = self.adapters.read();

        ResidencyStats {
            total_adapters: adapters.len(),
            hot_count: self.hot_count.load(Ordering::Relaxed) as usize,
            warm_count: self.warm_count.load(Ordering::Relaxed) as usize,
            cold_count: self.cold_count.load(Ordering::Relaxed) as usize,
            hot_budget: self.config.hot_tier_budget,
            warm_budget: self.config.warm_tier_budget,
        }
    }

    // Helper methods

    fn compute_initial_tier(&self) -> ResidencyTier {
        // New adapters start cold unless we have budget
        if (self.hot_count.load(Ordering::Acquire) as usize) < self.config.hot_tier_budget {
            ResidencyTier::Hot
        } else if (self.warm_count.load(Ordering::Acquire) as usize) < self.config.warm_tier_budget
        {
            ResidencyTier::Warm
        } else {
            ResidencyTier::Cold
        }
    }

    fn can_promote_to(&self, tier: ResidencyTier) -> bool {
        match tier {
            ResidencyTier::Hot => {
                (self.hot_count.load(Ordering::Acquire) as usize) < self.config.hot_tier_budget
            }
            ResidencyTier::Warm => {
                (self.warm_count.load(Ordering::Acquire) as usize) < self.config.warm_tier_budget
            }
            ResidencyTier::Cold => true,
        }
    }

    fn compute_residency_for_info(&self, info: &AdapterInfo) -> Option<ResidencyTier> {
        let accesses = info.accesses();

        if accesses >= self.config.hot_promotion_threshold as u64 {
            Some(ResidencyTier::Hot)
        } else if accesses >= self.config.warm_promotion_threshold as u64 {
            Some(ResidencyTier::Warm)
        } else {
            Some(ResidencyTier::Cold)
        }
    }

    fn increment_tier_count(&self, tier: ResidencyTier) {
        match tier {
            ResidencyTier::Hot => self.hot_count.fetch_add(1, Ordering::Relaxed),
            ResidencyTier::Warm => self.warm_count.fetch_add(1, Ordering::Relaxed),
            ResidencyTier::Cold => self.cold_count.fetch_add(1, Ordering::Relaxed),
        };
    }

    fn decrement_tier_count(&self, tier: ResidencyTier) {
        match tier {
            ResidencyTier::Hot => self.hot_count.fetch_sub(1, Ordering::Relaxed),
            ResidencyTier::Warm => self.warm_count.fetch_sub(1, Ordering::Relaxed),
            ResidencyTier::Cold => self.cold_count.fetch_sub(1, Ordering::Relaxed),
        };
    }
}

impl Default for ResidencyManager {
    fn default() -> Self {
        Self::new(ResidencyConfig::default())
    }
}

/// Result of a rebalancing operation
#[derive(Debug, Default)]
pub struct RebalanceResult {
    /// Number of adapters promoted
    pub promotions: usize,
    /// Number of adapters demoted
    pub demotions: usize,
}

/// Statistics for residency management
#[derive(Debug, Clone)]
pub struct ResidencyStats {
    /// Total number of registered adapters
    pub total_adapters: usize,
    /// Number of adapters in hot tier
    pub hot_count: usize,
    /// Number of adapters in warm tier
    pub warm_count: usize,
    /// Number of adapters in cold tier
    pub cold_count: usize,
    /// Hot tier budget
    pub hot_budget: usize,
    /// Warm tier budget
    pub warm_budget: usize,
}

impl ResidencyStats {
    /// Get hot tier utilization
    pub fn hot_utilization(&self) -> f64 {
        if self.hot_budget == 0 {
            0.0
        } else {
            self.hot_count as f64 / self.hot_budget as f64
        }
    }

    /// Get warm tier utilization
    pub fn warm_utilization(&self) -> f64 {
        if self.warm_budget == 0 {
            0.0
        } else {
            self.warm_count as f64 / self.warm_budget as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residency_tier_ordering() {
        assert!(ResidencyTier::Cold < ResidencyTier::Warm);
        assert!(ResidencyTier::Warm < ResidencyTier::Hot);
    }

    #[test]
    fn test_residency_manager_basic() {
        let manager = ResidencyManager::new(ResidencyConfig {
            hot_tier_budget: 2,
            warm_tier_budget: 3,
            ..Default::default()
        });

        // First adapters go to hot tier
        let t1 = manager.register_adapter("adapter1".into(), 1);
        let t2 = manager.register_adapter("adapter2".into(), 1);
        assert_eq!(t1, ResidencyTier::Hot);
        assert_eq!(t2, ResidencyTier::Hot);

        // Next go to warm tier
        let t3 = manager.register_adapter("adapter3".into(), 1);
        assert_eq!(t3, ResidencyTier::Warm);

        let stats = manager.stats();
        assert_eq!(stats.hot_count, 2);
        assert_eq!(stats.warm_count, 1);
    }

    #[test]
    fn test_access_tracking() {
        let manager = ResidencyManager::new(ResidencyConfig::default());

        manager.register_adapter("adapter1".into(), 1);

        // Record some accesses
        for _ in 0..15 {
            manager.record_access(&"adapter1".into());
        }

        let tier = manager.compute_residency(&"adapter1".into());
        assert_eq!(tier, Some(ResidencyTier::Hot));
    }

    #[test]
    fn test_unregister() {
        let manager = ResidencyManager::new(ResidencyConfig::default());

        manager.register_adapter("adapter1".into(), 1);
        let stats = manager.stats();
        assert_eq!(stats.total_adapters, 1);

        manager.unregister_adapter(&"adapter1".into());
        let stats = manager.stats();
        assert_eq!(stats.total_adapters, 0);
    }

    #[test]
    fn test_tier_latency() {
        assert!(ResidencyTier::Hot.load_latency() < ResidencyTier::Warm.load_latency());
        assert!(ResidencyTier::Warm.load_latency() < ResidencyTier::Cold.load_latency());
    }
}
