//! Unified Memory Pool and Paging System (ADR-006)
//!
//! This module implements a unified paged memory pool architecture designed for
//! high-performance LLM inference systems. Key features:
//!
//! - **Unified Pool**: Single arena for all tensor types (KV cache, LoRA weights, buffers)
//! - **Page-Granular Allocation**: 2MB pages (configurable) with best-fit allocation
//! - **Reference-Counted Pinning**: RAII-based pinning with automatic cleanup
//! - **LRU Eviction with Hysteresis**: Size-aware eviction with thrash prevention
//! - **Multi-Tenant Support**: Isolated memory spaces with residency tiers
//!
//! # Performance Targets (from ADR-006)
//!
//! | Operation | Target Latency | Throughput |
//! |-----------|----------------|------------|
//! | Allocate 1 page | <100ns | >10M/s |
//! | Allocate 100 pages | <1us | >1M/s |
//! | Pin page | <50ns | >20M/s |
//! | Unpin page | <50ns | >20M/s |
//! | Evict 1 page | <10us | >100K/s |
//!
//! # Architecture
//!
//! ```text
//! +------------------------------------------------------------------+
//! |                    UNIFIED MEMORY POOL                            |
//! +------------------------------------------------------------------+
//! |  Page 0   |  Page 1   |  Page 2   |   ...   |  Page N-1  |       |
//! |  [KV-A]   |  [KV-A]   |  [LoRA-1] |         |  [Temp]    |       |
//! |  pinned   |  pinned   |  pinned   |  free   |  unpinned  |       |
//! +------------------------------------------------------------------+
//!                               |
//!                               v
//! +------------------------------------------------------------------+
//! |                    PAGE METADATA TABLE                            |
//! +------------------------------------------------------------------+
//! ```
//!
//! # Example
//!
//! ```ignore
//! use ruvector_core::memory::{UnifiedPool, ContentType, PoolConfig};
//!
//! // Create a pool with default 2MB pages
//! let config = PoolConfig::default();
//! let pool = UnifiedPool::new(config)?;
//!
//! // Allocate pages for KV cache
//! let allocation = pool.allocate(4, ContentType::KvCache)?;
//!
//! // Pin pages during inference
//! let guard = pool.pin(&allocation);
//!
//! // Pages automatically unpinned and available for eviction when guard drops
//! ```

pub mod allocator;
pub mod eviction;
pub mod hysteresis;
pub mod metadata;
pub mod page;
pub mod pinning;
pub mod pool;
pub mod residency;
pub mod sync;
pub mod tenant;
pub mod thread_cache;

// Re-exports for convenient access
pub use allocator::{AllocationStrategy, BestFitAllocator, FirstFitAllocator, PageAllocator};
pub use eviction::{EvictionConfig, EvictionPolicy, LruEvictionPolicy};
pub use hysteresis::{HysteresisConfig, HysteresisController};
pub use metadata::{ContentType, PageMetadata, PageStatus};
pub use page::{Page, PageId, PageRange, PAGE_SIZE_2MB};
pub use pinning::{PinError, PinGuard, PinResult};
pub use pool::{AllocationHandle, PoolConfig, PoolError, PoolStats, UnifiedPool};
pub use residency::{ResidencyManager, ResidencyTier};
pub use sync::{EvictionCoordinator, LockHierarchy};
pub use tenant::{TenantConfig, TenantId, TenantManager};
pub use thread_cache::ThreadPageCache;

/// Default page size: 2MB (matches CUDA large page size)
pub const DEFAULT_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Minimum page size: 512KB
pub const MIN_PAGE_SIZE: usize = 512 * 1024;

/// Maximum page size: 4MB
pub const MAX_PAGE_SIZE: usize = 4 * 1024 * 1024;

/// Default alignment for GPU cache line
pub const DEFAULT_ALIGNMENT: usize = 256;

/// Default thread cache size (pages per thread)
pub const DEFAULT_THREAD_CACHE_SIZE: usize = 16;

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::{
        AllocationHandle, ContentType, EvictionPolicy, LruEvictionPolicy, PageId, PageRange,
        PageStatus, PinGuard, PoolConfig, PoolError, PoolStats, UnifiedPool,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_PAGE_SIZE, 2 * 1024 * 1024);
        assert!(MIN_PAGE_SIZE < DEFAULT_PAGE_SIZE);
        assert!(DEFAULT_PAGE_SIZE < MAX_PAGE_SIZE);
        assert!(DEFAULT_ALIGNMENT.is_power_of_two());
    }
}
