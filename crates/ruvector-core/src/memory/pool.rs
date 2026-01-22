//! Unified Memory Pool Implementation (ADR-006)
//!
//! This module provides the main unified memory pool that manages all page allocations,
//! pinning, and eviction for the memory system.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use super::allocator::{AllocationStrategy, BestFitAllocator, PageAllocator};
use super::eviction::{EvictionConfig, EvictionPolicy, LruEvictionPolicy};
use super::hysteresis::{HysteresisConfig, HysteresisController};
use super::metadata::{ContentType, MetadataTable, PageMetadata, PageStatus};
use super::page::{Page, PageId, PageRange};
use super::pinning::PinGuard;
use super::sync::EvictionCoordinator;
use super::thread_cache::ThreadPageCache;
use super::{DEFAULT_ALIGNMENT, DEFAULT_PAGE_SIZE, DEFAULT_THREAD_CACHE_SIZE};

/// Configuration for the unified memory pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Page size in bytes (default: 2MB)
    pub page_size: usize,
    /// Total number of pages
    pub total_pages: usize,
    /// Alignment for page boundaries (default: 256 bytes)
    pub alignment: usize,
    /// Allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Thread cache size (pages per thread)
    pub thread_cache_size: usize,
    /// Eviction configuration
    pub eviction: EvictionConfig,
    /// Hysteresis configuration
    pub hysteresis: HysteresisConfig,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            total_pages: 4096, // 8GB with 2MB pages
            alignment: DEFAULT_ALIGNMENT,
            allocation_strategy: AllocationStrategy::BestFit,
            thread_cache_size: DEFAULT_THREAD_CACHE_SIZE,
            eviction: EvictionConfig::default(),
            hysteresis: HysteresisConfig::default(),
        }
    }
}

impl PoolConfig {
    /// Create a config for the given total memory size
    pub fn with_memory_size(total_bytes: usize) -> Self {
        let page_size = DEFAULT_PAGE_SIZE;
        let total_pages = total_bytes / page_size;
        Self {
            total_pages,
            ..Default::default()
        }
    }

    /// Set the page size
    pub fn page_size(mut self, size: usize) -> Self {
        self.page_size = size;
        self
    }

    /// Set the allocation strategy
    pub fn allocation_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.allocation_strategy = strategy;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), PoolError> {
        if self.page_size < super::MIN_PAGE_SIZE || self.page_size > super::MAX_PAGE_SIZE {
            return Err(PoolError::InvalidConfig(format!(
                "Page size must be between {} and {} bytes",
                super::MIN_PAGE_SIZE,
                super::MAX_PAGE_SIZE
            )));
        }

        if !self.page_size.is_power_of_two() {
            return Err(PoolError::InvalidConfig(
                "Page size must be a power of 2".to_string(),
            ));
        }

        if !self.alignment.is_power_of_two() {
            return Err(PoolError::InvalidConfig(
                "Alignment must be a power of 2".to_string(),
            ));
        }

        if self.total_pages == 0 {
            return Err(PoolError::InvalidConfig(
                "Total pages must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Error type for pool operations
#[derive(Debug)]
pub enum PoolError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Out of memory
    OutOfMemory { requested: usize, available: usize },
    /// Allocation failed
    AllocationFailed(String),
    /// Page not found
    PageNotFound(PageId),
    /// Pin error
    PinError(super::pinning::PinError),
    /// System error
    SystemError(String),
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            PoolError::OutOfMemory {
                requested,
                available,
            } => write!(
                f,
                "Out of memory: requested {} pages, only {} available",
                requested, available
            ),
            PoolError::AllocationFailed(msg) => write!(f, "Allocation failed: {}", msg),
            PoolError::PageNotFound(id) => write!(f, "Page not found: {}", id),
            PoolError::PinError(e) => write!(f, "Pin error: {}", e),
            PoolError::SystemError(msg) => write!(f, "System error: {}", msg),
        }
    }
}

impl std::error::Error for PoolError {}

impl From<super::pinning::PinError> for PoolError {
    fn from(e: super::pinning::PinError) -> Self {
        PoolError::PinError(e)
    }
}

impl From<super::metadata::PinError> for PoolError {
    fn from(e: super::metadata::PinError) -> Self {
        PoolError::PinError(super::pinning::PinError::from(e))
    }
}

/// Statistics for the memory pool
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total pages in the pool
    pub total_pages: usize,
    /// Number of free pages
    pub free_pages: usize,
    /// Number of allocated (unpinned) pages
    pub unpinned_pages: usize,
    /// Number of pinned pages
    pub pinned_pages: usize,
    /// Total allocations
    pub total_allocations: u64,
    /// Total deallocations
    pub total_deallocations: u64,
    /// Total evictions
    pub total_evictions: u64,
    /// Memory utilization (0.0 - 1.0)
    pub utilization: f64,
    /// Fragmentation ratio (largest free block / total free)
    pub fragmentation_ratio: f64,
}

/// Unified memory pool implementation
pub struct UnifiedPool {
    /// Pool configuration
    config: PoolConfig,
    /// Raw memory buffer
    buffer: NonNull<u8>,
    /// Memory layout for deallocation
    layout: Layout,
    /// Page metadata table
    metadata: MetadataTable,
    /// Page allocator
    allocator: RwLock<BestFitAllocator>,
    /// Eviction policy
    eviction_policy: Mutex<LruEvictionPolicy>,
    /// Eviction coordinator
    eviction_coordinator: EvictionCoordinator,
    /// Hysteresis controller
    hysteresis: HysteresisController,
    /// Allocation sequence counter
    allocation_seq: AtomicU64,
    /// Statistics counters
    total_allocations: AtomicU64,
    total_deallocations: AtomicU64,
    total_evictions: AtomicU64,
}

// Safety: UnifiedPool is Send + Sync because:
// - buffer is only accessed through properly synchronized methods
// - metadata uses atomic operations
// - allocator is protected by RwLock
// - eviction_policy is protected by Mutex
unsafe impl Send for UnifiedPool {}
unsafe impl Sync for UnifiedPool {}

impl UnifiedPool {
    /// Create a new unified memory pool
    pub fn new(config: PoolConfig) -> Result<Arc<Self>, PoolError> {
        config.validate()?;

        // Calculate total memory size
        let total_size = config.total_pages * config.page_size;

        // Allocate the memory buffer with proper alignment
        let layout = Layout::from_size_align(total_size, config.alignment)
            .map_err(|e| PoolError::SystemError(format!("Invalid layout: {}", e)))?;

        let buffer = unsafe { alloc(layout) };
        if buffer.is_null() {
            return Err(PoolError::SystemError(
                "Failed to allocate memory".to_string(),
            ));
        }

        let buffer = unsafe { NonNull::new_unchecked(buffer) };

        // Initialize metadata table
        let metadata = MetadataTable::new(config.total_pages);

        // Initialize allocator
        let allocator = BestFitAllocator::new(config.total_pages);

        // Initialize eviction policy
        let eviction_policy = LruEvictionPolicy::new(config.eviction.clone());

        // Initialize hysteresis controller
        let hysteresis = HysteresisController::new(config.hysteresis.clone());

        // Initialize eviction coordinator
        let eviction_coordinator = EvictionCoordinator::new();

        Ok(Arc::new(Self {
            config,
            buffer,
            layout,
            metadata,
            allocator: RwLock::new(allocator),
            eviction_policy: Mutex::new(eviction_policy),
            eviction_coordinator,
            hysteresis,
            allocation_seq: AtomicU64::new(0),
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            total_evictions: AtomicU64::new(0),
        }))
    }

    /// Create a pool with default configuration
    pub fn with_defaults() -> Result<Arc<Self>, PoolError> {
        Self::new(PoolConfig::default())
    }

    /// Get the pool configuration
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// Get current timestamp in microseconds
    #[inline]
    fn current_timestamp_us(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0)
    }

    /// Allocate contiguous pages
    pub fn allocate(
        self: &Arc<Self>,
        num_pages: usize,
        content_type: ContentType,
    ) -> Result<AllocationHandle, PoolError> {
        self.allocate_with_owner(num_pages, content_type, 0, 0)
    }

    /// Allocate contiguous pages with owner and tenant
    pub fn allocate_with_owner(
        self: &Arc<Self>,
        num_pages: usize,
        content_type: ContentType,
        owner_id: u64,
        tenant_id: u32,
    ) -> Result<AllocationHandle, PoolError> {
        if num_pages == 0 {
            return Err(PoolError::InvalidConfig(
                "Cannot allocate 0 pages".to_string(),
            ));
        }

        // Try allocation from allocator
        let range = {
            let mut allocator = self.allocator.write();

            match allocator.allocate(num_pages) {
                Some(range) => range,
                None => {
                    // Need eviction
                    drop(allocator);
                    self.evict_for_allocation(num_pages)?;

                    // Retry allocation
                    let mut allocator = self.allocator.write();
                    allocator
                        .allocate(num_pages)
                        .ok_or_else(|| PoolError::OutOfMemory {
                            requested: num_pages,
                            available: allocator.free_pages(),
                        })?
                }
            }
        };

        // Update metadata for allocated pages
        let seq = self.allocation_seq.fetch_add(1, Ordering::Relaxed);
        let timestamp = self.current_timestamp_us();

        for page_id in range.iter() {
            if let Some(meta) = self.metadata.get(page_id) {
                meta.try_allocate(content_type, owner_id, tenant_id, seq, timestamp);
            }
        }

        // Update eviction policy
        {
            let mut policy = self.eviction_policy.lock();
            for page_id in range.iter() {
                policy.touch(page_id, timestamp);
            }
        }

        self.total_allocations.fetch_add(1, Ordering::Relaxed);

        Ok(AllocationHandle {
            pool: Arc::clone(self),
            range,
            content_type,
        })
    }

    /// Free pages back to the pool
    pub fn free(&self, range: &PageRange) {
        // Reset metadata
        for page_id in range.iter() {
            if let Some(meta) = self.metadata.get(page_id) {
                meta.reset();
            }
        }

        // Return to allocator
        {
            let mut allocator = self.allocator.write();
            allocator.free(range.clone());
        }

        // Remove from eviction tracking
        {
            let mut policy = self.eviction_policy.lock();
            for page_id in range.iter() {
                policy.remove(page_id);
            }
        }

        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
    }

    /// Pin pages to prevent eviction
    pub fn pin(self: &Arc<Self>, range: &PageRange) -> Result<PinGuard, PoolError> {
        for page_id in range.iter() {
            let meta = self
                .metadata
                .get(page_id)
                .ok_or(PoolError::PageNotFound(page_id))?;

            meta.pin()?;
        }

        Ok(PinGuard::new(Arc::clone(self), range.clone()))
    }

    /// Unpin pages (called by PinGuard on drop)
    pub(crate) fn unpin(&self, range: &PageRange) {
        for page_id in range.iter() {
            if let Some(meta) = self.metadata.get(page_id) {
                meta.unpin();
            }
        }
    }

    /// Touch pages to update access time
    pub fn touch(&self, range: &PageRange) {
        let timestamp = self.current_timestamp_us();

        for page_id in range.iter() {
            if let Some(meta) = self.metadata.get(page_id) {
                meta.touch(timestamp);
            }
        }

        let mut policy = self.eviction_policy.lock();
        for page_id in range.iter() {
            policy.touch(page_id, timestamp);
        }
    }

    /// Evict pages to free up space
    fn evict_for_allocation(&self, required_pages: usize) -> Result<(), PoolError> {
        // Use hysteresis to determine actual eviction target
        let target = self.hysteresis.eviction_target(required_pages);

        self.eviction_coordinator
            .maybe_evict(|| {
                let mut evicted = 0;

                while evicted < target {
                    // Get next victim from eviction policy
                    let victim = {
                        let mut policy = self.eviction_policy.lock();
                        policy.select_victim(&self.metadata)
                    };

                    let victim = match victim {
                        Some(v) => v,
                        None => break, // No more evictable pages
                    };

                    // Try to evict
                    if let Some(meta) = self.metadata.get(victim) {
                        if meta.try_evict() {
                            // Return to allocator
                            let mut allocator = self.allocator.write();
                            allocator.free(PageRange::single(victim));

                            // Remove from eviction tracking
                            let mut policy = self.eviction_policy.lock();
                            policy.remove(victim);

                            evicted += 1;
                            self.total_evictions.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                evicted >= required_pages
            })
            .then_some(())
            .ok_or_else(|| {
                let allocator = self.allocator.read();
                PoolError::OutOfMemory {
                    requested: required_pages,
                    available: allocator.free_pages(),
                }
            })
    }

    /// Get a raw pointer to a page's data
    ///
    /// # Safety
    /// The caller must ensure the page is allocated and properly pinned
    #[inline]
    pub unsafe fn page_ptr(&self, page_id: PageId) -> *mut u8 {
        let offset = page_id.byte_offset(self.config.page_size);
        self.buffer.as_ptr().add(offset)
    }

    /// Get page metadata
    #[inline]
    pub fn metadata(&self, page_id: PageId) -> Option<&PageMetadata> {
        self.metadata.get(page_id)
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let counts = self.metadata.count_by_status();
        let allocator = self.allocator.read();

        PoolStats {
            total_pages: self.config.total_pages,
            free_pages: counts.free,
            unpinned_pages: counts.unpinned,
            pinned_pages: counts.pinned,
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.total_deallocations.load(Ordering::Relaxed),
            total_evictions: self.total_evictions.load(Ordering::Relaxed),
            utilization: counts.utilization(),
            fragmentation_ratio: allocator.fragmentation_ratio(),
        }
    }

    /// Get the number of free pages
    #[inline]
    pub fn free_pages(&self) -> usize {
        self.allocator.read().free_pages()
    }

    /// Get the total number of pages
    #[inline]
    pub fn total_pages(&self) -> usize {
        self.config.total_pages
    }

    /// Get the page size
    #[inline]
    pub fn page_size(&self) -> usize {
        self.config.page_size
    }

    /// Get total memory size in bytes
    #[inline]
    pub fn total_memory(&self) -> usize {
        self.config.total_pages * self.config.page_size
    }
}

impl Drop for UnifiedPool {
    fn drop(&mut self) {
        // Deallocate the memory buffer
        unsafe {
            dealloc(self.buffer.as_ptr(), self.layout);
        }
    }
}

/// RAII handle for allocated pages
pub struct AllocationHandle {
    pool: Arc<UnifiedPool>,
    range: PageRange,
    content_type: ContentType,
}

impl AllocationHandle {
    /// Get the page range
    #[inline]
    pub fn range(&self) -> &PageRange {
        &self.range
    }

    /// Get the content type
    #[inline]
    pub fn content_type(&self) -> ContentType {
        self.content_type
    }

    /// Get the number of pages
    #[inline]
    pub fn num_pages(&self) -> usize {
        self.range.len()
    }

    /// Get the byte size
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.range.byte_size(self.pool.page_size())
    }

    /// Pin the allocation
    pub fn pin(&self) -> Result<PinGuard, PoolError> {
        self.pool.pin(&self.range)
    }

    /// Touch to update access time
    pub fn touch(&self) {
        self.pool.touch(&self.range);
    }

    /// Get a raw pointer to the data
    ///
    /// # Safety
    /// The caller must ensure the allocation is pinned before accessing
    #[inline]
    pub unsafe fn as_ptr(&self) -> *mut u8 {
        self.pool.page_ptr(self.range.start)
    }

    /// Get the allocation as a slice
    ///
    /// # Safety
    /// The caller must ensure the allocation is pinned and initialized
    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.as_ptr(), self.byte_size())
    }

    /// Get the allocation as a mutable slice
    ///
    /// # Safety
    /// The caller must ensure the allocation is pinned and have exclusive access
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.as_ptr(), self.byte_size())
    }
}

impl Drop for AllocationHandle {
    fn drop(&mut self) {
        self.pool.free(&self.range);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let config = PoolConfig {
            page_size: 4096, // 4KB for testing
            total_pages: 100,
            ..Default::default()
        };

        let pool = UnifiedPool::new(config).unwrap();
        assert_eq!(pool.total_pages(), 100);
        assert_eq!(pool.free_pages(), 100);
    }

    #[test]
    fn test_basic_allocation() {
        let config = PoolConfig {
            page_size: 4096,
            total_pages: 100,
            ..Default::default()
        };

        let pool = UnifiedPool::new(config).unwrap();

        let handle = pool.allocate(10, ContentType::KvCache).unwrap();
        assert_eq!(handle.num_pages(), 10);
        assert_eq!(pool.free_pages(), 90);

        drop(handle);
        assert_eq!(pool.free_pages(), 100);
    }

    #[test]
    fn test_pinning() {
        let config = PoolConfig {
            page_size: 4096,
            total_pages: 100,
            ..Default::default()
        };

        let pool = UnifiedPool::new(config).unwrap();
        let handle = pool.allocate(5, ContentType::TempBuffer).unwrap();

        // Pin the allocation
        let guard = handle.pin().unwrap();

        // Verify pages are pinned
        for page_id in handle.range().iter() {
            let meta = pool.metadata(page_id).unwrap();
            assert!(meta.is_pinned());
        }

        // Drop guard
        drop(guard);

        // Verify pages are unpinned
        for page_id in handle.range().iter() {
            let meta = pool.metadata(page_id).unwrap();
            assert!(!meta.is_pinned());
        }
    }

    #[test]
    fn test_stats() {
        let config = PoolConfig {
            page_size: 4096,
            total_pages: 100,
            ..Default::default()
        };

        let pool = UnifiedPool::new(config).unwrap();

        let _h1 = pool.allocate(30, ContentType::KvCache).unwrap();
        let h2 = pool.allocate(20, ContentType::LoraWeight).unwrap();
        let _guard = h2.pin().unwrap();

        let stats = pool.stats();
        assert_eq!(stats.total_pages, 100);
        assert_eq!(stats.free_pages, 50);
        assert_eq!(stats.total_allocations, 2);
    }

    #[test]
    fn test_invalid_config() {
        let config = PoolConfig {
            page_size: 1000, // Not power of 2
            total_pages: 100,
            ..Default::default()
        };

        assert!(UnifiedPool::new(config).is_err());
    }
}
