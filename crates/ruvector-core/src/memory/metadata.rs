//! Page metadata tracking
//!
//! This module provides per-page metadata for status tracking, content classification,
//! reference counting, and access time tracking.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use super::page::PageId;

/// Status of a page in the memory pool
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageStatus {
    /// Page is free and available for allocation
    Free = 0,
    /// Page is allocated but not pinned (eligible for eviction)
    Unpinned = 1,
    /// Page is pinned and cannot be evicted
    Pinned = 2,
}

impl PageStatus {
    /// Convert from raw u8 value
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(PageStatus::Free),
            1 => Some(PageStatus::Unpinned),
            2 => Some(PageStatus::Pinned),
            _ => None,
        }
    }
}

impl From<PageStatus> for u8 {
    fn from(status: PageStatus) -> Self {
        status as u8
    }
}

/// Content type stored in a page
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ContentType {
    /// Empty/uninitialized
    Empty = 0,
    /// Key-value cache for attention
    KvCache = 1,
    /// LoRA adapter A/B matrices
    LoraWeight = 2,
    /// Scratch space for computation
    TempBuffer = 3,
    /// Intermediate activations
    Activation = 4,
    /// Gradient buffers (training)
    Gradient = 5,
}

impl ContentType {
    /// Get the eviction priority (higher = less likely to evict)
    #[inline]
    pub const fn eviction_priority(self) -> u8 {
        match self {
            ContentType::Empty => 0,
            ContentType::TempBuffer => 1,
            ContentType::Activation => 2,
            ContentType::Gradient => 2,
            ContentType::LoraWeight => 3, // Cold LoRA; warm gets +1
            ContentType::KvCache => 5,
        }
    }

    /// Convert from raw u8 value
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(ContentType::Empty),
            1 => Some(ContentType::KvCache),
            2 => Some(ContentType::LoraWeight),
            3 => Some(ContentType::TempBuffer),
            4 => Some(ContentType::Activation),
            5 => Some(ContentType::Gradient),
            _ => None,
        }
    }

    /// Check if this content type requires manual unpin
    #[inline]
    pub const fn requires_manual_unpin(self) -> bool {
        matches!(self, ContentType::LoraWeight)
    }
}

impl From<ContentType> for u8 {
    fn from(content_type: ContentType) -> Self {
        content_type as u8
    }
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentType::Empty => write!(f, "Empty"),
            ContentType::KvCache => write!(f, "KV Cache"),
            ContentType::LoraWeight => write!(f, "LoRA Weight"),
            ContentType::TempBuffer => write!(f, "Temp Buffer"),
            ContentType::Activation => write!(f, "Activation"),
            ContentType::Gradient => write!(f, "Gradient"),
        }
    }
}

/// Atomic page metadata for lock-free access
#[repr(C, align(64))] // Cache line aligned
pub struct PageMetadata {
    /// Page status (FREE, UNPINNED, PINNED)
    status: AtomicU8,
    /// Content type stored in the page
    content_type: AtomicU8,
    /// Reference count for pinning
    ref_count: AtomicU32,
    /// Last access timestamp (monotonic, in microseconds from epoch)
    last_access: AtomicU64,
    /// Owner ID (request/adapter ID)
    owner_id: AtomicU64,
    /// Tenant ID for multi-tenant isolation
    tenant_id: AtomicU32,
    /// Allocation sequence number (for LRU ordering)
    allocation_seq: AtomicU64,
    // Padding to fill cache line
    _padding: [u8; 20],
}

impl PageMetadata {
    /// Create new metadata for a free page
    pub fn new_free() -> Self {
        Self {
            status: AtomicU8::new(PageStatus::Free as u8),
            content_type: AtomicU8::new(ContentType::Empty as u8),
            ref_count: AtomicU32::new(0),
            last_access: AtomicU64::new(0),
            owner_id: AtomicU64::new(0),
            tenant_id: AtomicU32::new(0),
            allocation_seq: AtomicU64::new(0),
            _padding: [0; 20],
        }
    }

    /// Get the current status
    #[inline]
    pub fn status(&self) -> PageStatus {
        PageStatus::from_u8(self.status.load(Ordering::Acquire)).unwrap_or(PageStatus::Free)
    }

    /// Set the status
    #[inline]
    pub fn set_status(&self, status: PageStatus) {
        self.status.store(status as u8, Ordering::Release);
    }

    /// Get the content type
    #[inline]
    pub fn content_type(&self) -> ContentType {
        ContentType::from_u8(self.content_type.load(Ordering::Acquire))
            .unwrap_or(ContentType::Empty)
    }

    /// Set the content type
    #[inline]
    pub fn set_content_type(&self, content_type: ContentType) {
        self.content_type
            .store(content_type as u8, Ordering::Release);
    }

    /// Get the reference count
    #[inline]
    pub fn ref_count(&self) -> u32 {
        self.ref_count.load(Ordering::Acquire)
    }

    /// Get the last access timestamp in microseconds
    #[inline]
    pub fn last_access_us(&self) -> u64 {
        self.last_access.load(Ordering::Acquire)
    }

    /// Update the last access timestamp
    #[inline]
    pub fn touch(&self, timestamp_us: u64) {
        self.last_access.store(timestamp_us, Ordering::Release);
    }

    /// Get the owner ID
    #[inline]
    pub fn owner_id(&self) -> u64 {
        self.owner_id.load(Ordering::Acquire)
    }

    /// Set the owner ID
    #[inline]
    pub fn set_owner_id(&self, owner: u64) {
        self.owner_id.store(owner, Ordering::Release);
    }

    /// Get the tenant ID
    #[inline]
    pub fn tenant_id(&self) -> u32 {
        self.tenant_id.load(Ordering::Acquire)
    }

    /// Set the tenant ID
    #[inline]
    pub fn set_tenant_id(&self, tenant: u32) {
        self.tenant_id.store(tenant, Ordering::Release);
    }

    /// Get the allocation sequence number
    #[inline]
    pub fn allocation_seq(&self) -> u64 {
        self.allocation_seq.load(Ordering::Acquire)
    }

    /// Set the allocation sequence number
    #[inline]
    pub fn set_allocation_seq(&self, seq: u64) {
        self.allocation_seq.store(seq, Ordering::Release);
    }

    /// Increment reference count (pin)
    ///
    /// Returns Ok(new_count) on success, Err if page is free
    #[inline]
    pub fn pin(&self) -> Result<u32, PinError> {
        loop {
            let count = self.ref_count.load(Ordering::Acquire);
            let status = self.status.load(Ordering::Acquire);

            if status == PageStatus::Free as u8 {
                return Err(PinError::PageFreed);
            }

            match self.ref_count.compare_exchange_weak(
                count,
                count + 1,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Update status to pinned
                    self.status
                        .store(PageStatus::Pinned as u8, Ordering::Release);
                    return Ok(count + 1);
                }
                Err(_) => continue,
            }
        }
    }

    /// Decrement reference count (unpin)
    ///
    /// Returns the new reference count
    #[inline]
    pub fn unpin(&self) -> u32 {
        let prev = self.ref_count.fetch_sub(1, Ordering::Release);
        debug_assert!(prev > 0, "Unpin called on page with zero ref count");

        if prev == 1 {
            // Transition from pinned to unpinned
            self.status
                .store(PageStatus::Unpinned as u8, Ordering::Release);
        }

        prev - 1
    }

    /// Try to transition from Unpinned to Free for eviction
    ///
    /// Returns true if successful
    #[inline]
    pub fn try_evict(&self) -> bool {
        // Only evict if ref_count is 0 and status is Unpinned
        let count = self.ref_count.load(Ordering::Acquire);
        if count > 0 {
            return false;
        }

        self.status
            .compare_exchange(
                PageStatus::Unpinned as u8,
                PageStatus::Free as u8,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Allocate this page (transition from Free to Unpinned)
    ///
    /// Returns true if successful
    #[inline]
    pub fn try_allocate(
        &self,
        content_type: ContentType,
        owner_id: u64,
        tenant_id: u32,
        seq: u64,
        timestamp_us: u64,
    ) -> bool {
        if self
            .status
            .compare_exchange(
                PageStatus::Free as u8,
                PageStatus::Unpinned as u8,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return false;
        }

        self.content_type
            .store(content_type as u8, Ordering::Release);
        self.owner_id.store(owner_id, Ordering::Release);
        self.tenant_id.store(tenant_id, Ordering::Release);
        self.allocation_seq.store(seq, Ordering::Release);
        self.last_access.store(timestamp_us, Ordering::Release);

        true
    }

    /// Reset metadata to free state
    pub fn reset(&self) {
        self.status.store(PageStatus::Free as u8, Ordering::Release);
        self.content_type
            .store(ContentType::Empty as u8, Ordering::Release);
        self.ref_count.store(0, Ordering::Release);
        self.last_access.store(0, Ordering::Release);
        self.owner_id.store(0, Ordering::Release);
        self.tenant_id.store(0, Ordering::Release);
        self.allocation_seq.store(0, Ordering::Release);
    }

    /// Check if page is eligible for eviction
    #[inline]
    pub fn is_evictable(&self) -> bool {
        self.status() == PageStatus::Unpinned && self.ref_count() == 0
    }

    /// Check if page is free
    #[inline]
    pub fn is_free(&self) -> bool {
        self.status() == PageStatus::Free
    }

    /// Check if page is pinned
    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.status() == PageStatus::Pinned || self.ref_count() > 0
    }
}

impl Default for PageMetadata {
    fn default() -> Self {
        Self::new_free()
    }
}

/// Error type for pin operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinError {
    /// Page has been freed
    PageFreed,
}

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinError::PageFreed => write!(f, "Cannot pin freed page"),
        }
    }
}

impl std::error::Error for PinError {}

/// Metadata table for all pages in the pool
pub struct MetadataTable {
    /// Metadata array (one entry per page)
    entries: Vec<PageMetadata>,
    /// Total number of pages
    num_pages: usize,
}

impl MetadataTable {
    /// Create a new metadata table
    pub fn new(num_pages: usize) -> Self {
        let mut entries = Vec::with_capacity(num_pages);
        for _ in 0..num_pages {
            entries.push(PageMetadata::new_free());
        }

        Self { entries, num_pages }
    }

    /// Get metadata for a page
    #[inline]
    pub fn get(&self, page: PageId) -> Option<&PageMetadata> {
        self.entries.get(page.raw() as usize)
    }

    /// Get the number of pages
    #[inline]
    pub fn len(&self) -> usize {
        self.num_pages
    }

    /// Check if table is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_pages == 0
    }

    /// Count pages by status
    pub fn count_by_status(&self) -> StatusCounts {
        let mut counts = StatusCounts::default();

        for entry in &self.entries {
            match entry.status() {
                PageStatus::Free => counts.free += 1,
                PageStatus::Unpinned => counts.unpinned += 1,
                PageStatus::Pinned => counts.pinned += 1,
            }
        }

        counts
    }

    /// Iterate over all evictable pages
    pub fn evictable_pages(&self) -> impl Iterator<Item = PageId> + '_ {
        self.entries.iter().enumerate().filter_map(|(i, meta)| {
            if meta.is_evictable() {
                Some(PageId::new(i as u32))
            } else {
                None
            }
        })
    }

    /// Iterate over all free pages
    pub fn free_pages(&self) -> impl Iterator<Item = PageId> + '_ {
        self.entries.iter().enumerate().filter_map(|(i, meta)| {
            if meta.is_free() {
                Some(PageId::new(i as u32))
            } else {
                None
            }
        })
    }
}

/// Counts of pages by status
#[derive(Debug, Default, Clone, Copy)]
pub struct StatusCounts {
    pub free: usize,
    pub unpinned: usize,
    pub pinned: usize,
}

impl StatusCounts {
    /// Total pages
    pub fn total(&self) -> usize {
        self.free + self.unpinned + self.pinned
    }

    /// Allocated pages (unpinned + pinned)
    pub fn allocated(&self) -> usize {
        self.unpinned + self.pinned
    }

    /// Utilization ratio
    pub fn utilization(&self) -> f64 {
        if self.total() == 0 {
            0.0
        } else {
            self.allocated() as f64 / self.total() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_status_conversion() {
        assert_eq!(PageStatus::from_u8(0), Some(PageStatus::Free));
        assert_eq!(PageStatus::from_u8(1), Some(PageStatus::Unpinned));
        assert_eq!(PageStatus::from_u8(2), Some(PageStatus::Pinned));
        assert_eq!(PageStatus::from_u8(3), None);
    }

    #[test]
    fn test_content_type_priority() {
        assert!(
            ContentType::KvCache.eviction_priority() > ContentType::TempBuffer.eviction_priority()
        );
        assert!(
            ContentType::LoraWeight.eviction_priority()
                > ContentType::Activation.eviction_priority()
        );
    }

    #[test]
    fn test_metadata_pin_unpin() {
        let meta = PageMetadata::new_free();

        // Allocate the page
        assert!(meta.try_allocate(ContentType::KvCache, 1, 0, 0, 1000));
        assert_eq!(meta.status(), PageStatus::Unpinned);

        // Pin
        assert!(meta.pin().is_ok());
        assert_eq!(meta.ref_count(), 1);
        assert_eq!(meta.status(), PageStatus::Pinned);

        // Pin again
        assert!(meta.pin().is_ok());
        assert_eq!(meta.ref_count(), 2);

        // Unpin
        meta.unpin();
        assert_eq!(meta.ref_count(), 1);
        assert_eq!(meta.status(), PageStatus::Pinned);

        // Unpin to zero
        meta.unpin();
        assert_eq!(meta.ref_count(), 0);
        assert_eq!(meta.status(), PageStatus::Unpinned);
    }

    #[test]
    fn test_metadata_eviction() {
        let meta = PageMetadata::new_free();

        // Allocate
        assert!(meta.try_allocate(ContentType::TempBuffer, 1, 0, 0, 1000));

        // Should be evictable (unpinned, ref_count 0)
        assert!(meta.is_evictable());
        assert!(meta.try_evict());
        assert_eq!(meta.status(), PageStatus::Free);
    }

    #[test]
    fn test_metadata_cannot_evict_pinned() {
        let meta = PageMetadata::new_free();

        // Allocate and pin
        assert!(meta.try_allocate(ContentType::KvCache, 1, 0, 0, 1000));
        meta.pin().unwrap();

        // Cannot evict pinned page
        assert!(!meta.is_evictable());
        assert!(!meta.try_evict());
    }

    #[test]
    fn test_metadata_table() {
        let table = MetadataTable::new(100);
        assert_eq!(table.len(), 100);

        let counts = table.count_by_status();
        assert_eq!(counts.free, 100);
        assert_eq!(counts.allocated(), 0);
    }
}
