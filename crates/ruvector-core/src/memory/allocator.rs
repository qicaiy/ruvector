//! Page allocation strategies
//!
//! This module implements various page allocation strategies including
//! best-fit and first-fit algorithms with support for contiguous allocations.

use std::collections::BTreeMap;

use super::page::{PageId, PageRange};

/// Allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Best-fit: Find smallest free block that fits (lower fragmentation)
    BestFit,
    /// First-fit: Use first free block that fits (faster)
    FirstFit,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        AllocationStrategy::BestFit
    }
}

/// Trait for page allocators
pub trait PageAllocator: Send + Sync {
    /// Allocate contiguous pages
    fn allocate(&mut self, num_pages: usize) -> Option<PageRange>;

    /// Free pages back to the allocator
    fn free(&mut self, range: PageRange);

    /// Get number of free pages
    fn free_pages(&self) -> usize;

    /// Get fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented)
    fn fragmentation_ratio(&self) -> f64;
}

/// Free block in the allocator
#[derive(Debug, Clone)]
struct FreeBlock {
    start: u32,
    size: u32,
}

impl FreeBlock {
    fn new(start: u32, size: u32) -> Self {
        Self { start, size }
    }

    fn to_range(&self) -> PageRange {
        PageRange::new(PageId::new(self.start), self.size)
    }
}

/// Best-fit allocator implementation
///
/// Maintains free blocks sorted by size for efficient best-fit search.
/// Uses O(log N) operations for allocation and deallocation.
pub struct BestFitAllocator {
    /// Free blocks indexed by starting page
    blocks_by_start: BTreeMap<u32, FreeBlock>,
    /// Free blocks indexed by (size, start) for best-fit search
    blocks_by_size: BTreeMap<(u32, u32), ()>,
    /// Total number of pages
    total_pages: usize,
    /// Number of free pages
    free_page_count: usize,
}

impl BestFitAllocator {
    /// Create a new best-fit allocator
    pub fn new(total_pages: usize) -> Self {
        let mut allocator = Self {
            blocks_by_start: BTreeMap::new(),
            blocks_by_size: BTreeMap::new(),
            total_pages,
            free_page_count: total_pages,
        };

        // Initialize with single free block
        if total_pages > 0 {
            let block = FreeBlock::new(0, total_pages as u32);
            allocator.insert_block(block);
        }

        allocator
    }

    /// Insert a free block into both indexes
    fn insert_block(&mut self, block: FreeBlock) {
        self.blocks_by_size.insert((block.size, block.start), ());
        self.blocks_by_start.insert(block.start, block);
    }

    /// Remove a free block from both indexes
    fn remove_block(&mut self, block: &FreeBlock) {
        self.blocks_by_size.remove(&(block.size, block.start));
        self.blocks_by_start.remove(&block.start);
    }

    /// Find the best-fit block for the requested size
    fn find_best_fit(&self, size: u32) -> Option<FreeBlock> {
        // Find smallest block that fits using range query
        self.blocks_by_size
            .range((size, 0)..)
            .next()
            .map(|((block_size, block_start), _)| FreeBlock::new(*block_start, *block_size))
    }

    /// Merge adjacent free blocks
    fn try_merge(&mut self, block: FreeBlock) -> FreeBlock {
        let mut result = block;

        // Try merge with previous block
        if result.start > 0 {
            let prev_end = result.start;
            // First check if we can merge - copy values to avoid borrow conflict
            let merge_prev: Option<FreeBlock> = self
                .blocks_by_start
                .range(..prev_end)
                .next_back()
                .filter(|(&prev_start, prev_block)| prev_start + prev_block.size == result.start)
                .map(|(&prev_start, prev_block)| FreeBlock::new(prev_start, prev_block.size));

            // Now do the merge if applicable (borrow has ended)
            if let Some(prev_block) = merge_prev {
                let merged = FreeBlock::new(prev_block.start, prev_block.size + result.size);
                self.remove_block(&prev_block);
                result = merged;
            }
        }

        // Try merge with next block
        let next_start = result.start + result.size;
        let next_block_copy = self.blocks_by_start.get(&next_start).cloned();
        if let Some(next_block) = next_block_copy {
            // Can merge
            let merged = FreeBlock::new(result.start, result.size + next_block.size);
            self.remove_block(&next_block);
            result = merged;
        }

        result
    }
}

impl PageAllocator for BestFitAllocator {
    fn allocate(&mut self, num_pages: usize) -> Option<PageRange> {
        if num_pages == 0 || num_pages > self.free_page_count {
            return None;
        }

        let size = num_pages as u32;

        // Find best-fit block
        let block = self.find_best_fit(size)?;
        self.remove_block(&block);

        // Split if necessary
        let allocated = PageRange::new(PageId::new(block.start), size);

        if block.size > size {
            // Create remainder block
            let remainder = FreeBlock::new(block.start + size, block.size - size);
            self.insert_block(remainder);
        }

        self.free_page_count -= num_pages;

        Some(allocated)
    }

    fn free(&mut self, range: PageRange) {
        if range.is_empty() {
            return;
        }

        let block = FreeBlock::new(range.start.raw(), range.count);

        // Try to merge with adjacent blocks
        let merged = self.try_merge(block);
        self.insert_block(merged);

        self.free_page_count += range.len();
    }

    fn free_pages(&self) -> usize {
        self.free_page_count
    }

    fn fragmentation_ratio(&self) -> f64 {
        if self.free_page_count == 0 {
            return 0.0;
        }

        // Find largest free block
        let largest = self
            .blocks_by_size
            .iter()
            .next_back()
            .map(|((size, _), _)| *size as usize)
            .unwrap_or(0);

        if largest == 0 {
            return 1.0;
        }

        1.0 - (largest as f64 / self.free_page_count as f64)
    }
}

/// First-fit allocator implementation
///
/// Simple allocator that uses the first free block that fits.
/// Faster allocation but potentially higher fragmentation.
pub struct FirstFitAllocator {
    /// Free blocks indexed by starting page
    blocks: BTreeMap<u32, FreeBlock>,
    /// Total number of pages
    total_pages: usize,
    /// Number of free pages
    free_page_count: usize,
}

impl FirstFitAllocator {
    /// Create a new first-fit allocator
    pub fn new(total_pages: usize) -> Self {
        let mut allocator = Self {
            blocks: BTreeMap::new(),
            total_pages,
            free_page_count: total_pages,
        };

        // Initialize with single free block
        if total_pages > 0 {
            allocator
                .blocks
                .insert(0, FreeBlock::new(0, total_pages as u32));
        }

        allocator
    }

    /// Try to merge a block with its neighbors
    fn try_merge(&mut self, block: FreeBlock) -> FreeBlock {
        let mut result = block;

        // Try merge with previous block
        if result.start > 0 {
            if let Some((&prev_start, prev_block)) = self.blocks.range(..result.start).next_back() {
                if prev_start + prev_block.size == result.start {
                    let merged = FreeBlock::new(prev_start, prev_block.size + result.size);
                    self.blocks.remove(&prev_start);
                    result = merged;
                }
            }
        }

        // Try merge with next block
        let next_start = result.start + result.size;
        if let Some(next_block) = self.blocks.remove(&next_start) {
            result = FreeBlock::new(result.start, result.size + next_block.size);
        }

        result
    }
}

impl PageAllocator for FirstFitAllocator {
    fn allocate(&mut self, num_pages: usize) -> Option<PageRange> {
        if num_pages == 0 || num_pages > self.free_page_count {
            return None;
        }

        let size = num_pages as u32;

        // Find first block that fits
        let block = self
            .blocks
            .iter()
            .find(|(_, b)| b.size >= size)
            .map(|(&start, b)| FreeBlock::new(start, b.size))?;

        self.blocks.remove(&block.start);

        // Split if necessary
        let allocated = PageRange::new(PageId::new(block.start), size);

        if block.size > size {
            let remainder = FreeBlock::new(block.start + size, block.size - size);
            self.blocks.insert(remainder.start, remainder);
        }

        self.free_page_count -= num_pages;

        Some(allocated)
    }

    fn free(&mut self, range: PageRange) {
        if range.is_empty() {
            return;
        }

        let block = FreeBlock::new(range.start.raw(), range.count);
        let merged = self.try_merge(block);
        self.blocks.insert(merged.start, merged);

        self.free_page_count += range.len();
    }

    fn free_pages(&self) -> usize {
        self.free_page_count
    }

    fn fragmentation_ratio(&self) -> f64 {
        if self.free_page_count == 0 {
            return 0.0;
        }

        let largest = self
            .blocks
            .values()
            .map(|b| b.size as usize)
            .max()
            .unwrap_or(0);

        if largest == 0 {
            return 1.0;
        }

        1.0 - (largest as f64 / self.free_page_count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_best_fit_basic() {
        let mut allocator = BestFitAllocator::new(100);

        let r1 = allocator.allocate(10).unwrap();
        assert_eq!(r1.len(), 10);
        assert_eq!(allocator.free_pages(), 90);

        let r2 = allocator.allocate(20).unwrap();
        assert_eq!(r2.len(), 20);
        assert_eq!(allocator.free_pages(), 70);

        allocator.free(r1);
        assert_eq!(allocator.free_pages(), 80);

        allocator.free(r2);
        assert_eq!(allocator.free_pages(), 100);
    }

    #[test]
    fn test_best_fit_fragmentation() {
        let mut allocator = BestFitAllocator::new(100);

        // Create fragmentation pattern
        let r1 = allocator.allocate(10).unwrap();
        let r2 = allocator.allocate(10).unwrap();
        let r3 = allocator.allocate(10).unwrap();

        allocator.free(r2); // Create hole in middle

        // Should have fragmentation
        let ratio = allocator.fragmentation_ratio();
        assert!(ratio > 0.0);

        // Best-fit should use the 10-page hole for a 10-page request
        let r4 = allocator.allocate(10).unwrap();
        assert_eq!(r4.start.raw(), 10); // Should fill the hole

        allocator.free(r1);
        allocator.free(r3);
        allocator.free(r4);

        // Should coalesce to single block
        assert!(allocator.fragmentation_ratio() < 0.01);
    }

    #[test]
    fn test_first_fit_basic() {
        let mut allocator = FirstFitAllocator::new(100);

        let r1 = allocator.allocate(10).unwrap();
        assert_eq!(r1.len(), 10);

        let r2 = allocator.allocate(20).unwrap();
        assert_eq!(r2.len(), 20);

        allocator.free(r1);
        allocator.free(r2);

        assert_eq!(allocator.free_pages(), 100);
    }

    #[test]
    fn test_coalescing() {
        let mut allocator = BestFitAllocator::new(100);

        let r1 = allocator.allocate(25).unwrap();
        let r2 = allocator.allocate(25).unwrap();
        let r3 = allocator.allocate(25).unwrap();
        let r4 = allocator.allocate(25).unwrap();

        // Free in order that allows coalescing
        allocator.free(r2);
        allocator.free(r3);

        // Should have coalesced to one 50-page block
        let large = allocator.allocate(50).unwrap();
        assert_eq!(large.len(), 50);

        allocator.free(r1);
        allocator.free(r4);
        allocator.free(large);

        assert_eq!(allocator.free_pages(), 100);
    }

    #[test]
    fn test_out_of_memory() {
        let mut allocator = BestFitAllocator::new(100);

        assert!(allocator.allocate(101).is_none());
        assert!(allocator.allocate(0).is_none());

        let _ = allocator.allocate(100).unwrap();
        assert!(allocator.allocate(1).is_none());
    }
}
