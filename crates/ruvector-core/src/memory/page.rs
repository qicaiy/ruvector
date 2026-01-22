//! Page structure and page range management
//!
//! This module defines the core page abstraction used throughout the memory system.

use std::ops::Range;

/// Page size constant: 2MB
pub const PAGE_SIZE_2MB: usize = 2 * 1024 * 1024;

/// Unique identifier for a page within the pool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub u32);

impl PageId {
    /// Create a new page ID
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Get the byte offset for this page given a page size
    #[inline]
    pub const fn byte_offset(self, page_size: usize) -> usize {
        self.0 as usize * page_size
    }
}

impl From<u32> for PageId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<PageId> for u32 {
    fn from(id: PageId) -> Self {
        id.0
    }
}

impl std::fmt::Display for PageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Page({})", self.0)
    }
}

/// A contiguous range of pages
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageRange {
    /// Starting page ID
    pub start: PageId,
    /// Number of pages in the range
    pub count: u32,
}

impl PageRange {
    /// Create a new page range
    #[inline]
    pub const fn new(start: PageId, count: u32) -> Self {
        Self { start, count }
    }

    /// Create a single-page range
    #[inline]
    pub const fn single(page: PageId) -> Self {
        Self {
            start: page,
            count: 1,
        }
    }

    /// Check if the range is empty
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the number of pages
    #[inline]
    pub const fn len(&self) -> usize {
        self.count as usize
    }

    /// Get the ending page ID (exclusive)
    #[inline]
    pub const fn end(&self) -> PageId {
        PageId::new(self.start.0 + self.count)
    }

    /// Check if this range contains a specific page
    #[inline]
    pub fn contains(&self, page: PageId) -> bool {
        page.0 >= self.start.0 && page.0 < self.start.0 + self.count
    }

    /// Check if this range overlaps with another
    #[inline]
    pub fn overlaps(&self, other: &PageRange) -> bool {
        let self_end = self.start.0 + self.count;
        let other_end = other.start.0 + other.count;
        self.start.0 < other_end && other.start.0 < self_end
    }

    /// Check if this range is contiguous with another (can be merged)
    #[inline]
    pub fn is_adjacent(&self, other: &PageRange) -> bool {
        self.end().0 == other.start.0 || other.end().0 == self.start.0
    }

    /// Merge two adjacent ranges
    pub fn merge(&self, other: &PageRange) -> Option<PageRange> {
        if self.end().0 == other.start.0 {
            Some(PageRange::new(self.start, self.count + other.count))
        } else if other.end().0 == self.start.0 {
            Some(PageRange::new(other.start, self.count + other.count))
        } else {
            None
        }
    }

    /// Split the range at a given offset, returning (left, right)
    pub fn split_at(&self, offset: u32) -> Option<(PageRange, PageRange)> {
        if offset == 0 || offset >= self.count {
            return None;
        }

        let left = PageRange::new(self.start, offset);
        let right = PageRange::new(PageId::new(self.start.0 + offset), self.count - offset);

        Some((left, right))
    }

    /// Get the total byte size of this range
    #[inline]
    pub const fn byte_size(&self, page_size: usize) -> usize {
        self.count as usize * page_size
    }

    /// Iterator over page IDs in this range
    pub fn iter(&self) -> PageRangeIter {
        PageRangeIter {
            current: self.start.0,
            end: self.start.0 + self.count,
        }
    }
}

impl IntoIterator for PageRange {
    type Item = PageId;
    type IntoIter = PageRangeIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a PageRange {
    type Item = PageId;
    type IntoIter = PageRangeIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over page IDs in a range
pub struct PageRangeIter {
    current: u32,
    end: u32,
}

impl Iterator for PageRangeIter {
    type Item = PageId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let id = PageId::new(self.current);
            self.current += 1;
            Some(id)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.end - self.current) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PageRangeIter {}

/// Physical page representation with data pointer
#[derive(Debug)]
pub struct Page {
    /// Page ID
    pub id: PageId,
    /// Pointer to the page data (if mapped)
    data: *mut u8,
    /// Page size in bytes
    size: usize,
}

impl Page {
    /// Create a new page with the given ID and size
    ///
    /// # Safety
    /// The caller must ensure the data pointer is valid for the given size
    pub unsafe fn new(id: PageId, data: *mut u8, size: usize) -> Self {
        Self { id, data, size }
    }

    /// Create a null (unmapped) page
    pub fn null(id: PageId, size: usize) -> Self {
        Self {
            id,
            data: std::ptr::null_mut(),
            size,
        }
    }

    /// Check if the page is mapped
    #[inline]
    pub fn is_mapped(&self) -> bool {
        !self.data.is_null()
    }

    /// Get the page size
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a pointer to the page data
    ///
    /// Returns None if the page is not mapped
    #[inline]
    pub fn as_ptr(&self) -> Option<*const u8> {
        if self.is_mapped() {
            Some(self.data as *const u8)
        } else {
            None
        }
    }

    /// Get a mutable pointer to the page data
    ///
    /// Returns None if the page is not mapped
    #[inline]
    pub fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        if self.is_mapped() {
            Some(self.data)
        } else {
            None
        }
    }

    /// Get the page data as a slice
    ///
    /// # Safety
    /// The caller must ensure the page is mapped and the data is valid
    #[inline]
    pub unsafe fn as_slice(&self) -> Option<&[u8]> {
        if self.is_mapped() {
            Some(std::slice::from_raw_parts(self.data, self.size))
        } else {
            None
        }
    }

    /// Get the page data as a mutable slice
    ///
    /// # Safety
    /// The caller must ensure the page is mapped and the data is valid
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        if self.is_mapped() {
            Some(std::slice::from_raw_parts_mut(self.data, self.size))
        } else {
            None
        }
    }

    /// Zero out the page contents
    ///
    /// # Safety
    /// The caller must ensure the page is mapped
    pub unsafe fn zero(&mut self) {
        if self.is_mapped() {
            std::ptr::write_bytes(self.data, 0, self.size);
        }
    }
}

// Page cannot be Send or Sync due to raw pointer
// This is intentional - pages should be accessed through proper synchronization

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_id() {
        let id = PageId::new(42);
        assert_eq!(id.raw(), 42);
        assert_eq!(id.byte_offset(PAGE_SIZE_2MB), 42 * PAGE_SIZE_2MB);
    }

    #[test]
    fn test_page_range() {
        let range = PageRange::new(PageId::new(10), 5);
        assert_eq!(range.len(), 5);
        assert!(!range.is_empty());
        assert_eq!(range.end(), PageId::new(15));

        assert!(range.contains(PageId::new(10)));
        assert!(range.contains(PageId::new(14)));
        assert!(!range.contains(PageId::new(15)));
        assert!(!range.contains(PageId::new(9)));
    }

    #[test]
    fn test_page_range_overlap() {
        let r1 = PageRange::new(PageId::new(0), 10);
        let r2 = PageRange::new(PageId::new(5), 10);
        let r3 = PageRange::new(PageId::new(10), 5);

        assert!(r1.overlaps(&r2));
        assert!(r2.overlaps(&r1));
        assert!(!r1.overlaps(&r3));
        assert!(r1.is_adjacent(&r3));
    }

    #[test]
    fn test_page_range_merge() {
        let r1 = PageRange::new(PageId::new(0), 5);
        let r2 = PageRange::new(PageId::new(5), 5);
        let r3 = PageRange::new(PageId::new(10), 5);

        let merged = r1.merge(&r2).unwrap();
        assert_eq!(merged.start, PageId::new(0));
        assert_eq!(merged.count, 10);

        assert!(r1.merge(&r3).is_none());
    }

    #[test]
    fn test_page_range_split() {
        let range = PageRange::new(PageId::new(0), 10);
        let (left, right) = range.split_at(3).unwrap();

        assert_eq!(left.start, PageId::new(0));
        assert_eq!(left.count, 3);
        assert_eq!(right.start, PageId::new(3));
        assert_eq!(right.count, 7);

        assert!(range.split_at(0).is_none());
        assert!(range.split_at(10).is_none());
    }

    #[test]
    fn test_page_range_iter() {
        let range = PageRange::new(PageId::new(5), 3);
        let pages: Vec<_> = range.iter().collect();

        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0], PageId::new(5));
        assert_eq!(pages[1], PageId::new(6));
        assert_eq!(pages[2], PageId::new(7));
    }

    #[test]
    fn test_page_null() {
        let page = Page::null(PageId::new(0), PAGE_SIZE_2MB);
        assert!(!page.is_mapped());
        assert!(page.as_ptr().is_none());
    }
}
