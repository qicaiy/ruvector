//! Witness types for cut certification
//!
//! A witness represents a connected set U ⊆ V with its boundary δ(U).
//! The witness certifies that a proper cut exists with value |δ(U)|.
//!
//! # Representation
//!
//! Witnesses use an implicit representation for memory efficiency:
//! - **Seed vertex**: The starting vertex that defines the connected component
//! - **Membership bitmap**: Compressed bitmap indicating which vertices are in U
//! - **Boundary size**: Pre-computed value |δ(U)| for O(1) queries
//! - **Hash**: Fast equality checking without full comparison
//!
//! # Performance
//!
//! - `WitnessHandle` uses `Arc` for cheap cloning (O(1))
//! - `contains()` is O(1) via bitmap lookup
//! - `boundary_size()` is O(1) via cached value
//! - `materialize_partition()` is O(|V|) and should be used sparingly

use crate::graph::VertexId;
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Handle to a witness (cheap to clone)
///
/// This is the primary type for passing witnesses around. It uses an `Arc`
/// internally so cloning is O(1) and witnesses can be shared across threads.
///
/// # Examples
///
/// ```
/// use ruvector_mincut::instance::witness::WitnessHandle;
/// use roaring::RoaringBitmap;
///
/// let mut membership = RoaringBitmap::new();
/// membership.insert(1);
/// membership.insert(2);
/// membership.insert(3);
///
/// let witness = WitnessHandle::new(1, membership, 4);
/// assert!(witness.contains(1));
/// assert!(witness.contains(2));
/// assert!(!witness.contains(5));
/// assert_eq!(witness.boundary_size(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct WitnessHandle {
    inner: Arc<ImplicitWitness>,
}

/// Implicit representation of a cut witness
///
/// The witness represents a connected set U ⊆ V where:
/// - U contains the seed vertex
/// - |δ(U)| = boundary_size
/// - membership[v] = true iff v ∈ U
#[derive(Debug)]
pub struct ImplicitWitness {
    /// Seed vertex that defines the cut (always in U)
    pub seed: VertexId,
    /// Membership bitmap (vertex v is in U iff bit v is set)
    pub membership: RoaringBitmap,
    /// Current boundary size |δ(U)|
    pub boundary_size: u64,
    /// Hash for quick equality checks
    pub hash: u64,
}

impl WitnessHandle {
    /// Create a new witness handle
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed vertex defining this cut (must be in membership)
    /// * `membership` - Bitmap of vertices in the cut set U
    /// * `boundary_size` - The size of the boundary |δ(U)|
    ///
    /// # Panics
    ///
    /// Panics if the seed vertex is not in the membership set (debug builds only)
    ///
    /// # Examples
    ///
    /// ```
    /// use ruvector_mincut::instance::witness::WitnessHandle;
    /// use roaring::RoaringBitmap;
    ///
    /// let mut membership = RoaringBitmap::new();
    /// membership.insert(0);
    /// membership.insert(1);
    ///
    /// let witness = WitnessHandle::new(0, membership, 5);
    /// assert_eq!(witness.seed(), 0);
    /// ```
    pub fn new(seed: VertexId, membership: RoaringBitmap, boundary_size: u64) -> Self {
        debug_assert!(
            seed <= u32::MAX as u64,
            "Seed vertex {} exceeds u32::MAX",
            seed
        );
        debug_assert!(
            membership.contains(seed as u32),
            "Seed vertex {} must be in membership set",
            seed
        );

        let hash = Self::compute_hash(seed, &membership);

        Self {
            inner: Arc::new(ImplicitWitness {
                seed,
                membership,
                boundary_size,
                hash,
            }),
        }
    }

    /// Compute hash for a witness
    ///
    /// The hash combines the seed vertex and membership bitmap for fast equality checks.
    fn compute_hash(seed: VertexId, membership: &RoaringBitmap) -> u64 {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);

        // Hash the membership bitmap by iterating its values
        for vertex in membership.iter() {
            vertex.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Check if vertex is in the cut set U
    ///
    /// # Time Complexity
    ///
    /// O(1) via bitmap lookup
    ///
    /// # Examples
    ///
    /// ```
    /// use ruvector_mincut::instance::witness::WitnessHandle;
    /// use roaring::RoaringBitmap;
    ///
    /// let mut membership = RoaringBitmap::new();
    /// membership.insert(5);
    /// membership.insert(10);
    ///
    /// let witness = WitnessHandle::new(5, membership, 3);
    /// assert!(witness.contains(5));
    /// assert!(witness.contains(10));
    /// assert!(!witness.contains(15));
    /// ```
    #[inline]
    pub fn contains(&self, v: VertexId) -> bool {
        if v > u32::MAX as u64 {
            return false;
        }
        self.inner.membership.contains(v as u32)
    }

    /// Get boundary size |δ(U)|
    ///
    /// Returns the pre-computed boundary size for O(1) access.
    ///
    /// # Examples
    ///
    /// ```
    /// use ruvector_mincut::instance::witness::WitnessHandle;
    /// use roaring::RoaringBitmap;
    ///
    /// let witness = WitnessHandle::new(1, RoaringBitmap::from_iter([1, 2, 3]), 7);
    /// assert_eq!(witness.boundary_size(), 7);
    /// ```
    #[inline]
    pub fn boundary_size(&self) -> u64 {
        self.inner.boundary_size
    }

    /// Get the seed vertex
    ///
    /// # Examples
    ///
    /// ```
    /// use ruvector_mincut::instance::witness::WitnessHandle;
    /// use roaring::RoaringBitmap;
    ///
    /// let witness = WitnessHandle::new(42, RoaringBitmap::from_iter([42u32]), 1);
    /// assert_eq!(witness.seed(), 42);
    /// ```
    #[inline]
    pub fn seed(&self) -> VertexId {
        self.inner.seed
    }

    /// Get the witness hash
    ///
    /// Used for fast equality checks without comparing full membership sets.
    #[inline]
    pub fn hash(&self) -> u64 {
        self.inner.hash
    }

    /// Materialize full partition (U, V \ U)
    ///
    /// This is an expensive operation (O(|V|)) that converts the implicit
    /// representation into explicit sets. Use sparingly, primarily for
    /// debugging or verification.
    ///
    /// # Returns
    ///
    /// A tuple `(U, V_minus_U)` where:
    /// - `U` is the set of vertices in the cut
    /// - `V_minus_U` is the complement set
    ///
    /// # Note
    ///
    /// This method assumes vertices are numbered 0..max_vertex. For sparse
    /// graphs, V \ U may contain vertex IDs that don't exist in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use ruvector_mincut::instance::witness::WitnessHandle;
    /// use roaring::RoaringBitmap;
    /// use std::collections::HashSet;
    ///
    /// let mut membership = RoaringBitmap::new();
    /// membership.insert(1);
    /// membership.insert(2);
    ///
    /// let witness = WitnessHandle::new(1, membership, 3);
    /// let (u, _v_minus_u) = witness.materialize_partition();
    ///
    /// assert!(u.contains(&1));
    /// assert!(u.contains(&2));
    /// assert!(!u.contains(&3));
    /// ```
    pub fn materialize_partition(&self) -> (HashSet<VertexId>, HashSet<VertexId>) {
        let u: HashSet<VertexId> = self.inner.membership.iter().map(|v| v as u64).collect();

        // Find the maximum vertex ID to determine graph size
        let max_vertex = self.inner.membership.max().unwrap_or(0) as u64;

        // Create complement set
        let v_minus_u: HashSet<VertexId> = (0..=max_vertex)
            .filter(|&v| !self.inner.membership.contains(v as u32))
            .collect();

        (u, v_minus_u)
    }

    /// Get the cardinality of the cut set U
    ///
    /// # Examples
    ///
    /// ```
    /// use ruvector_mincut::instance::witness::WitnessHandle;
    /// use roaring::RoaringBitmap;
    ///
    /// let witness = WitnessHandle::new(1, RoaringBitmap::from_iter([1u32, 2u32, 3u32]), 5);
    /// assert_eq!(witness.cardinality(), 3);
    /// ```
    #[inline]
    pub fn cardinality(&self) -> u64 {
        self.inner.membership.len()
    }
}

impl PartialEq for WitnessHandle {
    /// Fast equality check using hash
    ///
    /// First compares hashes (O(1)), then falls back to full comparison if needed.
    fn eq(&self, other: &Self) -> bool {
        // Fast path: compare hashes
        if self.inner.hash != other.inner.hash {
            return false;
        }

        // Slow path: compare actual membership
        self.inner.seed == other.inner.seed
            && self.inner.boundary_size == other.inner.boundary_size
            && self.inner.membership == other.inner.membership
    }
}

impl Eq for WitnessHandle {}

/// Trait for witness operations
///
/// This trait abstracts witness operations for generic programming.
/// The primary implementation is `WitnessHandle`.
pub trait Witness {
    /// Check if vertex is in the cut set U
    fn contains(&self, v: VertexId) -> bool;

    /// Get boundary size |δ(U)|
    fn boundary_size(&self) -> u64;

    /// Materialize full partition (expensive)
    fn materialize_partition(&self) -> (HashSet<VertexId>, HashSet<VertexId>);

    /// Get the seed vertex
    fn seed(&self) -> VertexId;

    /// Get cardinality of U
    fn cardinality(&self) -> u64;
}

impl Witness for WitnessHandle {
    #[inline]
    fn contains(&self, v: VertexId) -> bool {
        WitnessHandle::contains(self, v)
    }

    #[inline]
    fn boundary_size(&self) -> u64 {
        WitnessHandle::boundary_size(self)
    }

    fn materialize_partition(&self) -> (HashSet<VertexId>, HashSet<VertexId>) {
        WitnessHandle::materialize_partition(self)
    }

    #[inline]
    fn seed(&self) -> VertexId {
        WitnessHandle::seed(self)
    }

    #[inline]
    fn cardinality(&self) -> u64 {
        WitnessHandle::cardinality(self)
    }
}
