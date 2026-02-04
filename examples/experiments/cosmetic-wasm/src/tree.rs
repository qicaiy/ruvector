//! Computational Sparse Merkle Tree (SMT)
//!
//! ADR-002: Compact binary SMT with lazy evaluation and computation-aware
//! leaf encoding. Uses a HashMap-backed sparse representation where only
//! non-empty leaves and their ancestors are stored.
//!
//! Tree depth is fixed at 256 (matching SHA-256 key space). Each key is a
//! 256-bit address; the i-th bit selects left (0) or right (1) at depth i.

use crate::hasher::{self, Hash, DEFAULT_EMPTY};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Tree depth matching the 256-bit key space
pub const TREE_DEPTH: usize = 256;

/// Precomputed empty subtree hashes for each level.
/// `EMPTY_HASHES[0]` = hash of an empty leaf = DEFAULT_EMPTY
/// `EMPTY_HASHES[i]` = hash_internal(EMPTY_HASHES[i-1], EMPTY_HASHES[i-1])
/// `EMPTY_HASHES[256]` = root of a completely empty tree
pub static EMPTY_HASHES: LazyLock<Vec<Hash>> = LazyLock::new(|| {
    let mut hashes = vec![[0u8; 32]; TREE_DEPTH + 1];
    hashes[0] = DEFAULT_EMPTY;
    for i in 1..=TREE_DEPTH {
        hashes[i] = hasher::hash_internal(&hashes[i - 1], &hashes[i - 1]);
    }
    hashes
});

/// A leaf value stored in the tree, carrying optional computation metadata
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LeafData {
    /// The raw value bytes
    pub value: Vec<u8>,
    /// Optional computation tag indicating what produced this value
    pub computation_tag: Option<String>,
    /// Insertion timestamp (Unix ms)
    pub timestamp: u64,
}

/// Result of a tree mutation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationResult {
    /// The new root hash after mutation
    pub new_root: Hash,
    /// The old root hash before mutation
    pub old_root: Hash,
    /// The key that was mutated
    pub key: Hash,
}

/// Sparse Merkle Tree with computation-aware leaf encoding.
///
/// Stores only populated leaves and caches internal node hashes along
/// paths that have been touched. Empty subtrees are represented implicitly
/// using precomputed per-level empty hashes.
pub struct SparseMerkleTree {
    /// Populated leaf entries: key -> data
    leaves: HashMap<Hash, LeafData>,
    /// Cached internal node hashes keyed by (depth_from_root, prefix)
    /// where prefix has bits below that depth zeroed.
    /// "levels_from_leaf" in the cache: 1 = parent of leaves, 256 = root
    nodes: HashMap<(u16, Hash), Hash>,
    /// Current root hash
    root: Hash,
}

impl SparseMerkleTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            leaves: HashMap::new(),
            nodes: HashMap::new(),
            root: EMPTY_HASHES[TREE_DEPTH],
        }
    }

    /// Get the current root hash
    pub fn root(&self) -> Hash {
        self.root
    }

    /// Get the number of non-empty leaves
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Check if tree has no entries
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Insert or update a leaf. Returns the mutation result.
    pub fn insert(
        &mut self,
        key: Hash,
        value: Vec<u8>,
        computation_tag: Option<String>,
    ) -> MutationResult {
        let old_root = self.root;
        let timestamp = Self::current_timestamp();

        let leaf = LeafData {
            value,
            computation_tag,
            timestamp,
        };

        self.leaves.insert(key, leaf);
        self.update_path(&key);

        MutationResult {
            new_root: self.root,
            old_root,
            key,
        }
    }

    /// Remove a leaf. Returns the mutation result.
    pub fn remove(&mut self, key: &Hash) -> MutationResult {
        let old_root = self.root;
        self.leaves.remove(key);
        self.update_path(key);

        MutationResult {
            new_root: self.root,
            old_root,
            key: *key,
        }
    }

    /// Get a leaf value by key
    pub fn get(&self, key: &Hash) -> Option<&LeafData> {
        self.leaves.get(key)
    }

    /// Check whether a key exists in the tree
    pub fn contains(&self, key: &Hash) -> bool {
        self.leaves.contains_key(key)
    }

    /// Generate an inclusion proof for a key that exists in the tree.
    pub fn prove_inclusion(&self, key: &Hash) -> Option<InclusionProof> {
        let leaf_data = self.leaves.get(key)?.clone();
        let siblings = self.collect_siblings(key);

        Some(InclusionProof {
            key: *key,
            leaf_data,
            siblings,
            root: self.root,
        })
    }

    /// Generate an exclusion proof for a key that does NOT exist.
    pub fn prove_exclusion(&self, key: &Hash) -> Option<ExclusionProof> {
        if self.contains(key) {
            return None;
        }

        let siblings = self.collect_siblings(key);

        Some(ExclusionProof {
            key: *key,
            siblings,
            root: self.root,
        })
    }

    /// Collect sibling hashes from leaf level up to root.
    /// siblings[0] = sibling at leaf level (levels_from_leaf = 0)
    /// siblings[i] = sibling i levels above leaf
    fn collect_siblings(&self, key: &Hash) -> Vec<Hash> {
        let mut siblings = Vec::with_capacity(TREE_DEPTH);

        // Walk from leaf (depth_from_root = 255) up to root (depth_from_root = 0)
        for levels_up in 0..TREE_DEPTH {
            let depth_from_root = TREE_DEPTH - 1 - levels_up;
            let sibling = self.get_sibling_hash(key, depth_from_root, levels_up);
            siblings.push(sibling);
        }

        siblings
    }

    /// Get the hash of the sibling subtree at a given depth.
    ///
    /// At levels_up=0: sibling is a leaf at depth TREE_DEPTH (conceptual).
    /// At levels_up=k>0: sibling subtree is rooted at depth (depth_from_root + 1).
    fn get_sibling_hash(&self, key: &Hash, depth_from_root: usize, levels_up: usize) -> Hash {
        let mut sibling_key = *key;
        flip_bit(&mut sibling_key, depth_from_root);

        if levels_up == 0 {
            // Leaf level
            match self.leaves.get(&sibling_key) {
                Some(leaf) => hasher::hash_leaf(&sibling_key, &leaf.value),
                None => EMPTY_HASHES[0],
            }
        } else {
            // Internal level: the sibling subtree is rooted at depth_from_root + 1
            let sibling_depth = depth_from_root + 1;
            let cache_k = Self::make_cache_key(&sibling_key, sibling_depth);
            self.nodes
                .get(&cache_k)
                .copied()
                .unwrap_or(EMPTY_HASHES[levels_up])
        }
    }

    /// Update all cached internal nodes along the path from a leaf to the root.
    fn update_path(&mut self, key: &Hash) {
        let mut current = match self.leaves.get(key) {
            Some(leaf) => hasher::hash_leaf(key, &leaf.value),
            None => EMPTY_HASHES[0],
        };

        for levels_up in 0..TREE_DEPTH {
            let depth_from_root = TREE_DEPTH - 1 - levels_up;
            let bit = get_bit(key, depth_from_root);

            let sibling = self.get_sibling_hash(key, depth_from_root, levels_up);

            let parent = if bit == 0 {
                hasher::hash_internal(&current, &sibling)
            } else {
                hasher::hash_internal(&sibling, &current)
            };

            // Cache the parent at depth_from_root. The parent is the root of
            // the subtree spanning both current and sibling.
            let cache_k = Self::make_cache_key(key, depth_from_root);
            self.nodes.insert(cache_k, parent);

            current = parent;
        }

        self.root = current;
    }

    /// Create a cache key: zero bits below `depth_from_root` so that
    /// all keys sharing the same prefix at that depth map to the same entry.
    fn make_cache_key(key: &Hash, depth_from_root: usize) -> (u16, Hash) {
        let mut prefix = *key;
        // Zero out bit at depth_from_root and all bits below it
        for d in depth_from_root..TREE_DEPTH {
            let byte_idx = d / 8;
            let bit_idx = 7 - (d % 8);
            prefix[byte_idx] &= !(1 << bit_idx);
        }
        (depth_from_root as u16, prefix)
    }

    /// Get current timestamp in milliseconds
    fn current_timestamp() -> u64 {
        #[cfg(target_arch = "wasm32")]
        {
            js_sys::Date::now() as u64
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        }
    }

    /// Export all leaf keys
    pub fn keys(&self) -> Vec<Hash> {
        self.leaves.keys().copied().collect()
    }

    /// Export the full tree state as a serializable snapshot
    pub fn snapshot(&self) -> TreeSnapshot {
        TreeSnapshot {
            root: self.root,
            leaves: self.leaves.clone(),
            leaf_count: self.leaves.len(),
        }
    }
}

impl Default for SparseMerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable tree state snapshot
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeSnapshot {
    pub root: Hash,
    pub leaves: HashMap<Hash, LeafData>,
    pub leaf_count: usize,
}

/// Proof of inclusion: a key exists in the tree
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InclusionProof {
    /// The key being proved
    pub key: Hash,
    /// The leaf data at that key
    pub leaf_data: LeafData,
    /// Sibling hashes from leaf to root (256 entries)
    pub siblings: Vec<Hash>,
    /// The root hash this proof is valid against
    pub root: Hash,
}

/// Proof of exclusion: a key does NOT exist in the tree
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExclusionProof {
    /// The key being proved absent
    pub key: Hash,
    /// Sibling hashes from (empty) leaf to root (256 entries)
    pub siblings: Vec<Hash>,
    /// The root hash this proof is valid against
    pub root: Hash,
}

/// Extract the bit at a given depth from a 256-bit key.
/// Depth 0 = MSB of byte 0, depth 255 = LSB of byte 31.
#[inline]
pub(crate) fn get_bit(key: &Hash, depth: usize) -> u8 {
    let byte_idx = depth / 8;
    let bit_idx = 7 - (depth % 8);
    (key[byte_idx] >> bit_idx) & 1
}

/// Flip the bit at a given depth in a 256-bit key
#[inline]
fn flip_bit(key: &mut Hash, depth: usize) {
    let byte_idx = depth / 8;
    let bit_idx = 7 - (depth % 8);
    key[byte_idx] ^= 1 << bit_idx;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hasher::HASH_SIZE;

    #[test]
    fn test_empty_tree() {
        let tree = SparseMerkleTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        // Root should be the precomputed empty root
        assert_eq!(tree.root(), EMPTY_HASHES[TREE_DEPTH]);
    }

    #[test]
    fn test_insert_and_get() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"patient_001");
        let result = tree.insert(key, b"included".to_vec(), Some("enrollment".into()));

        assert_eq!(tree.len(), 1);
        assert!(tree.contains(&key));
        assert_ne!(result.old_root, result.new_root);

        let leaf = tree.get(&key).unwrap();
        assert_eq!(leaf.value, b"included");
        assert_eq!(leaf.computation_tag.as_deref(), Some("enrollment"));
    }

    #[test]
    fn test_remove() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"test");
        tree.insert(key, b"value".to_vec(), None);
        assert!(tree.contains(&key));

        tree.remove(&key);
        assert!(!tree.contains(&key));
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_remove_restores_empty_root() {
        let mut tree = SparseMerkleTree::new();
        let empty_root = tree.root();

        let key = hasher::compute_key(b"temp");
        tree.insert(key, b"val".to_vec(), None);
        assert_ne!(tree.root(), empty_root);

        tree.remove(&key);
        assert_eq!(tree.root(), empty_root);
    }

    #[test]
    fn test_root_changes_on_mutation() {
        let mut tree = SparseMerkleTree::new();
        let root0 = tree.root();

        let key1 = hasher::compute_key(b"k1");
        tree.insert(key1, b"v1".to_vec(), None);
        let root1 = tree.root();
        assert_ne!(root0, root1);

        let key2 = hasher::compute_key(b"k2");
        tree.insert(key2, b"v2".to_vec(), None);
        let root2 = tree.root();
        assert_ne!(root1, root2);
    }

    #[test]
    fn test_inclusion_proof_generated() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"exists");
        tree.insert(key, b"data".to_vec(), None);

        let proof = tree.prove_inclusion(&key);
        assert!(proof.is_some());

        let proof = proof.unwrap();
        assert_eq!(proof.key, key);
        assert_eq!(proof.root, tree.root());
        assert_eq!(proof.siblings.len(), TREE_DEPTH);
    }

    #[test]
    fn test_inclusion_proof_absent_key() {
        let tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"missing");
        assert!(tree.prove_inclusion(&key).is_none());
    }

    #[test]
    fn test_exclusion_proof_generated() {
        let tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"absent");

        let proof = tree.prove_exclusion(&key);
        assert!(proof.is_some());

        let proof = proof.unwrap();
        assert_eq!(proof.key, key);
        assert_eq!(proof.root, tree.root());
        assert_eq!(proof.siblings.len(), TREE_DEPTH);
    }

    #[test]
    fn test_exclusion_proof_present_key() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"present");
        tree.insert(key, b"val".to_vec(), None);
        assert!(tree.prove_exclusion(&key).is_none());
    }

    #[test]
    fn test_get_bit() {
        let mut key = [0u8; HASH_SIZE];
        key[0] = 0b10110010;
        assert_eq!(get_bit(&key, 0), 1);
        assert_eq!(get_bit(&key, 1), 0);
        assert_eq!(get_bit(&key, 2), 1);
        assert_eq!(get_bit(&key, 3), 1);
        assert_eq!(get_bit(&key, 4), 0);
        assert_eq!(get_bit(&key, 5), 0);
        assert_eq!(get_bit(&key, 6), 1);
        assert_eq!(get_bit(&key, 7), 0);
    }

    #[test]
    fn test_flip_bit() {
        let mut key = [0u8; HASH_SIZE];
        flip_bit(&mut key, 0);
        assert_eq!(get_bit(&key, 0), 1);
        flip_bit(&mut key, 0);
        assert_eq!(get_bit(&key, 0), 0);
    }

    #[test]
    fn test_snapshot() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"snap");
        tree.insert(key, b"shot".to_vec(), None);

        let snap = tree.snapshot();
        assert_eq!(snap.leaf_count, 1);
        assert_eq!(snap.root, tree.root());
    }
}
