//! Proof verification for CoSMeTIC
//!
//! Verifies inclusion and exclusion proofs against a known root hash.
//! ADR-003: Uses algebraic commitment-based verification (simplified,
//! no trusted setup). In production, this would integrate with a ZK
//! proof system like Halo2 or PLONK for privacy-preserving verification.

use crate::hasher::{self, Hash, DEFAULT_EMPTY, HASH_SIZE};
use crate::tree::{get_bit, ExclusionProof, InclusionProof, TREE_DEPTH};
use serde::{Deserialize, Serialize};

/// Result of proof verification
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum VerifyResult {
    /// Proof is valid
    Valid,
    /// Proof is invalid with a reason
    Invalid(String),
}

impl VerifyResult {
    #[inline]
    pub fn is_valid(&self) -> bool {
        matches!(self, VerifyResult::Valid)
    }
}

/// Walk a sibling path from a starting hash up to the root.
/// This is the common core of both inclusion and exclusion verification.
#[inline]
fn walk_to_root(key: &Hash, start_hash: Hash, siblings: &[Hash]) -> Hash {
    let mut current = start_hash;
    for (i, sibling) in siblings.iter().enumerate() {
        let depth = TREE_DEPTH - 1 - i;
        let bit = get_bit(key, depth);
        current = if bit == 0 {
            hasher::hash_internal(&current, sibling)
        } else {
            hasher::hash_internal(sibling, &current)
        };
    }
    current
}

/// Verify an inclusion proof: confirms that (key, value) is in the tree with the given root.
///
/// Algorithm:
/// 1. Compute the leaf hash from key + value
/// 2. Walk up the tree using sibling hashes
/// 3. Compare final computed root with the claimed root
pub fn verify_inclusion(proof: &InclusionProof) -> VerifyResult {
    if proof.siblings.len() != TREE_DEPTH {
        return VerifyResult::Invalid(format!(
            "Expected {} siblings, got {}",
            TREE_DEPTH,
            proof.siblings.len()
        ));
    }

    let leaf_hash = hasher::hash_leaf(&proof.key, &proof.leaf_data.value);
    let computed_root = walk_to_root(&proof.key, leaf_hash, &proof.siblings);

    if computed_root == proof.root {
        VerifyResult::Valid
    } else {
        VerifyResult::Invalid("Computed root does not match claimed root".into())
    }
}

/// Verify an exclusion proof: confirms that a key is NOT in the tree with the given root.
///
/// Algorithm:
/// 1. Start with DEFAULT_EMPTY (the hash of an absent leaf)
/// 2. Walk up the tree using sibling hashes
/// 3. Compare final computed root with the claimed root
pub fn verify_exclusion(proof: &ExclusionProof) -> VerifyResult {
    if proof.siblings.len() != TREE_DEPTH {
        return VerifyResult::Invalid(format!(
            "Expected {} siblings, got {}",
            TREE_DEPTH,
            proof.siblings.len()
        ));
    }

    let computed_root = walk_to_root(&proof.key, DEFAULT_EMPTY, &proof.siblings);

    if computed_root == proof.root {
        VerifyResult::Valid
    } else {
        VerifyResult::Invalid("Computed root does not match claimed root".into())
    }
}

/// Batch verification: verify multiple proofs against the same root.
/// Returns a vector of results, one per proof.
pub fn verify_batch_inclusion(proofs: &[InclusionProof]) -> Vec<VerifyResult> {
    proofs.iter().map(verify_inclusion).collect()
}

/// Batch verification for exclusion proofs.
pub fn verify_batch_exclusion(proofs: &[ExclusionProof]) -> Vec<VerifyResult> {
    proofs.iter().map(verify_exclusion).collect()
}

/// Check if all proofs in a batch are valid (short-circuits on first failure).
pub fn all_valid_inclusion(proofs: &[InclusionProof]) -> bool {
    proofs.iter().all(|p| verify_inclusion(p).is_valid())
}

/// Check if all exclusion proofs in a batch are valid (short-circuits on first failure).
pub fn all_valid_exclusion(proofs: &[ExclusionProof]) -> bool {
    proofs.iter().all(|p| verify_exclusion(p).is_valid())
}

/// Compact proof representation for serialization/transport.
/// Reduces proof size by omitting default-empty siblings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactProof {
    /// The key being proved
    pub key: Hash,
    /// Whether this is an inclusion (true) or exclusion (false) proof
    pub is_inclusion: bool,
    /// Leaf value (only for inclusion proofs)
    pub leaf_value: Option<Vec<u8>>,
    /// Computation tag (only for inclusion proofs)
    pub computation_tag: Option<String>,
    /// Bitmap indicating which siblings are non-default
    /// Bit i = 1 means siblings[i] is non-default and present in `non_default_siblings`
    pub sibling_bitmap: Vec<u8>,
    /// Only the non-default sibling hashes (sparse encoding)
    pub non_default_siblings: Vec<Hash>,
    /// The root hash
    pub root: Hash,
}

impl CompactProof {
    /// Create a compact proof from an inclusion proof
    pub fn from_inclusion(proof: &InclusionProof) -> Self {
        let (bitmap, non_defaults) = compress_siblings(&proof.siblings);
        CompactProof {
            key: proof.key,
            is_inclusion: true,
            leaf_value: Some(proof.leaf_data.value.clone()),
            computation_tag: proof.leaf_data.computation_tag.clone(),
            sibling_bitmap: bitmap,
            non_default_siblings: non_defaults,
            root: proof.root,
        }
    }

    /// Create a compact proof from an exclusion proof
    pub fn from_exclusion(proof: &ExclusionProof) -> Self {
        let (bitmap, non_defaults) = compress_siblings(&proof.siblings);
        CompactProof {
            key: proof.key,
            is_inclusion: false,
            leaf_value: None,
            computation_tag: None,
            sibling_bitmap: bitmap,
            non_default_siblings: non_defaults,
            root: proof.root,
        }
    }

    /// Decompress this compact proof back into a full sibling list.
    /// Uses per-level EMPTY_HASHES for missing siblings.
    pub fn decompress_siblings(&self) -> Vec<Hash> {
        use crate::tree::EMPTY_HASHES;

        let mut siblings = Vec::with_capacity(TREE_DEPTH);
        let mut nd_idx = 0;

        for i in 0..TREE_DEPTH {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);
            let is_set = byte_idx < self.sibling_bitmap.len()
                && (self.sibling_bitmap[byte_idx] >> bit_idx) & 1 == 1;

            if is_set && nd_idx < self.non_default_siblings.len() {
                siblings.push(self.non_default_siblings[nd_idx]);
                nd_idx += 1;
            } else {
                siblings.push(EMPTY_HASHES[i]);
            }
        }

        siblings
    }

    /// Verify this compact proof directly without full decompression overhead.
    pub fn verify(&self) -> VerifyResult {
        let siblings = self.decompress_siblings();
        if siblings.len() != TREE_DEPTH {
            return VerifyResult::Invalid("Decompressed siblings length mismatch".into());
        }

        let start_hash = if self.is_inclusion {
            match &self.leaf_value {
                Some(val) => hasher::hash_leaf(&self.key, val),
                None => return VerifyResult::Invalid("Inclusion proof missing leaf value".into()),
            }
        } else {
            DEFAULT_EMPTY
        };

        let computed_root = walk_to_root(&self.key, start_hash, &siblings);
        if computed_root == self.root {
            VerifyResult::Valid
        } else {
            VerifyResult::Invalid("Computed root does not match claimed root".into())
        }
    }

    /// Compute the byte size of this compact proof
    pub fn byte_size(&self) -> usize {
        HASH_SIZE  // key
        + 1  // is_inclusion flag
        + self.leaf_value.as_ref().map_or(0, |v| v.len())
        + self.sibling_bitmap.len()
        + self.non_default_siblings.len() * HASH_SIZE
        + HASH_SIZE  // root
    }

    /// Compression ratio: compact size / full proof size
    pub fn compression_ratio(&self) -> f64 {
        let full_size = HASH_SIZE // key
            + TREE_DEPTH * HASH_SIZE // siblings
            + self.leaf_value.as_ref().map_or(0, |v| v.len())
            + HASH_SIZE; // root
        self.byte_size() as f64 / full_size as f64
    }
}

/// Compress siblings by only keeping non-empty entries.
/// siblings[i] is at level i above the leaf; its empty hash is EMPTY_HASHES[i].
fn compress_siblings(siblings: &[Hash]) -> (Vec<u8>, Vec<Hash>) {
    use crate::tree::EMPTY_HASHES;

    let bitmap_bytes = (siblings.len() + 7) / 8;
    let mut bitmap = vec![0u8; bitmap_bytes];
    let mut non_defaults = Vec::new();

    for (i, sibling) in siblings.iter().enumerate() {
        if *sibling != EMPTY_HASHES[i] {
            bitmap[i / 8] |= 1 << (7 - (i % 8));
            non_defaults.push(*sibling);
        }
    }

    (bitmap, non_defaults)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::SparseMerkleTree;

    #[test]
    fn test_inclusion_proof_verifies() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"patient_included");
        tree.insert(key, b"enrolled".to_vec(), Some("trial_A".into()));

        let proof = tree.prove_inclusion(&key).unwrap();
        assert!(verify_inclusion(&proof).is_valid());
    }

    #[test]
    fn test_exclusion_proof_verifies() {
        let mut tree = SparseMerkleTree::new();
        // Insert some data so tree is not empty
        let k1 = hasher::compute_key(b"present");
        tree.insert(k1, b"here".to_vec(), None);

        // Prove absence of different key
        let absent_key = hasher::compute_key(b"absent");
        let proof = tree.prove_exclusion(&absent_key).unwrap();
        assert!(verify_exclusion(&proof).is_valid());
    }

    #[test]
    fn test_tampered_inclusion_proof_fails() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"patient");
        tree.insert(key, b"data".to_vec(), None);

        let mut proof = tree.prove_inclusion(&key).unwrap();
        // Tamper with the leaf value
        proof.leaf_data.value = b"tampered".to_vec();

        assert!(!verify_inclusion(&proof).is_valid());
    }

    #[test]
    fn test_wrong_root_fails() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"k1");
        tree.insert(key, b"v1".to_vec(), None);

        let mut proof = tree.prove_inclusion(&key).unwrap();
        // Set wrong root
        proof.root = [0xFFu8; HASH_SIZE];

        assert!(!verify_inclusion(&proof).is_valid());
    }

    #[test]
    fn test_compact_proof_compression() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"compact_test");
        tree.insert(key, b"val".to_vec(), None);

        let proof = tree.prove_inclusion(&key).unwrap();
        let full_size = HASH_SIZE + proof.siblings.len() * HASH_SIZE + proof.leaf_data.value.len();

        let compact = CompactProof::from_inclusion(&proof);
        let compact_size = compact.byte_size();

        // Compact should be smaller since most siblings are default-empty
        assert!(compact_size < full_size, "Compact {} should be < full {}", compact_size, full_size);
        assert!(compact.compression_ratio() < 1.0);
    }

    #[test]
    fn test_compact_proof_roundtrip() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"roundtrip_test");
        tree.insert(key, b"val".to_vec(), None);

        let proof = tree.prove_inclusion(&key).unwrap();
        let compact = CompactProof::from_inclusion(&proof);

        // Decompress and verify siblings match
        let decompressed = compact.decompress_siblings();
        assert_eq!(decompressed.len(), proof.siblings.len());
        assert_eq!(decompressed, proof.siblings);
    }

    #[test]
    fn test_compact_proof_verify() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"compact_verify");
        tree.insert(key, b"val".to_vec(), None);

        let proof = tree.prove_inclusion(&key).unwrap();
        let compact = CompactProof::from_inclusion(&proof);
        assert!(compact.verify().is_valid());

        // Exclusion compact proof
        let exc_key = hasher::compute_key(b"missing");
        let exc_proof = tree.prove_exclusion(&exc_key).unwrap();
        let exc_compact = CompactProof::from_exclusion(&exc_proof);
        assert!(exc_compact.verify().is_valid());
    }

    #[test]
    fn test_batch_verification() {
        let mut tree = SparseMerkleTree::new();
        let keys: Vec<Hash> = (0..5)
            .map(|i| hasher::compute_key(format!("key_{}", i).as_bytes()))
            .collect();

        for key in &keys {
            tree.insert(*key, b"value".to_vec(), None);
        }

        let proofs: Vec<InclusionProof> = keys
            .iter()
            .map(|k| tree.prove_inclusion(k).unwrap())
            .collect();

        let results = verify_batch_inclusion(&proofs);
        assert!(results.iter().all(|r| r.is_valid()));
        assert!(all_valid_inclusion(&proofs));
    }

    #[test]
    fn test_all_valid_short_circuits() {
        let mut tree = SparseMerkleTree::new();
        let key = hasher::compute_key(b"short");
        tree.insert(key, b"val".to_vec(), None);

        let mut proof = tree.prove_inclusion(&key).unwrap();
        proof.leaf_data.value = b"tampered".to_vec();

        assert!(!all_valid_inclusion(&[proof]));
    }
}
