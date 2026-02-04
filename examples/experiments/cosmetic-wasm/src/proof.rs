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
    pub fn is_valid(&self) -> bool {
        matches!(self, VerifyResult::Valid)
    }
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

    // Step 1: Compute leaf hash
    let mut current = hasher::hash_leaf(&proof.key, &proof.leaf_data.value);

    // Step 2: Walk up from leaf to root
    for (i, sibling) in proof.siblings.iter().enumerate() {
        let depth = TREE_DEPTH - 1 - i;
        let bit = get_bit(&proof.key, depth);

        current = if bit == 0 {
            hasher::hash_internal(&current, sibling)
        } else {
            hasher::hash_internal(sibling, &current)
        };
    }

    // Step 3: Compare roots
    if current == proof.root {
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
///
/// This works because if the key were present, the leaf hash would be different
/// from DEFAULT_EMPTY, producing a different root.
pub fn verify_exclusion(proof: &ExclusionProof) -> VerifyResult {
    if proof.siblings.len() != TREE_DEPTH {
        return VerifyResult::Invalid(format!(
            "Expected {} siblings, got {}",
            TREE_DEPTH,
            proof.siblings.len()
        ));
    }

    // Step 1: Empty leaf hash
    let mut current = DEFAULT_EMPTY;

    // Step 2: Walk up from leaf to root
    for (i, sibling) in proof.siblings.iter().enumerate() {
        let depth = TREE_DEPTH - 1 - i;
        let bit = get_bit(&proof.key, depth);

        current = if bit == 0 {
            hasher::hash_internal(&current, sibling)
        } else {
            hasher::hash_internal(sibling, &current)
        };
    }

    // Step 3: Compare roots
    if current == proof.root {
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

    /// Compute the byte size of this compact proof
    pub fn byte_size(&self) -> usize {
        HASH_SIZE  // key
        + 1  // is_inclusion flag
        + self.leaf_value.as_ref().map_or(0, |v| v.len())
        + self.sibling_bitmap.len()
        + self.non_default_siblings.len() * HASH_SIZE
        + HASH_SIZE  // root
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
    }
}
