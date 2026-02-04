//! # CoSMeTIC WASM
//!
//! Computational Sparse Merkle Trees with Inclusion/exclusion Certificates.
//!
//! Based on the CoSMeTIC framework (arXiv, January 2026), this crate implements:
//!
//! - **Sparse Merkle Trees (SMT)**: 256-bit address space with lazy evaluation,
//!   storing only non-empty leaves and their ancestors.
//!
//! - **Inclusion Proofs**: Merkle path from leaf to root proving a key exists
//!   in the tree, with the associated value.
//!
//! - **Exclusion Proofs**: Merkle path proving a key does NOT exist, using
//!   the default-empty leaf hash propagated through the sibling path.
//!
//! - **Computation Attestations**: Cryptographic commitments linking input
//!   state (input SMT root) to output state (output SMT root) through a
//!   named computation with parameters. Each attestation records individual
//!   inclusion/exclusion decisions with reasons.
//!
//! - **Attestation Chains**: Sequential attestations form a hash chain,
//!   enabling verifiable audit trails of multi-step computations.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    WASM API Layer                         │
//! │  CosmeticTree: insert, remove, prove, verify, attest     │
//! ├──────────────────────────────────────────────────────────┤
//! │                                                          │
//! │  ┌─────────────┐  ┌───────────┐  ┌──────────────────┐  │
//! │  │ SparseMerkle│  │  Proof    │  │  Computation     │  │
//! │  │ Tree        │  │  Verifier │  │  Attestation     │  │
//! │  │             │  │           │  │                  │  │
//! │  │ - insert    │  │ - verify  │  │  - build         │  │
//! │  │ - remove    │  │   incl.   │  │  - verify        │  │
//! │  │ - prove     │  │ - verify  │  │  - chain         │  │
//! │  │   inclusion │  │   excl.   │  │  - log           │  │
//! │  │ - prove     │  │ - batch   │  │  - audit         │  │
//! │  │   exclusion │  │ - compact │  │                  │  │
//! │  └─────────────┘  └───────────┘  └──────────────────┘  │
//! │                                                          │
//! ├──────────────────────────────────────────────────────────┤
//! │                  Hasher Layer                             │
//! │  SHA-256 with domain separation (leaf/internal/attest)   │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage (JavaScript)
//!
//! ```javascript
//! import { CosmeticTree } from 'cosmetic-wasm';
//!
//! const tree = new CosmeticTree();
//!
//! // Insert data
//! tree.insert(new TextEncoder().encode("patient_001"), new TextEncoder().encode("enrolled"), "trial_A");
//!
//! // Generate inclusion proof
//! const proof = tree.proveInclusion(new TextEncoder().encode("patient_001"));
//!
//! // Verify
//! const valid = tree.verifyInclusionProof(new TextEncoder().encode("patient_001"));
//! ```

use wasm_bindgen::prelude::*;

pub mod attestation;
pub mod hasher;
pub mod proof;
pub mod tree;
pub mod wasm_api;

pub use attestation::{AttestationBuilder, AttestationLog, ComputationAttestation};
pub use hasher::{Hash, HASH_SIZE};
pub use proof::{verify_exclusion, verify_inclusion, CompactProof, VerifyResult};
pub use tree::{ExclusionProof, InclusionProof, SparseMerkleTree};
pub use wasm_api::CosmeticTree;

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get the crate version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.1.0");
    }

    #[test]
    fn test_end_to_end_pipeline() {
        // 1. Create input tree with candidates
        let mut input_tree = SparseMerkleTree::new();
        let candidates = vec![
            ("alice", b"age:28,healthy:true" as &[u8], true),
            ("bob", b"age:15,healthy:true", false),    // too young
            ("carol", b"age:35,healthy:false", false),  // not healthy
            ("dave", b"age:42,healthy:true", true),
        ];

        for (name, data, _) in &candidates {
            let key = hasher::compute_key(name.as_bytes());
            input_tree.insert(key, data.to_vec(), Some("candidate".into()));
        }
        assert_eq!(input_tree.len(), 4);

        // 2. Build output tree with only included candidates
        let mut output_tree = SparseMerkleTree::new();
        let mut builder = AttestationBuilder::new("enrollment_filter")
            .with_parameters(b"min_age=18,require_healthy=true".to_vec());

        for (name, data, eligible) in &candidates {
            let key = hasher::compute_key(name.as_bytes());
            if *eligible {
                output_tree.insert(key, data.to_vec(), Some("enrolled".into()));
                builder = builder.include(key, "Meets criteria", "age>=18 AND healthy");
            } else {
                builder = builder.exclude(key, "Does not meet criteria", "age>=18 AND healthy");
            }
        }
        assert_eq!(output_tree.len(), 2);

        // 3. Create attestation
        let attestation = builder.build(input_tree.root(), output_tree.root());
        assert!(attestation::verify_attestation(&attestation));
        assert_eq!(attestation.decisions.len(), 4);

        // 4. Verify inclusion proofs for enrolled
        let alice_key = hasher::compute_key(b"alice");
        let inc_proof = output_tree.prove_inclusion(&alice_key).unwrap();
        assert!(verify_inclusion(&inc_proof).is_valid());

        // 5. Verify exclusion proofs for excluded
        let bob_key = hasher::compute_key(b"bob");
        let exc_proof = output_tree.prove_exclusion(&bob_key).unwrap();
        assert!(verify_exclusion(&exc_proof).is_valid());

        // 6. Compact proof is smaller
        let compact = CompactProof::from_inclusion(&inc_proof);
        let full_siblings_size = inc_proof.siblings.len() * HASH_SIZE;
        assert!(compact.non_default_siblings.len() * HASH_SIZE < full_siblings_size);
    }
}
