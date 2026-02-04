//! Computation Attestation for CoSMeTIC
//!
//! This module implements the "computational" aspect of the Computational SMT.
//! An attestation cryptographically links:
//! - The input data (via an SMT root representing the input set)
//! - The computation performed (identified by function_id + parameters)
//! - The output data (via an SMT root representing the output set)
//! - Inclusion/exclusion decisions with reasons
//!
//! This enables a verifier to check:
//! 1. What data was included/excluded (via SMT proofs)
//! 2. What computation was performed on it (via the attestation)
//! 3. That the computation was correct (via the commitment chain)

use crate::hasher::{self, Hash};
use serde::{Deserialize, Serialize};

/// A computation attestation linking input state to output state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComputationAttestation {
    /// Unique identifier for this attestation
    pub id: String,
    /// Root of the input SMT (the data that was processed)
    pub input_root: Hash,
    /// Root of the output SMT (the result of computation)
    pub output_root: Hash,
    /// Identifier for the computation function
    pub function_id: String,
    /// Serialized parameters used in the computation
    pub parameters: Vec<u8>,
    /// The attestation hash (commitment)
    pub commitment: Hash,
    /// Individual inclusion/exclusion decisions with reasons
    pub decisions: Vec<InclusionDecision>,
    /// Timestamp of attestation creation (Unix ms)
    pub timestamp: u64,
    /// Optional chain: hash of previous attestation for sequential processing
    pub previous_attestation: Option<Hash>,
}

/// A decision about whether a data item was included or excluded
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InclusionDecision {
    /// The key of the data item
    pub key: Hash,
    /// Whether the item was included (true) or excluded (false)
    pub included: bool,
    /// Human-readable reason for the decision
    pub reason: String,
    /// The criterion that was applied (e.g., "age >= 18", "no_prior_condition")
    pub criterion: String,
}

/// Builder for constructing attestations incrementally
pub struct AttestationBuilder {
    function_id: String,
    parameters: Vec<u8>,
    decisions: Vec<InclusionDecision>,
    previous: Option<Hash>,
}

impl AttestationBuilder {
    /// Create a new attestation builder
    pub fn new(function_id: impl Into<String>) -> Self {
        Self {
            function_id: function_id.into(),
            parameters: Vec::new(),
            decisions: Vec::new(),
            previous: None,
        }
    }

    /// Set computation parameters
    pub fn with_parameters(mut self, params: Vec<u8>) -> Self {
        self.parameters = params;
        self
    }

    /// Chain to a previous attestation
    pub fn chain_after(mut self, previous_commitment: Hash) -> Self {
        self.previous = Some(previous_commitment);
        self
    }

    /// Record an inclusion decision
    pub fn include(mut self, key: Hash, reason: impl Into<String>, criterion: impl Into<String>) -> Self {
        self.decisions.push(InclusionDecision {
            key,
            included: true,
            reason: reason.into(),
            criterion: criterion.into(),
        });
        self
    }

    /// Record an exclusion decision
    pub fn exclude(mut self, key: Hash, reason: impl Into<String>, criterion: impl Into<String>) -> Self {
        self.decisions.push(InclusionDecision {
            key,
            included: false,
            reason: reason.into(),
            criterion: criterion.into(),
        });
        self
    }

    /// Build the attestation given input and output tree roots
    pub fn build(self, input_root: Hash, output_root: Hash) -> ComputationAttestation {
        let commitment = hasher::hash_attestation(
            &input_root,
            &output_root,
            self.function_id.as_bytes(),
            &self.parameters,
        );

        let timestamp = current_timestamp();

        let id = format!(
            "attest-{}",
            hasher::to_hex(&commitment)[..16].to_string()
        );

        ComputationAttestation {
            id,
            input_root,
            output_root,
            function_id: self.function_id,
            parameters: self.parameters,
            commitment,
            decisions: self.decisions,
            timestamp,
            previous_attestation: self.previous,
        }
    }
}

/// Verify that an attestation's commitment matches its contents
pub fn verify_attestation(attestation: &ComputationAttestation) -> bool {
    let expected = hasher::hash_attestation(
        &attestation.input_root,
        &attestation.output_root,
        attestation.function_id.as_bytes(),
        &attestation.parameters,
    );
    attestation.commitment == expected
}

/// Verify a chain of attestations (each one references the previous)
pub fn verify_chain(attestations: &[ComputationAttestation]) -> Result<(), ChainError> {
    if attestations.is_empty() {
        return Ok(());
    }

    // First attestation should have no previous
    if attestations[0].previous_attestation.is_some() {
        return Err(ChainError::InvalidStart);
    }

    // Verify each attestation's commitment
    for (i, att) in attestations.iter().enumerate() {
        if !verify_attestation(att) {
            return Err(ChainError::InvalidCommitment { index: i });
        }
    }

    // Verify chain linkage
    for i in 1..attestations.len() {
        match attestations[i].previous_attestation {
            Some(prev) if prev == attestations[i - 1].commitment => {}
            Some(_) => return Err(ChainError::BrokenChain { index: i }),
            None => return Err(ChainError::MissingLink { index: i }),
        }
    }

    // Verify output->input continuity: each output_root should match next input_root
    for i in 0..attestations.len() - 1 {
        if attestations[i].output_root != attestations[i + 1].input_root {
            return Err(ChainError::StateMismatch { index: i });
        }
    }

    Ok(())
}

/// Errors in attestation chain verification
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ChainError {
    /// First attestation has a previous reference
    InvalidStart,
    /// Attestation commitment doesn't match contents
    InvalidCommitment { index: usize },
    /// Previous reference doesn't match prior attestation
    BrokenChain { index: usize },
    /// Missing previous reference in non-first attestation
    MissingLink { index: usize },
    /// Output root doesn't match next input root
    StateMismatch { index: usize },
}

/// Attestation log: ordered sequence of attestations for audit
#[derive(Default)]
pub struct AttestationLog {
    attestations: Vec<ComputationAttestation>,
}

impl AttestationLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an attestation to the log
    pub fn append(&mut self, attestation: ComputationAttestation) {
        self.attestations.push(attestation);
    }

    /// Get the latest attestation commitment (for chaining)
    pub fn latest_commitment(&self) -> Option<Hash> {
        self.attestations.last().map(|a| a.commitment)
    }

    /// Verify the entire chain
    pub fn verify(&self) -> Result<(), ChainError> {
        verify_chain(&self.attestations)
    }

    /// Get all decisions across all attestations
    pub fn all_decisions(&self) -> Vec<&InclusionDecision> {
        self.attestations
            .iter()
            .flat_map(|a| a.decisions.iter())
            .collect()
    }

    /// Count total inclusions and exclusions
    pub fn decision_counts(&self) -> (usize, usize) {
        let decisions = self.all_decisions();
        let included = decisions.iter().filter(|d| d.included).count();
        let excluded = decisions.iter().filter(|d| !d.included).count();
        (included, excluded)
    }

    /// Get number of attestations
    pub fn len(&self) -> usize {
        self.attestations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.attestations.is_empty()
    }

    /// Export as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.attestations)
    }
}

/// Get current timestamp
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hasher::HASH_SIZE;
    use crate::tree::SparseMerkleTree;

    #[test]
    fn test_attestation_builder() {
        let input_root = hasher::sha256(b"input_state");
        let output_root = hasher::sha256(b"output_state");

        let key1 = hasher::compute_key(b"patient_001");
        let key2 = hasher::compute_key(b"patient_002");

        let attestation = AttestationBuilder::new("enrollment_filter")
            .with_parameters(b"min_age=18,condition=healthy".to_vec())
            .include(key1, "Met all criteria", "age >= 18 AND healthy")
            .exclude(key2, "Below minimum age", "age >= 18")
            .build(input_root, output_root);

        assert!(attestation.id.starts_with("attest-"));
        assert_eq!(attestation.decisions.len(), 2);
        assert!(attestation.decisions[0].included);
        assert!(!attestation.decisions[1].included);
    }

    #[test]
    fn test_attestation_verification() {
        let input_root = hasher::sha256(b"in");
        let output_root = hasher::sha256(b"out");

        let attestation = AttestationBuilder::new("compute")
            .build(input_root, output_root);

        assert!(verify_attestation(&attestation));
    }

    #[test]
    fn test_tampered_attestation_fails() {
        let input_root = hasher::sha256(b"in");
        let output_root = hasher::sha256(b"out");

        let mut attestation = AttestationBuilder::new("compute")
            .build(input_root, output_root);

        // Tamper with function_id
        attestation.function_id = "tampered".into();
        assert!(!verify_attestation(&attestation));
    }

    #[test]
    fn test_attestation_chain() {
        let root0 = hasher::sha256(b"state_0");
        let root1 = hasher::sha256(b"state_1");
        let root2 = hasher::sha256(b"state_2");

        let att1 = AttestationBuilder::new("step_1")
            .build(root0, root1);

        let att2 = AttestationBuilder::new("step_2")
            .chain_after(att1.commitment)
            .build(root1, root2);

        let chain = vec![att1, att2];
        assert!(verify_chain(&chain).is_ok());
    }

    #[test]
    fn test_broken_chain_detected() {
        let root0 = hasher::sha256(b"s0");
        let root1 = hasher::sha256(b"s1");
        let root2 = hasher::sha256(b"s2");

        let att1 = AttestationBuilder::new("step_1")
            .build(root0, root1);

        // Wrong previous commitment
        let att2 = AttestationBuilder::new("step_2")
            .chain_after([0xFF; HASH_SIZE])
            .build(root1, root2);

        let chain = vec![att1, att2];
        assert_eq!(verify_chain(&chain), Err(ChainError::BrokenChain { index: 1 }));
    }

    #[test]
    fn test_attestation_log() {
        let mut log = AttestationLog::new();
        let root0 = hasher::sha256(b"s0");
        let root1 = hasher::sha256(b"s1");

        let key = hasher::compute_key(b"patient");
        let att = AttestationBuilder::new("filter")
            .include(key, "Eligible", "criteria_met")
            .build(root0, root1);

        log.append(att);
        assert_eq!(log.len(), 1);
        assert_eq!(log.decision_counts(), (1, 0));
        assert!(log.verify().is_ok());
    }

    #[test]
    fn test_full_pipeline() {
        // Simulate: input tree -> computation -> output tree -> attestation
        let mut input_tree = SparseMerkleTree::new();
        let mut output_tree = SparseMerkleTree::new();

        // Candidates
        let candidates = vec![
            (hasher::compute_key(b"p1"), b"age:25,healthy:true".to_vec(), true),
            (hasher::compute_key(b"p2"), b"age:16,healthy:true".to_vec(), false),
            (hasher::compute_key(b"p3"), b"age:30,healthy:false".to_vec(), false),
            (hasher::compute_key(b"p4"), b"age:22,healthy:true".to_vec(), true),
        ];

        // Insert all into input tree
        let mut builder = AttestationBuilder::new("enrollment_screen")
            .with_parameters(b"min_age=18,require_healthy=true".to_vec());

        for (key, data, _) in &candidates {
            input_tree.insert(*key, data.clone(), Some("candidate".into()));
        }

        // Apply filter: only include eligible
        for (key, data, eligible) in &candidates {
            if *eligible {
                output_tree.insert(*key, data.clone(), Some("enrolled".into()));
                builder = builder.include(*key, "Met all criteria", "age>=18 AND healthy");
            } else {
                builder = builder.exclude(*key, "Did not meet criteria", "age>=18 AND healthy");
            }
        }

        let attestation = builder.build(input_tree.root(), output_tree.root());

        // Verify attestation
        assert!(verify_attestation(&attestation));
        assert_eq!(attestation.decisions.len(), 4);

        // Verify inclusion proofs for enrolled patients
        for (key, _, eligible) in &candidates {
            if *eligible {
                let proof = output_tree.prove_inclusion(key).unwrap();
                assert_eq!(proof.root, attestation.output_root);
            } else {
                let proof = output_tree.prove_exclusion(key).unwrap();
                assert_eq!(proof.root, attestation.output_root);
            }
        }
    }
}
