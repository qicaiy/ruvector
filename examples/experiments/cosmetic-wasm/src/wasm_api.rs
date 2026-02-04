//! WASM bindings for CoSMeTIC
//!
//! Exposes the Sparse Merkle Tree, proof generation/verification,
//! and computation attestation to JavaScript via wasm-bindgen.

use crate::attestation::{AttestationBuilder, AttestationLog};
use crate::hasher;
use crate::proof::{self, CompactProof};
use crate::tree::SparseMerkleTree;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// WASM-exposed Computational Sparse Merkle Tree
#[wasm_bindgen]
pub struct CosmeticTree {
    tree: SparseMerkleTree,
    log: AttestationLog,
}

#[wasm_bindgen]
impl CosmeticTree {
    /// Create a new empty CoSMeTIC tree
    #[wasm_bindgen(constructor)]
    pub fn new() -> CosmeticTree {
        CosmeticTree {
            tree: SparseMerkleTree::new(),
            log: AttestationLog::new(),
        }
    }

    /// Create a new tree with pre-allocated capacity
    #[wasm_bindgen(js_name = withCapacity)]
    pub fn with_capacity(n: usize) -> CosmeticTree {
        CosmeticTree {
            tree: SparseMerkleTree::with_capacity(n),
            log: AttestationLog::new(),
        }
    }

    /// Get the current root hash as a hex string
    #[wasm_bindgen(js_name = rootHex)]
    pub fn root_hex(&self) -> String {
        hasher::to_hex(&self.tree.root())
    }

    /// Get the current root hash as bytes
    #[wasm_bindgen(js_name = rootBytes)]
    pub fn root_bytes(&self) -> Vec<u8> {
        self.tree.root().to_vec()
    }

    /// Get the number of leaves in the tree
    #[wasm_bindgen(js_name = leafCount)]
    pub fn leaf_count(&self) -> usize {
        self.tree.len()
    }

    /// Check if tree is empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Insert a key-value pair. The key is derived from `key_data` by hashing.
    /// Returns the new root hash as hex.
    #[wasm_bindgen]
    pub fn insert(
        &mut self,
        key_data: &[u8],
        value: &[u8],
        computation_tag: Option<String>,
    ) -> String {
        let key = hasher::compute_key(key_data);
        let result = self.tree.insert(key, value.to_vec(), computation_tag);
        hasher::to_hex(&result.new_root)
    }

    /// Insert with an explicit hex key (must be 64 hex chars = 32 bytes)
    #[wasm_bindgen(js_name = insertWithKey)]
    pub fn insert_with_key(
        &mut self,
        key_hex: &str,
        value: &[u8],
        computation_tag: Option<String>,
    ) -> Result<String, JsValue> {
        let key =
            hasher::from_hex(key_hex).map_err(|e| JsValue::from_str(&format!("Bad key: {}", e)))?;
        let result = self.tree.insert(key, value.to_vec(), computation_tag);
        Ok(hasher::to_hex(&result.new_root))
    }

    /// Insert multiple key-value pairs in batch.
    /// `entries_json` format: [{"key_data": "...", "value": "...", "tag": "..."}]
    /// Returns JSON with new root and count.
    #[wasm_bindgen(js_name = insertBatch)]
    pub fn insert_batch(&mut self, entries_json: &str) -> Result<String, JsValue> {
        let entries: Vec<BatchEntry> = serde_json::from_str(entries_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

        let batch: Vec<_> = entries
            .into_iter()
            .map(|e| {
                let key = hasher::compute_key(e.key_data.as_bytes());
                (key, e.value.into_bytes(), e.tag)
            })
            .collect();

        let result = self.tree.insert_batch(batch);
        let json = serde_json::json!({
            "new_root": hasher::to_hex(&result.new_root),
            "old_root": hasher::to_hex(&result.old_root),
            "count": result.count,
        });
        Ok(json.to_string())
    }

    /// Remove a key. Key is derived from `key_data` by hashing.
    #[wasm_bindgen]
    pub fn remove(&mut self, key_data: &[u8]) -> String {
        let key = hasher::compute_key(key_data);
        let result = self.tree.remove(&key);
        hasher::to_hex(&result.new_root)
    }

    /// Check if a key exists. Key is derived from `key_data` by hashing.
    #[wasm_bindgen]
    pub fn contains(&self, key_data: &[u8]) -> bool {
        let key = hasher::compute_key(key_data);
        self.tree.contains(&key)
    }

    /// Get the value at a key as JSON. Returns null if absent.
    #[wasm_bindgen(js_name = getValue)]
    pub fn get_value(&self, key_data: &[u8]) -> JsValue {
        let key = hasher::compute_key(key_data);
        match self.tree.get(&key) {
            Some(leaf) => {
                let json = serde_json::json!({
                    "value": String::from_utf8_lossy(&leaf.value),
                    "computation_tag": leaf.computation_tag,
                    "timestamp": leaf.timestamp,
                });
                JsValue::from_str(&json.to_string())
            }
            None => JsValue::NULL,
        }
    }

    /// Generate an inclusion proof as JSON. Returns null if key absent.
    #[wasm_bindgen(js_name = proveInclusion)]
    pub fn prove_inclusion(&self, key_data: &[u8]) -> JsValue {
        let key = hasher::compute_key(key_data);
        match self.tree.prove_inclusion(&key) {
            Some(proof) => {
                let compact = CompactProof::from_inclusion(&proof);
                match serde_json::to_string(&compact) {
                    Ok(json) => JsValue::from_str(&json),
                    Err(_) => JsValue::NULL,
                }
            }
            None => JsValue::NULL,
        }
    }

    /// Generate an exclusion proof as JSON. Returns null if key present.
    #[wasm_bindgen(js_name = proveExclusion)]
    pub fn prove_exclusion(&self, key_data: &[u8]) -> JsValue {
        let key = hasher::compute_key(key_data);
        match self.tree.prove_exclusion(&key) {
            Some(proof) => {
                let compact = CompactProof::from_exclusion(&proof);
                match serde_json::to_string(&compact) {
                    Ok(json) => JsValue::from_str(&json),
                    Err(_) => JsValue::NULL,
                }
            }
            None => JsValue::NULL,
        }
    }

    /// Verify an inclusion proof (re-generates from tree state)
    #[wasm_bindgen(js_name = verifyInclusionProof)]
    pub fn verify_inclusion_proof(&self, key_data: &[u8]) -> bool {
        let key = hasher::compute_key(key_data);
        match self.tree.prove_inclusion(&key) {
            Some(proof) => proof::verify_inclusion(&proof).is_valid(),
            None => false,
        }
    }

    /// Verify an exclusion proof
    #[wasm_bindgen(js_name = verifyExclusionProof)]
    pub fn verify_exclusion_proof(&self, key_data: &[u8]) -> bool {
        let key = hasher::compute_key(key_data);
        match self.tree.prove_exclusion(&key) {
            Some(proof) => proof::verify_exclusion(&proof).is_valid(),
            None => false,
        }
    }

    /// Create a computation attestation.
    ///
    /// Takes the current tree state as "input", applies inclusion/exclusion
    /// decisions, and records the attestation.
    ///
    /// `decisions_json` format: [{"key_data": "...", "included": true, "reason": "...", "criterion": "..."}]
    #[wasm_bindgen(js_name = createAttestation)]
    pub fn create_attestation(
        &mut self,
        function_id: &str,
        parameters: &[u8],
        decisions_json: &str,
    ) -> Result<String, JsValue> {
        let decisions: Vec<DecisionInput> = serde_json::from_str(decisions_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid decisions JSON: {}", e)))?;

        let input_root = self.tree.root();

        let mut builder = AttestationBuilder::new(function_id)
            .with_parameters(parameters.to_vec());

        if let Some(prev) = self.log.latest_commitment() {
            builder = builder.chain_after(prev);
        }

        // Process decisions and build output tree state
        for decision in &decisions {
            let key = hasher::compute_key(decision.key_data.as_bytes());
            if decision.included {
                builder = builder.include(key, &decision.reason, &decision.criterion);
            } else {
                builder = builder.exclude(key, &decision.reason, &decision.criterion);
                // Remove excluded keys from tree
                self.tree.remove(&key);
            }
        }

        let output_root = self.tree.root();
        let attestation = builder.build(input_root, output_root);

        let result = serde_json::json!({
            "id": attestation.id,
            "commitment": hasher::to_hex(&attestation.commitment),
            "input_root": hasher::to_hex(&attestation.input_root),
            "output_root": hasher::to_hex(&attestation.output_root),
            "decisions_count": attestation.decisions.len(),
        });

        self.log.append(attestation);

        Ok(result.to_string())
    }

    /// Verify the entire attestation chain
    #[wasm_bindgen(js_name = verifyAttestationChain)]
    pub fn verify_attestation_chain(&self) -> bool {
        self.log.verify().is_ok()
    }

    /// Get attestation log statistics
    #[wasm_bindgen(js_name = attestationStats)]
    pub fn attestation_stats(&self) -> String {
        let (included, excluded) = self.log.decision_counts();
        let json = serde_json::json!({
            "total_attestations": self.log.len(),
            "total_inclusions": included,
            "total_exclusions": excluded,
        });
        json.to_string()
    }

    /// Export tree state as JSON snapshot
    #[wasm_bindgen(js_name = exportSnapshot)]
    pub fn export_snapshot(&self) -> String {
        let snap = self.tree.snapshot();
        serde_json::to_string(&snap).unwrap_or_else(|_| "{}".into())
    }

    /// Export attestation log as JSON
    #[wasm_bindgen(js_name = exportAttestationLog)]
    pub fn export_attestation_log(&self) -> String {
        self.log.to_json().unwrap_or_else(|_| "[]".into())
    }

    /// Compute the hash of arbitrary data (utility for JS)
    #[wasm_bindgen(js_name = computeHash)]
    pub fn compute_hash(data: &[u8]) -> String {
        hasher::to_hex(&hasher::sha256(data))
    }

    /// Get memory usage statistics as JSON
    #[wasm_bindgen(js_name = memoryStats)]
    pub fn memory_stats(&self) -> String {
        let stats = self.tree.memory_stats();
        serde_json::to_string(&stats).unwrap_or_else(|_| "{}".into())
    }
}

/// Input format for inclusion/exclusion decisions from JS
#[derive(Deserialize)]
struct DecisionInput {
    key_data: String,
    included: bool,
    reason: String,
    criterion: String,
}

/// Input format for batch insert entries from JS
#[derive(Deserialize)]
struct BatchEntry {
    key_data: String,
    value: String,
    tag: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_tree_lifecycle() {
        let mut tree = CosmeticTree::new();
        assert!(tree.is_empty());

        let root1 = tree.insert(b"patient_1", b"data_1", Some("enrollment".into()));
        assert!(!tree.is_empty());
        assert_eq!(tree.leaf_count(), 1);

        let root2 = tree.insert(b"patient_2", b"data_2", None);
        assert_ne!(root1, root2);
        assert_eq!(tree.leaf_count(), 2);

        assert!(tree.contains(b"patient_1"));
        assert!(!tree.contains(b"patient_3"));

        tree.remove(b"patient_1");
        assert!(!tree.contains(b"patient_1"));
        assert_eq!(tree.leaf_count(), 1);
    }

    #[test]
    fn test_wasm_proof_verification() {
        let mut tree = CosmeticTree::new();
        tree.insert(b"key_1", b"val_1", None);

        // Verify inclusion proof for existing key
        assert!(tree.verify_inclusion_proof(b"key_1"));

        // Verify exclusion proof for missing key
        assert!(tree.verify_exclusion_proof(b"key_2"));
    }

    #[test]
    fn test_compute_hash() {
        let h1 = CosmeticTree::compute_hash(b"hello");
        let h2 = CosmeticTree::compute_hash(b"hello");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // 32 bytes = 64 hex chars
    }

    #[test]
    fn test_wasm_batch_insert() {
        let mut tree = CosmeticTree::new();
        let json = r#"[
            {"key_data": "p1", "value": "enrolled", "tag": "trial"},
            {"key_data": "p2", "value": "enrolled", "tag": null},
            {"key_data": "p3", "value": "pending", "tag": null}
        ]"#;
        let result = tree.insert_batch(json).unwrap();
        assert_eq!(tree.leaf_count(), 3);
        assert!(result.contains("\"count\":3"));
    }

    #[test]
    fn test_wasm_memory_stats() {
        let mut tree = CosmeticTree::new();
        tree.insert(b"key", b"val", None);
        let stats = tree.memory_stats();
        assert!(stats.contains("leaf_count"));
        assert!(stats.contains("cached_node_count"));
    }

    #[test]
    fn test_wasm_with_capacity() {
        let tree = CosmeticTree::with_capacity(100);
        assert!(tree.is_empty());
    }
}
