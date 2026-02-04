# DDD: CoSMeTIC Domain Model

**Status**: Proposed
**Date**: 2026-02-04
**Authors**: ruv.io, RuVector Team
**Related ADRs**: ADR-001 (Hash Function Selection), ADR-002 (SMT Architecture), ADR-003 (ZK Proof System)

---

## Overview

This document defines the Domain-Driven Design model for the CoSMeTIC framework -- Computational Sparse Merkle Trees with Inclusion/exclusion Certificates. The domain model captures the concepts, relationships, and invariants necessary to implement a Rust WASM crate that provides privacy-preserving verifiable data structures for clinical research and beyond.

The design is informed by the CoSMeTIC paper (Ramanan et al., arXiv:2601.12136, January 2026), which introduces Computational Sparse Merkle Trees (CSMTs) as a zero-knowledge framework for generating verifiable inclusion and exclusion proofs for individual participants' data in clinical studies.

---

## Strategic Design

### Domain Vision Statement

> The CoSMeTIC domain provides cryptographically verifiable, privacy-preserving proof generation and verification for sparse Merkle tree membership, enabling applications where data custodians must prove properties about their datasets (presence, absence, computation correctness) without revealing the underlying data.

### Core Domain

**Proof Generation and Verification** is the core domain. This is the novel capability that differentiates CoSMeTIC from standard authenticated data structures:

- Not just data storage (that is a supporting concern)
- Not just hashing (that is infrastructure)
- **The novel capability**: Generating and verifying cryptographic certificates that prove a key-value pair is included in (or excluded from) a committed dataset, with optional computation correctness attestation, while preserving the privacy of the data

### Supporting Domains

| Domain | Role | Boundary |
|--------|------|----------|
| **Tree Management** | Maintain the sparse Merkle tree state (insert, remove, lookup) | Infrastructure, data structure |
| **Cryptographic Hashing** | Provide hash function abstractions (SHA-256, Poseidon) | Infrastructure, cryptographic primitive |
| **Serialization** | Encode/decode proofs, tree snapshots, keys for transport | Generic, infrastructure |
| **Computation Attestation** | Bind computation results to tree state | Supporting, extends core domain |
| **Clinical Protocol** | Map clinical research operations to tree operations | Application, external |

### Generic Subdomains

- Timestamp management (WASM/native abstraction)
- Hex encoding/decoding for key display
- Error handling and result types
- Configuration and feature flag management

---

## Ubiquitous Language

### Core Terms

| Term | Definition | Context |
|------|------------|---------|
| **Sparse Merkle Tree (SMT)** | A complete binary tree of depth 256 where only populated leaves and their ancestors are explicitly stored; all other positions hold a canonical default empty hash | Data structure |
| **Computational SMT (CSMT)** | An SMT where internal nodes apply an aggregation function (not just hash concatenation) at each level, embedding computation semantics into the tree structure | CoSMeTIC paper |
| **Leaf** | A populated position in the tree holding a key-value pair with optional computation metadata | Data structure |
| **Default Empty Hash** | The canonical hash value `H("cosmetic_empty_leaf")` assigned to all unpopulated leaf positions; propagated up through internal nodes of empty subtrees | Cryptographic constant |
| **Root Hash** | The single 256-bit hash at depth 0 that cryptographically commits to the entire tree state; any change to any leaf changes the root | Commitment |
| **Key** | A 256-bit address derived by hashing the raw data; determines the leaf's position in the tree via bit-by-bit path selection | Addressing |
| **Sibling Path** | The ordered sequence of 256 sibling node hashes from a leaf's position up to the root; constitutes the essential content of a Merkle proof | Proof structure |
| **Inclusion Proof** | A cryptographic certificate demonstrating that a specific key-value pair exists at a specific position in a tree with a known root hash | Core domain |
| **Exclusion Proof** | A cryptographic certificate demonstrating that a specific key does NOT exist in a tree with a known root hash, by showing the position holds the default empty hash | Core domain |
| **Computation Tag** | A string identifier recording which transformation function produced the leaf value; enables auditors to verify not just data presence but processing provenance | Computation attestation |
| **Computation Witness** | Evidence (inputs, intermediate values) that a specific computation was correctly performed to produce a leaf value; used inside ZK proofs | ZK context |
| **Computation Attestation** | A signed binding between an input tree root, an output tree root, and the identity of the computation that transformed one into the other | Attestation context |
| **Tree Depth** | The fixed number of levels in the tree (256), matching the bit width of the hash function output | Structural constant |
| **Path Bit** | A single bit (0 or 1) extracted from the key at a given depth; 0 selects left child, 1 selects right child | Path traversal |
| **Domain Separator** | A prefix byte appended before hashing to ensure leaf hashes, internal node hashes, and attestation hashes cannot collide: `0x00` for leaves, `0x01` for internal nodes, `0x02` for attestations | Security mechanism |
| **Compact Proof** | A space-optimized proof encoding that uses a bitmap to indicate which of the 256 siblings are non-default, followed by only the non-default hashes | Optimization |
| **Mutation Result** | The output of a tree write operation, containing the old root, new root, and the key that was mutated | Event |
| **Tree Snapshot** | A serializable representation of the complete tree state (root, all leaves) at a point in time | Persistence |
| **Verification Key** | In the ZK context, a public artifact derived from the circuit definition that allows any party to verify a ZK proof without knowing the witness | ZK infrastructure |
| **Proving Key** | In the ZK context, a secret artifact used by the prover to generate ZK proofs; derived from the circuit definition during setup | ZK infrastructure |
| **Leaf Transform Ratio (LTR)** | In the CoSMeTIC paper, a ZK proof that the leaf transformation function `L^s` was correctly evaluated for a specific patient's data | CoSMeTIC paper |
| **Merkle Record Path (MRP)** | In the CoSMeTIC paper, a ZK proof that the per-level aggregation functions `A^l` were correctly evaluated along the path from leaf to root | CoSMeTIC paper |
| **Salt** | A secret random value mixed into the leaf transformation to ensure unique identity binding (`mu`) and leaf distinguishability (`tau`) | Privacy mechanism |

### Clinical Research Terms (Application Layer)

| Term | Definition | Context |
|------|------------|---------|
| **Participant** | A clinical trial participant whose data is represented as a leaf in the CSMT | Application |
| **Enrollment** | The act of inserting a participant's transformed data as a leaf in the tree | Application |
| **Exclusion Criterion** | A clinical rule that determines a participant should NOT be in the dataset; mapped to an exclusion proof | Application |
| **Inclusion Criterion** | A clinical rule that determines a participant SHOULD be in the dataset; mapped to an inclusion proof | Application |
| **Audit Trail** | A chronological sequence of tree roots, mutation results, and attestations recording all changes to the study dataset | Regulatory |
| **Statistical Fidelity** | The property that computations performed on the privacy-protected CSMT data produce the same statistical results as computations on raw data | CoSMeTIC paper |
| **Protected Health Information (PHI)** | Personally identifiable health data that must not be revealed to verifiers | Regulatory |

---

## Bounded Contexts

### Context Map

```
+-----------------------------------------------------------------------------+
|                     PROOF GENERATION & VERIFICATION CONTEXT                   |
|                              (Core Domain)                                   |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
|  |  Inclusion    |  |  Exclusion    |  |   Compact     |  |    Batch      |  |
|  |  Prover       |  |  Prover       |  |   Encoder     |  |   Verifier    |  |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
+-----------------------------------------------------------------------------+
         |                   |                   |                   |
         | Uses              | Uses              | Encodes           | Verifies
         v                   v                   v                   v
+-----------------------------------------------------------------------------+
|                       TREE MANAGEMENT CONTEXT                                |
|                        (Supporting Domain)                                   |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
|  |    Sparse     |  |    Leaf       |  |    Path       |  |   Snapshot    |  |
|  |  Merkle Tree  |  |   Storage     |  | Recomputation |  |   Manager     |  |
|  +---------------+  +---------------+  +---------------+  +---------------+  |
+-----------------------------------------------------------------------------+
         |                                        |
         | Hashes via                              | Commits via
         v                                        v
+--------------------------------------+  +--------------------------------------+
|    CRYPTOGRAPHIC HASHING CONTEXT     |  | COMPUTATION ATTESTATION CONTEXT      |
|         (Generic Subdomain)          |  |       (Supporting Domain)            |
|  +------------+  +------------+      |  |  +-------------+  +-------------+   |
|  |  SHA-256   |  |  Poseidon  |      |  |  | Attestation |  |   Witness   |   |
|  |  Hasher    |  |  Hasher    |      |  |  |   Binder    |  |  Generator  |   |
|  +------------+  +------------+      |  |  +-------------+  +-------------+   |
+--------------------------------------+  +--------------------------------------+
                                                   |
                                                   | Proves via (feature-gated)
                                                   v
                                          +--------------------------------------+
                                          |  ZERO-KNOWLEDGE PROOF CONTEXT        |
                                          |     (Supporting Domain, Optional)    |
                                          |  +-------------+  +-------------+   |
                                          |  |   Circuit    |  |    Proof    |   |
                                          |  |  Compiler    |  | Aggregator  |   |
                                          |  +-------------+  +-------------+   |
                                          +--------------------------------------+
```

### Proof Generation and Verification Context (Core)

**Responsibility**: Generate and verify cryptographic certificates of membership (inclusion) and non-membership (exclusion) in a sparse Merkle tree.

**Key Aggregates**:
- InclusionProof
- ExclusionProof
- VerifyResult

**Invariants**:
- An inclusion proof is only valid if the recomputed root (from leaf hash + sibling path) equals the claimed root
- An exclusion proof is only valid if the recomputed root (from DEFAULT_EMPTY + sibling path) equals the claimed root
- Proof sibling count must equal TREE_DEPTH (256)
- A key cannot have both a valid inclusion and a valid exclusion proof against the same root

**Anti-Corruption Layers**:
- Compact Proof Encoder (translates between full and bitmap-compressed proof formats)
- ZK Proof Bridge (when zk-full feature is enabled, translates standard proofs into ZK circuit witnesses)

### Tree Management Context (Supporting)

**Responsibility**: Maintain the sparse Merkle tree data structure, support CRUD operations, and provide sibling path collection for proof generation.

**Key Aggregates**:
- SparseMerkleTree
- LeafData
- MutationResult
- TreeSnapshot

**Invariants**:
- The root hash is always consistent with all stored leaves and cached internal nodes
- Only populated leaves and their ancestor nodes are stored
- The empty tree root is deterministic (DEFAULT_EMPTY propagated up 256 levels)
- Path recomputation after mutation must update all 256 levels from leaf to root

**Relationship**: Conforms to Proof Context (provides sibling paths on demand)

### Cryptographic Hashing Context (Generic Subdomain)

**Responsibility**: Provide cryptographic hash function implementations with domain separation guarantees.

**Key Value Objects**:
- Hash (32-byte digest)
- DEFAULT_EMPTY (canonical empty leaf hash)

**Key Domain Services**:
- sha256(), hash_leaf(), hash_internal(), hash_attestation()
- poseidon_hash() (behind feature flag)
- to_hex(), from_hex(), compute_key()

**Invariants**:
- Domain separation: leaf hashes (0x00 prefix), internal hashes (0x01 prefix), and attestation hashes (0x02 prefix) never collide for the same input data
- Hash output is always exactly 32 bytes (256 bits)
- DEFAULT_EMPTY is computed deterministically from a fixed label string

**Relationship**: Shared Kernel (used by all other contexts)

### Computation Attestation Context (Supporting)

**Responsibility**: Create and verify bindings between computation results and tree state, recording what functions were applied to produce leaf values.

**Key Aggregates**:
- ComputationAttestation

**Key Value Objects**:
- AttestationHash
- FunctionIdentifier

**Invariants**:
- An attestation must reference valid input and output tree roots
- The function_id must identify a known, registered computation
- The attestation hash deterministically binds all attestation components

**Relationship**: Customer-Supplier with Tree Management Context (reads tree roots) and with ZK Proof Context (supplies witnesses for ZK computation proofs)

### Zero-Knowledge Proof Context (Supporting, Optional)

**Responsibility**: When enabled via `zk-full` feature, compile ZK circuits for CSMT verification, generate ZK proofs, and verify them.

**Key Aggregates**:
- ZkCircuit (defines the constraint system)
- ZkProof (the generated proof artifact)
- VerificationKey (public verification parameters)

**Key Domain Services**:
- Circuit compilation (CSMT membership circuit)
- Proof generation (prover)
- Proof verification (verifier)
- Recursive proof aggregation

**Invariants**:
- A valid ZK proof must verify against its verification key
- The circuit must encode exactly 256 levels of Poseidon hashing (matching tree depth)
- Recursive aggregation preserves soundness (if any sub-proof is invalid, the aggregate is invalid)

**Relationship**: Downstream consumer of Proof Generation Context (converts standard proofs into ZK witnesses) and Computation Attestation Context (adds computation correctness to membership proofs)

---

## Aggregates

### SparseMerkleTree (Root Aggregate)

The central aggregate representing the authenticated data structure.

```
+---------------------------------------------------------------------+
|                      SPARSE MERKLE TREE                              |
|                    (Aggregate Root)                                   |
+---------------------------------------------------------------------+
|  root: Hash                                                          |
|  leaves: HashMap<Hash, LeafData>                                     |
|  nodes: HashMap<(u16, Hash), Hash>                                   |
+---------------------------------------------------------------------+
|  +---------------------------------------------------------------+  |
|  | LeafData (Value Object)                                       |  |
|  |  value: Vec<u8>                                               |  |
|  |  computation_tag: Option<String>                              |  |
|  |  timestamp: u64                                               |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|  Commands:                                                           |
|  - insert(key, value, computation_tag) -> MutationResult             |
|  - remove(key) -> MutationResult                                     |
|  - get(key) -> Option<LeafData>                                      |
|  - contains(key) -> bool                                             |
|  - prove_inclusion(key) -> Option<InclusionProof>                    |
|  - prove_exclusion(key) -> Option<ExclusionProof>                    |
|  - keys() -> Vec<Hash>                                               |
|  - snapshot() -> TreeSnapshot                                        |
+---------------------------------------------------------------------+
|  Invariants:                                                         |
|  - root == recompute_from_all_leaves() (always consistent)           |
|  - prove_inclusion returns None for absent keys                      |
|  - prove_exclusion returns None for present keys                     |
|  - insert/remove always recompute the full 256-level path            |
|  - len() == leaves.len() (accurate count)                            |
+---------------------------------------------------------------------+
```

### InclusionProof (Value Object)

An immutable certificate that a key-value pair exists in the tree.

```
+---------------------------------------------------------------------+
|                      INCLUSION PROOF                                 |
|                    (Value Object)                                    |
+---------------------------------------------------------------------+
|  key: Hash                          // 32 bytes                      |
|  leaf_data: LeafData                // Variable size                 |
|  siblings: Vec<Hash>                // 256 * 32 = 8,192 bytes        |
|  root: Hash                         // 32 bytes                      |
+---------------------------------------------------------------------+
|  Invariants:                                                         |
|  - siblings.len() == 256                                             |
|  - Recomputing root from leaf hash + siblings must equal root        |
|  - key matches the path described by sibling ordering                |
+---------------------------------------------------------------------+
```

### ExclusionProof (Value Object)

An immutable certificate that a key does NOT exist in the tree.

```
+---------------------------------------------------------------------+
|                      EXCLUSION PROOF                                 |
|                    (Value Object)                                    |
+---------------------------------------------------------------------+
|  key: Hash                          // 32 bytes                      |
|  siblings: Vec<Hash>                // 256 * 32 = 8,192 bytes        |
|  root: Hash                         // 32 bytes                      |
+---------------------------------------------------------------------+
|  Invariants:                                                         |
|  - siblings.len() == 256                                             |
|  - Recomputing root from DEFAULT_EMPTY + siblings must equal root    |
|  - The leaf at position `key` holds DEFAULT_EMPTY (implicit)         |
+---------------------------------------------------------------------+
```

### CompactProof (Value Object)

A space-optimized proof encoding that omits default-empty siblings.

```
+---------------------------------------------------------------------+
|                      COMPACT PROOF                                   |
|                    (Value Object)                                    |
+---------------------------------------------------------------------+
|  key: Hash                          // 32 bytes                      |
|  is_inclusion: bool                 // 1 byte                        |
|  leaf_value: Option<Vec<u8>>        // Variable (inclusion only)     |
|  computation_tag: Option<String>    // Variable (inclusion only)     |
|  sibling_bitmap: Vec<u8>            // 32 bytes (256-bit bitmap)     |
|  non_default_siblings: Vec<Hash>    // M * 32 bytes (M << 256)       |
|  root: Hash                         // 32 bytes                      |
+---------------------------------------------------------------------+
|  Invariants:                                                         |
|  - popcount(sibling_bitmap) == non_default_siblings.len()            |
|  - Expanding via bitmap + DEFAULT_EMPTY fill yields full proof       |
|  - byte_size() is always less than the full proof size               |
+---------------------------------------------------------------------+
```

### ComputationAttestation (Entity)

A signed binding between computation identity and tree state transitions.

```
+---------------------------------------------------------------------+
|                  COMPUTATION ATTESTATION                             |
|                       (Entity)                                       |
+---------------------------------------------------------------------+
|  attestation_id: Hash               // Unique identifier (derived)   |
|  input_root: Hash                   // Tree root before computation  |
|  output_root: Hash                  // Tree root after computation   |
|  function_id: Vec<u8>               // Identifier of the function    |
|  params: Vec<u8>                    // Serialized function params    |
|  attestation_hash: Hash             // H(0x02 || inputs...)          |
|  timestamp: u64                     // When attestation was created  |
+---------------------------------------------------------------------+
|  Invariants:                                                         |
|  - attestation_hash == hash_attestation(input_root, output_root,     |
|                                         function_id, params)         |
|  - input_root and output_root must reference valid tree states       |
|  - function_id must identify a registered computation                |
+---------------------------------------------------------------------+
```

### MutationResult (Value Object)

The output of any tree write operation.

```
+---------------------------------------------------------------------+
|                     MUTATION RESULT                                   |
|                    (Value Object)                                    |
+---------------------------------------------------------------------+
|  new_root: Hash                     // Root after mutation           |
|  old_root: Hash                     // Root before mutation          |
|  key: Hash                          // The key that was mutated      |
+---------------------------------------------------------------------+
|  Invariants:                                                         |
|  - If tree was actually modified, new_root != old_root               |
|  - key must be a valid 256-bit hash                                  |
+---------------------------------------------------------------------+
```

### TreeSnapshot (Value Object)

A serializable capture of the complete tree state.

```
+---------------------------------------------------------------------+
|                     TREE SNAPSHOT                                     |
|                    (Value Object)                                    |
+---------------------------------------------------------------------+
|  root: Hash                         // Root hash at snapshot time    |
|  leaves: HashMap<Hash, LeafData>    // All populated leaves          |
|  leaf_count: usize                  // Count of populated leaves     |
+---------------------------------------------------------------------+
|  Invariants:                                                         |
|  - leaf_count == leaves.len()                                        |
|  - Reconstructing a tree from this snapshot yields the same root     |
+---------------------------------------------------------------------+
```

### VerifyResult (Value Object)

The output of proof verification.

```
+---------------------------------------------------------------------+
|                      VERIFY RESULT                                   |
|                    (Value Object)                                    |
+---------------------------------------------------------------------+
|  Valid                               // Proof verified successfully  |
|  Invalid(reason: String)             // Proof verification failed    |
+---------------------------------------------------------------------+
|  Invariants:                                                         |
|  - Exactly one variant at a time                                     |
|  - Invalid always carries a non-empty reason string                  |
+---------------------------------------------------------------------+
```

---

## Domain Events

### Tree Lifecycle Events

| Event | Trigger | Payload | Consumers |
|-------|---------|---------|-----------|
| `LeafInserted` | `insert()` called | key, value, computation_tag, old_root, new_root | Audit trail, attestation context |
| `LeafRemoved` | `remove()` called | key, old_root, new_root | Audit trail, attestation context |
| `LeafUpdated` | `insert()` on existing key | key, old_value, new_value, old_root, new_root | Audit trail |
| `RootUpdated` | Any mutation | old_root, new_root, cause (insert/remove) | All contexts |
| `TreeCreated` | `new()` called | empty_root | Session management |

### Proof Events

| Event | Trigger | Payload | Consumers |
|-------|---------|---------|-----------|
| `InclusionProofGenerated` | `prove_inclusion()` succeeds | key, root, proof_size | Audit trail, metrics |
| `ExclusionProofGenerated` | `prove_exclusion()` succeeds | key, root, proof_size | Audit trail, metrics |
| `ProofVerified` | `verify_inclusion/exclusion()` returns Valid | key, root, proof_type | Audit trail |
| `ProofRejected` | `verify_inclusion/exclusion()` returns Invalid | key, root, proof_type, reason | Security monitoring |
| `BatchVerified` | `verify_batch_*()` completes | count_valid, count_invalid, root | Metrics |

### Attestation Events

| Event | Trigger | Payload | Consumers |
|-------|---------|---------|-----------|
| `ComputationAttested` | Attestation created | attestation_hash, function_id, input_root, output_root | Audit trail, regulatory |
| `AttestationVerified` | Attestation hash checked | attestation_hash, valid | Audit trail |

### ZK Events (Feature-Gated)

| Event | Trigger | Payload | Consumers |
|-------|---------|---------|-----------|
| `ZkCircuitCompiled` | Circuit setup complete | circuit_id, constraint_count, setup_time_ms | Metrics |
| `ZkProofGenerated` | Proving completes | proof_id, proof_size_bytes, generation_time_ms | Metrics, transport |
| `ZkProofVerified` | Verification completes | proof_id, valid, verification_time_ms | Audit trail |
| `ZkProofsAggregated` | Recursive aggregation | aggregate_proof_id, sub_proof_count | Metrics |

---

## Domain Services

### ProofService

The primary domain service for generating and verifying proofs.

```rust
trait ProofService {
    /// Verify an inclusion proof against its claimed root
    fn verify_inclusion(&self, proof: &InclusionProof) -> VerifyResult;

    /// Verify an exclusion proof against its claimed root
    fn verify_exclusion(&self, proof: &ExclusionProof) -> VerifyResult;

    /// Batch verify multiple inclusion proofs
    fn verify_batch_inclusion(&self, proofs: &[InclusionProof]) -> Vec<VerifyResult>;

    /// Batch verify multiple exclusion proofs
    fn verify_batch_exclusion(&self, proofs: &[ExclusionProof]) -> Vec<VerifyResult>;

    /// Compress a proof to compact encoding
    fn compact_inclusion(&self, proof: &InclusionProof) -> CompactProof;

    /// Compress an exclusion proof to compact encoding
    fn compact_exclusion(&self, proof: &ExclusionProof) -> CompactProof;
}
```

### TreeService

Orchestrates tree operations and proof generation.

```rust
trait TreeService {
    /// Insert a leaf and return the mutation result
    fn insert(&mut self, key: Hash, value: Vec<u8>, tag: Option<String>) -> MutationResult;

    /// Remove a leaf and return the mutation result
    fn remove(&mut self, key: &Hash) -> MutationResult;

    /// Generate an inclusion proof for a present key
    fn prove_inclusion(&self, key: &Hash) -> Option<InclusionProof>;

    /// Generate an exclusion proof for an absent key
    fn prove_exclusion(&self, key: &Hash) -> Option<ExclusionProof>;

    /// Get the current root hash
    fn root(&self) -> Hash;

    /// Export tree state as a snapshot
    fn snapshot(&self) -> TreeSnapshot;

    /// Restore tree from a snapshot
    fn restore(snapshot: TreeSnapshot) -> Self;
}
```

### AttestationService

Creates and verifies computation attestations.

```rust
trait AttestationService {
    /// Create an attestation binding input/output roots to a computation
    fn attest(
        &self,
        input_root: &Hash,
        output_root: &Hash,
        function_id: &[u8],
        params: &[u8],
    ) -> ComputationAttestation;

    /// Verify an attestation hash
    fn verify(&self, attestation: &ComputationAttestation) -> bool;

    /// Build a chain of attestations (audit trail)
    fn chain(&self, attestations: &[ComputationAttestation]) -> Vec<Hash>;
}
```

### ZkProofService (Feature-Gated)

Full ZK proof generation and verification when `zk-full` is enabled.

```rust
#[cfg(feature = "zk-full")]
trait ZkProofService {
    /// Compile the CSMT membership circuit
    fn setup(&self) -> (ProvingKey, VerificationKey);

    /// Generate a ZK proof of inclusion
    fn prove_inclusion_zk(
        &self,
        pk: &ProvingKey,
        key: &Hash,
        value: &[u8],
        siblings: &[Hash],
        root: &Hash,
    ) -> ZkProof;

    /// Generate a ZK proof of exclusion
    fn prove_exclusion_zk(
        &self,
        pk: &ProvingKey,
        key: &Hash,
        siblings: &[Hash],
        root: &Hash,
    ) -> ZkProof;

    /// Verify a ZK proof
    fn verify_zk(&self, vk: &VerificationKey, proof: &ZkProof) -> bool;

    /// Aggregate multiple ZK proofs into one
    fn aggregate(&self, proofs: &[ZkProof]) -> ZkProof;
}
```

---

## Repositories

### TreeRepository

Persistence interface for tree state.

```rust
trait TreeRepository {
    /// Store a tree snapshot
    async fn store_snapshot(&self, snapshot: TreeSnapshot) -> Result<(), StoreError>;

    /// Load the latest snapshot
    async fn load_latest(&self) -> Option<TreeSnapshot>;

    /// Load a snapshot by its root hash
    async fn load_by_root(&self, root: &Hash) -> Option<TreeSnapshot>;

    /// List all stored snapshot roots (for audit trail)
    async fn list_roots(&self) -> Vec<(Hash, u64)>; // (root, timestamp)
}
```

### ProofRepository

Storage interface for generated proofs (for caching and audit).

```rust
trait ProofRepository {
    /// Store an inclusion proof
    async fn store_inclusion(&self, proof: &InclusionProof) -> Result<(), StoreError>;

    /// Store an exclusion proof
    async fn store_exclusion(&self, proof: &ExclusionProof) -> Result<(), StoreError>;

    /// Retrieve a proof by key and root
    async fn find_by_key_and_root(
        &self,
        key: &Hash,
        root: &Hash,
    ) -> Option<CompactProof>;
}
```

### AttestationRepository

Storage interface for computation attestations.

```rust
trait AttestationRepository {
    /// Store an attestation
    async fn store(&self, attestation: &ComputationAttestation) -> Result<(), StoreError>;

    /// Retrieve attestations for a given output root
    async fn find_by_output_root(&self, root: &Hash) -> Vec<ComputationAttestation>;

    /// Retrieve the full attestation chain leading to a given root
    async fn chain_to_root(&self, root: &Hash) -> Vec<ComputationAttestation>;
}
```

---

## Factories

### ProofFactory

Constructs proof objects from tree state.

```rust
impl ProofFactory {
    /// Create an inclusion proof from tree internals
    fn create_inclusion(
        key: Hash,
        leaf_data: LeafData,
        siblings: Vec<Hash>,
        root: Hash,
    ) -> InclusionProof {
        InclusionProof {
            key,
            leaf_data,
            siblings,
            root,
        }
    }

    /// Create an exclusion proof from tree internals
    fn create_exclusion(
        key: Hash,
        siblings: Vec<Hash>,
        root: Hash,
    ) -> ExclusionProof {
        ExclusionProof {
            key,
            siblings,
            root,
        }
    }

    /// Create a compact proof from either proof type
    fn compact_from_inclusion(proof: &InclusionProof) -> CompactProof {
        CompactProof::from_inclusion(proof)
    }

    fn compact_from_exclusion(proof: &ExclusionProof) -> CompactProof {
        CompactProof::from_exclusion(proof)
    }
}
```

### AttestationFactory

Constructs computation attestation objects.

```rust
impl AttestationFactory {
    fn create(
        input_root: &Hash,
        output_root: &Hash,
        function_id: &[u8],
        params: &[u8],
    ) -> ComputationAttestation {
        let attestation_hash = hasher::hash_attestation(
            input_root, output_root, function_id, params
        );
        ComputationAttestation {
            attestation_id: attestation_hash,
            input_root: *input_root,
            output_root: *output_root,
            function_id: function_id.to_vec(),
            params: params.to_vec(),
            attestation_hash,
            timestamp: current_timestamp(),
        }
    }
}
```

---

## Invariants and Business Rules

### Proof Invariants

1. **Sibling Count**: Every proof (inclusion and exclusion) must contain exactly 256 sibling hashes, matching the tree depth. Any other count is immediately invalid.

2. **Root Consistency**: The root hash recomputed from the proof data (leaf hash + sibling path) must exactly equal the claimed root hash. A single bit difference means the proof is invalid.

3. **Mutual Exclusivity**: For a given key and root, a valid inclusion proof and a valid exclusion proof cannot both exist. If inclusion is valid, the leaf is populated; if exclusion is valid, the leaf is empty. These are mutually exclusive states.

4. **Deterministic Verification**: Given the same proof bytes, verification must always produce the same result. There is no randomness in verification.

5. **Compact Fidelity**: A compact proof must be expandable back to a full proof that verifies identically. The bitmap + non-default siblings + DEFAULT_EMPTY fill must reconstruct the original sibling vector.

### Tree Invariants

1. **Root-Leaf Consistency**: The root hash must always be consistent with all stored leaves. After any mutation, the path from the mutated leaf to the root must be recomputed.

2. **Empty Tree Root**: A tree with zero populated leaves must have a root equal to DEFAULT_EMPTY propagated up 256 levels. This root is deterministic and serves as the genesis root.

3. **Insertion Determinism**: Inserting the same key-value pairs in any order must produce the same root hash. The tree structure is determined entirely by the key addresses, not insertion order.

4. **Domain Separation**: Leaf hashes must use the `0x00` domain separator. Internal hashes must use `0x01`. Attestation hashes must use `0x02`. No domain may use another's separator.

5. **Key Uniqueness**: Each key maps to exactly one leaf position. Inserting at an existing key overwrites the previous value (update semantics, not multi-value).

### Attestation Invariants

1. **Hash Binding**: The attestation hash must deterministically equal `H(0x02 || input_root || output_root || function_id || params)`. Any mismatch between the stored hash and the recomputed hash invalidates the attestation.

2. **Root Referential Integrity**: Both input_root and output_root must reference tree states that existed (or exist). An attestation referencing a root that was never computed is meaningless.

3. **Function Identity**: The function_id must identify a known, registered computation. Unknown function IDs should be flagged during verification.

---

## Context Boundary Summary

| Boundary | Upstream | Downstream | Integration Pattern |
|----------|----------|------------|---------------------|
| Tree -> Proof | Tree Management | Proof Generation | Published Language (InclusionProof, ExclusionProof) |
| Hash -> Tree | Cryptographic Hashing | Tree Management | Shared Kernel (Hash type, hash functions) |
| Hash -> Proof | Cryptographic Hashing | Proof Verification | Shared Kernel (Hash type, hash functions) |
| Tree -> Attestation | Tree Management | Computation Attestation | Domain Events (RootUpdated, LeafInserted) |
| Attestation -> ZK | Computation Attestation | ZK Proof Context | Customer-Supplier (witnesses for circuits) |
| Proof -> ZK | Proof Generation | ZK Proof Context | Conformist (standard proofs become ZK witnesses) |
| Proof -> Compact | Proof Generation | Serialization | ACL (CompactProof adapter) |

---

## Module Mapping

The domain model maps to the `cosmetic-wasm` crate source modules as follows:

| Module | DDD Context | Aggregates / Services |
|--------|-------------|----------------------|
| `src/hasher.rs` | Cryptographic Hashing Context | Hash, DEFAULT_EMPTY, sha256, hash_leaf, hash_internal, hash_attestation, poseidon_hash |
| `src/tree.rs` | Tree Management Context | SparseMerkleTree, LeafData, MutationResult, TreeSnapshot, InclusionProof, ExclusionProof |
| `src/proof.rs` | Proof Generation & Verification Context | verify_inclusion, verify_exclusion, verify_batch_*, CompactProof |
| `src/attestation.rs` (planned) | Computation Attestation Context | ComputationAttestation, AttestationService |
| `src/zk.rs` (planned, feature-gated) | Zero-Knowledge Proof Context | ZkCircuit, ZkProof, ZkProofService |
| `src/lib.rs` | WASM API surface | wasm_bindgen exports bridging all contexts |

---

## Clinical Research Application Mapping

The following table maps CoSMeTIC paper concepts to domain model elements:

| Paper Concept | Domain Model Element | Notes |
|---------------|---------------------|-------|
| Patient data tuple (delta, mu, tau) | LeafData.value (serialized) + Salt (in witness) | Raw PHI is never stored in cleartext |
| Leaf transformation L^s | computation_tag = "ks_transform" etc. | Identifies which statistical transform was applied |
| Leaf index N_u = Decimal[H(L^s(...))] | Key = compute_key(transformed_data) | 256-bit address derived from transformed data hash |
| Aggregation A^l at level l | Aggregator trait (CSMT extension) | Default: hash_internal; CSMT: per-level reduction |
| LTR proof | ZkProof (type: LTR) | Proves leaf transformation correctness |
| MRP proof | ZkProof (type: MRP) | Proves per-level aggregation correctness |
| Inclusion verification (VerInc) | verify_inclusion() + verify_zk() | Standard proof + optional ZK proof |
| Exclusion verification | verify_exclusion() + verify_zk() | Leaf at position is DEFAULT_EMPTY |
| CosmeticCROBuild | AttestationService.attest() + tree mutations | Orchestrates the full CRO build process |
| EZKL scale parameter | ZK circuit configuration | Controls fixed-point precision in ZK circuits |
| KS/LRT/ACC test circuits | ZkCircuit instances per test type | Each statistical test compiles to a separate circuit |

---

## References

- Ramanan, P. et al. "CoSMeTIC: Zero-Knowledge Computational Sparse Merkle Trees with Inclusion-Exclusion Proofs for Clinical Research." arXiv:2601.12136, January 2026.
- Evans, Eric. "Domain-Driven Design." Addison-Wesley, 2003.
- Vernon, Vaughn. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
- ADR-001: Cryptographic Hash Function Selection (this crate)
- ADR-002: Sparse Merkle Tree Architecture (this crate)
- ADR-003: Zero-Knowledge Proof System Selection (this crate)
