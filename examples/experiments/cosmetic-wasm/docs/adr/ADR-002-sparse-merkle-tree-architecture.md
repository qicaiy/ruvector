# ADR-002: Sparse Merkle Tree Architecture

**Status**: Accepted
**Date**: 2026-02-04
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-04 | ruv.io | Initial architecture based on CoSMeTIC paper analysis |

---

## Context

### What is a Sparse Merkle Tree?

A Sparse Merkle Tree (SMT) is an authenticated data structure built on a complete binary tree of fixed depth K, where K is typically 256 (matching the output size of the hash function). The tree has 2^256 possible leaf positions, but only a tiny fraction are populated with actual data. All unpopulated leaves hold a canonical "default empty" hash value.

The key properties of an SMT that make it suitable for the CoSMeTIC framework are:

1. **Deterministic addressing**: A key (256-bit hash of the data) directly determines the leaf position. The i-th bit of the key selects left (0) or right (1) at depth i. No balancing or reordering is needed.

2. **Inclusion proofs**: To prove a key K exists with value V, provide the sibling hashes along the path from the leaf at position K to the root. The verifier recomputes the root from the leaf hash and siblings, and checks it matches the published root. This requires exactly K=256 hash evaluations and K=256 sibling hashes.

3. **Exclusion proofs**: To prove a key K does NOT exist, provide the sibling hashes along the path from position K (which holds the default empty hash) to the root. The verifier starts from DEFAULT_EMPTY instead of a leaf hash and walks up. If the recomputed root matches, the key is provably absent.

4. **Root commitment**: The root hash is a binding commitment to the entire tree state. Any change to any leaf changes the root hash. This enables compact state attestation.

### What Makes a SMT "Computational" (CSMT)?

The CoSMeTIC paper (Ramanan et al., arXiv:2601.12136) introduces a critical distinction: **Computational Sparse Merkle Trees (CSMTs)** embed reduction operations at each recursion level.

In a conventional SMT:
```
internal_node = H(left_child || right_child)
```

In a CSMT:
```
internal_node = A^l(left_child, right_child)
```

Where `A^l` is an **aggregation function** parameterized per level `l`. The aggregation function can perform:

- **Numeric reduction**: Sum, mean, variance accumulation across subtrees
- **Statistical computation**: Running sufficient statistics for hypothesis tests
- **Transformation chaining**: Each level applies a different step of a multi-step computation

At the leaf level, a **leaf transformation function** `L^s` converts raw individual data into the transformed space:
```
leaf_value = L^s(delta, mu, tau)
```

Where:
- `delta` is the raw datum (e.g., patient clinical data)
- `mu` is a secret salt vector ensuring unique identity binding
- `tau` is a transform salt ensuring leaf distinguishability

The leaf position is determined by:
```
leaf_index = Decimal[H(L^s(delta, mu, tau))]
```

This means the tree encodes not just data presence but **computation state** -- the root hash attests to the result of a specific computation applied across all included data.

### Design Requirements

| Requirement | Description | Source |
|-------------|-------------|--------|
| R1 | 256-bit address space (2^256 leaf positions) | SMT standard |
| R2 | O(log n) insert/lookup/proof-generation where n = populated leaves | Performance |
| R3 | Sparse storage: only populated leaves and their ancestors consume memory | WASM memory constraint |
| R4 | Computation-aware leaf encoding with metadata | CoSMeTIC paper |
| R5 | Domain-separated hashing (leaf vs. internal vs. attestation) | Security (second-preimage resistance) |
| R6 | Compact proof encoding (omit default-empty siblings) | Bandwidth/WASM |
| R7 | Deterministic root computation (same leaves -> same root) | Correctness |
| R8 | WASM-compatible (no OS dependencies, no file I/O) | Deployment target |
| R9 | Aggregation function extensibility at internal nodes | CSMT spec |
| R10 | Batch proof generation for multiple keys | Efficiency |

### Options Evaluated

#### Option A: Standard Binary SMT (HashMap-backed)

A complete binary tree of depth 256 represented sparsely using a HashMap. Only non-empty leaves and the internal nodes along their root paths are stored. Empty subtrees resolve to the pre-computed DEFAULT_EMPTY hash.

- **Pros**: Simple, deterministic, well-understood. Direct mapping to SMT proofs. O(log n) operations where n = populated leaves.
- **Cons**: Each leaf insertion recomputes 256 internal nodes along the path. HashMap overhead per node. No structural sharing between paths.

#### Option B: Patricia/Radix Trie (Compressed Path)

A radix trie that compresses sequences of single-child internal nodes into "extension nodes" containing the shared key prefix. Only branching points and leaves are stored.

- **Pros**: Much fewer internal nodes for sparse trees. Used by Ethereum's Modified Merkle Patricia Trie.
- **Cons**: Variable-depth proofs (proof size depends on tree population density). More complex proof structure. Harder to verify in ZK circuits due to variable-length paths. Does not match the fixed-depth model assumed by the CoSMeTIC paper's MRP (Merkle Record Path) proof protocol.

#### Option C: Jellyfish Merkle Tree (from Diem/Aptos)

A variant of a sparse Merkle tree with internal nodes, leaf nodes, and null nodes. Uses 4-bit nibbles instead of individual bits for path traversal, reducing tree depth to 64 levels.

- **Pros**: Proven at scale (Aptos blockchain). Fewer levels mean shorter proofs. Good caching behavior.
- **Cons**: 4-bit nibble addressing changes the tree structure and proof format. Not directly compatible with the CoSMeTIC paper's binary path model. More complex node types.

#### Option D: Compact Binary SMT with Lazy Evaluation

A standard 256-depth binary SMT with the following optimizations:

1. **Lazy path computation**: Only recompute internal nodes along the modified path, not the entire tree.
2. **Sparse HashMap storage**: Store only populated leaves and cached internal node hashes. Default-empty subtrees are never materialized.
3. **Computation-aware leaf encoding**: Each leaf carries metadata (`computation_tag`, `timestamp`) alongside the raw value.
4. **Compact proof encoding**: Bitmap-based sibling compression that omits default-empty siblings from serialized proofs.
5. **Cache key design**: Internal nodes are cached at `(depth, prefix_bits)` positions, enabling efficient subtree hash lookups.

- **Pros**: Fixed-depth proofs (exactly 256 siblings per proof) matching the CoSMeTIC paper. O(K) path recomputation per mutation (K=256). Sparse storage proportional to populated leaves. Straightforward extension to aggregation functions. WASM-safe.
- **Cons**: 256-level paths mean proofs are larger than Patricia trie proofs for very sparse trees. HashMap per-entry overhead.

---

## Decision

### Adopt Option D: Compact Binary SMT with Lazy Evaluation and Computation-Aware Leaf Encoding

We implement the Computational Sparse Merkle Tree as a compact binary SMT with the following architecture:

### Tree Structure

```
                         Root (depth 0)
                        /              \
                   H(0x01||L||R)    H(0x01||L||R)
                   /        \       /        \
                 ...        ...   ...        ...

           depth 254        ...        ...
            /     \
     depth 255   depth 255
      /             \
  Leaf(key,val)   DEFAULT_EMPTY
```

- **Depth**: Fixed at 256 levels (matching 256-bit SHA-256 / Poseidon output)
- **Address space**: 2^256 possible leaf positions
- **Path selection**: Bit `i` of the key selects left (0) or right (1) at depth `i`
- **Domain separation**: `0x00` prefix for leaf hashes, `0x01` prefix for internal nodes, `0x02` prefix for attestations

### Storage Model

```
SparseMerkleTree {
    leaves: HashMap<Hash, LeafData>        // Only populated leaves
    nodes:  HashMap<(u16, Hash), Hash>     // Cached internal node hashes
    root:   Hash                           // Current root hash
}
```

**Memory analysis for a tree with N populated leaves**:

| Component | Size per entry | Total for N leaves |
|-----------|---------------|-------------------|
| `leaves` map entries | ~120 bytes (key + LeafData + HashMap overhead) | ~120N bytes |
| `nodes` cache entries | ~100 bytes (key tuple + hash + HashMap overhead) | ~100 * 256 * N bytes (worst case) |
| `root` | 32 bytes | 32 bytes |

For N=10,000 leaves: ~256MB worst case (all paths unique), ~25MB typical (shared path prefixes).

**WASM memory budget**: With a 4GB WASM linear memory ceiling, the tree supports approximately 150,000 fully-unique-path leaves in worst case, or approximately 1.5 million leaves with typical path sharing.

### Computation-Aware Leaf Encoding

Each leaf stores metadata that links the value to the computation that produced it:

```rust
struct LeafData {
    value: Vec<u8>,                    // Raw value bytes
    computation_tag: Option<String>,   // What computation produced this value
    timestamp: u64,                    // Insertion timestamp (Unix ms)
}
```

The `computation_tag` field is critical for the CoSMeTIC framework: it records the identity of the leaf transformation function `L^s` that was applied. This enables auditors to verify not just that data is present, but what processing was applied to produce the stored value.

In the clinical research context from the paper:
- `computation_tag = "ks_transform"` indicates a Kolmogorov-Smirnov test transform
- `computation_tag = "lrt_transform"` indicates a likelihood-ratio test transform
- `computation_tag = "logistic_regression"` indicates logistic regression feature extraction

### Path Recomputation Algorithm

When a leaf is inserted, updated, or removed, only the 256 internal nodes along the path from that leaf to the root are recomputed:

```
recompute_path(key):
    current_hash = hash_leaf(key, leaf.value) OR DEFAULT_EMPTY
    for depth in (255, 254, ..., 1, 0):
        sibling_hash = lookup_sibling(key, depth)
        if bit(key, depth) == 0:
            parent = hash_internal(current_hash, sibling_hash)
        else:
            parent = hash_internal(sibling_hash, current_hash)
        cache[(depth, prefix(key, depth))] = parent
        current_hash = parent
    root = current_hash
```

**Complexity**: O(K) = O(256) hash evaluations per mutation, independent of the number of populated leaves.

### Proof Structures

#### Inclusion Proof

```rust
struct InclusionProof {
    key: Hash,              // 32 bytes: the key being proved
    leaf_data: LeafData,    // Variable: the value + metadata at that key
    siblings: Vec<Hash>,    // 256 * 32 = 8,192 bytes: sibling hashes leaf-to-root
    root: Hash,             // 32 bytes: the root this proof is valid against
}
```

**Full inclusion proof size**: ~8,260 bytes + leaf value size

#### Exclusion Proof

```rust
struct ExclusionProof {
    key: Hash,              // 32 bytes: the key being proved absent
    siblings: Vec<Hash>,    // 256 * 32 = 8,192 bytes: sibling hashes
    root: Hash,             // 32 bytes: the root this proof is valid against
}
```

**Full exclusion proof size**: ~8,256 bytes

#### Compact Proof (Bitmap Compression)

For sparse trees, most siblings are DEFAULT_EMPTY. The compact encoding uses a 32-byte bitmap (256 bits, one per level) to indicate which siblings are non-default, followed by only the non-default sibling hashes:

```rust
struct CompactProof {
    key: Hash,                          // 32 bytes
    is_inclusion: bool,                 // 1 byte
    leaf_value: Option<Vec<u8>>,        // Variable (inclusion only)
    computation_tag: Option<String>,    // Variable (inclusion only)
    sibling_bitmap: Vec<u8>,            // 32 bytes (256 bits)
    non_default_siblings: Vec<Hash>,    // M * 32 bytes where M = popcount(bitmap)
    root: Hash,                         // 32 bytes
}
```

**Compact proof size for a tree with N populated leaves**: Approximately `97 + 32 * ceil(log2(N))` bytes, since only ~log2(N) siblings along the path are non-default.

| Populated Leaves (N) | Non-default Siblings (M) | Compact Proof Size |
|----------------------|--------------------------|-------------------|
| 1 | 1 | ~129 bytes |
| 100 | ~7 | ~321 bytes |
| 10,000 | ~14 | ~545 bytes |
| 1,000,000 | ~20 | ~737 bytes |

This represents a **91-98% reduction** from the full 8,256-byte proof for typical tree populations.

### Aggregation Function Extension Point

To support the CSMT's per-level aggregation functions, the tree provides an extension point:

```rust
/// Aggregation function applied at internal nodes
/// Default implementation: standard hash concatenation
/// CSMT override: computational reduction per the CoSMeTIC paper
trait Aggregator: Clone + Send + Sync {
    /// Aggregate two child values at a given tree level
    fn aggregate(&self, level: u16, left: &Hash, right: &Hash) -> Hash;
}

/// Default aggregator: standard Merkle hash
struct MerkleAggregator;

impl Aggregator for MerkleAggregator {
    fn aggregate(&self, _level: u16, left: &Hash, right: &Hash) -> Hash {
        hasher::hash_internal(left, right)
    }
}

/// Computational aggregator: per-level reduction (CoSMeTIC)
struct ComputationalAggregator {
    /// Reduction functions indexed by tree level
    level_functions: Vec<Box<dyn Fn(&Hash, &Hash) -> Hash>>,
}
```

### Empty Root Optimization

The root of a completely empty tree is computed by propagating DEFAULT_EMPTY up 256 levels:

```
empty_root = DEFAULT_EMPTY
for _ in 0..256:
    empty_root = hash_internal(empty_root, empty_root)
```

This value is computed once and cached. It serves as the initial root and as the verification target for exclusion proofs on empty subtrees.

### Batch Operation Support

For inserting multiple leaves (e.g., enrolling a cohort of clinical trial participants), the tree supports batch insertion that shares path recomputation:

1. Sort keys by their binary prefix to maximize shared path segments
2. Insert leaves in sorted order
3. Recompute shared ancestors only once (after all leaves sharing that ancestor are inserted)

**Complexity**: O(M * K) worst case (all paths disjoint), O(M * K / log M) amortized (shared prefixes), where M = batch size and K = tree depth.

---

## Consequences

### Positive

1. **Fixed-depth proofs**: Every proof (inclusion or exclusion) contains exactly 256 sibling hashes. This uniformity simplifies ZK circuit design since the circuit has a fixed, known structure -- no conditional logic for variable-depth paths.

2. **Deterministic addressing**: The same key always maps to the same leaf position. No tree rebalancing, no key ordering dependencies. Concurrent readers can verify proofs against any historical root without needing the full tree state.

3. **WASM-safe implementation**: The entire tree uses `HashMap`, `Vec`, and fixed-size arrays. No file I/O, no system calls, no OS dependencies. The `js_sys::Date::now()` call for timestamps is behind `#[cfg(target_arch = "wasm32")]`.

4. **Computation metadata preserved**: The `computation_tag` on each leaf creates an auditable record of what transformation was applied, directly supporting the CoSMeTIC paper's requirement for verifiable computation attestation in clinical research.

5. **Compact proofs reduce bandwidth**: For typical populations (100-100,000 leaves), compact proofs are 500-750 bytes -- small enough for blockchain storage, API responses, or QR code embedding.

6. **Aggregation extensibility**: The `Aggregator` trait allows the standard Merkle tree to be upgraded to a full CSMT with per-level reduction functions without changing the core tree structure.

### Negative

1. **256-level depth is fixed overhead**: Even a tree with 2 leaves produces 256-sibling proofs. The compact encoding mitigates this for serialization, but inside ZK circuits the full 256-level path must be verified.

2. **HashMap memory overhead**: Rust's `HashMap` has ~60 bytes overhead per entry (bucket metadata, hash, Robin Hood probing). For millions of internal node cache entries, this is significant. An arena-allocated flat hash map would reduce this but adds implementation complexity.

3. **No persistent storage**: The current implementation is entirely in-memory. For long-running clinical trials with large populations, the tree state must be serialized to external storage. The `TreeSnapshot` structure supports this but requires integration with a persistence layer.

4. **Path recomputation not amortized**: Each single-leaf mutation recomputes all 256 levels. If mutations arrive one-at-a-time, this is optimal. If a batch of M mutations arrives simultaneously, the naive approach performs M * 256 hash evaluations instead of the potentially amortized M * 256 / log(M).

5. **No concurrent mutation**: The `&mut self` borrow on `insert` and `remove` prevents concurrent writes. A read-write lock or persistent data structure (e.g., COW-based tree) would be needed for concurrent access patterns.

### Performance Characteristics

| Operation | Complexity | Estimated Latency (WASM) | Estimated Latency (Native) |
|-----------|-----------|-------------------------|---------------------------|
| Insert single leaf | O(256) hashes | ~500us | ~50us |
| Remove single leaf | O(256) hashes | ~500us | ~50us |
| Lookup leaf by key | O(1) HashMap | ~100ns | ~50ns |
| Generate inclusion proof | O(256) lookups | ~200us | ~20us |
| Generate exclusion proof | O(256) lookups | ~200us | ~20us |
| Verify inclusion proof | O(256) hashes | ~500us | ~50us |
| Verify exclusion proof | O(256) hashes | ~500us | ~50us |
| Batch insert M leaves | O(M * 256) hashes | ~500us * M | ~50us * M |
| Compact proof encoding | O(256) bitmap ops | ~10us | ~1us |

---

## References

- Ramanan, P. et al. "CoSMeTIC: Zero-Knowledge Computational Sparse Merkle Trees with Inclusion-Exclusion Proofs for Clinical Research." arXiv:2601.12136, January 2026.
- Dahlberg, R., Pulls, T., and Peeters, R. "Efficient Sparse Merkle Trees: Caching Strategies and Secure (Non-)Membership Proofs." ePrint 2016/683.
- Ethereum Research. "Optimizing Sparse Merkle Trees." https://ethresear.ch/t/optimizing-sparse-merkle-trees/3751
- Amsden, Z. et al. "The Diem Blockchain." (Jellyfish Merkle Tree specification).
- Wood, G. "Ethereum Yellow Paper." (Modified Merkle Patricia Trie specification).
