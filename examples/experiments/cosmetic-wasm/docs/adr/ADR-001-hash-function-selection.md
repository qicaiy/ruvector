# ADR-001: Cryptographic Hash Function Selection

**Status**: Accepted
**Date**: 2026-02-04
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-04 | ruv.io | Initial proposal based on CoSMeTIC paper analysis |

---

## Context

### The Hash Function Challenge in Computational Sparse Merkle Trees

The CoSMeTIC framework (Computational Sparse Merkle Trees with Inclusion/exclusion Certificates) requires cryptographic hash functions in two fundamentally different operational contexts:

1. **Standard tree operations**: Leaf hashing, internal node hashing, key derivation, domain separation, attestation binding. These run on conventional hardware (native x86_64, ARM64) and in WebAssembly runtimes (browser, edge). Performance is measured in wall-clock throughput.

2. **Zero-knowledge circuit operations**: When a prover must demonstrate knowledge of a valid Merkle path (inclusion) or the absence of a leaf (exclusion) inside a ZK circuit, every hash invocation translates into arithmetic constraints over a prime field. The dominant cost metric shifts from CPU cycles to **constraint count** (R1CS gates or PLONKish rows).

The CoSMeTIC paper (Ramanan et al., arXiv:2601.12136, January 2026) defines computational sparse Merkle trees that embed reduction operations at each recursion level. Unlike conventional SMTs that merely concatenate-and-hash child nodes, CSMTs apply an aggregation function at each internal node. This means hash functions are invoked at every level of the tree (up to 256 levels deep for a 256-bit address space), making hash performance critical in both contexts.

### Constraint Cost Comparison

The following table summarizes the constraint overhead of candidate hash functions when arithmetized inside a ZK-SNARK circuit:

| Hash Function | R1CS Constraints (per invocation) | Relative Cost | ZK-Native Design | Maturity |
|---------------|-----------------------------------|---------------|-------------------|----------|
| **SHA-256** | ~25,000-30,000 | 1x (baseline) | No | Very High |
| **SHA-3 / Keccak** | ~75,000-90,000 | 3x | No | High |
| **Blake3** | ~15,000-20,000 | 0.6x | No | High |
| **Poseidon** | ~250-350 | **0.01x** | Yes | Medium |
| **MiMC** | ~300-700 | 0.02x | Yes | Medium |
| **Rescue** | ~400-600 | 0.02x | Yes | Medium |
| **Anemoi** | ~150-250 | 0.007x | Yes | Low |
| **Griffin** | ~200-300 | 0.009x | Yes | Low |

Traditional hash functions (SHA-256, Blake3) rely on bitwise operations (XOR, rotation, shifts) that are fundamentally expensive to arithmetize because arithmetic circuits operate over finite field elements using addition and multiplication. Each XOR gate must be decomposed into field operations, inflating constraint counts by orders of magnitude.

ZK-native hash functions (Poseidon, MiMC, Rescue) are designed around algebraic operations (field addition and exponentiation) that map directly to arithmetic circuit primitives, yielding 50-100x fewer constraints.

### WASM Deployment Constraints

The `cosmetic-wasm` crate targets `wasm32-unknown-unknown` with `crate-type = ["cdylib", "rlib"]`. This imposes:

- **Binary size budget**: Compressed WASM should remain under 200KB for reasonable browser load times. Full ZK libraries (e.g., arkworks, halo2_proofs with gadgets) can exceed 5MB compiled to WASM.
- **No native intrinsics**: WASM lacks SIMD instructions in the base spec (though WASM SIMD 128-bit is increasingly available). SHA-256 benefits from hardware acceleration (SHA-NI on x86_64) that is unavailable in WASM.
- **Memory model**: 32-bit linear memory with a 4GB ceiling. Large circuit witness generation can pressure memory limits.
- **Execution model**: Single-threaded unless Web Workers are used via `wasm-bindgen-rayon`. Proof generation benefits greatly from parallelism.

### Existing Implementation State

The current `hasher.rs` implements:
- SHA-256 via the `sha2` crate for all standard operations
- Domain-separated hashing: `0x00` prefix for leaves, `0x01` for internal nodes, `0x02` for attestations
- A placeholder `poseidon_hash` behind the `poseidon` feature flag that wraps SHA-256 (for API shape testing)
- A pre-computed `DEFAULT_EMPTY` constant for absent leaf positions

---

## Decision

### Dual-Hash Architecture: Poseidon for ZK Circuits, SHA-256/Blake3 for Non-ZK Operations

We adopt a **stratified hash function architecture** with the following assignments:

#### Layer 1: Standard Tree Operations (Non-ZK)

**SHA-256** remains the default hash function for all non-ZK tree operations:

- Leaf hashing: `H(0x00 || key || value)`
- Internal node hashing: `H(0x01 || left || right)`
- Key derivation: `H(data)` to compute 256-bit leaf addresses
- Attestation binding: `H(0x02 || input_root || output_root || function_id || params)`
- Default empty leaf: Pre-computed `SHA-256("cosmetic_empty_leaf")`

**Rationale**: SHA-256 is battle-tested, widely available in WASM (`sha2` crate compiles cleanly to `wasm32`), and provides the 256-bit output that directly maps to the tree's address space. The `sha2` crate adds approximately 15KB to WASM binary size.

**Future consideration**: Blake3 offers approximately 3-5x higher throughput than SHA-256 on conventional hardware and compiles well to WASM. If non-ZK hashing becomes a bottleneck (e.g., bulk tree construction with thousands of leaves), Blake3 can be introduced as an optional backend behind a feature flag without changing the tree architecture, since both produce 256-bit outputs.

#### Layer 2: Zero-Knowledge Circuit Operations

**Poseidon** is the designated hash function for all operations that must be verified inside a ZK proof circuit:

- Leaf Transform Ratio (LTR) proofs: Proving correct leaf-level computation
- Merkle Record Path (MRP) proofs: Proving valid paths from leaf to root
- Inclusion certificate generation: Proving a key exists with its value
- Exclusion certificate generation: Proving a key is absent (default-empty at its position)

**Rationale**: The CoSMeTIC paper's use of the ezkl framework with Halo2 backend confirms algebraic-operation-native hashing is essential for practical proof generation. Poseidon requires approximately 250-350 R1CS constraints per hash invocation versus approximately 25,000-30,000 for SHA-256 -- a reduction of roughly 80-100x. For a tree of depth K, each inclusion/exclusion proof requires K hash evaluations inside the circuit. At K=256, this means:

- **SHA-256 in circuit**: ~6.4-7.7 million constraints per proof path
- **Poseidon in circuit**: ~64,000-89,600 constraints per proof path

This difference determines whether proof generation completes in seconds (Poseidon) or minutes (SHA-256) on consumer hardware, and whether it is feasible at all in WASM with its memory constraints.

**Poseidon configuration**:
- Field: BN254 scalar field (matching existing Rust ecosystem: `ark-bn254`, `halo2curves`)
- Arity: 2 (binary tree, width-3 sponge: rate=2, capacity=1)
- Security level: 128 bits
- Full rounds: 8, partial rounds: 56 (per Poseidon reference specification)

#### Layer 3: Bridge Layer (Hash Compatibility)

A **bridge hash** mechanism links the two layers when a proof must attest to the same data that exists in the SHA-256-based tree:

1. The prover holds the pre-image data (leaf value, key)
2. Inside the ZK circuit, the prover re-hashes using Poseidon to produce a Poseidon-based root
3. A separate algebraic commitment binds the Poseidon root to the SHA-256 root via a signed attestation outside the circuit
4. The verifier checks: (a) the ZK proof is valid against the Poseidon root, and (b) the attestation correctly binds the Poseidon root to the published SHA-256 root

This avoids the need to hash SHA-256 inside the ZK circuit while maintaining a verifiable link between the two hash domains.

### Feature Flag Design

```toml
[features]
default = ["console_error_panic_hook"]
# Enable Poseidon hash for ZK-friendly circuits
poseidon = ["dep:poseidon-permutation"]
# Enable Blake3 as an alternative non-ZK hash
blake3 = ["dep:blake3"]
# Enable computation attestation proofs
attestation = []
# Enable all features
full = ["poseidon", "attestation", "blake3"]
```

### Hash Trait Abstraction

```rust
/// Trait abstracting over hash function implementations
pub trait TreeHasher: Clone + Send + Sync {
    /// Hash output size in bytes
    const OUTPUT_SIZE: usize;

    /// Hash arbitrary data
    fn hash(data: &[u8]) -> Hash;

    /// Hash a leaf node with domain separation
    fn hash_leaf(key: &Hash, value: &[u8]) -> Hash;

    /// Hash an internal node with domain separation
    fn hash_internal(left: &Hash, right: &Hash) -> Hash;

    /// The default hash for empty/absent leaves
    fn default_empty() -> Hash;
}
```

This trait allows the tree implementation to be generic over the hash function, enabling testing with fast non-cryptographic hashes and production use with either SHA-256 or Poseidon depending on context.

---

## Consequences

### Positive

1. **Practical ZK proof generation**: Poseidon reduces circuit constraint count by ~80-100x compared to SHA-256, making in-browser WASM proof generation feasible for trees of practical depth.

2. **Backward compatibility**: SHA-256 remains the default for non-ZK operations, preserving compatibility with the existing `hasher.rs` implementation and all current tests.

3. **Minimal WASM binary impact**: The `sha2` crate adds ~15KB. Poseidon is opt-in behind a feature flag and only pulled in when ZK functionality is needed.

4. **Domain separation preserved**: Both hash layers use domain-separated inputs (leaf vs. internal vs. attestation), preventing cross-domain collision attacks regardless of which hash function is active.

5. **Ecosystem alignment**: Poseidon over BN254 aligns with the dominant Rust ZK ecosystem (arkworks, halo2, circom). Multiple production Rust crates exist: `poseidon-merkle`, `dusk-poseidon`, `light-poseidon`.

### Negative

1. **Dual-root complexity**: The bridge layer introduces a second root hash (Poseidon-based) that must be kept in sync with the SHA-256 root. This adds implementation complexity and a potential consistency failure mode.

2. **Poseidon security maturity**: Poseidon has received significant cryptanalysis but has not been subjected to the decades of scrutiny that SHA-256 has. The security margin of ZK-friendly hash functions remains an active area of research. Newer candidates (Anemoi, Griffin) claim better efficiency but have even less cryptanalytic history.

3. **Performance asymmetry**: Poseidon is optimized for arithmetic circuits, not for raw throughput on conventional hardware. On native/WASM without ZK circuits, Poseidon is approximately 10-50x slower than SHA-256 for the same input size. It should never be used as a general-purpose hash outside of ZK contexts.

4. **Library dependency surface**: Adding Poseidon introduces dependencies on finite field arithmetic libraries (e.g., `ark-ff`, `pasta_curves`, or `halo2curves`), which increase WASM binary size by 100-500KB depending on the implementation chosen.

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Poseidon algebraic attack discovered | Low | High | Monitor ePrint/IACR; trait abstraction allows swapping to MiMC/Rescue/Anemoi |
| WASM binary exceeds budget with Poseidon deps | Medium | Medium | Feature-flag isolation; consider vendoring minimal Poseidon implementation |
| Dual-root consistency bug | Medium | High | Extensive property-based testing; root binding attestation includes both roots |
| Blake3 needed for bulk operations | Low | Low | Already designed as feature flag; drop-in compatible via `TreeHasher` trait |

---

## References

- Ramanan, P. et al. "CoSMeTIC: Zero-Knowledge Computational Sparse Merkle Trees with Inclusion-Exclusion Proofs for Clinical Research." arXiv:2601.12136, January 2026.
- Grassi, L. et al. "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems." USENIX Security 2021.
- Bowe, S. et al. "Halo: Recursive Proof Composition without a Trusted Setup." IACR ePrint 2019/1021.
- Dahlberg, R. et al. "Efficient Sparse Merkle Trees." Nordic Conference on Secure IT Systems, 2016.
- ZK-Plus. "Benchmarks of Hashing Algorithms in ZoKrates." https://zk-plus.github.io/tutorials/basics/hashing-algorithms-benchmarks
- Zellic Research. "ZK-Friendly Hash Functions." https://www.zellic.io/blog/zk-friendly-hash-functions/
