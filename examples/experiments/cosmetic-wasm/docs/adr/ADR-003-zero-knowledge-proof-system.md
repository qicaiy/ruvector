# ADR-003: Zero-Knowledge Proof System Selection

**Status**: Accepted
**Date**: 2026-02-04
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-04 | ruv.io | Initial proposal based on CoSMeTIC paper and WASM constraints |

---

## Context

### Why Zero-Knowledge Proofs for Sparse Merkle Trees?

Standard Merkle proofs (as implemented in the current `proof.rs`) are already efficient: the verifier recomputes 256 hashes and checks the result against a known root. However, standard proofs have fundamental limitations that the CoSMeTIC framework must address:

1. **Privacy**: A standard inclusion proof reveals the leaf key, value, and all sibling hashes along the path. In clinical research, this exposes patient data to the verifier. Zero-knowledge proofs allow the prover to demonstrate "this patient's data is included in the study" without revealing which patient, what data, or the tree structure.

2. **Computation verification**: The CoSMeTIC paper's CSMT performs leaf transformations (`L^s`) and per-level aggregations (`A^l`). A standard Merkle proof only proves data presence, not that the transformation was correctly applied. A ZK proof can attest to both: "this leaf exists AND it was produced by correctly applying function L^s to the raw data."

3. **Batch efficiency**: Verifying 10,000 individual Merkle proofs requires 10,000 * 256 hash evaluations. A ZK proof system can batch all 10,000 membership claims into a single proof that verifies in constant time.

4. **Regulatory compliance**: The CoSMeTIC paper targets ACM CCS 2026 and addresses FDA/EMA regulatory requirements for clinical research data integrity. Regulators need to verify that all enrolled patients' data was correctly processed without accessing the raw protected health information (PHI). ZK proofs are the cryptographic mechanism that makes this possible.

### The CoSMeTIC Paper's Proof Architecture

The paper (Ramanan et al., arXiv:2601.12136) describes a three-phase zk-SNARK protocol:

**Phase 1 -- Setup**: Generate proving key (pk) and verification key (vk) for:
- The leaf transformation function `L^s` (parametrized by model weights theta)
- The aggregation function `A^l` at each tree level

Setup is executed once per circuit instantiation and must be re-executed if the function parameters change.

**Phase 2 -- Proving**: Generate proof artifacts Pi attesting to correct function evaluation:
- **LTR (Leaf Transform Ratio) proofs**: One per leaf, proving `L^s(delta, mu, tau)` was correctly computed
- **MRP (Merkle Record Path) proofs**: One per tree hop from leaf to root, proving `A^l(left, right)` was correctly evaluated

**Phase 3 -- Verification**: Use vk to validate proof Pi, outputting a Boolean flag.

The paper uses the **ezkl** framework with the **Halo2** proof generation backend. ezkl compiles PyTorch models into ZK circuits via ONNX-to-circuit translation, using lookup table arguments and efficient einsum operations.

### Proof System Requirements for WASM

| Requirement | Description | Weight |
|-------------|-------------|--------|
| R1 | No trusted setup (or universal/updatable setup) | Critical |
| R2 | Rust implementation that compiles to `wasm32-unknown-unknown` | Critical |
| R3 | Proof size under 1KB for practical transport | High |
| R4 | Verification time under 50ms in WASM | High |
| R5 | Proof generation feasible in WASM (under 30s for single proof) | Medium |
| R6 | Supports Poseidon hash in circuit (per ADR-001) | High |
| R7 | Constant-time verification (independent of statement complexity) | Medium |
| R8 | Active maintenance and ecosystem support | Medium |
| R9 | Memory usage under 2GB during proof generation (WASM ceiling) | High |
| R10 | Composable proofs (can aggregate multiple proofs) | Medium |

### Options Evaluated

#### Option A: Groth16

The most widely deployed SNARK. Produces the smallest proofs (~128-192 bytes) with the fastest verification (~10ms). Used by Zcash Sapling.

| Attribute | Value |
|-----------|-------|
| Proof size | ~128 bytes (2 G1 + 1 G2 elements on BN254) |
| Verification time | ~10ms (3 pairings) |
| Proving time | ~1-10s per proof (native) |
| Trusted setup | **Required** (circuit-specific, toxic waste) |
| Rust crates | `ark-groth16`, `bellman` |
| WASM support | Partial (`ark-groth16` compiles but pairing is slow) |

**Disqualified**: The circuit-specific trusted setup (R1) is a critical barrier. Each new circuit (each new statistical test or clinical computation) would require a new multi-party computation ceremony. This is operationally infeasible for a general-purpose clinical research framework where new analyses are defined regularly.

#### Option B: PLONK (KZG-based)

A universal SNARK with a single structured reference string (SRS) that works for all circuits up to a fixed size. Larger proofs than Groth16 but no circuit-specific setup.

| Attribute | Value |
|-----------|-------|
| Proof size | ~400-700 bytes |
| Verification time | ~15-25ms |
| Proving time | ~2-20s (native) |
| Trusted setup | **Universal** (one ceremony, reusable for all circuits) |
| Rust crates | `plonk` (dusk-network), `halo2_proofs` (with KZG) |
| WASM support | Feasible but SRS download is large (~100MB for circuits >2^20) |

**Concern**: The universal SRS is a significant improvement over Groth16's per-circuit setup, but the SRS is still a trusted artifact. The size of the SRS for large circuits (millions of gates, as needed for full CSMT path verification) can exceed the WASM memory budget.

#### Option C: Halo2 (IPA-based, no trusted setup)

The Halo2 proof system, developed by the Electric Coin Company for Zcash, combines PLONK arithmetization with an Inner Product Argument (IPA) polynomial commitment scheme. The IPA eliminates the trusted setup entirely.

| Attribute | Value |
|-----------|-------|
| Proof size | ~2-10KB (larger than Groth16/PLONK due to IPA) |
| Verification time | ~50-200ms (IPA verification is logarithmic, not constant) |
| Proving time | ~5-60s (native), ~30-300s (WASM) |
| Trusted setup | **None** |
| Rust crates | `halo2_proofs`, `halo2_gadgets` |
| WASM support | **Official WASM guide** in The Halo2 Book |
| Recursive composition | Supported via accumulation |

**Key advantages**:
- No trusted setup at all -- purely transparent
- Official Rust implementation with documented WASM compilation path
- Extensive ecosystem adoption: Zcash, Protocol Labs, Ethereum PSE, Scroll, Taiko
- The CoSMeTIC paper's ezkl framework builds on an improved version of Halo2
- Recursive proof composition via nested amortization (one proof attests to unlimited others)
- Active development and security audits (Kudelski Security)

**Key challenges for WASM**:
- Uses `rayon` for parallelism (requires `wasm-bindgen-rayon` adapter for Web Workers)
- Memory limit: default WASM memory (2GB) insufficient for K > 10; must increase to 4GB maximum
- Requires nightly Rust with WASM atomics enabled
- Not supported in Safari (Web Worker spawning limitation)
- Proving time 5-10x slower in WASM versus native due to lack of SIMD and memory constraints

#### Option D: Bulletproofs

A transparent proof system (no trusted setup) based on the discrete log assumption. Produces compact proofs but with linear verification time.

| Attribute | Value |
|-----------|-------|
| Proof size | ~700 bytes (logarithmic in witness size) |
| Verification time | ~50-500ms (linear in proof size) |
| Proving time | ~1-5s (native) |
| Trusted setup | **None** |
| Rust crates | `bulletproofs` (dalek-cryptography) |
| WASM support | Good (curve25519-dalek compiles to WASM) |

**Concern**: Verification time scales linearly with the number of multiplication gates. For CSMT path verification (256 levels of Poseidon hashing, each ~300 constraints), the total circuit has ~80,000+ gates. Bulletproof verification would take seconds, violating R4.

#### Option E: STARKs (FRI-based)

Transparent, post-quantum-secure proof system based on hash functions and the FRI (Fast Reed-Solomon IOP of Proximity) protocol.

| Attribute | Value |
|-----------|-------|
| Proof size | ~50-200KB |
| Verification time | ~10-50ms (polylogarithmic) |
| Proving time | ~1-30s (native) |
| Trusted setup | **None** |
| Rust crates | `winterfell`, `stwo` |
| WASM support | Possible but proof sizes are very large |

**Disqualified**: Proof sizes of 50-200KB violate R3. While STARKs have excellent asymptotic properties and post-quantum security, the concrete proof sizes are impractical for transport in clinical research scenarios (API payloads, blockchain anchoring, QR codes).

#### Option F: Simplified Algebraic Commitment-Based Verification (Lightweight)

A pragmatic approach that does not implement a full general-purpose SNARK, but instead uses algebraic commitments (Pedersen commitments, polynomial commitments) to provide verifiable computation attestation with privacy properties:

1. **Pedersen commitment** to the leaf value: `C = g^v * h^r` where v is the value and r is the blinding factor
2. **Polynomial commitment** to the Merkle path: represent the 256-level path as coefficients of a polynomial, commit to the polynomial, and prove evaluation at the challenge point
3. **Fiat-Shamir transform** to make the protocol non-interactive

This provides:
- **Privacy**: The committed value is hidden behind the commitment
- **Binding**: The prover cannot change the value after committing
- **Verifiable path**: The polynomial commitment proves the Merkle path without revealing all siblings
- **No trusted setup**: Pedersen commitments use a random group generator selection, not a toxic waste ceremony

| Attribute | Value |
|-----------|-------|
| Proof size | ~256-512 bytes |
| Verification time | ~5-20ms |
| Proving time | ~100ms-1s (native), ~500ms-5s (WASM) |
| Trusted setup | **None** |
| Rust crates | Custom, using `curve25519-dalek` or `pasta_curves` |
| WASM support | Excellent (minimal dependencies) |

**Trade-off**: This is NOT a general-purpose ZK-SNARK. It cannot prove arbitrary computations. It specifically proves: (1) a committed value is at a specific position in a tree with a known root, or (2) a specific position is empty. It does not prove that the leaf transformation `L^s` was correctly applied -- that would require a full SNARK.

---

## Decision

### Adopt a Two-Tier Proof Architecture

Given the competing requirements -- full SNARK capability (CoSMeTIC paper compliance) versus WASM feasibility and implementation simplicity -- we adopt a **two-tier architecture**:

#### Tier 1: Simplified Algebraic Commitment-Based Verification (Default)

The default proof system shipped with `cosmetic-wasm` is a lightweight algebraic commitment scheme that provides:

- **Privacy-preserving membership proofs**: Prove inclusion/exclusion without revealing leaf values or sibling hashes
- **Non-interactive via Fiat-Shamir**: No interaction required between prover and verifier
- **WASM-native performance**: Sub-second proof generation, sub-50ms verification
- **Minimal binary size**: ~50KB additional WASM (curve arithmetic only)
- **No trusted setup**: Transparent, deterministic parameter generation

**Protocol sketch for inclusion**:

```
Prover(key, value, merkle_path, root):
    1. Commit to leaf value: C_leaf = Commit(value, r_leaf)
    2. For each level i in 0..255:
        a. Commit to sibling: C_sib[i] = Commit(sibling[i], r_sib[i])
        b. Compute challenge: e[i] = H(C_leaf, C_sib[i], i)
        c. Compute response: z[i] = value_function(e[i], ...)
    3. Output proof: (C_leaf, {C_sib[i], z[i]}, root)

Verifier(proof, root):
    1. Recompute challenges from commitments
    2. Verify commitment opening consistency
    3. Verify computed root matches claimed root
    4. Output: Valid / Invalid
```

**What Tier 1 proves**: A committed value exists at a committed position in a tree whose root is `root` (or is absent, for exclusion). The verifier learns only the root hash and the validity of the claim.

**What Tier 1 does NOT prove**: That the leaf value was produced by a specific computation `L^s`. For that, Tier 2 is needed.

#### Tier 2: Halo2-Based Full ZK-SNARK (Feature-Gated)

Behind the `zk-full` feature flag, the crate integrates with the Halo2 proof system to provide full zk-SNARK capability:

- **Arbitrary computation proofs**: Prove that `L^s(delta, mu, tau)` was correctly evaluated AND the result is at the correct tree position AND the Merkle path is valid
- **Recursive proof composition**: Aggregate multiple per-leaf proofs into a single proof via Halo2's accumulation scheme
- **ezkl compatibility**: The circuit design is compatible with the ezkl framework used in the CoSMeTIC paper, enabling ML model verification in ZK

**Feature flag design**:

```toml
[features]
default = ["console_error_panic_hook"]
# Tier 1: Lightweight algebraic commitments (always available)
# No feature flag needed -- built into the base crate

# Tier 2: Full Halo2 ZK-SNARK
zk-full = ["dep:halo2_proofs", "dep:halo2_gadgets", "poseidon"]

# Poseidon hash for ZK circuits
poseidon = ["dep:poseidon-permutation"]

# All features
full = ["zk-full", "poseidon", "attestation"]
```

**Halo2 circuit structure for CSMT verification**:

```
Circuit: CsmtMembershipCircuit
    Public inputs:
        - root: Fp          (the tree root hash)
        - claim: bool        (inclusion=true, exclusion=false)

    Private inputs (witness):
        - key: [bool; 256]   (path selection bits)
        - value: Fp          (leaf value, or 0 for exclusion)
        - salt: Fp           (blinding factor)
        - siblings: [Fp; 256] (sibling hashes at each level)

    Constraints:
        Region 1: Leaf hash
            leaf_hash = Poseidon(domain_sep, key, value, salt)
            OR leaf_hash = DEFAULT_EMPTY (for exclusion)

        Region 2: Path verification (256 iterations)
            for i in 0..256:
                left, right = select(key[i], current, siblings[i])
                current = Poseidon(internal_sep, left, right)

        Region 3: Root check
            assert current == root

    Total constraints: ~256 * 300 + 300 = ~77,100
        (256 Poseidon hashes for path + 1 for leaf)
```

**Halo2 WASM compilation requirements**:

| Requirement | Solution |
|-------------|----------|
| Parallelism (rayon) | `wasm-bindgen-rayon` adapter for Web Workers |
| Memory (>2GB) | Set `--max-memory=4294967296` in wasm-bindgen |
| Atomics | Nightly Rust with `target-feature=+atomics,+bulk-memory,+mutable-globals` |
| Safari compat | Fallback to single-threaded mode (slower but functional) |

### Verification Key Distribution

For Tier 2, the verification key (vk) is a public artifact that must be distributed to all verifiers. Since Halo2 has no trusted setup, the vk is deterministically derived from the circuit definition:

1. **Embed in WASM**: For small circuits (K <= 12), the vk can be compiled into the WASM binary as a constant (~2-10KB)
2. **Fetch on demand**: For larger circuits, the vk is fetched from a content-addressed store (IPFS, HTTP) using its hash as the lookup key
3. **Cache in IndexedDB**: Once fetched, the vk is cached in the browser's IndexedDB for subsequent verifications

### Proof Composition Strategy

The CoSMeTIC paper requires generating proofs for potentially thousands of patients in a clinical trial. The proof composition strategy:

1. **Per-leaf proofs**: Generate one LTR proof per patient (proving correct leaf transformation)
2. **Per-path proofs**: Generate one MRP proof per patient (proving valid Merkle path)
3. **Batch aggregation**: Use Halo2's recursive accumulation to compress all per-patient proofs into a single aggregate proof
4. **Single verification**: The regulatory auditor verifies one aggregate proof instead of thousands of individual proofs

**Aggregation complexity**:

| Patients (N) | Individual Proofs | Aggregate Proof Size | Aggregate Verification Time |
|--------------|-------------------|---------------------|-----------------------------|
| 100 | 200 (100 LTR + 100 MRP) | ~10KB | ~200ms |
| 1,000 | 2,000 | ~10KB | ~200ms |
| 10,000 | 20,000 | ~10KB | ~200ms |
| 100,000 | 200,000 | ~10KB | ~200ms |

The aggregate proof size and verification time are **constant** regardless of the number of patients, thanks to recursive composition. Only proving time scales linearly.

---

## Consequences

### Positive

1. **No trusted setup at either tier**: Tier 1 uses Pedersen/polynomial commitments (transparent). Tier 2 uses Halo2 IPA (transparent). No multi-party computation ceremony is ever required.

2. **Incremental adoption**: Developers can start with Tier 1 (lightweight, fast, small binary) and upgrade to Tier 2 (full SNARK) when they need computation verification proofs. The API surface is designed so Tier 1 proofs can be replaced by Tier 2 proofs without changing the verifier interface.

3. **WASM-first design**: Tier 1 runs efficiently in any WASM environment (including Safari, mobile browsers, edge runtimes). Tier 2 has documented WASM support with known constraints.

4. **CoSMeTIC paper alignment**: Tier 2 directly implements the paper's three-phase protocol (Setup, Prove, Verify) with LTR and MRP proofs. The ezkl/Halo2 backend choice matches the paper's implementation.

5. **Constant-time aggregate verification**: Recursive proof composition means a clinical trial with 100,000 patients produces an aggregate proof that verifies in the same time as a trial with 100 patients.

6. **Post-Groth16 security**: Halo2's IPA avoids the trusted setup attack surface entirely. If the SRS of a KZG-based system is compromised, soundness is broken. Halo2 has no SRS to compromise.

### Negative

1. **Larger proofs than Groth16**: Tier 1 proofs are ~256-512 bytes (acceptable). Tier 2 (Halo2 IPA) proofs are ~2-10KB -- an order of magnitude larger than Groth16's ~128 bytes. For on-chain storage, this increases gas costs proportionally.

2. **Slower verification than Groth16**: Halo2 IPA verification is O(log n) in the proof size, not O(1) like Groth16's 3-pairing check. Tier 2 verification takes ~50-200ms versus Groth16's ~10ms. Still acceptable for clinical audit workflows but not for high-frequency on-chain verification.

3. **WASM proving is slow**: Halo2 proof generation in WASM is 5-10x slower than native due to lack of SIMD, single-threaded execution (without rayon adapter), and WASM interpreter overhead. A proof that takes 5 seconds native may take 30-60 seconds in WASM. This limits the practical circuit size (K <= 15 in Halo2 terms).

4. **Tier 2 binary size**: The `halo2_proofs` + `halo2_gadgets` + Poseidon dependencies add approximately 1-3MB to the WASM binary (compressed). This is acceptable for applications that need ZK but oversized for lightweight use cases (hence the feature-flag isolation).

5. **Tier 1 does not prove computation**: The simplified algebraic commitment scheme proves data presence/absence with privacy but does NOT prove that the leaf value was produced by a specific function. Applications requiring computation verification (the core CoSMeTIC use case for regulatory compliance) MUST use Tier 2.

6. **Nightly Rust requirement for WASM Tier 2**: Halo2 in WASM requires nightly Rust with atomics support. This complicates CI/CD and may conflict with other crates in the workspace that target stable Rust.

### Decision Matrix Summary

| Criterion | Groth16 | PLONK (KZG) | Halo2 (IPA) | Bulletproofs | STARKs | Tier 1 (Algebraic) |
|-----------|---------|-------------|-------------|--------------|--------|---------------------|
| Trusted setup | Circuit-specific | Universal | **None** | **None** | **None** | **None** |
| Proof size | **128B** | 400-700B | 2-10KB | 700B | 50-200KB | 256-512B |
| Verification time | **10ms** | 15-25ms | 50-200ms | 50-500ms | 10-50ms | 5-20ms |
| WASM compilation | Partial | Partial | **Documented** | Good | Possible | **Excellent** |
| WASM binary size | ~2MB | ~2MB | ~1-3MB | ~500KB | ~2MB | **~50KB** |
| Computation proofs | Yes | Yes | Yes | Limited | Yes | **No** |
| Recursive composition | No | Limited | **Yes** | No | Yes | No |
| Ecosystem maturity | High | Medium | **High** | Medium | Medium | Custom |

**Selected**: Tier 1 (Algebraic) as default + Tier 2 (Halo2 IPA) as `zk-full` feature

---

## References

- Ramanan, P. et al. "CoSMeTIC: Zero-Knowledge Computational Sparse Merkle Trees with Inclusion-Exclusion Proofs for Clinical Research." arXiv:2601.12136, January 2026.
- Bowe, S., Grigg, J., and Hopwood, D. "Halo: Recursive Proof Composition without a Trusted Setup." ePrint 2019/1021.
- Electric Coin Company. "The Halo2 Proving System." https://halo2.dev/
- Zcash. "WASM Guide -- The halo2 Book." https://zcash.github.io/halo2/user/wasm-port.html
- Kudelski Security. "On the Security of Halo2 Proof System." https://kudelskisecurity.com/research/on-the-security-of-halo2-proof-system
- Gabizon, A., Williamson, Z., and Ciobotaru, O. "PLONK: Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge." ePrint 2019/953.
- Groth, J. "On the Size of Pairing-Based Non-interactive Arguments." EUROCRYPT 2016.
- Bunz, B., Bootle, J., et al. "Bulletproofs: Short Proofs for Confidential Transactions and More." IEEE S&P 2018.
- Ben-Sasson, E. et al. "Scalable, Transparent, and Post-quantum Secure Computational Integrity." ePrint 2018/046 (STARKs).
- EZKL. "The EZKL System." https://docs.ezkl.xyz/
- Tham, Y.J. "Building a Zero Knowledge Web App with Halo 2 and Wasm." https://medium.com/@yujiangtham/building-a-zero-knowledge-web-app-with-halo-2-and-wasm-part-1-80858c8d16ee
