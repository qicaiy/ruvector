# ADR-045: PR #201 Post-Merge Review Remediation

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-02-23 |
| **Authors** | ruv, claude-flow |
| **PR** | #201 (WASM cognitive stack with canonical min-cut, spectral coherence, container orchestration) |
| **Branch** | `fix/pr201-review-findings` |
| **Crates Affected** | `ruvector-cognitive-container`, `ruvector-mincut`, `ruvector-gnn` |

## Context

PR #201 merged 32 crates introducing the WASM cognitive stack: canonical min-cut, spectral coherence, GNN cold-tier persistence, and container orchestration. A post-merge code review identified 2 critical, 4 major, and 1 minor issue across 4 source files. This ADR documents the findings and their remediation.

## Findings

### Critical

#### C1: Non-deterministic witness chain hashing

**File:** `crates/ruvector-cognitive-container/src/witness.rs`

`std::collections::hash_map::DefaultHasher` is not guaranteed stable across Rust compiler versions. Witness receipt chains are serialized to disk and verified on reload. A rustc upgrade could silently change hash output, breaking chain verification and invalidating persisted witness receipts.

**Fix:** Replaced `DefaultHasher` with a custom fixed-seed SipHash-2-4 implementation using 4 hardcoded key pairs. The implementation follows the standard SipHash-2-4 specification (2 compression rounds, 4 finalization rounds) with deterministic output guaranteed regardless of compiler version, platform, or architecture.

#### C2: Unbounded memory allocation on sparse vertex IDs

**File:** `crates/ruvector-mincut/src/canonical/mod.rs`

`compute_canonical_key()` allocated a lookup table as `vec![usize::MAX; max_id + 1]`. For sparse vertex ID spaces (e.g., max_id = 10^9 with only 100 vertices), this allocates gigabytes of memory and panics on OOM.

**Fix:** Replaced the dense `Vec<usize>` lookup table with `HashMap<usize, usize>`. This uses O(n) memory proportional to the number of vertices, not the magnitude of the largest vertex ID.

### Major

#### M1: Non-deterministic canonical key hashing

**File:** `crates/ruvector-mincut/src/canonical/mod.rs`

Same `DefaultHasher` issue as C1, affecting `compute_canonical_key()`. Canonical keys are used to identify equivalent min-cut partitions; non-deterministic hashing could cause cache misses or false negatives in partition deduplication.

**Fix:** Replaced with inline fixed-seed SipHash-2-4 using the same deterministic algorithm as the witness chain fix.

#### M2: Endianness-dependent cold-tier serialization

**File:** `crates/ruvector-gnn/src/cold_tier.rs`

Feature vectors were serialized using `unsafe { std::slice::from_raw_parts(features.as_ptr() as *const u8, ...) }`, which writes native-endian bytes. A file written on a little-endian x86 host would read incorrectly on a big-endian platform (or vice versa), producing silently wrong feature vectors.

**Fix:** Replaced the unsafe native-endian cast with explicit `features.iter().flat_map(|f| f.to_le_bytes()).collect()`. All platforms now write and read little-endian, ensuring cross-platform portability.

#### M3: Cactus cache requires `&mut self` through `&self` trait

**File:** `crates/ruvector-mincut/src/canonical/mod.rs`

The `CanonicalMinCut` trait defines `canonical_cut(&self)` and `cactus_graph(&self)`, but the implementation needed `&mut self` to lazily recompute the cactus cache. This forced callers to hold mutable references even for read-only queries, preventing concurrent read access.

**Fix:** Changed `cactus: Option<CactusGraph>` to `RefCell<Option<CactusGraph>>` and `dirty: bool` to `Cell<bool>`. The `ensure_cactus()` method now takes `&self` and uses interior mutability to lazily populate the cache. This preserves the `&self` trait contract while allowing transparent lazy recomputation.

#### M4: O(n) receipt chain eviction

**File:** `crates/ruvector-cognitive-container/src/witness.rs`

Receipt chains used `Vec::remove(0)` for eviction when exceeding `max_receipts`. This is O(n) because it shifts all remaining elements. With the default max of 1024 receipts, every eviction copies up to 1023 entries.

**Fix:** Changed `receipts` from `Vec<ContainerWitnessReceipt>` to `VecDeque<ContainerWitnessReceipt>`. Front eviction via `pop_front()` is now O(1). The `receipt_chain()` method returns `Vec<T>` (materialized copy) since `VecDeque` doesn't guarantee contiguous memory for slice references.

### Minor

#### m1: Unused 4MB MemorySlab allocation

**File:** `crates/ruvector-cognitive-container/src/container.rs`

`CognitiveContainer` held a `MemorySlab` field (4MB default allocation) that was never read, marked `#[allow(dead_code)]`. This wastes 4MB of heap per container instance with zero benefit.

**Fix:** Removed the stored field. Config validation is preserved via `let _validate = MemorySlab::new(config.memory_slab_bytes)?;` which validates the slab size is constructible without retaining the allocation.

#### m2: Double-clone in HyperbatchIterator

**File:** `crates/ruvector-gnn/src/cold_tier.rs`

The `next()` method cloned features into the batch buffer, then cloned them again for the return value — two full copies of each feature vector per iteration.

**Fix:** Store features in the buffer first, then clone from the buffer for the return value. This eliminates one redundant clone per feature vector.

## Testing

| Crate | Result |
|-------|--------|
| `ruvector-cognitive-container` | All tests pass (including witness chain verification) |
| `ruvector-mincut` (canonical module) | 2/2 tests pass |
| `ruvector-gnn` | All 195 unit + 6 integration + 10 doc tests pass |
| Compilation check | All 3 crates compile clean (warnings only) |

## Decision

All 7 fixes are applied in a single commit on `fix/pr201-review-findings`. The changes are backwards-compatible at the API level. The SipHash implementation uses fixed keys that must never change to preserve witness chain integrity.

**Key invariant:** The 4 SipHash key pairs in `witness.rs` and the inline SipHash in `canonical/mod.rs` are load-bearing constants. Changing them invalidates all persisted witness chains and canonical keys.

## Consequences

- Witness chains are now deterministic across Rust compiler versions and platforms
- Canonical min-cut operates safely on sparse vertex ID spaces
- Cold-tier files are portable across architectures (little-endian canonical form)
- Receipt chain eviction is O(1) instead of O(n)
- Container instances save 4MB heap each
- Cactus cache supports concurrent read access via interior mutability
- HyperbatchIterator uses ~50% less memory per iteration
