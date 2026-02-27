# Reflex-on-Seed, Heavy-on-Cluster (with Cryptographic Witnesses)

**Research Date**: 2026-02-27
**Status**: Research
**Related ADRs**: ADR-030, ADR-032, ADR-034, ADR-047, ADR-049, ADR-CE-014, ADR-CE-012

---

## Overview

This document explores the architecture pattern of splitting computation between a resource-constrained **Seed** device and a cluster of compute nodes, using cryptographic witnesses to bridge trust across the boundary. The pattern enables safety-critical edge deployments where the Seed must accept results from remote heavy compute without trusting raw outputs.

The core invariant: **the Seed never trusts computation it did not verify**. It delegates heavy work and validates compact, signed proofs before committing any state change.

---

## 1. Motivation and Problem Statement

### 1.1 The Split-Brain Problem in Edge AI

Current ruvector deployments assume a runtime-host split:
- `cognitum-gate-kernel` tiles operate in ≤64 KB WASM sandboxes with a bump allocator and no heap
- `ruvector-mincut` maintains dynamic graph state in tens of MB of RAM
- `ruvector-gnn` requires GPU-class memory for full forward passes
- Training via `ruvector-verified`'s `VerifiedTrainer` consumes hundreds of MB

A Seed device (embedded controller, field node, QR cognitive seed per ADR-034) cannot host these workloads. It can, however, host:

| Component | Memory Budget | Capability |
|-----------|--------------|------------|
| WASM tile kernel | ≤64 KB | Delta ingestion, e-value accumulation, local witness fragment |
| Ed25519 verifier | ~4 KB | Signature verification in ~40 µs |
| Merkle verifier | ~2 KB | Root hash recompute for N leaves in O(N log N) |
| Policy gate | ~1 KB | Threshold checks on metrics |
| Audit log (ring) | 8–32 KB | Append-only circular buffer |

The gap: **min-cut computation, GNN steps, training, and sparsification** live entirely on the cluster side and must cross a trust boundary before the Seed acts on their outputs.

### 1.2 Existing Primitives in ruvector

The workspace already provides building blocks for this pattern:

**On the cluster (heavy) side:**
- `ruvector-mincut`: subpolynomial dynamic min-cut with `CutCertificate`, `WitnessTree`, and `AuditLogger`
- `ruvector-mincut/src/fragment/`: fragment sharding for distributed min-cut
- `ruvector-gnn/src/training.rs`: `VerifiedTrainer` issuing 82-byte `ProofAttestation` per gradient step
- `ruvector-verified/src/gated.rs`: three-tier proof routing (`Reflex` / `Standard` / `Deep`)
- `cognitum-gate-kernel/src/canonical_witness.rs`: `CanonicalWitness` using `FixedPointWeight` for deterministic, hash-stable outputs

**On the Seed (reflex) side:**
- `cognitum-gate-kernel/src/evidence.rs`: e-value accumulator with pre-computed log thresholds
- `ruvector-verified/src/proof_store.rs`: `ProofAttestation` (82-byte verified proof with hash-consing)
- `ruvector-verified/src/gated.rs`: `ProofGate<T>` type that gates any value behind a proof
- `cognitum-gate-kernel/src/shard.rs`: `CompactGraph` fitting in the tile's 42 KB budget
- ADR-CE-012 (`gate-refusal-witness`): a Seed may reject a witness and produce its own signed refusal record
- ADR-CE-014 (`reflex-lane-default`): reflex lane is the safe default; cluster results must earn trust

---

## 2. Architecture

### 2.1 High-Level Flow

```
┌─────────────────────────────────┐      ┌──────────────────────────────────────┐
│           SEED                  │      │             CLUSTER                   │
│  (embedded / field / QR node)   │      │   (Cognitum / ruvector compute)       │
│                                 │      │                                      │
│  Reflex Loop:                   │      │  Heavy Kernels:                      │
│  ┌──────────────────────────┐   │ MCP  │  ┌────────────────────────────────┐  │
│  │ cognitum-gate-kernel     │──────────▶  │ ruvector-mincut                │  │
│  │  - ingest_delta          │   │      │  │  - Contract-and-verify min-cut │  │
│  │  - tick (deterministic)  │◀──────────  │  - Emit CutCertificate         │  │
│  │  - get_witness_fragment  │   │      │  └────────────────────────────────┘  │
│  └──────────────────────────┘   │      │  ┌────────────────────────────────┐  │
│         │                       │      │  │ ruvector-gnn (VerifiedTrainer) │  │
│  Verify Layer:                  │      │  │  - Gradient step + invariants  │  │
│  ┌──────────────────────────┐   │      │  │  - Emit ProofAttestation(82B)  │  │
│  │ WitnessCert verifier     │   │      │  └────────────────────────────────┘  │
│  │  - Ed25519 check         │   │      │  ┌────────────────────────────────┐  │
│  │  - Merkle root recompute │   │      │  │ RVF node signer                │  │
│  │  - Policy gate           │   │      │  │  - Bundle metrics + root       │  │
│  └──────────────────────────┘   │      │  │  - Sign WitnessCert (Ed25519)  │  │
│         │                       │      │  └────────────────────────────────┘  │
│  State Gate:                    │      │                                      │
│  ┌──────────────────────────┐   │      │                                      │
│  │ ProofGate<StateChange>   │   │      │                                      │
│  │  - commit_state_change() │   │      │                                      │
│  │  - append audit log      │   │      │                                      │
│  └──────────────────────────┘   │      │                                      │
└─────────────────────────────────┘      └──────────────────────────────────────┘
```

### 2.2 Trust Boundary Protocol

The MCP surface is minimal by design. The Seed exposes a single tool invocation that returns two fields:

```yaml
# rvf-to-mcp.yaml (minimal tool schema)
version: 1
tools:
  - name: rvf.invoke
    description: Submit a heavy RVF task and receive a signed witness.
    input_schema:
      type: object
      properties:
        task:
          type: string
          enum: ["mincut.contract", "gnn.step", "sparsify.segment"]
        params:
          type: object
    output_schema:
      type: object
      properties:
        witness_cert:
          type: string
          description: "base64-encoded WitnessCert"
        fragment_url:
          type: string
          description: "URL to fetch the full fragment for Merkle verification"
```

The Seed-side protocol after receiving the MCP response:

1. Decode `witness_cert` from base64 into `WitnessCert`
2. Fetch `fragment_url` → raw bytes
3. Recompute `Merkle(root(fragment_bytes))` and compare to `cert.merkle_root`
4. Verify `Ed25519(cert.sig_ed25519, encode(merkle_root || metrics || recourse_count || timestamp_ms), cert.pubkey_ed25519)`
5. Run `policy_check(cert.metrics, cert.recourse_count, cert.timestamp_ms)`
6. If all pass → call `commit_state_change()` and append to audit log
7. If any fail → reject, log `(merkle_root, metrics, sig)` as a refusal witness (per ADR-CE-012)

---

## 3. Data Structures

### 3.1 WitnessCert

The canonical witness certificate sent from cluster to Seed. Fixed-size fields only — no heap allocation required on the Seed for deserialization.

```rust
/// Compact, fixed-size witness certificate for Seed-side verification.
///
/// Wire format: 164 bytes total (all fields fixed-size, no padding).
/// Compatible with no_std environments and the cognitum-gate-kernel's
/// 64 KB memory budget.
#[repr(C)]
pub struct WitnessCert {
    /// SHA-256 Merkle root of the fragment (32 bytes).
    /// Seed recomputes this from the fetched fragment and compares.
    pub merkle_root: [u8; 32],

    /// Fixed-size metrics summary — task-specific, always 24 bytes.
    /// For mincut.contract: (lambda_value: f32, coherence: f32, edge_count: u32,
    ///                        vertex_count: u32, partition_hash: u64)
    /// For gnn.step:        (loss: f32, grad_norm: f32, step_idx: u32,
    ///                        layer_count: u32, weight_hash: u64)
    /// For sparsify.segment:(sparsity: f32, retained: f32, segment_id: u32,
    ///                        edge_count: u32, graph_hash: u64)
    pub metrics: [u8; 24],

    /// Number of times the cluster had to fall back to a recourse path.
    /// Policy gate: reject if recourse_count > threshold (e.g., 3).
    pub recourse_count: u32,

    /// Milliseconds since Unix epoch when the cluster signed this cert.
    /// Policy gate: reject if |now_ms - timestamp_ms| > 120_000 (±2 min).
    pub timestamp_ms: u64,

    /// Ed25519 signature over encode(merkle_root || metrics || recourse_count_le32
    ///                                || timestamp_ms_le64).
    /// 64 bytes.
    pub sig_ed25519: [u8; 64],

    /// Signer's Ed25519 public key (32 bytes).
    /// Must appear in the Seed's trusted-key roster for the cert to be accepted.
    pub pubkey_ed25519: [u8; 32],
}

/// Task-specific metrics decoded from WitnessCert.metrics.
pub enum WitnessMetrics {
    MinCut {
        lambda: f32,       // min-cut value (λ); policy: λ >= threshold
        coherence: f32,    // coherence score [0,1]
        edge_count: u32,
        vertex_count: u32,
        partition_hash: u64, // xxhash of sorted partition assignment
    },
    GnnStep {
        loss: f32,
        grad_norm: f32,
        step_idx: u32,
        layer_count: u32,
        weight_hash: u64,  // hash of updated weight tensor
    },
    SparsifySegment {
        sparsity: f32,     // fraction of edges retained after sparsification
        retained: f32,
        segment_id: u32,
        edge_count: u32,
        graph_hash: u64,
    },
}
```

**Total wire size: 164 bytes.** Fits trivially in the tile's 64 KB budget. Deserializes with zero allocation via `core::mem::transmute` from a `[u8; 164]` after signature verification.

### 3.2 Encoding for Signature

The signed byte sequence is deterministic and order-fixed. No schema versioning needed — field order is frozen at wire version 1:

```rust
fn encode_for_signing(cert: &WitnessCert) -> [u8; 72] {
    let mut buf = [0u8; 72];
    buf[0..32].copy_from_slice(&cert.merkle_root);
    buf[32..56].copy_from_slice(&cert.metrics);
    buf[56..60].copy_from_slice(&cert.recourse_count.to_le_bytes());
    buf[60..68].copy_from_slice(&cert.timestamp_ms.to_le_bytes());
    // 4 bytes padding for alignment (zeroed, not signed over)
    buf
}
```

---

## 4. Seed-Side Verifier (WASM)

### 4.1 Module Design

The Seed-side verifier compiles to a ≤48 KB WASM module. It exports exactly two functions:

```rust
// verifier_wasm.rs — compiles to WASM with Wasmtime + WASI caps

/// Verify a WitnessCert against a fetched fragment.
///
/// # Arguments
/// - `cert_bytes`: 164-byte WitnessCert (as defined above)
/// - `fragment_bytes`: raw fragment from fragment_url
/// - `trusted_keys`: length-prefixed list of trusted 32-byte Ed25519 pubkeys
///
/// # Returns
/// - 0: valid — Seed may commit state change
/// - 1: invalid signature
/// - 2: Merkle root mismatch
/// - 3: untrusted pubkey
/// - 4: timestamp out of window
/// - 5: policy violation (recourse_count or metrics threshold)
#[no_mangle]
pub extern "C" fn verify_witness(
    cert_ptr: *const u8, cert_len: usize,
    fragment_ptr: *const u8, fragment_len: usize,
    trusted_keys_ptr: *const u8, trusted_keys_len: usize,
) -> u32;

/// Check metrics and recourse_count against policy thresholds.
///
/// Called after verify_witness returns 0 for a secondary policy pass.
///
/// # Returns
/// - 0: policy satisfied
/// - non-zero: specific policy violation code
#[no_mangle]
pub extern "C" fn policy_check(
    cert_ptr: *const u8, cert_len: usize,
    policy_ptr: *const u8, policy_len: usize,
) -> u32;
```

### 4.2 Wasmtime Sandbox Configuration

```rust
// Host-side Wasmtime setup — runs on the Seed's native runtime
use wasmtime::{Engine, Module, Store, Linker, ResourceLimiter};
use wasmtime_wasi::WasiCtxBuilder;

pub fn build_verifier_store(wasm_bytes: &[u8]) -> Result<(Store<WasiCtx>, Module)> {
    let mut config = wasmtime::Config::new();
    config
        .consume_fuel(true)            // enable fuel metering
        .epoch_interruption(false);    // use fuel not epoch for determinism

    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, wasm_bytes)?;

    // WASI: stdin/stdout only — no FS, no network, no env
    let wasi = WasiCtxBuilder::new()
        .inherit_stdin()
        .inherit_stdout()
        .build();

    let mut store = Store::new(&engine, wasi);

    // Memory cap: 2 MiB (verifier needs <256 KB in practice)
    store.limiter(|_| &mut MemoryLimiter { max_bytes: 2 * 1024 * 1024 });

    // Fuel cap: ~10M instructions (~5 ms at 2 GHz, well above Ed25519+Merkle cost)
    store.add_fuel(10_000_000)?;

    Ok((store, module))
}

/// ResourceLimiter capping memory growth
struct MemoryLimiter { max_bytes: usize }

impl ResourceLimiter for MemoryLimiter {
    fn memory_growing(&mut self, current: usize, desired: usize, _max: Option<usize>)
        -> Result<bool> {
        Ok(desired <= self.max_bytes)
    }
    fn table_growing(&mut self, _c: u32, _d: u32, _m: Option<u32>) -> Result<bool> {
        Ok(true)
    }
}
```

**Key properties:**
- WASI caps: stdin/stdout only. No filesystem access (`no_fs`). No network (`no_net`). No environment variables.
- Fuel budget: 10M instructions. Ed25519 verification costs ~500K–1M instructions; Merkle recompute for a 256-leaf tree costs ~200K. Total headroom is 10x.
- Memory cap: 2 MiB. The `cognitum-gate-kernel` tile already fits in 64 KB; the verifier WASM needs even less.
- Module size target: ≤48 KB compiled WASM. Achieved by using `ed25519-dalek` in `no_std` mode with `alloc` disabled, and a simple iterative Merkle implementation.

---

## 5. Cluster-Side Signer

### 5.1 Signing After Heavy Computation

The cluster side uses existing ruvector primitives and adds an Ed25519 signing step over the result summary.

**For `mincut.contract`:**
```rust
// Uses ruvector-mincut CutCertificate + canonical fragment from cognitum-gate-kernel
use ruvector_mincut::{DynamicMinCut, MinCutBuilder};
use cognitum_gate_kernel::canonical_witness::ArenaCactus;
use ed25519_dalek::{SigningKey, Signer};

pub fn run_mincut_task(
    graph: &DynamicMinCut,
    signing_key: &SigningKey,
    fragment_store: &mut FragmentStore,
) -> WitnessCert {
    // 1. Compute min-cut (subpolynomial time via ruvector-mincut)
    let cut = graph.min_cut_value();
    let cert = graph.certificate(); // CutCertificate from ruvector-mincut

    // 2. Serialize fragment and compute Merkle root
    let fragment_bytes = cert.serialize_fragment();
    let merkle_root = merkle_sha256_root(&fragment_bytes);

    // 3. Build metrics (24 bytes fixed)
    let metrics = MinCutMetrics {
        lambda: cut as f32,
        coherence: compute_coherence(&cert),
        edge_count: graph.edge_count() as u32,
        vertex_count: graph.vertex_count() as u32,
        partition_hash: xxhash(&cert.partition_assignment()),
    }.to_bytes();

    // 4. Sign
    let encoded = encode_for_signing_raw(&merkle_root, &metrics, recourse_count, timestamp_ms);
    let sig = signing_key.sign(&encoded);

    // 5. Store fragment, return cert
    let fragment_url = fragment_store.put(fragment_bytes);
    WitnessCert {
        merkle_root,
        metrics,
        recourse_count,
        timestamp_ms: unix_ms(),
        sig_ed25519: sig.to_bytes(),
        pubkey_ed25519: signing_key.verifying_key().to_bytes(),
    }
}
```

**For `gnn.step`:**
```rust
// Uses ruvector-verified VerifiedTrainer (ADR-049)
// The VerifiedTrainer already emits ProofAttestation (82 bytes) per step.
// The cluster bundles the attestation hash into WitnessMetrics.weight_hash.
use ruvector_graph_transformer::verified_training::VerifiedTrainer;

pub fn run_gnn_step(
    trainer: &mut VerifiedTrainer,
    batch: &TrainingBatch,
    signing_key: &SigningKey,
    fragment_store: &mut FragmentStore,
) -> WitnessCert {
    let (loss, attestation) = trainer.step(batch)?;  // invariant-verified step
    let weight_hash = xxhash(attestation.proof_hash.as_bytes());
    let fragment_bytes = attestation.serialize();
    let merkle_root = merkle_sha256_root(&fragment_bytes);
    // ... sign and return WitnessCert
}
```

### 5.2 Key Management on the Cluster

Node keys are Ed25519 keypairs managed per RVF node. The Seed maintains a roster of trusted pubkeys (see §6.2 on key rotation).

```
Cluster node  →  Ed25519 keypair in secure enclave or HSM
             →  Signs WitnessCert with node_signing_key
             →  Publishes verifying_key via OTA manifest signed by root key
Seed          →  Trusts only pubkeys in its local roster
             →  Roster updated via OTA, signed by root (offline key)
```

---

## 6. Operations and Policy

### 6.1 Policy Gate Parameters

```rust
pub struct VerifierPolicy {
    /// Maximum allowed clock skew between Seed and cluster (milliseconds).
    /// Default: 120_000 (±2 min).
    pub clock_skew_ms: u64,

    /// Maximum recourse_count before cert is rejected.
    /// Default: 3.
    pub max_recourse: u32,

    /// Minimum λ (min-cut value) required to commit a topology change.
    /// Task-specific. Default for mincut.contract: 1.0.
    pub min_lambda: f32,

    /// Maximum coherence drift allowed per step for gnn.step.
    /// Default: 0.05 (5% drift).
    pub max_coherence_drift: f32,

    /// Minimum sparsity retention for sparsify.segment.
    /// Default: 0.1 (retain at least 10% of edges).
    pub min_sparsity_retained: f32,
}
```

All defaults are conservative. A Seed policy is compiled into the WASM verifier module at build time, making policy bypasses require a re-flash.

### 6.2 Key Rotation

```
1. Root operator generates new node keypair offline.
2. Operator publishes `key_roster_v{N}.json`:
   { "version": N, "keys": [<pubkey_hex>, ...], "expires_ms": <T> }
   Signed by root offline key (separate from node keys).
3. OTA manifest includes roster file + root signature.
4. Seed verifies root signature before installing new roster.
5. Old keys remain valid until `expires_ms`; new keys are immediately trusted.
6. Seed logs every key rotation event to its append-only audit trail.
```

**Key rotation does not require a Seed reboot** — the trusted-key roster is a runtime parameter passed to `verify_witness`.

### 6.3 Audit Trail

The Seed appends a fixed-size entry to a circular ring buffer for every witness verdict (accepted or rejected):

```rust
/// 64-byte audit entry — fixed size for ring buffer allocation.
#[repr(C)]
pub struct AuditEntry {
    pub timestamp_ms: u64,      // 8 bytes
    pub merkle_root: [u8; 32],  // 32 bytes
    pub verdict: u8,            // 0=accept, 1=reject_sig, 2=reject_merkle, ...
    pub task_type: u8,          // 0=mincut, 1=gnn, 2=sparsify
    pub recourse_count: u32,    // 4 bytes
    pub lambda_or_loss: f32,    // 4 bytes (task-specific metric)
    pub pubkey_prefix: [u8; 8], // 8 bytes (first 8 bytes of signer pubkey)
    pub _pad: [u8; 4],          // 4 bytes padding to 64 bytes
}
```

Ring buffer size: 512 entries × 64 bytes = 32 KB. At one witness per minute, this covers ~8.5 hours of offline operation. When the Seed comes online, it ships the full ring buffer to upstream storage and truncates.

---

## 7. Crate Mapping and File Layout

### 7.1 New Files (Drop-In Paths)

```
seed/rvf-bridge/
├── src/
│   ├── witness_chain.rs     # WitnessCert struct, encode_for_signing, AuditEntry
│   ├── verifier_wasm.rs     # Wasmtime host: build_verifier_store, run_verify
│   └── lib.rs               # Re-exports: WitnessCert, VerifierPolicy, AuditEntry
└── rvf-to-mcp.yaml          # MCP tool schema (rvf.invoke)
```

### 7.2 Existing Crate Integration

| New component | Uses existing crate | Integration point |
|--------------|--------------------|--------------------|
| Cluster min-cut signer | `ruvector-mincut` | `CutCertificate`, `AuditLogger`, `WitnessTree` |
| Cluster GNN step signer | `ruvector-graph-transformer` | `VerifiedTrainer` (ADR-049), `ProofAttestation` |
| Cluster canonical fragment | `cognitum-gate-kernel` | `canonical_witness::ArenaCactus`, `FixedPointWeight` |
| Seed verifier (native) | `ruvector-verified` | `ProofAttestation`, `ProofGate<T>`, `FastTermArena` |
| Seed gate | `ruvector-verified` | `ProofGate<StateChange>` wraps commit |
| Seed audit | `cognitum-gate-kernel` | `AuditData`, `AuditEntry`, `AuditLogger` |
| WASM verifier module | Standalone | Compiles separately; exports `verify_witness`, `policy_check` |

### 7.3 npm Package

For browser/Node.js Seed implementations, publish `rvf-wasm-verifier` (dev bridge):

```
npm/rvf-wasm-verifier/
├── src/
│   ├── verifier.ts     # TypeScript wrapper over WASM verify_witness
│   └── policy.ts       # VerifierPolicy type and defaults
├── wasm/               # Pre-compiled WASM artifact (≤48 KB)
└── package.json        # { "name": "rvf-wasm-verifier", "version": "0.1.0" }
```

Complements the existing `@ruvector/rvf-wasm` package. The `rvf-wasm-verifier` is Seed-side only — it cannot run heavy tasks.

---

## 8. Use Cases

### 8.1 Min-Cut–Gated Actuation (Robotics)

The `agentic-robotics` crates (`agentic-robotics-embedded`, `agentic-robotics-rt`) provide the control loop context. Today, actuation decisions are locally computed. With this pattern:

1. Seed's reflex loop detects a topology change request (e.g., regraph after obstacle).
2. Seed calls `rvf.invoke(task: "mincut.contract", params: {graph_fragment_id: X})`.
3. Cluster computes canonical min-cut, emits `WitnessCert`.
4. Seed verifies: λ ≥ 1.5 (safety threshold), coherence ≥ 0.8, recourse ≤ 2.
5. If valid → Seed toggles GPIO / updates motor command buffer.
6. If rejected → Seed executes conservative fallback (halt or prior trajectory).

The Seed never acts on raw cluster output. It acts only on **verified summaries** of cluster output.

### 8.2 Safe Online Training (VerifiedTrainer + WitnessCert)

The `ruvector-verified` `VerifiedTrainer` (ADR-049) issues `ProofAttestation` per gradient step. This pattern extends that to cross-device training:

1. Model weights live on the cluster; a read-only summary lives on the Seed.
2. Cluster runs a training step → emits `WitnessCert` with `weight_hash` and `loss` metrics.
3. Seed verifies the cert, checks loss stability (per `LossStabilityBound` invariant), and records the new weight hash.
4. Only a signed "stable step" cert can advance the Seed's authoritative weight reference.
5. If the cluster emits an unstable step (loss spike, gradient explosion), the Seed retains the last valid weight hash and logs a refusal witness.

This prevents a compromised or glitchy cluster from pushing corrupted weights to production Seed policy.

### 8.3 Field Autonomy (Offline Operation)

Seeds cache their most recent accepted `WitnessCert` in persistent storage (e.g., the QR cognitive seed's RVQS segment, ADR-034). When offline:

- Seed operates on the last verified graph topology.
- New heavy tasks queue locally; execution is deferred until cluster connectivity resumes.
- On reconnect, Seed submits queued tasks, verifies returned certs in sequence, and replays deferred state changes in audit-log order.

This gives deterministic, auditable behavior across connectivity gaps — critical for submarine, satellite, and field hospital deployments (per ADR-030 motivation).

---

## 9. Security Properties

### 9.1 Threat Model

| Threat | Mitigation |
|--------|-----------|
| Malicious cluster returns crafted metrics | Ed25519 signature verification. Forging a sig requires the node's private key. |
| Cluster replays an old valid cert | `timestamp_ms` check with ±2 min window. |
| Cluster presents a valid cert with wrong fragment | Merkle root recompute. If `fragment_url` returns tampered bytes, root mismatch → reject. |
| Unknown/expired cluster key | Seed's trusted-key roster rejects certs from unrecognized pubkeys. |
| Runaway WASM in verifier | Wasmtime `ResourceLimiter` (2 MiB) + fuel cap (10M instrs). Module cannot escape sandbox. |
| Seed audit log tampering | Ring buffer is append-only; entries include `merkle_root` and `sig` from the original cert. A tampered entry would require regenerating a valid Ed25519 signature. |
| Clock manipulation | Seed maintains a monotonic counter alongside wall clock. If wall clock regresses, Seed flags it and applies conservative timestamp policy. |

### 9.2 What This Pattern Does NOT Protect Against

- **Compromised node signing key**: If the cluster's Ed25519 private key leaks, an attacker can forge `WitnessCert`s. Mitigated by HSM storage and key rotation.
- **Quantum adversary**: Ed25519 is not post-quantum. ADR-034 mentions ML-DSA-65 as a post-quantum alternative (3,309-byte signature). For Seed deployments with PQ requirements, the cert structure accommodates a `sig_algo` field extension.
- **Correct-but-pathological cluster behavior**: A cluster that computes the right min-cut but with adversarial graph manipulation still produces a valid cert. The Seed's policy gate (`min_lambda`, `max_coherence_drift`) provides a semantic layer, but cannot prove the cluster operated on the correct input graph.

---

## 10. Alignment with Existing ADRs

| ADR | Relevant principle | How this pattern aligns |
|-----|-------------------|------------------------|
| ADR-CE-014 (reflex-lane-default) | Reflex is the safe default; cluster results must earn entry | Seed reflex loop runs independently. WitnessCert verification gates cluster influence. |
| ADR-CE-012 (gate-refusal-witness) | A gate may refuse and produce its own signed refusal | Seed produces `AuditEntry` with `verdict=reject` on any failed verification. |
| ADR-CE-017 (unified-audit-trail) | All events flow through a single audit log | Seed ring buffer captures every accept/reject with merkle_root + sig prefix. |
| ADR-047 (proof-gated-mutation) | No mutation without proof | `ProofGate<StateChange>` wraps the commit path; cluster cert maps to a proof obligation. |
| ADR-049 (verified-training-pipeline) | Per-step invariant proofs for training | `VerifiedTrainer` attestations feed into `WitnessCert.metrics.weight_hash`. |
| ADR-034 (QR cognitive seed) | Offline-first Seed with signed bootstrap | Cached WitnessCerts enable offline autonomy; Seed resumes from last verified state. |
| ADR-030 (cognitive container) | Self-booting, attested compute | WitnessCert's `pubkey_ed25519` participates in the attestation chain of the container. |

---

## 11. Implementation Sequencing

### Phase 1 — Structures and Signing (Week 1–2)

1. Define `WitnessCert`, `WitnessMetrics`, `AuditEntry` in `seed/rvf-bridge/src/witness_chain.rs`
2. Implement `encode_for_signing` (deterministic byte encoding)
3. Add cluster-side signing to `ruvector-mincut`'s `CutCertificate` path
4. Add cluster-side signing to `ruvector-graph-transformer`'s `VerifiedTrainer`
5. Unit test: sign → serialize → deserialize → verify (round-trip)

### Phase 2 — Seed Verifier WASM (Week 2–3)

1. Implement `verify_witness` and `policy_check` in `seed/rvf-bridge/src/verifier_wasm.rs`
2. Compile to WASM targeting `wasm32-unknown-unknown` with `no_std`
3. Measure compiled size (target: ≤48 KB)
4. Measure fuel consumption for Ed25519 + 256-leaf Merkle (target: ≤2M instructions)
5. Write Wasmtime host harness in `seed/rvf-bridge/src/lib.rs`

### Phase 3 — MCP Integration and Policy (Week 3–4)

1. Write `rvf-to-mcp.yaml` tool schema
2. Implement MCP client in Seed runtime that calls `rvf.invoke` and feeds response to verifier
3. Implement `VerifierPolicy` with configurable thresholds
4. Implement `AuditEntry` ring buffer with overflow semantics
5. Integration test: full round-trip Seed→MCP→cluster→Seed with `mincut.contract`

### Phase 4 — Key Rotation and OTA (Week 4–5)

1. Design `key_roster_v{N}.json` schema
2. Implement roster-loading in the Seed's trusted-key store
3. Implement root-signature verification for OTA roster updates
4. Test key rotation without Seed reboot

### Phase 5 — npm Bridge Package (Week 5–6)

1. Bundle WASM verifier as `rvf-wasm-verifier` npm package
2. Write TypeScript wrapper (`verifier.ts`, `policy.ts`)
3. Integration test with `@ruvector/rvf-wasm` on both Node.js and browser (via rvlite)

---

## 12. Open Questions

1. **Fragment URL scheme**: Should `fragment_url` be an HTTP endpoint, an IPFS CID, or an RVF segment address? IPFS CIDs provide content-addressed retrieval (Merkle root is the CID), eliminating the separate Merkle recompute step. Tradeoff: IPFS gateway availability vs. simpler HTTP.

2. **Cert aggregation for multi-tile Cognitum gates**: The 256-tile Cognitum gate (`cognitum-gate-kernel`) produces 256 witness fragments. Should the cluster aggregate these into a single `WitnessCert` (one Merkle root over all 256 fragment roots), or should the Seed verify 256 individual certs? Single aggregated cert is cheaper for the Seed; per-tile certs give finer-grained attribution.

3. **Recourse semantics**: `recourse_count` tracks fallback invocations. Should this be per-attempt (reset on each `rvf.invoke` call) or cumulative (grows across a session until a successful cert resets it)? Cumulative recourse count gives the Seed a health signal for the cluster; per-attempt is simpler.

4. **Seed-side Merkle implementation**: Should the WASM verifier implement SHA-256 from scratch (zero external deps, ≤2 KB) or import `sha2` crate (larger but audited)? The `no_std` `sha2` crate compiles to ~6 KB; a hand-rolled SHA-256 is ~1.5 KB. Both are reasonable for the 48 KB budget.

5. **Temporal tensor integration**: `ruvector-temporal-tensor` maintains time-indexed graph state. Should `timestamp_ms` in `WitnessCert` align with temporal tensor epochs, allowing the Seed to precisely associate verified state with a point in the temporal graph?

---

## 13. Summary

The Reflex-on-Seed, Heavy-on-Cluster pattern provides a clean separation between:

- **What runs on the Seed**: tiny WASM verifier (≤48 KB), deterministic reflex loop, append-only audit log, policy gate
- **What runs on the cluster**: min-cut, GNN steps, sparsification, training — all existing ruvector crates
- **What crosses the trust boundary**: a 164-byte `WitnessCert` with Ed25519 signature and Merkle root, verifiable in microseconds

The pattern reuses `ruvector-mincut`'s `CutCertificate`, `ruvector-verified`'s `ProofGate<T>` and `VerifiedTrainer`, `cognitum-gate-kernel`'s canonical witness and evidence accumulator, and the audit trail infrastructure from ADR-CE-017. No new cryptographic primitives are introduced; the Seed verifier uses `ed25519-dalek` in `no_std` mode.

The result: **Seed devices can operate safely offline, accept cluster results selectively, and maintain a tamper-evident audit trail** — without running any untrusted compute locally beyond the sandboxed WASM verifier module.
