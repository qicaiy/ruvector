# ADR-001: Anytime-Valid Coherence Gate

**Status**: Proposed
**Date**: 2026-01-17
**Authors**: Research Team
**Deciders**: Architecture Review Board

## Context

The RuVector ecosystem requires a principled mechanism for controlling autonomous agent actions with:
- **Formal safety guarantees** under distribution shift
- **Computational efficiency** suitable for real-time enforcement
- **Auditable decision trails** with cryptographic receipts

Current approaches (threshold classifiers, rule-based systems, periodic audits) lack one or more of these properties. This ADR proposes the **Anytime-Valid Coherence Gate (AVCG)** - a 3-way algorithmic combination that converts coherence measurement into a deterministic control loop.

## Decision

We will implement an Anytime-Valid Coherence Gate that integrates three cutting-edge algorithmic components:

### 1. Dynamic Min-Cut with Witness Partitions

**Source**: El-Hayek, Henzinger, Li (arXiv:2512.13105, December 2025)

**Key Innovation**: Exact deterministic n^{o(1)} update time for cuts up to 2^{Θ(log^{3/4-c}n)}

**Integration**:
- Extends existing `SubpolynomialMinCut` in `ruvector-mincut/src/subpolynomial/mod.rs`
- Leverages existing `WitnessTree` for explicit partition certificates
- Uses deterministic `LocalKCut` for local cut verification

**Role in Gate**: Provides the **structural coherence signal** - identifies minimal intervention points in the agent action graph with explicit witness partitions showing which actions form the critical boundary to unsafe states.

### 2. Online Conformal Prediction with Shift-Awareness

**Sources**:
- Retrospective Adjustment (arXiv:2511.04275, November 2025)
- Conformal Optimistic Prediction (COP) (December 2025)
- CORE: RL-based Conformal Regression (October 2025)

**Key Innovation**: Distribution-free coverage guarantees that adapt to arbitrary distribution shift with faster recalibration via retrospective adjustment.

**Integration**:
- New module: `ruvector-mincut/src/conformal/` for prediction sets
- Interfaces with existing `GatePolicy` thresholds
- Wraps action outcome predictions with calibrated uncertainty

**Role in Gate**: Provides the **predictive uncertainty signal** - quantifies confidence in action outcomes, triggering DEFER when prediction sets are too large.

### 3. E-Values and E-Processes for Anytime-Valid Inference

**Sources**:
- Ramdas & Wang "Hypothesis Testing with E-values" (FnTStA 2025)
- ICML 2025 Tutorial on SAVI
- Sequential Randomization Tests (arXiv:2512.04366, December 2025)

**Key Innovation**: Evidence accumulation that remains valid at any stopping time, with multiplicative composition across experiments.

**Definition**: E-value e satisfies E[e] ≤ 1 under null hypothesis. E-processes are nonnegative supermartingales with E_0 = 1.

**Integration**:
- New module: `ruvector-mincut/src/eprocess/` for evidence tracking
- Integrates with existing `CutCertificate` for audit trails
- Enables anytime-valid stopping decisions

**Role in Gate**: Provides the **evidential validity signal** - accumulates statistical evidence for/against coherence with formal Type I error control at any stopping time.

## Gate Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ANYTIME-VALID COHERENCE GATE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│   │  DYNAMIC MIN-CUT │    │    CONFORMAL     │    │   E-PROCESS      │ │
│   │    (Structural)  │    │   (Predictive)   │    │  (Evidential)    │ │
│   │                  │    │                  │    │                  │ │
│   │  SubpolynomialMC │    │  ShiftAdaptive   │    │  CoherenceTest   │ │
│   │  WitnessTree     │───▶│  PredictionSet   │───▶│  EvidenceAccum   │ │
│   │  LocalKCut       │    │  COP/CORE        │    │  StoppingRule    │ │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘ │
│            │                       │                       │           │
│            ▼                       ▼                       ▼           │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    DECISION LOGIC                              │   │
│   │                                                                │   │
│   │   PERMIT: E_t > τ_permit ∧ action ∉ CriticalCut ∧ |C_t| small │   │
│   │   DEFER:  |C_t| large ∨ τ_deny < E_t < τ_permit               │   │
│   │   DENY:   E_t < τ_deny ∨ action ∈ WitnessPartition(unsafe)    │   │
│   │                                                                │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                               │                                        │
│                               ▼                                        │
│                    ┌─────────────────────┐                            │
│                    │   WITNESS RECEIPT   │                            │
│                    │  (cut + conf + e)   │                            │
│                    └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Integration with Existing Architecture

### Extension Points

| Component | Current Implementation | AVCG Extension |
|-----------|----------------------|----------------|
| `GatePacket` | λ as point estimate | Add `lambda_confidence_q15`, `e_value_log_q15` |
| `GateController` | Rule-based thresholds | Add `AnytimeGatePolicy` with adaptive thresholds |
| `WitnessTree` | Cut value only | Add `ConfidenceWitness` with staleness tracking |
| `CutCertificate` | Static verification | Add `EvidenceReceipt` with e-value trace |
| `TierDecision` | Fixed tiers | Add `required_confidence_for_tier` |

### New Modules

```
ruvector-mincut/
├── src/
│   ├── conformal/           # NEW: Online conformal prediction
│   │   ├── mod.rs
│   │   ├── prediction_set.rs
│   │   ├── cop.rs           # Conformal Optimistic Prediction
│   │   ├── retrospective.rs # Retrospective adjustment
│   │   └── core.rs          # RL-based conformal
│   ├── eprocess/            # NEW: E-value and e-process tracking
│   │   ├── mod.rs
│   │   ├── evalue.rs
│   │   ├── evidence_accum.rs
│   │   ├── stopping.rs
│   │   └── mixture.rs
│   ├── anytime_gate/        # NEW: Integrated gate controller
│   │   ├── mod.rs
│   │   ├── policy.rs
│   │   ├── decision.rs
│   │   └── receipt.rs
│   └── ...existing modules...
```

## Decision Rules

### Permit Conditions (all must hold)
1. E-process value E_t > τ_permit (sufficient evidence of coherence)
2. Action not in witness partition of critical cut
3. Conformal prediction set |C_t| < θ_confidence (confident prediction)

### Defer Conditions (any triggers)
1. Conformal prediction set |C_t| > θ_uncertainty (uncertain outcome)
2. E-process in indeterminate range: τ_deny < E_t < τ_permit
3. Deadline approaching without sufficient confidence

### Deny Conditions (any triggers)
1. E-process value E_t < τ_deny (strong evidence of incoherence)
2. Action in witness partition crossing to unsafe states
3. Structural impossibility via min-cut topology

## Threshold Configuration

| Threshold | Meaning | Recommended Default |
|-----------|---------|---------------------|
| τ_deny | E-process level indicating incoherence | 0.01 (1% false alarm) |
| τ_permit | E-process level indicating coherence | 100 (strong evidence) |
| θ_uncertainty | Conformal set size requiring deferral | Task-dependent |
| θ_confidence | Conformal set size for confident permit | Task-dependent |

## Witness Receipt Structure

```rust
pub struct WitnessReceipt {
    /// Timestamp of decision
    pub timestamp: u64,
    /// Action that was evaluated
    pub action_id: ActionId,
    /// Gate decision
    pub decision: GateDecision,

    // Structural witness (from min-cut)
    pub cut_value: f64,
    pub witness_partition: (Vec<VertexId>, Vec<VertexId>),
    pub critical_edges: Vec<EdgeId>,

    // Predictive witness (from conformal)
    pub prediction_set: ConformalSet,
    pub coverage_target: f32,
    pub shift_adaptation_rate: f32,

    // Evidential witness (from e-process)
    pub e_value: f64,
    pub e_process_cumulative: f64,
    pub stopping_valid: bool,

    // Cryptographic seal
    pub receipt_hash: [u8; 32],
}
```

## Consequences

### Benefits

1. **Formal Guarantees**: Type I error control at any stopping time
2. **Distribution Shift Robustness**: Conformal prediction adapts without retraining
3. **Computational Efficiency**: O(n^{o(1)}) update time from subpolynomial min-cut
4. **Audit Trail**: Every decision has cryptographic witness receipt
5. **Defense in Depth**: Three independent signals must concur for permit

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Computational overhead | Lazy evaluation; batch updates; SIMD optimization |
| E-value power under uncertainty | Mixture e-values for robustness |
| Graph model mismatch | Learn graph structure from trajectories |
| Threshold tuning | Adaptive thresholds via meta-learning |

### Complexity Analysis

| Operation | Current | With AVCG |
|-----------|---------|-----------|
| Edge update | O(n^{o(1)}) | O(n^{o(1)}) (unchanged) |
| Gate evaluation | O(1) | O(k) where k = prediction set computation |
| Witness generation | O(m) | O(m) (amortized) |
| Certificate verification | O(n) | O(n + log T) where T = history length |

## References

### Dynamic Min-Cut
1. El-Hayek, Henzinger, Li. "Deterministic and Exact Fully-dynamic Minimum Cut of Superpolylogarithmic Size in Subpolynomial Time." arXiv:2512.13105, December 2025.
2. Jin, Sun, Thorup. "Fully Dynamic Exact Minimum Cut in Subpolynomial Time." SODA 2024.

### Online Conformal Prediction
3. "Online Conformal Inference with Retrospective Adjustment for Faster Adaptation to Distribution Shift." arXiv:2511.04275, November 2025.
4. "Distribution-informed Online Conformal Prediction (COP)." December 2025.
5. "CORE: Conformal Regression under Distribution Shift via Reinforcement Learning." October 2025.

### E-Values and E-Processes
6. Ramdas, Wang. "Hypothesis Testing with E-values." Foundations and Trends in Statistics, 2025.
7. ICML 2025 Tutorial: "Game-theoretic Statistics and Sequential Anytime-Valid Inference."
8. "Sequential Randomization Tests Using e-values." arXiv:2512.04366, December 2025.

### AI Agent Control
9. "Bounded Autonomy: A Pragmatic Response to Concerns About Fully Autonomous AI Agents." XMPRO, 2025.
10. "Customizable Runtime Enforcement for Safe and Reliable LLM Agents." arXiv:2503.18666, 2025.

## Appendix: Mathematical Foundations

### E-Value Composition

For independent e-values e₁, e₂:
```
e_combined = e₁ · e₂
E[e_combined] = E[e₁] · E[e₂] ≤ 1 · 1 = 1
```

This enables **optional continuation**: evidence accumulates validly across sessions.

### Conformal Coverage

Under exchangeability or bounded distribution shift:
```
P(Y_{t+1} ∈ C_t(X_{t+1})) ≥ 1 - α - δ_t
```

Where δ_t → 0 as the algorithm adapts via retrospective adjustment.

### Anytime-Valid Stopping

For any stopping time τ (possibly data-dependent):
```
P_H₀(E_τ ≥ 1/α) ≤ α
```

This holds because E_t is a nonnegative supermartingale with E[E_0] = 1.
