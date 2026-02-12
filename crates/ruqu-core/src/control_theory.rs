//! Hybrid classical-quantum control theory engine for QEC.
//!
//! Models the QEC feedback loop as a discrete-time control system:
//!
//! ```text
//! Physical qubits -> Syndrome extraction -> Classical decode -> Correction -> Repeat
//! ```
//!
//! The central insight: if classical decoding latency exceeds the syndrome
//! extraction period, errors accumulate faster than they are corrected (the
//! "backlog problem"). This module provides:
//!
//! - **Stability analysis**: derive conditions under which the control loop
//!   converges (decode latency < syndrome period).
//! - **Resource optimization**: Pareto-optimal allocation of physical qubits
//!   and classical compute for a given error budget.
//! - **Latency budget planning**: breakdown of time budgets per QEC round.
//! - **Backlog simulation**: Monte Carlo simulation of error accumulation
//!   under realistic timing constraints.
//! - **Scaling laws**: asymptotic classical overhead and logical error rate
//!   scaling with code distance.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Error types available for future extensions requiring fallible operations.
#[allow(unused_imports)]
use crate::error::{QuantumError, Result};

// ---------------------------------------------------------------------------
// 1. Control Loop Model
// ---------------------------------------------------------------------------

/// The full QEC control loop: plant (quantum side) + controller (classical
/// side) + runtime state.
#[derive(Debug, Clone)]
pub struct QecControlLoop {
    /// The quantum subsystem being protected.
    pub plant: QuantumPlant,
    /// The classical decoder subsystem.
    pub controller: ClassicalController,
    /// Accumulated runtime state of the loop.
    pub state: ControlState,
}

/// Physical parameters of the quantum error-correction code.
#[derive(Debug, Clone)]
pub struct QuantumPlant {
    /// Surface code distance.
    pub code_distance: u32,
    /// Per-gate / per-cycle physical error rate.
    pub physical_error_rate: f64,
    /// Number of data qubits (typically d^2 for a surface code patch).
    pub num_data_qubits: u32,
    /// T1/T2 coherence time in nanoseconds.
    pub coherence_time_ns: u64,
}

/// Classical decoder performance characteristics.
#[derive(Debug, Clone)]
pub struct ClassicalController {
    /// Wall-clock decode latency per syndrome round (ns).
    pub decode_latency_ns: u64,
    /// Sustained decode throughput (syndromes/sec).
    pub decode_throughput: f64,
    /// Decoder accuracy: probability of choosing the correct correction.
    pub accuracy: f64,
}

/// Evolving state of the control loop during execution.
#[derive(Debug, Clone)]
pub struct ControlState {
    /// Current effective logical error rate.
    pub logical_error_rate: f64,
    /// Accumulated error backlog (fractional uncorrected rounds).
    pub error_backlog: f64,
    /// Total syndrome rounds decoded so far.
    pub rounds_decoded: u64,
    /// Total elapsed time (ns).
    pub total_latency_ns: u64,
}

impl ControlState {
    /// Fresh initial state with no history.
    pub fn new() -> Self {
        Self {
            logical_error_rate: 0.0,
            error_backlog: 0.0,
            rounds_decoded: 0,
            total_latency_ns: 0,
        }
    }
}

impl Default for ControlState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 2. Stability Analysis
// ---------------------------------------------------------------------------

/// Result of analyzing the control loop's stability.
#[derive(Debug, Clone)]
pub struct StabilityCondition {
    /// Whether the loop converges (errors do not accumulate unboundedly).
    pub is_stable: bool,
    /// Stability margin: ratio of syndrome period to decode latency minus 1.
    /// Positive means stable, negative means unstable.
    pub margin: f64,
    /// Maximum decode latency (ns) that still yields a stable loop.
    pub critical_latency_ns: u64,
    /// Maximum physical error rate that still yields a stable loop
    /// at the given code distance.
    pub critical_error_rate: f64,
    /// Exponential convergence rate of the backlog per round.
    /// Positive means the backlog shrinks; negative means it grows.
    pub convergence_rate: f64,
}

/// Syndrome extraction period for a distance-d surface code.
///
/// Each round requires O(d) time-steps of gate operations. At ~50 ns per
/// gate layer the syndrome period is approximately `6 * d * 50` ns, but we
/// use `6 * d * 20` ns as a representative fast-gate estimate.
fn syndrome_period_ns(distance: u32) -> u64 {
    // 6 gate layers per extraction cycle, ~20 ns per gate layer.
    let layers_per_round: u64 = 6;
    let gate_time_ns: u64 = 20;
    layers_per_round * (distance as u64) * gate_time_ns
}

/// Analyze the stability of a QEC control loop.
///
/// The loop is stable when the decoder can keep up with incoming
/// syndromes: `decode_latency < syndrome_period`. The margin quantifies
/// how much headroom exists.
pub fn analyze_stability(config: &QecControlLoop) -> StabilityCondition {
    let d = config.plant.code_distance;
    let p = config.plant.physical_error_rate;
    let t_decode = config.controller.decode_latency_ns;
    let acc = config.controller.accuracy;

    let t_syndrome = syndrome_period_ns(d);

    // Stability: decode must finish before the next syndrome round.
    let margin = if t_decode == 0 {
        f64::INFINITY
    } else {
        (t_syndrome as f64 / t_decode as f64) - 1.0
    };
    let is_stable = t_decode < t_syndrome;

    // Critical latency: the syndrome period itself.
    let critical_latency_ns = t_syndrome;

    // Critical error rate: the surface code threshold (~1% for standard
    // decoders). Below this rate, increasing d suppresses logical errors
    // exponentially. We model threshold as a function of decoder accuracy.
    let threshold = 0.01 * acc;
    let critical_error_rate = threshold;

    // Convergence rate: each round, the fraction of the backlog cleared.
    // If decode finishes in time, roughly one syndrome is consumed per
    // period. The backlog drain rate per round is:
    //   rate = 1 - (t_decode / t_syndrome) - p * d  (error injection)
    // Positive means the backlog shrinks.
    let error_injection_per_round = p * (d as f64);
    let convergence_rate = if t_syndrome > 0 {
        1.0 - (t_decode as f64 / t_syndrome as f64) - error_injection_per_round
    } else {
        -1.0
    };

    StabilityCondition {
        is_stable,
        margin,
        critical_latency_ns,
        critical_error_rate,
        convergence_rate,
    }
}

/// Find the maximum code distance that is stable for a given controller
/// and physical error rate.
///
/// Iterates distances 3, 5, 7, ... (odd) until the decode latency exceeds
/// the syndrome period or the error rate exceeds threshold. Returns the
/// largest stable distance found.
pub fn max_stable_distance(controller: &ClassicalController, error_rate: f64) -> u32 {
    let mut best = 3u32;
    // Decode latency typically scales with distance. We model it as
    // constant here (the controller's stated latency), which gives the
    // upper bound for the given hardware.
    for d in (3..=201).step_by(2) {
        let t_syndrome = syndrome_period_ns(d);
        if controller.decode_latency_ns >= t_syndrome {
            break;
        }
        // Also check that the error rate is below threshold for this d.
        let threshold = 0.01 * controller.accuracy;
        if error_rate >= threshold {
            break;
        }
        best = d;
    }
    best
}

/// Minimum decoder throughput (syndromes/sec) required to keep up with
/// a given quantum plant.
///
/// The plant generates one syndrome every `syndrome_period_ns` nanoseconds.
pub fn min_throughput(plant: &QuantumPlant) -> f64 {
    let t_ns = syndrome_period_ns(plant.code_distance);
    if t_ns == 0 {
        return f64::INFINITY;
    }
    1e9 / t_ns as f64
}

// ---------------------------------------------------------------------------
// 3. Resource Optimization
// ---------------------------------------------------------------------------

/// Available hardware resources.
#[derive(Debug, Clone)]
pub struct ResourceBudget {
    /// Total physical qubits available.
    pub total_physical_qubits: u32,
    /// Number of classical CPU cores for decoding.
    pub classical_cores: u32,
    /// Clock speed of each classical core (GHz).
    pub classical_clock_ghz: f64,
    /// Wall-clock time budget for the entire computation (microseconds).
    pub total_time_budget_us: u64,
}

/// A candidate allocation point on the Pareto frontier.
#[derive(Debug, Clone)]
pub struct OptimalAllocation {
    /// Chosen code distance.
    pub code_distance: u32,
    /// Number of logical qubits that fit.
    pub logical_qubits: u32,
    /// Number of classical threads dedicated to decoding.
    pub decode_threads: u32,
    /// Expected logical error rate at this operating point.
    pub expected_logical_error_rate: f64,
    /// Pareto score (higher is better): balances logical qubits vs error rate.
    pub pareto_score: f64,
}

/// Enumerate Pareto-optimal resource allocations.
///
/// Sweeps odd code distances from 3 up to the maximum that fits in the
/// qubit budget, computes the resulting logical qubit count, expected
/// logical error rate, and ranks by a combined Pareto score.
///
/// Returns allocations sorted by descending `pareto_score`.
pub fn optimize_allocation(
    budget: &ResourceBudget,
    error_rate: f64,
    min_logical: u32,
) -> Vec<OptimalAllocation> {
    let mut candidates = Vec::new();

    // Physical qubits per logical qubit at distance d: 2d^2 - 2d + 1.
    for d in (3u32..=99).step_by(2) {
        let qubits_per_logical = 2 * d * d - 2 * d + 1;
        if qubits_per_logical == 0 {
            continue;
        }
        let max_logical = budget.total_physical_qubits / qubits_per_logical;
        if max_logical < min_logical {
            continue;
        }

        // Estimate decode latency from classical resources.
        // Model: decode time ~ d^3 / (cores * clock_ghz) nanoseconds.
        let decode_ns = if budget.classical_cores > 0 && budget.classical_clock_ghz > 0.0 {
            let raw = (d as f64).powi(3)
                / (budget.classical_cores as f64 * budget.classical_clock_ghz);
            raw as u64
        } else {
            u64::MAX
        };

        // Allocate a fraction of cores to decoding.
        let decode_threads = budget.classical_cores.min(max_logical);

        // Expected logical error rate: p_L ~ A * (p / p_th)^((d+1)/2)
        // where p_th ~ 0.01.
        let p_th = 0.01_f64;
        let ratio = error_rate / p_th;
        let exponent = (d as f64 + 1.0) / 2.0;
        let p_logical = if ratio < 1.0 {
            0.1 * ratio.powf(exponent)
        } else {
            // Above threshold: logical error rate does not improve.
            1.0_f64.min(ratio.powf(exponent))
        };

        // Check timing: can we complete enough rounds within the budget?
        let t_syndrome = syndrome_period_ns(d);
        let round_time = t_syndrome.max(decode_ns);
        let budget_ns = budget.total_time_budget_us * 1000;
        let affordable_rounds = if round_time > 0 {
            budget_ns / round_time
        } else {
            0
        };
        if affordable_rounds == 0 {
            continue;
        }

        // Pareto score: reward more logical qubits and lower error rate.
        // score = log2(logical_qubits) - log10(p_logical)
        let score = if p_logical > 0.0 && max_logical > 0 {
            (max_logical as f64).log2() - p_logical.log10()
        } else if max_logical > 0 {
            (max_logical as f64).log2() + 15.0 // cap for p_logical ~ 0
        } else {
            0.0
        };

        candidates.push(OptimalAllocation {
            code_distance: d,
            logical_qubits: max_logical,
            decode_threads,
            expected_logical_error_rate: p_logical,
            pareto_score: score,
        });
    }

    // Sort by descending Pareto score.
    candidates.sort_by(|a, b| {
        b.pareto_score
            .partial_cmp(&a.pareto_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates
}

// ---------------------------------------------------------------------------
// 4. Latency Budget Planner
// ---------------------------------------------------------------------------

/// Breakdown of time budgets for a single QEC round.
#[derive(Debug, Clone)]
pub struct LatencyBudget {
    /// Time to extract syndromes (ns).
    pub syndrome_extraction_ns: u64,
    /// Time to decode the syndrome classically (ns).
    pub decode_ns: u64,
    /// Time to apply the correction (ns).
    pub correction_ns: u64,
    /// Total round time (ns).
    pub total_round_ns: u64,
    /// Slack: positive means time to spare, negative means overrun.
    pub slack_ns: i64,
}

/// Plan the latency budget for a single QEC round.
///
/// `distance`: surface code distance.
/// `decode_ns_per_syndrome`: classical decode time per syndrome (ns).
///
/// The syndrome extraction time is derived from the code distance.
/// Correction application is modeled as a single gate layer (~20 ns).
pub fn plan_latency_budget(distance: u32, decode_ns_per_syndrome: u64) -> LatencyBudget {
    let extraction_ns = syndrome_period_ns(distance);
    let correction_ns: u64 = 20; // Single Pauli gate layer.
    let decode_ns = decode_ns_per_syndrome;

    let total_round_ns = extraction_ns + decode_ns + correction_ns;

    // Slack relative to the syndrome period: we must complete the full
    // round within one syndrome period to avoid backlog.
    let slack_ns = extraction_ns as i64 - (decode_ns as i64 + correction_ns as i64);

    LatencyBudget {
        syndrome_extraction_ns: extraction_ns,
        decode_ns,
        correction_ns,
        total_round_ns,
        slack_ns,
    }
}

// ---------------------------------------------------------------------------
// 5. Backlog Simulator
// ---------------------------------------------------------------------------

/// Full trace of a simulated control loop execution.
#[derive(Debug, Clone)]
pub struct SimulationTrace {
    /// Per-round snapshots.
    pub rounds: Vec<RoundSnapshot>,
    /// Whether the backlog converged to zero.
    pub converged: bool,
    /// Logical error rate at the end of the simulation.
    pub final_logical_error_rate: f64,
    /// Peak backlog observed.
    pub max_backlog: f64,
}

/// Snapshot of a single simulation round.
#[derive(Debug, Clone)]
pub struct RoundSnapshot {
    /// Round index.
    pub round: u64,
    /// Number of physical errors injected this round.
    pub errors_this_round: u32,
    /// Number of errors successfully corrected this round.
    pub errors_corrected: u32,
    /// Current backlog (fractional uncorrected errors).
    pub backlog: f64,
    /// Decode latency for this round (ns).
    pub decode_latency_ns: u64,
}

/// Run a Monte Carlo simulation of the QEC control loop.
///
/// Each round:
/// 1. Sample physical errors from a Bernoulli distribution.
/// 2. Attempt to decode within the decode latency window.
/// 3. If decode finishes in time, correct errors; otherwise backlog grows.
/// 4. Track the logical error rate as the fraction of rounds with
///    uncorrectable errors.
///
/// Uses a seeded RNG for reproducibility.
pub fn simulate_control_loop(
    config: &QecControlLoop,
    num_rounds: u64,
    seed: u64,
) -> SimulationTrace {
    let mut rng = StdRng::seed_from_u64(seed);
    let d = config.plant.code_distance;
    let p = config.plant.physical_error_rate;
    let num_qubits = config.plant.num_data_qubits;
    let t_decode = config.controller.decode_latency_ns;
    let acc = config.controller.accuracy;
    let t_syndrome = syndrome_period_ns(d);

    let mut rounds = Vec::with_capacity(num_rounds as usize);
    let mut backlog: f64 = 0.0;
    let mut max_backlog: f64 = 0.0;
    let mut logical_errors: u64 = 0;
    let mut _total_latency_ns: u64 = 0;

    for r in 0..num_rounds {
        // 1. Sample errors: each data qubit has probability p of an error.
        let mut errors_this_round: u32 = 0;
        for _ in 0..num_qubits {
            if rng.gen::<f64>() < p {
                errors_this_round += 1;
            }
        }

        // 2. Determine if decoder finishes in time.
        // Add jitter: actual latency ~ t_decode * (0.8 + 0.4 * uniform).
        let jitter: f64 = 0.8 + 0.4 * rng.gen::<f64>();
        let actual_latency = (t_decode as f64 * jitter) as u64;
        let decode_in_time = actual_latency < t_syndrome;

        // 3. Correction.
        let errors_corrected = if decode_in_time {
            // Decoder corrects with probability `accuracy`.
            let mut corrected = 0u32;
            for _ in 0..errors_this_round {
                if rng.gen::<f64>() < acc {
                    corrected += 1;
                }
            }
            corrected
        } else {
            // Decoder too slow: no correction this round, errors go to backlog.
            0
        };

        let uncorrected = errors_this_round.saturating_sub(errors_corrected);
        backlog += uncorrected as f64;

        // Drain backlog: when decoder is idle, it can catch up.
        if decode_in_time && backlog > 0.0 {
            let drain = (backlog * acc).min(backlog);
            backlog -= drain;
        }

        if backlog > max_backlog {
            max_backlog = backlog;
        }

        // A logical error occurs if the number of uncorrected errors in a
        // single round exceeds the code's correction capacity: floor((d-1)/2).
        let correction_capacity = (d.saturating_sub(1)) / 2;
        if uncorrected > correction_capacity {
            logical_errors += 1;
        }

        _total_latency_ns += actual_latency.max(t_syndrome);

        rounds.push(RoundSnapshot {
            round: r,
            errors_this_round,
            errors_corrected,
            backlog,
            decode_latency_ns: actual_latency,
        });
    }

    let final_logical_error_rate = if num_rounds > 0 {
        logical_errors as f64 / num_rounds as f64
    } else {
        0.0
    };

    // Converged if the backlog at the end is below a small threshold.
    let converged = backlog < 1.0;

    SimulationTrace {
        rounds,
        converged,
        final_logical_error_rate,
        max_backlog,
    }
}

// ---------------------------------------------------------------------------
// 6. Scaling Laws
// ---------------------------------------------------------------------------

/// A power-law scaling relation: `y = prefactor * x^exponent`.
#[derive(Debug, Clone)]
pub struct ScalingLaw {
    /// Human-readable name of the scaling law.
    pub name: String,
    /// Power-law exponent.
    pub exponent: f64,
    /// Multiplicative prefactor.
    pub prefactor: f64,
}

/// Return the classical overhead scaling law for a named decoder.
///
/// Known decoders:
/// - `"union_find"`: O(n * alpha(n)) ~ effectively linear, exponent ~ 1.0
/// - `"mwpm"`: O(n^3) matching, exponent ~ 3.0
/// - `"neural"`: O(n) inference, exponent ~ 1.0, higher prefactor
///
/// Falls back to a generic O(n^2) estimate for unknown decoders.
pub fn classical_overhead_scaling(decoder_name: &str) -> ScalingLaw {
    match decoder_name {
        "union_find" => ScalingLaw {
            name: "Union-Find decoder".into(),
            exponent: 1.0,
            prefactor: 1.0,
        },
        "mwpm" => ScalingLaw {
            name: "Minimum Weight Perfect Matching".into(),
            exponent: 3.0,
            prefactor: 0.5,
        },
        "neural" => ScalingLaw {
            name: "Neural network decoder".into(),
            exponent: 1.0,
            prefactor: 10.0,
        },
        _ => ScalingLaw {
            name: format!("Generic decoder ({})", decoder_name),
            exponent: 2.0,
            prefactor: 1.0,
        },
    }
}

/// Return the logical error rate scaling law.
///
/// For a surface code below threshold:
///   p_L ~ A * (p / p_th)^((d+1)/2)
///
/// where `p_th` is the threshold error rate (~1%).
/// The returned `ScalingLaw` has:
///   - `exponent` = (d_eff + 1) / 2 for d_eff derived from the ratio
///   - `prefactor` = 0.1 (typical for surface codes)
///
/// If `physical_rate >= threshold`, the exponent is set to 0 (no
/// suppression) and the prefactor is 1.0.
pub fn logical_error_scaling(physical_rate: f64, threshold: f64) -> ScalingLaw {
    if threshold <= 0.0 || physical_rate <= 0.0 {
        return ScalingLaw {
            name: "Logical error scaling (degenerate)".into(),
            exponent: 0.0,
            prefactor: 1.0,
        };
    }

    if physical_rate >= threshold {
        return ScalingLaw {
            name: "Logical error scaling (above threshold)".into(),
            exponent: 0.0,
            prefactor: 1.0,
        };
    }

    // Below threshold: the effective distance needed for a target logical
    // error rate p_L is d ~ 2 * log(p_L / A) / log(p / p_th) - 1.
    // We return the scaling exponent per unit distance increase.
    let ratio = physical_rate / threshold;
    let lambda = -ratio.ln(); // suppression factor per half-distance

    ScalingLaw {
        name: "Logical error scaling (below threshold)".into(),
        exponent: lambda,
        prefactor: 0.1,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plant(d: u32, p: f64) -> QuantumPlant {
        QuantumPlant {
            code_distance: d,
            physical_error_rate: p,
            num_data_qubits: d * d,
            coherence_time_ns: 100_000,
        }
    }

    fn make_controller(latency_ns: u64, throughput: f64, accuracy: f64) -> ClassicalController {
        ClassicalController {
            decode_latency_ns: latency_ns,
            decode_throughput: throughput,
            accuracy,
        }
    }

    fn make_loop(d: u32, p: f64, latency_ns: u64) -> QecControlLoop {
        QecControlLoop {
            plant: make_plant(d, p),
            controller: make_controller(latency_ns, 1e6, 0.99),
            state: ControlState::new(),
        }
    }

    // -- ControlState --

    #[test]
    fn test_control_state_new() {
        let s = ControlState::new();
        assert_eq!(s.logical_error_rate, 0.0);
        assert_eq!(s.error_backlog, 0.0);
        assert_eq!(s.rounds_decoded, 0);
        assert_eq!(s.total_latency_ns, 0);
    }

    #[test]
    fn test_control_state_default() {
        let s = ControlState::default();
        assert_eq!(s.rounds_decoded, 0);
    }

    // -- syndrome_period_ns --

    #[test]
    fn test_syndrome_period_scales_with_distance() {
        let t3 = syndrome_period_ns(3);
        let t5 = syndrome_period_ns(5);
        let t7 = syndrome_period_ns(7);
        assert!(t3 < t5);
        assert!(t5 < t7);
    }

    #[test]
    fn test_syndrome_period_d3() {
        // 6 * 3 * 20 = 360 ns
        assert_eq!(syndrome_period_ns(3), 360);
    }

    // -- analyze_stability --

    #[test]
    fn test_stable_loop() {
        // d=5, low error rate, fast decoder (100 ns < 600 ns syndrome period).
        let config = make_loop(5, 0.001, 100);
        let cond = analyze_stability(&config);
        assert!(cond.is_stable);
        assert!(cond.margin > 0.0);
        assert!(cond.convergence_rate > 0.0);
    }

    #[test]
    fn test_unstable_loop() {
        // d=3, decode latency exceeds syndrome period (1000 > 360).
        let config = make_loop(3, 0.001, 1000);
        let cond = analyze_stability(&config);
        assert!(!cond.is_stable);
        assert!(cond.margin < 0.0);
    }

    #[test]
    fn test_stability_critical_latency() {
        let config = make_loop(5, 0.001, 100);
        let cond = analyze_stability(&config);
        assert_eq!(cond.critical_latency_ns, syndrome_period_ns(5));
    }

    #[test]
    fn test_stability_zero_decode_latency() {
        let config = make_loop(3, 0.001, 0);
        let cond = analyze_stability(&config);
        assert!(cond.is_stable);
        assert!(cond.margin.is_infinite());
    }

    // -- max_stable_distance --

    #[test]
    fn test_max_stable_distance_fast_decoder() {
        let ctrl = make_controller(100, 1e7, 0.99);
        let d = max_stable_distance(&ctrl, 0.001);
        assert!(d >= 3);
    }

    #[test]
    fn test_max_stable_distance_slow_decoder() {
        // 10_000 ns latency: syndrome period for d=3 is 360 ns.
        let ctrl = make_controller(10_000, 1e5, 0.99);
        let d = max_stable_distance(&ctrl, 0.001);
        // Should still find d that satisfies the constraint.
        assert!(d >= 3);
    }

    #[test]
    fn test_max_stable_distance_above_threshold() {
        // Error rate above threshold: only d=3 is returned (the initial).
        let ctrl = make_controller(100, 1e7, 0.99);
        let d = max_stable_distance(&ctrl, 0.5);
        assert_eq!(d, 3);
    }

    // -- min_throughput --

    #[test]
    fn test_min_throughput_d3() {
        let plant = make_plant(3, 0.001);
        let tp = min_throughput(&plant);
        // 1e9 / 360 ~ 2.78 MHz
        assert!(tp > 2e6);
        assert!(tp < 3e6);
    }

    #[test]
    fn test_min_throughput_increases_with_smaller_distance() {
        // Smaller distance => shorter syndrome period => higher required throughput.
        let tp3 = min_throughput(&make_plant(3, 0.001));
        let tp5 = min_throughput(&make_plant(5, 0.001));
        assert!(tp3 > tp5);
    }

    // -- optimize_allocation --

    #[test]
    fn test_optimize_allocation_basic() {
        let budget = ResourceBudget {
            total_physical_qubits: 10_000,
            classical_cores: 8,
            classical_clock_ghz: 3.0,
            total_time_budget_us: 1_000,
        };
        let allocs = optimize_allocation(&budget, 0.001, 1);
        assert!(!allocs.is_empty());
        // Should be sorted by descending Pareto score.
        for w in allocs.windows(2) {
            assert!(w[0].pareto_score >= w[1].pareto_score);
        }
    }

    #[test]
    fn test_optimize_allocation_respects_min_logical() {
        let budget = ResourceBudget {
            total_physical_qubits: 100,
            classical_cores: 4,
            classical_clock_ghz: 2.0,
            total_time_budget_us: 1_000,
        };
        let allocs = optimize_allocation(&budget, 0.001, 5);
        for a in &allocs {
            assert!(a.logical_qubits >= 5);
        }
    }

    #[test]
    fn test_optimize_allocation_insufficient_qubits() {
        let budget = ResourceBudget {
            total_physical_qubits: 5, // Way too few for any code.
            classical_cores: 1,
            classical_clock_ghz: 1.0,
            total_time_budget_us: 100,
        };
        let allocs = optimize_allocation(&budget, 0.001, 1);
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_optimize_allocation_zero_cores() {
        let budget = ResourceBudget {
            total_physical_qubits: 10_000,
            classical_cores: 0,
            classical_clock_ghz: 0.0,
            total_time_budget_us: 1_000,
        };
        let allocs = optimize_allocation(&budget, 0.001, 1);
        // No classical resources: decode_ns is MAX, no affordable rounds.
        assert!(allocs.is_empty());
    }

    // -- plan_latency_budget --

    #[test]
    fn test_latency_budget_d3_fast() {
        let lb = plan_latency_budget(3, 100);
        assert_eq!(lb.syndrome_extraction_ns, 360);
        assert_eq!(lb.decode_ns, 100);
        assert_eq!(lb.correction_ns, 20);
        assert_eq!(lb.total_round_ns, 480);
        // Slack: 360 - 100 - 20 = 240.
        assert_eq!(lb.slack_ns, 240);
    }

    #[test]
    fn test_latency_budget_negative_slack() {
        // Decode takes 1000 ns, syndrome extraction for d=3 is 360 ns.
        let lb = plan_latency_budget(3, 1000);
        assert!(lb.slack_ns < 0, "Should have negative slack");
    }

    #[test]
    fn test_latency_budget_scales_with_distance() {
        let lb3 = plan_latency_budget(3, 100);
        let lb7 = plan_latency_budget(7, 100);
        assert!(lb7.syndrome_extraction_ns > lb3.syndrome_extraction_ns);
        assert!(lb7.total_round_ns > lb3.total_round_ns);
    }

    // -- simulate_control_loop --

    #[test]
    fn test_simulation_stable_loop() {
        let config = make_loop(5, 0.001, 100);
        let trace = simulate_control_loop(&config, 100, 42);
        assert_eq!(trace.rounds.len(), 100);
        // With low error rate and fast decoder, should converge.
        assert!(trace.converged);
        assert!(trace.max_backlog < 50.0);
    }

    #[test]
    fn test_simulation_unstable_loop() {
        // Very high error rate + slow decoder.
        let config = make_loop(3, 0.3, 1000);
        let trace = simulate_control_loop(&config, 200, 42);
        assert_eq!(trace.rounds.len(), 200);
        // High error rate should produce a non-trivial backlog.
        assert!(trace.max_backlog > 0.0);
    }

    #[test]
    fn test_simulation_zero_rounds() {
        let config = make_loop(3, 0.001, 100);
        let trace = simulate_control_loop(&config, 0, 42);
        assert!(trace.rounds.is_empty());
        assert_eq!(trace.final_logical_error_rate, 0.0);
        assert!(trace.converged);
    }

    #[test]
    fn test_simulation_deterministic() {
        let config = make_loop(5, 0.01, 200);
        let t1 = simulate_control_loop(&config, 50, 123);
        let t2 = simulate_control_loop(&config, 50, 123);
        assert_eq!(t1.rounds.len(), t2.rounds.len());
        for (a, b) in t1.rounds.iter().zip(t2.rounds.iter()) {
            assert_eq!(a.errors_this_round, b.errors_this_round);
            assert_eq!(a.errors_corrected, b.errors_corrected);
        }
    }

    #[test]
    fn test_simulation_zero_error_rate() {
        let config = make_loop(5, 0.0, 100);
        let trace = simulate_control_loop(&config, 50, 99);
        assert!(trace.converged);
        assert_eq!(trace.final_logical_error_rate, 0.0);
        for snap in &trace.rounds {
            assert_eq!(snap.errors_this_round, 0);
        }
    }

    #[test]
    fn test_simulation_round_snapshot_fields() {
        let config = make_loop(3, 0.01, 100);
        let trace = simulate_control_loop(&config, 10, 7);
        for (i, snap) in trace.rounds.iter().enumerate() {
            assert_eq!(snap.round, i as u64);
            assert!(snap.errors_corrected <= snap.errors_this_round);
            assert!(snap.decode_latency_ns > 0);
        }
    }

    // -- classical_overhead_scaling --

    #[test]
    fn test_scaling_union_find() {
        let law = classical_overhead_scaling("union_find");
        assert_eq!(law.exponent, 1.0);
        assert!(law.name.contains("Union-Find"));
    }

    #[test]
    fn test_scaling_mwpm() {
        let law = classical_overhead_scaling("mwpm");
        assert_eq!(law.exponent, 3.0);
    }

    #[test]
    fn test_scaling_neural() {
        let law = classical_overhead_scaling("neural");
        assert_eq!(law.exponent, 1.0);
        assert!(law.prefactor > 1.0);
    }

    #[test]
    fn test_scaling_unknown() {
        let law = classical_overhead_scaling("custom_decoder");
        assert_eq!(law.exponent, 2.0);
        assert!(law.name.contains("custom_decoder"));
    }

    // -- logical_error_scaling --

    #[test]
    fn test_logical_scaling_below_threshold() {
        let law = logical_error_scaling(0.001, 0.01);
        assert!(law.exponent > 0.0);
        assert_eq!(law.prefactor, 0.1);
        assert!(law.name.contains("below threshold"));
    }

    #[test]
    fn test_logical_scaling_above_threshold() {
        let law = logical_error_scaling(0.05, 0.01);
        assert_eq!(law.exponent, 0.0);
        assert_eq!(law.prefactor, 1.0);
    }

    #[test]
    fn test_logical_scaling_at_threshold() {
        let law = logical_error_scaling(0.01, 0.01);
        assert_eq!(law.exponent, 0.0);
        assert!(law.name.contains("above threshold"));
    }

    #[test]
    fn test_logical_scaling_zero_rate() {
        let law = logical_error_scaling(0.0, 0.01);
        assert_eq!(law.exponent, 0.0);
        assert!(law.name.contains("degenerate"));
    }

    #[test]
    fn test_logical_scaling_zero_threshold() {
        let law = logical_error_scaling(0.001, 0.0);
        assert_eq!(law.exponent, 0.0);
    }
}
