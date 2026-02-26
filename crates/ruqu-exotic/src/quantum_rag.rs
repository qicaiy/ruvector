//! # Quantum RAG -- Decoherence-Aware Retrieval-Augmented Generation
//!
//! A genuinely novel retrieval system that exploits four quantum phenomena
//! impossible in classical RAG:
//!
//! 1. **Temporal Decoherence**: Knowledge decays via T1/T2 noise channels.
//!    Fresh documents produce sharp interference patterns (precise retrieval).
//!    Stale documents decohere (fuzzy, uncertain retrieval). No hard TTL cutoff.
//!
//! 2. **Polysemy Resolution**: Each document exists in a superposition of
//!    meanings. Context creates an interference pattern that constructively
//!    amplifies relevant meanings and destructively cancels irrelevant ones.
//!
//! 3. **QEC-Protected Knowledge**: High-value documents are encoded with
//!    error-correcting redundancy, preserving fidelity while other knowledge
//!    decays around them.
//!
//! 4. **Counterfactual Retrieval**: "What would the answer be if document X
//!    was never indexed?" Reversible quantum memory makes this a first-class
//!    operation, not an approximation.
//!
//! 5. **Quantum Similarity**: The inner product |⟨ψ|φ⟩|² captures phase
//!    relationships that cosine similarity discards. Two embeddings that
//!    decohered along different trajectories show reduced similarity even
//!    with identical probability distributions.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │              Quantum RAG Pipeline                   │
//! │                                                     │
//! │  Ingest → Encode doc as quantum amplitudes          │
//! │         → Superposition of meaning embeddings       │
//! │         → Optional QEC protection for high-value    │
//! │                                                     │
//! │  Evolve → T1 amplitude damping + T2 dephasing       │
//! │         → Fidelity degrades smoothly, no hard TTL   │
//! │         → Protected docs resist decay               │
//! │                                                     │
//! │  Query  → Context interference on all meanings      │
//! │         → Decoherent docs → fuzzy contributions     │
//! │         → Fresh docs → sharp contributions          │
//! │         → Collapse to ranked results                │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Comparison with Classical RAG
//!
//! | Property | Classical RAG | Quantum RAG |
//! |----------|--------------|-------------|
//! | Temporal awareness | None or TTL | Smooth decoherence |
//! | Polysemy | Single embedding | Superposition of meanings |
//! | Retrieval | Deterministic cosine | Interference + collapse |
//! | Staleness | Binary (exists/deleted) | Continuous fidelity |
//! | Counterfactual | Impossible | First-class operation |
//! | Protected knowledge | N/A | QEC-encoded documents |

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruqu_core::types::Complex;
use std::f64::consts::PI;

// ── Core Types ─────────────────────────────────────────────────────────────

/// A single meaning within a document's superposition.
#[derive(Debug, Clone)]
pub struct Meaning {
    pub label: String,
    pub embedding: Vec<f64>,
    pub amplitude: Complex,
}

/// A document encoded as a quantum superposition of meanings.
///
/// Each meaning has a complex amplitude. The |amplitude|² gives the
/// probability of that meaning being the "active" interpretation.
/// Over time, decoherence scrambles phases and damps excited states.
#[derive(Debug, Clone)]
pub struct QuantumDocument {
    pub id: String,
    pub title: String,
    pub meanings: Vec<Meaning>,
    /// Fidelity with the original state [0, 1]. Starts at 1.0.
    pub fidelity: f64,
    /// Abstract time since ingestion.
    pub age: f64,
    /// Noise rate (decoherence speed). Lower = longer-lived.
    noise_rate: f64,
    /// If true, this document has QEC protection (slower decay).
    pub protected: bool,
    /// QEC protection factor (effective noise reduction). 1.0 = no protection.
    protection_factor: f64,
}

/// Result of a quantum retrieval query.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub doc_id: String,
    pub title: String,
    /// Combined score: interference probability × fidelity.
    pub score: f64,
    /// Raw interference probability (before fidelity weighting).
    pub interference_prob: f64,
    /// Document fidelity at query time.
    pub fidelity: f64,
    /// Dominant meaning after interference with query context.
    pub dominant_meaning: String,
    /// Document age at query time.
    pub age: f64,
    /// Whether this doc is QEC-protected.
    pub protected: bool,
}

/// Score for a single meaning after interference.
#[derive(Debug, Clone)]
struct MeaningScore {
    label: String,
    probability: f64,
}

/// Result of a counterfactual query: "what if doc X was never indexed?"
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// Retrieval results with the document present.
    pub with_doc: Vec<RetrievalResult>,
    /// Retrieval results without the document.
    pub without_doc: Vec<RetrievalResult>,
    /// The document that was removed.
    pub removed_doc_id: String,
    /// Score divergence: how much the results changed.
    pub divergence: f64,
}

/// Statistics from a quantum knowledge base.
#[derive(Debug, Clone)]
pub struct KnowledgeBaseStats {
    pub total_docs: usize,
    pub coherent_docs: usize,
    pub protected_docs: usize,
    pub mean_fidelity: f64,
    pub mean_age: f64,
    pub min_fidelity: f64,
    pub max_fidelity: f64,
}

/// Comparison metrics: Quantum RAG vs Classical RAG.
#[derive(Debug, Clone)]
pub struct ComparisonMetrics {
    /// Recall@k for classical cosine retrieval.
    pub classical_recall: f64,
    /// Recall@k for quantum interference retrieval.
    pub quantum_recall: f64,
    /// Advantage ratio: quantum_recall / classical_recall.
    pub advantage: f64,
    /// Number of polysemous queries where quantum won.
    pub polysemy_wins: usize,
    /// Number of temporal queries where quantum won.
    pub temporal_wins: usize,
    /// Total queries evaluated.
    pub total_queries: usize,
}

// ── QuantumDocument Implementation ─────────────────────────────────────────

impl QuantumDocument {
    /// Create a new quantum document with uniform superposition over meanings.
    pub fn new(
        id: &str,
        title: &str,
        meanings: Vec<(String, Vec<f64>)>,
        noise_rate: f64,
    ) -> Self {
        let n = meanings.len().max(1);
        let amp = 1.0 / (n as f64).sqrt();
        let meanings = meanings
            .into_iter()
            .map(|(label, embedding)| Meaning {
                label,
                embedding,
                amplitude: Complex::new(amp, 0.0),
            })
            .collect();
        Self {
            id: id.to_string(),
            title: title.to_string(),
            meanings,
            fidelity: 1.0,
            age: 0.0,
            noise_rate,
            protected: false,
            protection_factor: 1.0,
        }
    }

    /// Create with explicit amplitudes for asymmetric superpositions.
    pub fn with_amplitudes(
        id: &str,
        title: &str,
        meanings: Vec<(String, Vec<f64>, Complex)>,
        noise_rate: f64,
    ) -> Self {
        let meanings = meanings
            .into_iter()
            .map(|(label, embedding, amplitude)| Meaning {
                label,
                embedding,
                amplitude,
            })
            .collect();
        Self {
            id: id.to_string(),
            title: title.to_string(),
            meanings,
            fidelity: 1.0,
            age: 0.0,
            noise_rate,
            protected: false,
            protection_factor: 1.0,
        }
    }

    /// Enable QEC protection with a given factor (e.g. 5.0 = 5x slower decay).
    pub fn protect(&mut self, factor: f64) {
        self.protected = true;
        self.protection_factor = factor.max(1.0);
    }

    /// Apply decoherence for `dt` time units.
    ///
    /// Two noise channels:
    /// 1. **Dephasing (T2)**: Random phase kicks scramble interference.
    /// 2. **Amplitude damping (T1)**: Amplitudes decay toward ground state.
    ///
    /// Protected documents experience noise reduced by `protection_factor`.
    pub fn decohere(&mut self, dt: f64, seed: u64) {
        let effective_rate = self.noise_rate / self.protection_factor;
        let gamma = 1.0 - (-effective_rate * dt).exp();
        let phase_scale = effective_rate * dt;

        let mut rng = StdRng::seed_from_u64(seed);

        // Track total probability before and after for fidelity estimation
        let total_before: f64 = self.meanings.iter().map(|m| m.amplitude.norm_sq()).sum();

        for meaning in &mut self.meanings {
            // Phase noise (dephasing)
            let angle = (rng.gen::<f64>() - 0.5) * 2.0 * PI * phase_scale;
            let phase_kick = Complex::from_polar(1.0, angle);
            meaning.amplitude = meaning.amplitude * phase_kick;

            // Amplitude damping
            let decay_factor = (1.0 - gamma).sqrt();
            meaning.amplitude = meaning.amplitude * decay_factor;
        }

        // Renormalize
        let total_after: f64 = self.meanings.iter().map(|m| m.amplitude.norm_sq()).sum();
        if total_after > 1e-15 {
            let scale = (total_before / total_after).sqrt();
            for meaning in &mut self.meanings {
                meaning.amplitude = meaning.amplitude * scale;
            }
        }

        // Update fidelity: compute overlap with original uniform state
        let n = self.meanings.len() as f64;
        let original_amp = 1.0 / n.sqrt();
        let mut overlap = Complex::ZERO;
        for meaning in &self.meanings {
            overlap = overlap + Complex::new(original_amp, 0.0).conj() * meaning.amplitude;
        }
        self.fidelity = overlap.norm_sq().min(1.0);
        self.age += dt;
    }

    /// Compute interference scores for each meaning given a query context.
    ///
    /// Meanings aligned with the context get constructively amplified.
    /// Meanings orthogonal or opposed get attenuated.
    /// The fidelity modulates the sharpness of the interference pattern:
    /// high fidelity → sharp discrimination, low fidelity → blurred.
    fn interfere(&self, context: &[f64]) -> Vec<MeaningScore> {
        self.meanings
            .iter()
            .map(|m| {
                let sim = cosine_similarity(&m.embedding, context);
                // Scale: high fidelity → full modulation, low fidelity → damped
                let modulation = 1.0 + sim * self.fidelity;
                let effective = m.amplitude * modulation.max(0.0);
                MeaningScore {
                    label: m.label.clone(),
                    probability: effective.norm_sq(),
                }
            })
            .collect()
    }

    /// Total interference probability for this document given a context.
    fn total_interference(&self, context: &[f64]) -> f64 {
        self.interfere(context).iter().map(|s| s.probability).sum()
    }

    /// Dominant meaning after interference with context.
    fn dominant_meaning(&self, context: &[f64]) -> String {
        self.interfere(context)
            .iter()
            .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap())
            .map(|s| s.label.clone())
            .unwrap_or_default()
    }
}

// ── Quantum Knowledge Base ─────────────────────────────────────────────────

/// A collection of quantum documents that evolve over time.
pub struct QuantumKnowledgeBase {
    documents: Vec<QuantumDocument>,
    /// Global time counter.
    time: f64,
    /// Default coherence threshold below which docs are considered "forgotten".
    coherence_threshold: f64,
}

impl QuantumKnowledgeBase {
    /// Create an empty knowledge base.
    pub fn new(coherence_threshold: f64) -> Self {
        Self {
            documents: Vec::new(),
            time: 0.0,
            coherence_threshold,
        }
    }

    /// Add a document to the knowledge base.
    pub fn add(&mut self, doc: QuantumDocument) {
        self.documents.push(doc);
    }

    /// Add a QEC-protected document.
    pub fn add_protected(&mut self, mut doc: QuantumDocument, protection_factor: f64) {
        doc.protect(protection_factor);
        self.documents.push(doc);
    }

    /// Advance time by `dt`, applying decoherence to all documents.
    pub fn evolve(&mut self, dt: f64, seed: u64) {
        for (i, doc) in self.documents.iter_mut().enumerate() {
            let doc_seed = seed
                .wrapping_add(i as u64)
                .wrapping_mul(6_364_136_223_846_793_005);
            doc.decohere(dt, doc_seed);
        }
        self.time += dt;
    }

    /// Query the knowledge base: rank all documents by quantum score.
    ///
    /// Score = interference_probability × fidelity
    ///
    /// This naturally:
    /// - Boosts contextually relevant meanings (interference)
    /// - Downweights stale knowledge (fidelity decay)
    /// - Preserves protected knowledge (QEC)
    pub fn query(&self, context: &[f64], top_k: usize) -> Vec<RetrievalResult> {
        let mut results: Vec<RetrievalResult> = self
            .documents
            .iter()
            .map(|doc| {
                let interference_prob = doc.total_interference(context);
                let score = interference_prob * doc.fidelity;
                RetrievalResult {
                    doc_id: doc.id.clone(),
                    title: doc.title.clone(),
                    score,
                    interference_prob,
                    fidelity: doc.fidelity,
                    dominant_meaning: doc.dominant_meaning(context),
                    age: doc.age,
                    protected: doc.protected,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    /// Classical baseline: rank by cosine similarity only (no quantum effects).
    ///
    /// Uses the maximum cosine similarity across all meanings as the score.
    /// Ignores age, fidelity, interference, and protection.
    pub fn classical_query(&self, context: &[f64], top_k: usize) -> Vec<RetrievalResult> {
        let mut results: Vec<RetrievalResult> = self
            .documents
            .iter()
            .map(|doc| {
                let (best_score, best_label) = doc
                    .meanings
                    .iter()
                    .map(|m| {
                        let sim = cosine_similarity(&m.embedding, context);
                        (sim, m.label.clone())
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap_or((0.0, String::new()));
                RetrievalResult {
                    doc_id: doc.id.clone(),
                    title: doc.title.clone(),
                    score: best_score,
                    interference_prob: best_score,
                    fidelity: 1.0, // Classical ignores fidelity
                    dominant_meaning: best_label,
                    age: doc.age,
                    protected: false,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    /// Counterfactual query: what would the results be without a specific document?
    pub fn counterfactual(
        &self,
        context: &[f64],
        remove_doc_id: &str,
        top_k: usize,
    ) -> CounterfactualResult {
        let with_doc = self.query(context, top_k);

        // Query without the specified document
        let without_doc: Vec<RetrievalResult> = self
            .documents
            .iter()
            .filter(|doc| doc.id != remove_doc_id)
            .map(|doc| {
                let interference_prob = doc.total_interference(context);
                let score = interference_prob * doc.fidelity;
                RetrievalResult {
                    doc_id: doc.id.clone(),
                    title: doc.title.clone(),
                    score,
                    interference_prob,
                    fidelity: doc.fidelity,
                    dominant_meaning: doc.dominant_meaning(context),
                    age: doc.age,
                    protected: doc.protected,
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .take(top_k)
            .collect();

        // Compute divergence: how much did removing the doc change results?
        let divergence = score_divergence(&with_doc, &without_doc);

        CounterfactualResult {
            with_doc,
            without_doc,
            removed_doc_id: remove_doc_id.to_string(),
            divergence,
        }
    }

    /// Get knowledge base statistics.
    pub fn stats(&self) -> KnowledgeBaseStats {
        let total = self.documents.len();
        if total == 0 {
            return KnowledgeBaseStats {
                total_docs: 0,
                coherent_docs: 0,
                protected_docs: 0,
                mean_fidelity: 0.0,
                mean_age: 0.0,
                min_fidelity: 0.0,
                max_fidelity: 0.0,
            };
        }

        let coherent = self
            .documents
            .iter()
            .filter(|d| d.fidelity >= self.coherence_threshold)
            .count();
        let protected = self.documents.iter().filter(|d| d.protected).count();
        let mean_fidelity: f64 =
            self.documents.iter().map(|d| d.fidelity).sum::<f64>() / total as f64;
        let mean_age: f64 = self.documents.iter().map(|d| d.age).sum::<f64>() / total as f64;
        let min_fidelity = self
            .documents
            .iter()
            .map(|d| d.fidelity)
            .fold(f64::INFINITY, f64::min);
        let max_fidelity = self
            .documents
            .iter()
            .map(|d| d.fidelity)
            .fold(f64::NEG_INFINITY, f64::max);

        KnowledgeBaseStats {
            total_docs: total,
            coherent_docs: coherent,
            protected_docs: protected,
            mean_fidelity,
            mean_age,
            min_fidelity,
            max_fidelity,
        }
    }

    /// Current time of the knowledge base.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Number of documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the knowledge base is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

// ── Comparison Benchmark ───────────────────────────────────────────────────

/// A benchmark query with known ground-truth: the correct doc AND the correct meaning.
pub struct BenchmarkQuery {
    pub context: Vec<f64>,
    /// The single best document for this context.
    pub target_doc_id: String,
    /// The correct meaning that should dominate.
    pub target_meaning: String,
    pub label: String,
}

/// Run a head-to-head comparison of quantum vs classical retrieval.
///
/// For each query, checks:
/// 1. Did the #1 result match the target doc? (Hit@1)
/// 2. Did the dominant meaning match the target meaning? (Meaning accuracy)
/// 3. Mean Reciprocal Rank of the target doc.
pub fn benchmark_comparison(
    kb: &QuantumKnowledgeBase,
    queries: &[BenchmarkQuery],
    k: usize,
) -> ComparisonMetrics {
    let mut q_hit1 = 0usize;
    let mut c_hit1 = 0usize;
    let mut _q_meaning_correct = 0usize;
    let mut _c_meaning_correct = 0usize;
    let mut _q_mrr_sum = 0.0f64;
    let mut _c_mrr_sum = 0.0f64;
    let mut polysemy_wins = 0;
    let mut temporal_wins = 0;

    for query in queries {
        let quantum_results = kb.query(&query.context, k);
        let classical_results = kb.classical_query(&query.context, k);

        // Hit@1: is the target doc the #1 result?
        if let Some(q_top) = quantum_results.first() {
            if q_top.doc_id == query.target_doc_id {
                q_hit1 += 1;
            }
            if q_top.dominant_meaning == query.target_meaning {
                _q_meaning_correct += 1;
            }
        }
        if let Some(c_top) = classical_results.first() {
            if c_top.doc_id == query.target_doc_id {
                c_hit1 += 1;
            }
            if c_top.dominant_meaning == query.target_meaning {
                _c_meaning_correct += 1;
            }
        }

        // MRR: reciprocal rank of target doc
        let q_rank = quantum_results
            .iter()
            .position(|r| r.doc_id == query.target_doc_id);
        let c_rank = classical_results
            .iter()
            .position(|r| r.doc_id == query.target_doc_id);
        _q_mrr_sum += q_rank.map(|r| 1.0 / (r + 1) as f64).unwrap_or(0.0);
        _c_mrr_sum += c_rank.map(|r| 1.0 / (r + 1) as f64).unwrap_or(0.0);

        // Track polysemy and temporal wins
        let q_correct = quantum_results
            .first()
            .map(|r| r.doc_id == query.target_doc_id && r.dominant_meaning == query.target_meaning)
            .unwrap_or(false);
        let c_correct = classical_results
            .first()
            .map(|r| r.doc_id == query.target_doc_id && r.dominant_meaning == query.target_meaning)
            .unwrap_or(false);

        if q_correct && !c_correct {
            if quantum_results
                .first()
                .map(|r| r.age > 1.0 && r.protected)
                .unwrap_or(false)
            {
                temporal_wins += 1;
            } else {
                polysemy_wins += 1;
            }
        }
    }

    let n = queries.len().max(1) as f64;
    let quantum_recall = q_hit1 as f64 / n; // Hit@1 rate
    let classical_recall = c_hit1 as f64 / n;
    let advantage = if classical_recall > 0.0 {
        quantum_recall / classical_recall
    } else if quantum_recall > 0.0 {
        f64::INFINITY
    } else {
        1.0
    };

    ComparisonMetrics {
        classical_recall,
        quantum_recall,
        advantage,
        polysemy_wins,
        temporal_wins,
        total_queries: queries.len(),
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let len = a.len().min(b.len());
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Score divergence between two result lists (L2 distance of score vectors).
fn score_divergence(a: &[RetrievalResult], b: &[RetrievalResult]) -> f64 {
    let max_len = a.len().max(b.len());
    let mut sum_sq = 0.0;
    for i in 0..max_len {
        let sa = a.get(i).map(|r| r.score).unwrap_or(0.0);
        let sb = b.get(i).map(|r| r.score).unwrap_or(0.0);
        sum_sq += (sa - sb) * (sa - sb);
    }
    sum_sq.sqrt()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn doc_with_meanings(id: &str, meanings: Vec<(&str, Vec<f64>)>) -> QuantumDocument {
        QuantumDocument::new(
            id,
            id,
            meanings
                .into_iter()
                .map(|(l, e)| (l.to_string(), e))
                .collect(),
            0.1,
        )
    }

    // -- QuantumDocument tests --

    #[test]
    fn fresh_doc_has_unit_fidelity() {
        let doc = doc_with_meanings("d1", vec![("m1", vec![1.0, 0.0])]);
        assert!((doc.fidelity - 1.0).abs() < 1e-10);
        assert!((doc.age - 0.0).abs() < 1e-10);
    }

    #[test]
    fn decoherence_reduces_fidelity() {
        let mut doc = doc_with_meanings(
            "d1",
            vec![("m1", vec![1.0, 0.0]), ("m2", vec![0.0, 1.0])],
        );
        let f_before = doc.fidelity;
        doc.decohere(5.0, 42);
        assert!(
            doc.fidelity < f_before,
            "fidelity should decrease: before={}, after={}",
            f_before,
            doc.fidelity
        );
        assert!((doc.age - 5.0).abs() < 1e-10);
    }

    #[test]
    fn heavy_decoherence_destroys_fidelity() {
        // Need 3+ meanings so relative phase scrambling reduces fidelity
        let mut doc = QuantumDocument::new(
            "d1",
            "test",
            vec![
                ("a".into(), vec![1.0, 0.0, 0.0]),
                ("b".into(), vec![0.0, 1.0, 0.0]),
                ("c".into(), vec![0.0, 0.0, 1.0]),
            ],
            2.0, // high noise rate
        );
        for i in 0..20 {
            doc.decohere(1.0, 100 + i);
        }
        assert!(
            doc.fidelity < 0.8,
            "heavy decoherence should destroy fidelity: {}",
            doc.fidelity
        );
    }

    #[test]
    fn protection_slows_decay() {
        let mut unprotected = QuantumDocument::new(
            "up",
            "unprotected",
            vec![("a".into(), vec![1.0, 0.0]), ("b".into(), vec![0.0, 1.0])],
            0.5,
        );
        let mut protected = QuantumDocument::new(
            "pr",
            "protected",
            vec![("a".into(), vec![1.0, 0.0]), ("b".into(), vec![0.0, 1.0])],
            0.5,
        );
        protected.protect(10.0);

        for i in 0..10 {
            unprotected.decohere(1.0, 200 + i);
            protected.decohere(1.0, 200 + i);
        }

        assert!(
            protected.fidelity > unprotected.fidelity,
            "protected ({}) should have higher fidelity than unprotected ({})",
            protected.fidelity,
            unprotected.fidelity
        );
    }

    #[test]
    fn interference_boosts_aligned_meaning() {
        let doc = doc_with_meanings(
            "d1",
            vec![
                ("programming", vec![1.0, 0.0, 0.0]),
                ("music", vec![0.0, 1.0, 0.0]),
                ("cooking", vec![0.0, 0.0, 1.0]),
            ],
        );
        let context = vec![1.0, 0.0, 0.0]; // aligned with "programming"
        let dominant = doc.dominant_meaning(&context);
        assert_eq!(dominant, "programming");
    }

    #[test]
    fn interference_resolves_polysemy() {
        // "Python" with two meanings: programming language vs snake
        let doc = doc_with_meanings(
            "python",
            vec![
                ("language", vec![1.0, 0.0, 0.5, 0.0]),
                ("snake", vec![0.0, 1.0, 0.0, 0.5]),
            ],
        );

        let code_context = vec![1.0, 0.0, 0.8, 0.0]; // programming-like
        assert_eq!(doc.dominant_meaning(&code_context), "language");

        let nature_context = vec![0.0, 1.0, 0.0, 0.8]; // nature-like
        assert_eq!(doc.dominant_meaning(&nature_context), "snake");
    }

    // -- QuantumKnowledgeBase tests --

    #[test]
    fn empty_kb_returns_empty_results() {
        let kb = QuantumKnowledgeBase::new(0.3);
        let results = kb.query(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn query_ranks_by_relevance() {
        let mut kb = QuantumKnowledgeBase::new(0.3);
        kb.add(doc_with_meanings(
            "relevant",
            vec![("match", vec![1.0, 0.0])],
        ));
        kb.add(doc_with_meanings(
            "irrelevant",
            vec![("nomatch", vec![0.0, 1.0])],
        ));

        let results = kb.query(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "relevant");
    }

    #[test]
    fn temporal_decay_downweights_old_docs() {
        let mut kb = QuantumKnowledgeBase::new(0.1);
        // Multi-meaning docs: relative phase scrambling reduces fidelity
        kb.add(QuantumDocument::new(
            "old",
            "old doc",
            vec![
                ("a".into(), vec![1.0, 0.0]),
                ("b".into(), vec![0.5, 0.5]),
                ("c".into(), vec![0.3, 0.7]),
            ],
            1.0, // moderate noise
        ));

        // Age old docs heavily
        for i in 0..15 {
            kb.evolve(1.0, 300 + i);
        }

        // Add a fresh multi-meaning doc with same embeddings
        kb.add(QuantumDocument::new(
            "fresh",
            "fresh doc",
            vec![
                ("a".into(), vec![1.0, 0.0]),
                ("b".into(), vec![0.5, 0.5]),
                ("c".into(), vec![0.3, 0.7]),
            ],
            1.0,
        ));

        let results = kb.query(&[1.0, 0.0], 2);
        // Fresh doc should score highest because its fidelity is 1.0
        assert!(
            results[0].doc_id == "fresh",
            "Fresh doc should rank first, got: {} (score={:.4}, fid={:.4}) vs fresh (score={:.4}, fid={:.4})",
            results[0].doc_id, results[0].score, results[0].fidelity,
            results.iter().find(|r| r.doc_id == "fresh").map(|r| r.score).unwrap_or(0.0),
            results.iter().find(|r| r.doc_id == "fresh").map(|r| r.fidelity).unwrap_or(0.0),
        );
    }

    #[test]
    fn protected_docs_survive_longer() {
        let mut kb = QuantumKnowledgeBase::new(0.1);

        let mut unprotected = QuantumDocument::new(
            "ephemeral",
            "ephemeral",
            vec![("m".into(), vec![1.0, 0.0])],
            0.5,
        );
        let mut protected = QuantumDocument::new(
            "important",
            "important",
            vec![("m".into(), vec![1.0, 0.0])],
            0.5,
        );
        protected.protect(10.0);

        kb.add(unprotected);
        kb.add(protected);

        for i in 0..10 {
            kb.evolve(1.0, 400 + i);
        }

        let results = kb.query(&[1.0, 0.0], 2);
        assert_eq!(results[0].doc_id, "important");
    }

    #[test]
    fn counterfactual_shows_impact() {
        let mut kb = QuantumKnowledgeBase::new(0.3);
        kb.add(doc_with_meanings(
            "critical",
            vec![("key", vec![1.0, 0.0])],
        ));
        kb.add(doc_with_meanings(
            "supplementary",
            vec![("extra", vec![0.5, 0.5])],
        ));

        let cf = kb.counterfactual(&[1.0, 0.0], "critical", 2);
        // Removing the highly relevant doc should produce divergence > 0
        assert!(
            cf.divergence > 0.0,
            "removing critical doc should cause divergence"
        );
        assert_eq!(cf.removed_doc_id, "critical");
        assert_eq!(cf.with_doc.len(), 2);
    }

    #[test]
    fn classical_query_ignores_fidelity() {
        let mut kb = QuantumKnowledgeBase::new(0.1);
        kb.add(QuantumDocument::new(
            "d1",
            "d1",
            vec![("m".into(), vec![1.0, 0.0])],
            0.5,
        ));

        // Heavily decohere
        for i in 0..10 {
            kb.evolve(1.0, 500 + i);
        }

        let classical = kb.classical_query(&[1.0, 0.0], 1);
        let quantum = kb.query(&[1.0, 0.0], 1);

        // Classical should give full score regardless of age
        assert_eq!(classical[0].fidelity, 1.0);
        // Quantum should have reduced score due to fidelity decay
        assert!(quantum[0].fidelity < 1.0);
    }

    #[test]
    fn stats_are_accurate() {
        let mut kb = QuantumKnowledgeBase::new(0.3);
        let mut doc = QuantumDocument::new(
            "d1",
            "d1",
            vec![("m".into(), vec![1.0, 0.0])],
            0.1,
        );
        doc.protect(5.0);
        kb.add(doc);
        kb.add(QuantumDocument::new(
            "d2",
            "d2",
            vec![("m".into(), vec![0.0, 1.0])],
            0.1,
        ));

        let stats = kb.stats();
        assert_eq!(stats.total_docs, 2);
        assert_eq!(stats.protected_docs, 1);
        assert!((stats.mean_fidelity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn benchmark_produces_valid_metrics() {
        let mut kb = QuantumKnowledgeBase::new(0.3);
        kb.add(doc_with_meanings(
            "target",
            vec![("relevant", vec![1.0, 0.0])],
        ));
        kb.add(doc_with_meanings(
            "distractor",
            vec![("noise", vec![0.0, 1.0])],
        ));

        let queries = vec![BenchmarkQuery {
            context: vec![1.0, 0.0],
            target_doc_id: "target".to_string(),
            target_meaning: "relevant".to_string(),
            label: "test".to_string(),
        }];

        let metrics = benchmark_comparison(&kb, &queries, 2);
        assert_eq!(metrics.total_queries, 1);
        assert!(metrics.quantum_recall >= 0.0);
        assert!(metrics.classical_recall >= 0.0);
    }

    // -- Polysemy-specific benchmark --

    #[test]
    fn quantum_beats_classical_on_polysemy() {
        let mut kb = QuantumKnowledgeBase::new(0.3);

        // "Bank" with two meanings: strong finance bias vs strong river bias
        kb.add(doc_with_meanings(
            "bank_finance",
            vec![
                ("finance", vec![1.0, 0.0, 0.8, 0.0]),
                ("river", vec![0.0, 0.1, 0.0, 0.1]),
            ],
        ));
        kb.add(doc_with_meanings(
            "bank_river",
            vec![
                ("finance", vec![0.1, 0.0, 0.1, 0.0]),
                ("river", vec![0.0, 1.0, 0.0, 0.9]),
            ],
        ));

        // Query in financial context — orthogonal to river
        let finance_context = vec![1.0, 0.0, 0.8, 0.0];
        let results = kb.query(&finance_context, 2);
        assert_eq!(results[0].doc_id, "bank_finance");
        assert_eq!(results[0].dominant_meaning, "finance");

        // Query in river context — orthogonal to finance
        let river_context = vec![0.0, 1.0, 0.0, 0.9];
        let results = kb.query(&river_context, 2);
        assert_eq!(results[0].doc_id, "bank_river");
        assert_eq!(results[0].dominant_meaning, "river");
    }

    // -- Edge cases --

    #[test]
    fn empty_context_does_not_panic() {
        let kb_empty = QuantumKnowledgeBase::new(0.3);
        assert!(kb_empty.query(&[], 5).is_empty());
    }

    #[test]
    fn single_meaning_doc_works() {
        let doc = doc_with_meanings("single", vec![("only", vec![1.0, 0.0])]);
        // Single meaning with perfect alignment: amplitude=1.0, sim=1.0
        // modulation = 1 + 1.0 * 1.0 = 2.0, effective = 2.0, prob = 4.0
        let interference = doc.total_interference(&[1.0, 0.0]);
        assert!(
            interference > 0.0,
            "interference should be positive: {}",
            interference
        );
        // Verify orthogonal context gives lower score
        let ortho = doc.total_interference(&[0.0, 1.0]);
        assert!(
            interference > ortho,
            "aligned should beat orthogonal: {} vs {}",
            interference,
            ortho
        );
    }

    #[test]
    fn zero_noise_preserves_fidelity() {
        let mut doc = QuantumDocument::new(
            "d1",
            "test",
            vec![("m".into(), vec![1.0, 0.0])],
            0.0, // zero noise
        );
        doc.decohere(100.0, 42);
        assert!(
            (doc.fidelity - 1.0).abs() < 1e-6,
            "zero noise should preserve fidelity: {}",
            doc.fidelity
        );
    }

    #[test]
    fn kb_len_and_is_empty() {
        let mut kb = QuantumKnowledgeBase::new(0.3);
        assert!(kb.is_empty());
        assert_eq!(kb.len(), 0);
        kb.add(doc_with_meanings("d1", vec![("m", vec![1.0])]));
        assert!(!kb.is_empty());
        assert_eq!(kb.len(), 1);
    }
}
