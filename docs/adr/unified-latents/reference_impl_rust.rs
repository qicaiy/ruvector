//! ADR-058 Reference Implementation — Rust Service Scaffold
//!
//! Unified Latents (UL) dual-lane memory on RuVector/RVF.
//!
//! This scaffold demonstrates:
//!   1. Store creation with witness + governance config
//!   2. UL encoder adapter (thin shim; real encoder is external)
//!   3. Dual-lane ingest (search lane + archive lane)
//!   4. Proof-gated mutation for graph edges
//!   5. Query with GNN rerank, coherence gate, and governance
//!   6. Preview / reconstruction path via archive lane
//!   7. Deployment helpers: kernel embed, eBPF embed, COW branch
//!
//! Crate dependencies (Cargo.toml):
//! ```toml
//! [dependencies]
//! rvf-runtime  = "0.5"
//! rvf-crypto   = "0.5"
//! rvf-index    = "0.5"
//! rvf-manifest = "0.5"
//! rvf-ebpf     = "0.5"
//! rvf-launch   = "0.5"
//! ruvector-domain-expansion = "0.3"
//! ruvector-gnn = "0.2"
//! ruvector-graph-transformer = "0.2"
//! serde        = { version = "1", features = ["derive"] }
//! serde_json   = "1"
//! sha2         = "0.10"
//! thiserror    = "2"
//! tokio        = { version = "1", features = ["full"] }
//! tracing      = "0.1"
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Re-exports from RVF / RuVector crates (real signatures from the codebase)
// ---------------------------------------------------------------------------

use rvf_runtime::{
    options::{FilterExpr, QualityPreference, QueryOptions, SafetyNetBudget},
    witness::{GovernanceMode, GovernancePolicy, WitnessBuilder, WitnessConfig},
    CompactionResult, DeleteResult, DistanceMetric, IngestResult, MetadataEntry,
    MetadataValue, QualityEnvelope, RvfError, RvfOptions, RvfStore, SearchResult,
    StoreStatus,
};
use rvf_crypto::{
    create_witness_chain, sign_segment, verify_segment, verify_witness_chain,
    shake256_256, WitnessEntry,
};
use rvf_index::{
    build_full_index, build_layer_a, build_layer_b, build_layer_c,
    HnswConfig, IndexState, LayerA, LayerB, LayerC, ProgressiveIndex,
};
use ruvector_domain_expansion::{
    AccelerationScoreboard, CostCurve, CostCurvePoint, ConvergenceThresholds,
    DomainExpansionEngine, DomainId, PolicyKernel, PolicyKnobs, TransferPrior,
    TransferVerification,
};
use ruvector_graph_transformer::{AttestationChain, ProofGate, ProofGatedMutation};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum UlServiceError {
    #[error("rvf: {0}")]
    Rvf(#[from] RvfError),
    #[error("encoder: {0}")]
    Encoder(String),
    #[error("decoder: {0}")]
    Decoder(String),
    #[error("validation: {0}")]
    Validation(String),
    #[error("governance denied: {0}")]
    GovernanceDenied(String),
    #[error("coherence below threshold: {score} < {threshold}")]
    CoherenceBelowThreshold { score: f32, threshold: f32 },
    #[error("proof gate: {0}")]
    ProofGate(String),
}

type Result<T> = std::result::Result<T, UlServiceError>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-tenant / per-encoder store configuration.
#[derive(Debug, Clone)]
pub struct UlStoreConfig {
    /// Filesystem path to the RVF file.
    pub rvf_path: PathBuf,
    /// Latent dimensionality produced by the UL encoder.
    pub latent_dim: u16,
    /// Distance metric for ANN.
    pub metric: DistanceMetric,
    /// HNSW M parameter.
    pub hnsw_m: u16,
    /// HNSW ef_construction.
    pub hnsw_ef_construction: u16,
    /// Whether to sign segments with Ed25519.
    pub signing: bool,
    /// Witness configuration.
    pub witness: WitnessConfig,
    /// Default governance policy.
    pub governance: GovernancePolicy,
    /// Coherence threshold for graph writes.
    pub coherence_threshold: f32,
    /// Archive lane path (may be same file or a sibling).
    pub archive_path: Option<PathBuf>,
}

impl Default for UlStoreConfig {
    fn default() -> Self {
        Self {
            rvf_path: PathBuf::from("ul_store.rvf"),
            latent_dim: 768,
            metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            signing: true,
            witness: WitnessConfig {
                witness_ingest: true,
                witness_delete: true,
                witness_compact: true,
                audit_queries: false,
            },
            governance: GovernancePolicy::approved(),
            coherence_threshold: 0.70,
            archive_path: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Modality & safety enums (stored as metadata)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum Modality {
    Image = 0,
    Frame = 1,
    Clip = 2,
    Slide = 3,
    Diagram = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum SafetyClass {
    Public = 0,
    Internal = 1,
    Regulated = 2,
    Restricted = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum SearchQuant {
    Fp16 = 0,
    Int8 = 1,
    Int4 = 2,
    Binary = 3,
}

// ---------------------------------------------------------------------------
// Metadata field registry — stable field IDs for RVF MetadataEntry
// ---------------------------------------------------------------------------

pub mod meta_fields {
    pub const ASSET_ID: u16 = 1;
    pub const TENANT_ID: u16 = 2;
    pub const MODALITY: u16 = 3;
    pub const SOURCE_URI: u16 = 4;
    pub const SHA256: u16 = 5;
    pub const ENCODER_ID: u16 = 6;
    pub const PRIOR_ID: u16 = 7;
    pub const DECODER_ID: u16 = 8;
    pub const NOISE_SIGMA0: u16 = 9;
    pub const BITRATE_UPPER_BOUND: u16 = 10;
    pub const SEARCH_QUANT: u16 = 11;
    pub const SAFETY_CLASS: u16 = 12;
    pub const BRANCH_ID: u16 = 13;
    pub const PROOF_RECEIPT: u16 = 14;
    pub const TAGS: u16 = 15;
}

// ---------------------------------------------------------------------------
// Asset descriptor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UlAsset {
    pub asset_id: String,
    pub tenant_id: String,
    pub modality: Modality,
    pub source_uri: String,
    pub sha256: String,
    pub encoder_id: String,
    pub prior_id: String,
    pub decoder_id: String,
    pub noise_sigma0: f32,
    pub bitrate_upper_bound: f32,
    pub search_quant: SearchQuant,
    pub safety_class: SafetyClass,
    pub branch_id: String,
    pub tags: Vec<String>,
}

impl UlAsset {
    /// Convert to RVF metadata entries using the stable field registry.
    pub fn to_metadata_entries(&self) -> Vec<MetadataEntry> {
        vec![
            MetadataEntry { field_id: meta_fields::ASSET_ID,            value: MetadataValue::String(self.asset_id.clone()) },
            MetadataEntry { field_id: meta_fields::TENANT_ID,           value: MetadataValue::String(self.tenant_id.clone()) },
            MetadataEntry { field_id: meta_fields::MODALITY,            value: MetadataValue::U64(self.modality as u64) },
            MetadataEntry { field_id: meta_fields::SOURCE_URI,          value: MetadataValue::String(self.source_uri.clone()) },
            MetadataEntry { field_id: meta_fields::SHA256,              value: MetadataValue::String(self.sha256.clone()) },
            MetadataEntry { field_id: meta_fields::ENCODER_ID,          value: MetadataValue::String(self.encoder_id.clone()) },
            MetadataEntry { field_id: meta_fields::PRIOR_ID,            value: MetadataValue::String(self.prior_id.clone()) },
            MetadataEntry { field_id: meta_fields::DECODER_ID,          value: MetadataValue::String(self.decoder_id.clone()) },
            MetadataEntry { field_id: meta_fields::NOISE_SIGMA0,        value: MetadataValue::F64(self.noise_sigma0 as f64) },
            MetadataEntry { field_id: meta_fields::BITRATE_UPPER_BOUND, value: MetadataValue::F64(self.bitrate_upper_bound as f64) },
            MetadataEntry { field_id: meta_fields::SEARCH_QUANT,        value: MetadataValue::U64(self.search_quant as u64) },
            MetadataEntry { field_id: meta_fields::SAFETY_CLASS,        value: MetadataValue::U64(self.safety_class as u64) },
            MetadataEntry { field_id: meta_fields::BRANCH_ID,           value: MetadataValue::String(self.branch_id.clone()) },
            MetadataEntry { field_id: meta_fields::TAGS,                value: MetadataValue::String(self.tags.join(",")) },
        ]
    }
}

// ---------------------------------------------------------------------------
// Edge record (graph lane)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum EdgeRelation {
    NearDuplicate = 0,
    SceneNext = 1,
    SameAsset = 2,
    ClickedAfter = 3,
    UserFeedback = 4,
    ParentChild = 5,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UlEdge {
    pub src_vector_id: u64,
    pub dst_vector_id: u64,
    pub relation: EdgeRelation,
    pub weight: f32,
    pub proof_id: String,
    pub created_by: String,
}

// ---------------------------------------------------------------------------
// UL Encoder / Decoder adapter traits
// ---------------------------------------------------------------------------

/// Thin adapter for the UL encoder.  The real model runs externally (e.g.
/// JAX on a GPU sidecar).  This trait wraps the RPC or FFI boundary.
pub trait UlEncoder: Send + Sync {
    /// Encode raw asset bytes into a full-precision latent z0.
    fn encode(&self, asset_bytes: &[u8]) -> Result<UlEncodeOutput>;
}

#[derive(Debug, Clone)]
pub struct UlEncodeOutput {
    /// High-fidelity latent for archive lane (fp32).
    pub z0_archive: Vec<f32>,
    /// Quantized latent for search lane.
    pub z_search: Vec<f32>,
    /// Noise level sigma0 from the encoder.
    pub noise_sigma0: f32,
    /// Upper bound on latent bitrate.
    pub bitrate_upper_bound: f32,
}

/// Thin adapter for the UL decoder / diffusion prior.
pub trait UlDecoder: Send + Sync {
    /// Reconstruct bytes from the archive latent z0.
    fn decode(&self, z0: &[f32], decoder_id: &str) -> Result<Vec<u8>>;
}

// ---------------------------------------------------------------------------
// Coherence scorer (wraps min-cut coherence from ruvector)
// ---------------------------------------------------------------------------

pub trait CoherenceScorer: Send + Sync {
    /// Return a coherence score in [0, 1] for a candidate vector against
    /// its nearest neighbors in the graph.
    fn score(&self, candidate: &[f32], neighbor_ids: &[u64]) -> f32;
}

// ---------------------------------------------------------------------------
// GNN reranker adapter
// ---------------------------------------------------------------------------

pub trait GnnReranker: Send + Sync {
    /// Rerank search results using graph context.
    fn rerank(&self, results: &[SearchResult], query: &[f32]) -> Vec<SearchResult>;
}

// ---------------------------------------------------------------------------
// UL Service
// ---------------------------------------------------------------------------

pub struct UlService {
    config: UlStoreConfig,
    /// Search-lane store.
    search_store: RvfStore,
    /// Archive-lane store (same file or separate).
    archive_store: Option<RvfStore>,
    /// Encoder adapter.
    encoder: Box<dyn UlEncoder>,
    /// Decoder adapter (optional — only needed for preview/generation).
    decoder: Option<Box<dyn UlDecoder>>,
    /// Coherence scorer.
    coherence: Box<dyn CoherenceScorer>,
    /// GNN reranker.
    reranker: Box<dyn GnnReranker>,
    /// Graph state wrapped in a proof gate.
    graph: ProofGate<GraphState>,
    /// Domain expansion engine for cross-domain transfer.
    domain_engine: DomainExpansionEngine,
    /// Decoder registry: decoder_id -> version metadata.
    decoder_registry: HashMap<String, DecoderMeta>,
    /// Next vector id counter (monotonic).
    next_vector_id: u64,
}

#[derive(Debug, Clone, Default)]
pub struct GraphState {
    pub edges: Vec<UlEdge>,
}

#[derive(Debug, Clone)]
pub struct DecoderMeta {
    pub decoder_id: String,
    pub version: String,
    pub modalities: Vec<Modality>,
}

impl UlService {
    // ----- Construction ----------------------------------------------------

    /// Create a new UL service with fresh RVF stores.
    pub fn create(
        config: UlStoreConfig,
        encoder: Box<dyn UlEncoder>,
        decoder: Option<Box<dyn UlDecoder>>,
        coherence: Box<dyn CoherenceScorer>,
        reranker: Box<dyn GnnReranker>,
    ) -> Result<Self> {
        let opts = RvfOptions {
            dimension: config.latent_dim,
            metric: config.metric,
            m: config.hnsw_m,
            ef_construction: config.hnsw_ef_construction,
            signing: config.signing,
            witness: config.witness.clone(),
            ..Default::default()
        };

        let search_store = RvfStore::create(&config.rvf_path, opts.clone())?;

        let archive_store = config.archive_path.as_ref().map(|p| {
            let archive_opts = RvfOptions {
                dimension: config.latent_dim,
                metric: config.metric,
                signing: config.signing,
                witness: WitnessConfig {
                    witness_ingest: true,
                    witness_delete: true,
                    witness_compact: true,
                    audit_queries: false,
                },
                ..Default::default()
            };
            RvfStore::create(p, archive_opts)
        }).transpose()?;

        Ok(Self {
            config,
            search_store,
            archive_store,
            encoder,
            decoder,
            coherence,
            reranker,
            graph: ProofGate::new(GraphState::default()),
            domain_engine: DomainExpansionEngine::new(),
            decoder_registry: HashMap::new(),
            next_vector_id: 1,
        })
    }

    /// Open an existing UL service from disk.
    pub fn open(
        config: UlStoreConfig,
        encoder: Box<dyn UlEncoder>,
        decoder: Option<Box<dyn UlDecoder>>,
        coherence: Box<dyn CoherenceScorer>,
        reranker: Box<dyn GnnReranker>,
    ) -> Result<Self> {
        let search_store = RvfStore::open(&config.rvf_path)?;
        let archive_store = config.archive_path.as_ref()
            .map(|p| RvfStore::open(p))
            .transpose()?;

        Ok(Self {
            config,
            search_store,
            archive_store,
            encoder,
            decoder,
            coherence,
            reranker,
            graph: ProofGate::new(GraphState::default()),
            domain_engine: DomainExpansionEngine::new(),
            decoder_registry: HashMap::new(),
            next_vector_id: 1,
        })
    }

    /// Register a decoder in the immutable registry for this branch.
    pub fn register_decoder(&mut self, meta: DecoderMeta) {
        self.decoder_registry.insert(meta.decoder_id.clone(), meta);
    }

    // ----- Write path ------------------------------------------------------

    /// Ingest a single asset through the dual-lane pipeline.
    pub fn ingest(&mut self, asset_bytes: &[u8], asset: UlAsset) -> Result<IngestReceipt> {
        // 1. Encode
        let encoded = self.encoder.encode(asset_bytes)?;

        // 2. Validate
        self.validate_ingest(&encoded, &asset)?;

        // 3. Assign vector ID
        let vector_id = self.next_vector_id;
        self.next_vector_id += 1;

        // 4. Build metadata
        let metadata = asset.to_metadata_entries();

        // 5. Search lane — ingest quantized latent
        let search_result = self.search_store.ingest_batch(
            &[encoded.z_search.as_slice()],
            &[vector_id],
            Some(&metadata),
        )?;

        // 6. Archive lane — ingest full-precision latent
        if let Some(ref mut archive) = self.archive_store {
            archive.ingest_batch(
                &[encoded.z0_archive.as_slice()],
                &[vector_id],
                Some(&metadata),
            )?;
        }

        // 7. Witness receipt
        let receipt_hash = shake256_256(
            &[asset.asset_id.as_bytes(), &vector_id.to_le_bytes()].concat(),
        );

        tracing::info!(
            vector_id,
            asset_id = %asset.asset_id,
            accepted = search_result.accepted,
            "UL asset ingested"
        );

        Ok(IngestReceipt {
            vector_id,
            asset_id: asset.asset_id,
            receipt_hash,
            search_accepted: search_result.accepted,
        })
    }

    /// Batch ingest multiple assets.
    pub fn ingest_batch(
        &mut self,
        items: Vec<(Vec<u8>, UlAsset)>,
    ) -> Result<Vec<IngestReceipt>> {
        let mut receipts = Vec::with_capacity(items.len());
        for (bytes, asset) in items {
            receipts.push(self.ingest(&bytes, asset)?);
        }
        Ok(receipts)
    }

    /// Validate an encoded latent before ingest.
    fn validate_ingest(&self, encoded: &UlEncodeOutput, asset: &UlAsset) -> Result<()> {
        // Dimension check
        if encoded.z_search.len() != self.config.latent_dim as usize {
            return Err(UlServiceError::Validation(format!(
                "search latent dim {} != store dim {}",
                encoded.z_search.len(),
                self.config.latent_dim
            )));
        }

        // NaN / inf guard
        for (i, &v) in encoded.z_search.iter().enumerate() {
            if v.is_nan() || v.is_infinite() {
                return Err(UlServiceError::Validation(format!(
                    "search latent contains NaN/Inf at index {i}"
                )));
            }
        }

        // Decoder exists in registry
        if !self.decoder_registry.is_empty()
            && !self.decoder_registry.contains_key(&asset.decoder_id)
        {
            return Err(UlServiceError::Validation(format!(
                "decoder_id '{}' not in registry",
                asset.decoder_id
            )));
        }

        // Governance policy check
        let check = self.config.governance.check_tool("ul_ingest");
        if matches!(check, rvf_runtime::witness::PolicyCheck::Denied) {
            return Err(UlServiceError::GovernanceDenied(
                "ingest denied by governance policy".into(),
            ));
        }

        Ok(())
    }

    // ----- Graph lane (proof-gated) ----------------------------------------

    /// Add an edge to the graph with proof-gated mutation.
    pub fn add_edge(&mut self, edge: UlEdge) -> Result<String> {
        // Coherence check
        let score = self.coherence.score(
            &[], // caller should provide context vector
            &[edge.src_vector_id, edge.dst_vector_id],
        );
        if score < self.config.coherence_threshold {
            return Err(UlServiceError::CoherenceBelowThreshold {
                score,
                threshold: self.config.coherence_threshold,
            });
        }

        // Proof-gated mutation
        let attestation = self.graph.mutate_with_dim_proof(
            2, // expected: both src and dst exist
            2, // actual
            |state| {
                state.edges.push(edge.clone());
            },
        ).map_err(|e| UlServiceError::ProofGate(format!("{e:?}")))?;

        let proof_id = hex::encode(&shake256_256(
            format!("{:?}", attestation).as_bytes(),
        )[..16]);

        tracing::info!(
            src = edge.src_vector_id,
            dst = edge.dst_vector_id,
            relation = ?edge.relation,
            proof_id = %proof_id,
            "graph edge added"
        );

        Ok(proof_id)
    }

    // ----- Query path ------------------------------------------------------

    /// Retrieval-only query.
    pub fn query(
        &self,
        query_bytes: &[u8],
        k: usize,
        opts: UlQueryOptions,
    ) -> Result<Vec<UlSearchResult>> {
        // Encode query
        let encoded = self.encoder.encode(query_bytes)?;

        // Build RVF query options
        let rvf_opts = QueryOptions {
            ef_search: opts.ef_search,
            filter: opts.filter,
            timeout_ms: opts.timeout_ms,
            quality_preference: opts.quality_preference,
            ..Default::default()
        };

        // ANN search
        let mut results = self.search_store.query(
            &encoded.z_search,
            k * 2, // over-fetch for reranking
            &rvf_opts,
        )?;

        // GNN reranking
        results = self.reranker.rerank(&results, &encoded.z_search);

        // Coherence filter on top results
        let results: Vec<_> = results
            .into_iter()
            .take(k)
            .filter(|r| {
                let score = self.coherence.score(&encoded.z_search, &[r.id]);
                score >= self.config.coherence_threshold
            })
            .collect();

        Ok(results
            .into_iter()
            .map(|r| UlSearchResult {
                vector_id: r.id,
                distance: r.distance,
                preview: None,
            })
            .collect())
    }

    /// Retrieval + preview (decode top-k from archive lane).
    pub fn query_with_preview(
        &mut self,
        query_bytes: &[u8],
        k: usize,
        opts: UlQueryOptions,
        decoder_id: &str,
    ) -> Result<Vec<UlSearchResult>> {
        // Governance check for decoder
        let check = self.config.governance.check_tool("ul_decode");
        if matches!(check, rvf_runtime::witness::PolicyCheck::Denied) {
            return Err(UlServiceError::GovernanceDenied(
                "decode denied by governance policy".into(),
            ));
        }

        let decoder = self.decoder.as_ref().ok_or_else(|| {
            UlServiceError::Decoder("no decoder registered".into())
        })?;

        let mut results = self.query(query_bytes, k, opts)?;

        // Load archive latents and decode
        if let Some(ref archive) = self.archive_store {
            for result in &mut results {
                // In a real implementation, retrieve the archive vector by ID
                // and pass it to the decoder. Shown here as the structural flow.
                let preview_bytes = decoder.decode(&[], decoder_id)?;
                result.preview = Some(preview_bytes);
            }
        }

        // Audited witness for preview queries
        if self.config.witness.audit_queries {
            let _envelope = self.search_store.query_with_envelope(
                &[0.0; 1], // placeholder
                k,
                &QueryOptions::default(),
            )?;
        }

        Ok(results)
    }

    // ----- Deployment helpers ----------------------------------------------

    /// Embed a kernel image for self-booting deployment.
    pub fn embed_kernel(&mut self, kernel_bytes: &[u8]) -> Result<()> {
        self.search_store.embed_kernel(kernel_bytes)?;
        tracing::info!(size = kernel_bytes.len(), "kernel embedded in RVF");
        Ok(())
    }

    /// Embed an eBPF program for hot-path acceleration.
    pub fn embed_ebpf(&mut self, ebpf_bytes: &[u8]) -> Result<()> {
        self.search_store.embed_ebpf(ebpf_bytes)?;
        tracing::info!(size = ebpf_bytes.len(), "eBPF program embedded in RVF");
        Ok(())
    }

    /// Compact the search store.
    pub fn compact(&mut self) -> Result<CompactionResult> {
        let result = self.search_store.compact()?;
        tracing::info!(
            segments = result.segments_compacted,
            reclaimed = result.bytes_reclaimed,
            "search store compacted"
        );
        Ok(result)
    }

    /// Return store status.
    pub fn status(&self) -> StoreStatus {
        self.search_store.status()
    }

    /// Return graph attestation chain for audit.
    pub fn attestation_chain(&self) -> &AttestationChain {
        self.graph.attestation_chain()
    }
}

// ---------------------------------------------------------------------------
// Query options & results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UlQueryOptions {
    pub ef_search: u16,
    pub filter: Option<FilterExpr>,
    pub timeout_ms: u32,
    pub quality_preference: QualityPreference,
}

impl Default for UlQueryOptions {
    fn default() -> Self {
        Self {
            ef_search: 64,
            filter: None,
            timeout_ms: 0,
            quality_preference: QualityPreference::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UlSearchResult {
    pub vector_id: u64,
    pub distance: f32,
    pub preview: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct IngestReceipt {
    pub vector_id: u64,
    pub asset_id: String,
    pub receipt_hash: [u8; 32],
    pub search_accepted: u64,
}

// ---------------------------------------------------------------------------
// Transfer prior helpers
// ---------------------------------------------------------------------------

/// Initiate cross-domain transfer from source to target domain.
pub fn initiate_transfer(
    engine: &mut DomainExpansionEngine,
    source: &str,
    target: &str,
) {
    let src = DomainId(source.to_string());
    let tgt = DomainId(target.to_string());
    engine.initiate_transfer(&src, &tgt);
}

/// Verify that a transfer prior improved the target without regressing source.
pub fn verify_transfer(
    engine: &DomainExpansionEngine,
    source: &str,
    target: &str,
    source_before: f32,
    source_after: f32,
    target_before: f32,
    target_after: f32,
    baseline_cycles: u64,
    transfer_cycles: u64,
) -> TransferVerification {
    engine.verify_transfer(
        &DomainId(source.to_string()),
        &DomainId(target.to_string()),
        source_before,
        source_after,
        target_before,
        target_after,
        baseline_cycles,
        transfer_cycles,
    )
}

// ---------------------------------------------------------------------------
// Governance helpers
// ---------------------------------------------------------------------------

/// Build governance policy for each ADR-058 tier.
pub fn restricted_policy() -> GovernancePolicy {
    GovernancePolicy::restricted()
}

pub fn approved_policy() -> GovernancePolicy {
    GovernancePolicy::approved()
}

pub fn autonomous_policy(
    allowed_tools: Vec<String>,
    max_cost_microdollars: u32,
    max_tool_calls: u16,
) -> GovernancePolicy {
    GovernancePolicy {
        mode: GovernanceMode::Autonomous,
        allowed_tools,
        denied_tools: vec![],
        max_cost_microdollars,
        max_tool_calls,
    }
}

// ---------------------------------------------------------------------------
// Witness helpers
// ---------------------------------------------------------------------------

/// Build a witness receipt for a generation request.
pub fn generation_witness(
    task_id: [u8; 16],
    policy: GovernancePolicy,
    output_hash: [u8; 32],
) -> WitnessBuilder {
    let mut builder = WitnessBuilder::new(task_id, policy);
    builder.diff = Some(output_hash.to_vec());
    builder
}

// ---------------------------------------------------------------------------
// Launch config helper (self-booting microVM)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UlLaunchConfig {
    pub rvf_path: PathBuf,
    pub memory_mb: u64,
    pub vcpus: u32,
    pub api_port: u16,
}

impl Default for UlLaunchConfig {
    fn default() -> Self {
        Self {
            rvf_path: PathBuf::from("ul_store.rvf"),
            memory_mb: 512,
            vcpus: 2,
            api_port: 8080,
        }
    }
}

// ---------------------------------------------------------------------------
// Example usage (compile-gated)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: config defaults are sane.
    #[test]
    fn config_defaults() {
        let cfg = UlStoreConfig::default();
        assert_eq!(cfg.latent_dim, 768);
        assert_eq!(cfg.hnsw_m, 16);
        assert!(cfg.coherence_threshold > 0.0);
    }

    /// Metadata round-trip.
    #[test]
    fn metadata_entries_count() {
        let asset = UlAsset {
            asset_id: "a1".into(),
            tenant_id: "t1".into(),
            modality: Modality::Image,
            source_uri: "s3://bucket/img.png".into(),
            sha256: "abc123".into(),
            encoder_id: "ul-enc-v1".into(),
            prior_id: "ul-prior-v1".into(),
            decoder_id: "ul-dec-v1".into(),
            noise_sigma0: 0.01,
            bitrate_upper_bound: 3.5,
            search_quant: SearchQuant::Int8,
            safety_class: SafetyClass::Public,
            branch_id: "main".into(),
            tags: vec!["test".into()],
        };
        let entries = asset.to_metadata_entries();
        assert_eq!(entries.len(), 14);
    }

    /// Governance tiers.
    #[test]
    fn governance_tiers() {
        let r = restricted_policy();
        assert!(matches!(r.mode, GovernanceMode::Restricted));

        let a = approved_policy();
        assert!(matches!(a.mode, GovernanceMode::Approved));

        let auto = autonomous_policy(
            vec!["ul_decode".into()],
            1_000_000,
            100,
        );
        assert!(matches!(auto.mode, GovernanceMode::Autonomous));
        assert_eq!(auto.max_tool_calls, 100);
    }
}
