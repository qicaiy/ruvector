/**
 * ADR-058 Reference Implementation — TypeScript Client Scaffold
 *
 * Unified Latents (UL) dual-lane client for Node.js and edge runtimes.
 *
 * This scaffold demonstrates:
 *   1. RVF store creation and configuration via @ruvector/rvf-node
 *   2. UL encoder adapter (thin shim; real encoder is external)
 *   3. Dual-lane ingest with metadata and witness receipts
 *   4. Search with filter expressions and quality preferences
 *   5. Preview path via archive lane
 *   6. Governance policy enforcement
 *   7. WASM-based browser/edge search via @ruvector/rvf-wasm
 *
 * Package dependencies:
 *   npm install @ruvector/rvf @ruvector/rvf-node @ruvector/rvf-wasm @ruvector/rvlite
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type DistanceMetric = "l2" | "cosine" | "inner_product";

export type QualityPreference =
  | "auto"
  | "prefer_quality"
  | "prefer_latency"
  | "accept_degraded";

export type GovernanceMode = "restricted" | "approved" | "autonomous";

export type Modality = "image" | "frame" | "clip" | "slide" | "diagram";

export type SafetyClass = "public" | "internal" | "regulated" | "restricted";

export type SearchQuant = "fp16" | "int8" | "int4" | "binary";

export type EdgeRelation =
  | "near_duplicate"
  | "scene_next"
  | "same_asset"
  | "clicked_after"
  | "user_feedback"
  | "parent_child";

// ---------------------------------------------------------------------------
// Metadata field registry — stable field IDs matching Rust implementation
// ---------------------------------------------------------------------------

export const META_FIELDS = {
  ASSET_ID: 1,
  TENANT_ID: 2,
  MODALITY: 3,
  SOURCE_URI: 4,
  SHA256: 5,
  ENCODER_ID: 6,
  PRIOR_ID: 7,
  DECODER_ID: 8,
  NOISE_SIGMA0: 9,
  BITRATE_UPPER_BOUND: 10,
  SEARCH_QUANT: 11,
  SAFETY_CLASS: 12,
  BRANCH_ID: 13,
  PROOF_RECEIPT: 14,
  TAGS: 15,
} as const;

// ---------------------------------------------------------------------------
// Asset & Edge interfaces
// ---------------------------------------------------------------------------

export interface UlAsset {
  assetId: string;
  tenantId: string;
  modality: Modality;
  sourceUri: string;
  sha256: string;
  encoderId: string;
  priorId: string;
  decoderId: string;
  noiseSigma0: number;
  bitrateUpperBound: number;
  searchQuant: SearchQuant;
  safetyClass: SafetyClass;
  branchId: string;
  tags: string[];
}

export interface UlEdge {
  srcVectorId: number;
  dstVectorId: number;
  relation: EdgeRelation;
  weight: number;
  proofId: string;
  createdBy: string;
}

// ---------------------------------------------------------------------------
// Encode / Decode output
// ---------------------------------------------------------------------------

export interface UlEncodeOutput {
  /** High-fidelity latent for archive lane (fp32). */
  z0Archive: Float32Array;
  /** Quantized latent for search lane. */
  zSearch: Float32Array;
  /** Noise level sigma0 from the encoder. */
  noiseSigma0: number;
  /** Upper bound on latent bitrate. */
  bitrateUpperBound: number;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface WitnessConfig {
  witnessIngest: boolean;
  witnessDelete: boolean;
  witnessCompact: boolean;
  auditQueries: boolean;
}

export interface GovernancePolicy {
  mode: GovernanceMode;
  allowedTools: string[];
  deniedTools: string[];
  maxCostMicrodollars: number;
  maxToolCalls: number;
}

export interface UlStoreConfig {
  rvfPath: string;
  latentDim: number;
  metric: DistanceMetric;
  hnswM: number;
  hnswEfConstruction: number;
  signing: boolean;
  witness: WitnessConfig;
  governance: GovernancePolicy;
  coherenceThreshold: number;
  archivePath?: string;
}

export const DEFAULT_STORE_CONFIG: UlStoreConfig = {
  rvfPath: "ul_store.rvf",
  latentDim: 768,
  metric: "cosine",
  hnswM: 16,
  hnswEfConstruction: 200,
  signing: true,
  witness: {
    witnessIngest: true,
    witnessDelete: true,
    witnessCompact: true,
    auditQueries: false,
  },
  governance: {
    mode: "approved",
    allowedTools: ["ul_ingest", "ul_query", "ul_decode"],
    deniedTools: [],
    maxCostMicrodollars: 1_000_000,
    maxToolCalls: 1000,
  },
  coherenceThreshold: 0.7,
};

// ---------------------------------------------------------------------------
// Query options & results
// ---------------------------------------------------------------------------

export interface MetadataEntry {
  fieldId: number;
  value: string | number | Uint8Array;
}

export interface FilterExpr {
  field: number;
  op: "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "in" | "contains";
  value: string | number | number[];
}

export interface UlQueryOptions {
  efSearch: number;
  filters?: FilterExpr[];
  timeoutMs: number;
  qualityPreference: QualityPreference;
}

export const DEFAULT_QUERY_OPTIONS: UlQueryOptions = {
  efSearch: 64,
  timeoutMs: 0,
  qualityPreference: "auto",
};

export interface SearchResult {
  vectorId: number;
  distance: number;
  metadata?: Record<string, unknown>;
}

export interface UlSearchResult extends SearchResult {
  preview?: Uint8Array;
}

export interface IngestReceipt {
  vectorId: number;
  assetId: string;
  receiptHash: string;
  searchAccepted: number;
}

export interface QualityEnvelope {
  results: SearchResult[];
  quality: string;
  layersUsed: { layerA: boolean; layerB: boolean; layerC: boolean };
  totalUs: number;
}

// ---------------------------------------------------------------------------
// Encoder / Decoder adapters (thin shims for external models)
// ---------------------------------------------------------------------------

export interface UlEncoder {
  encode(assetBytes: Uint8Array): Promise<UlEncodeOutput>;
}

export interface UlDecoder {
  decode(z0: Float32Array, decoderId: string): Promise<Uint8Array>;
}

// ---------------------------------------------------------------------------
// RVF Store wrapper (uses @ruvector/rvf-node bindings)
// ---------------------------------------------------------------------------

/**
 * Wraps the native RVF store from @ruvector/rvf-node.
 *
 * In production, import the actual bindings:
 *   import { RvfStore } from "@ruvector/rvf-node";
 */
export class RvfStoreClient {
  private store: unknown; // RvfStore from native binding
  private config: UlStoreConfig;

  constructor(config: UlStoreConfig) {
    this.config = config;
    // In production: this.store = RvfStore.create(config.rvfPath, { ... });
  }

  static async create(config: UlStoreConfig): Promise<RvfStoreClient> {
    const client = new RvfStoreClient(config);
    // Native call: RvfStore.create(config.rvfPath, {
    //   dimension: config.latentDim,
    //   metric: config.metric,
    //   m: config.hnswM,
    //   efConstruction: config.hnswEfConstruction,
    //   signing: config.signing,
    //   witness: config.witness,
    // });
    return client;
  }

  static async open(config: UlStoreConfig): Promise<RvfStoreClient> {
    const client = new RvfStoreClient(config);
    // Native call: RvfStore.open(config.rvfPath);
    return client;
  }

  async ingestBatch(
    vectors: Float32Array[],
    ids: number[],
    metadata?: MetadataEntry[][],
  ): Promise<{ accepted: number; rejected: number }> {
    // Native call: this.store.ingestBatch(vectors, ids, metadata);
    return { accepted: ids.length, rejected: 0 };
  }

  async query(
    vector: Float32Array,
    k: number,
    options: UlQueryOptions,
  ): Promise<SearchResult[]> {
    // Native call: this.store.query(vector, k, {
    //   efSearch: options.efSearch,
    //   filter: options.filters,
    //   timeoutMs: options.timeoutMs,
    //   qualityPreference: options.qualityPreference,
    // });
    return [];
  }

  async queryWithEnvelope(
    vector: Float32Array,
    k: number,
    options: UlQueryOptions,
  ): Promise<QualityEnvelope> {
    // Native call: this.store.queryWithEnvelope(vector, k, options);
    return {
      results: [],
      quality: "verified",
      layersUsed: { layerA: true, layerB: true, layerC: true },
      totalUs: 0,
    };
  }

  async compact(): Promise<{ segmentsCompacted: number; bytesReclaimed: number }> {
    // Native call: this.store.compact();
    return { segmentsCompacted: 0, bytesReclaimed: 0 };
  }

  async embedKernel(kernelBytes: Uint8Array): Promise<void> {
    // Native call: this.store.embedKernel(kernelBytes);
  }

  async embedEbpf(ebpfBytes: Uint8Array): Promise<void> {
    // Native call: this.store.embedEbpf(ebpfBytes);
  }
}

// ---------------------------------------------------------------------------
// UL Service
// ---------------------------------------------------------------------------

export class UlService {
  private searchStore: RvfStoreClient;
  private archiveStore: RvfStoreClient | null;
  private encoder: UlEncoder;
  private decoder: UlDecoder | null;
  private config: UlStoreConfig;
  private decoderRegistry: Map<string, { version: string; modalities: Modality[] }>;
  private nextVectorId: number;

  private constructor(
    config: UlStoreConfig,
    searchStore: RvfStoreClient,
    archiveStore: RvfStoreClient | null,
    encoder: UlEncoder,
    decoder: UlDecoder | null,
  ) {
    this.config = config;
    this.searchStore = searchStore;
    this.archiveStore = archiveStore;
    this.encoder = encoder;
    this.decoder = decoder;
    this.decoderRegistry = new Map();
    this.nextVectorId = 1;
  }

  /** Create a new UL service with fresh RVF stores. */
  static async create(
    config: UlStoreConfig,
    encoder: UlEncoder,
    decoder?: UlDecoder,
  ): Promise<UlService> {
    const searchStore = await RvfStoreClient.create(config);
    let archiveStore: RvfStoreClient | null = null;
    if (config.archivePath) {
      archiveStore = await RvfStoreClient.create({
        ...config,
        rvfPath: config.archivePath,
      });
    }
    return new UlService(config, searchStore, archiveStore, encoder, decoder ?? null);
  }

  /** Open an existing UL service from disk. */
  static async open(
    config: UlStoreConfig,
    encoder: UlEncoder,
    decoder?: UlDecoder,
  ): Promise<UlService> {
    const searchStore = await RvfStoreClient.open(config);
    let archiveStore: RvfStoreClient | null = null;
    if (config.archivePath) {
      archiveStore = await RvfStoreClient.open({
        ...config,
        rvfPath: config.archivePath,
      });
    }
    return new UlService(config, searchStore, archiveStore, encoder, decoder ?? null);
  }

  /** Register a decoder version for this branch. */
  registerDecoder(decoderId: string, version: string, modalities: Modality[]): void {
    this.decoderRegistry.set(decoderId, { version, modalities });
  }

  // ----- Write path --------------------------------------------------------

  /** Ingest a single asset through the dual-lane pipeline. */
  async ingest(assetBytes: Uint8Array, asset: UlAsset): Promise<IngestReceipt> {
    // 1. Encode
    const encoded = await this.encoder.encode(assetBytes);

    // 2. Validate
    this.validateIngest(encoded, asset);

    // 3. Assign vector ID
    const vectorId = this.nextVectorId++;

    // 4. Build metadata entries
    const metadata = this.assetToMetadata(asset);

    // 5. Search lane ingest
    const searchResult = await this.searchStore.ingestBatch(
      [encoded.zSearch],
      [vectorId],
      [metadata],
    );

    // 6. Archive lane ingest
    if (this.archiveStore) {
      await this.archiveStore.ingestBatch(
        [encoded.z0Archive],
        [vectorId],
        [metadata],
      );
    }

    // 7. Compute receipt hash
    const receiptData = new TextEncoder().encode(`${asset.assetId}:${vectorId}`);
    const receiptHash = await crypto.subtle.digest("SHA-256", receiptData);
    const receiptHex = Array.from(new Uint8Array(receiptHash))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");

    return {
      vectorId,
      assetId: asset.assetId,
      receiptHash: receiptHex,
      searchAccepted: searchResult.accepted,
    };
  }

  /** Batch ingest multiple assets. */
  async ingestBatch(
    items: Array<{ bytes: Uint8Array; asset: UlAsset }>,
  ): Promise<IngestReceipt[]> {
    const receipts: IngestReceipt[] = [];
    for (const { bytes, asset } of items) {
      receipts.push(await this.ingest(bytes, asset));
    }
    return receipts;
  }

  // ----- Query path --------------------------------------------------------

  /** Retrieval-only query. */
  async query(
    queryBytes: Uint8Array,
    k: number,
    opts: Partial<UlQueryOptions> = {},
  ): Promise<UlSearchResult[]> {
    const options: UlQueryOptions = { ...DEFAULT_QUERY_OPTIONS, ...opts };

    // Encode query
    const encoded = await this.encoder.encode(queryBytes);

    // ANN search (over-fetch for reranking)
    const results = await this.searchStore.query(encoded.zSearch, k * 2, options);

    // In production: apply GNN reranking and coherence filtering here
    return results.slice(0, k).map((r) => ({
      vectorId: r.vectorId,
      distance: r.distance,
      metadata: r.metadata,
    }));
  }

  /** Retrieval + preview (decode top-k from archive lane). */
  async queryWithPreview(
    queryBytes: Uint8Array,
    k: number,
    decoderId: string,
    opts: Partial<UlQueryOptions> = {},
  ): Promise<UlSearchResult[]> {
    // Governance check
    this.checkGovernance("ul_decode");

    if (!this.decoder) {
      throw new Error("No decoder registered for preview");
    }

    const results = await this.query(queryBytes, k, opts);

    // Decode previews from archive lane
    if (this.archiveStore) {
      for (const result of results) {
        // In production: load archive vector by ID, then decode
        const preview = await this.decoder.decode(
          new Float32Array(this.config.latentDim),
          decoderId,
        );
        result.preview = preview;
      }
    }

    return results;
  }

  /** Get quality envelope for diagnostic queries. */
  async queryDiagnostic(
    queryBytes: Uint8Array,
    k: number,
    opts: Partial<UlQueryOptions> = {},
  ): Promise<QualityEnvelope> {
    const options: UlQueryOptions = { ...DEFAULT_QUERY_OPTIONS, ...opts };
    const encoded = await this.encoder.encode(queryBytes);
    return this.searchStore.queryWithEnvelope(encoded.zSearch, k, options);
  }

  // ----- Deployment helpers ------------------------------------------------

  /** Embed a kernel for self-booting microVM deployment. */
  async embedKernel(kernelBytes: Uint8Array): Promise<void> {
    await this.searchStore.embedKernel(kernelBytes);
  }

  /** Embed an eBPF program for hot-path acceleration. */
  async embedEbpf(ebpfBytes: Uint8Array): Promise<void> {
    await this.searchStore.embedEbpf(ebpfBytes);
  }

  /** Compact the search store. */
  async compact(): Promise<{ segmentsCompacted: number; bytesReclaimed: number }> {
    return this.searchStore.compact();
  }

  // ----- Internal helpers --------------------------------------------------

  private validateIngest(encoded: UlEncodeOutput, asset: UlAsset): void {
    // Dimension check
    if (encoded.zSearch.length !== this.config.latentDim) {
      throw new Error(
        `search latent dim ${encoded.zSearch.length} != store dim ${this.config.latentDim}`,
      );
    }

    // NaN/Inf guard
    for (let i = 0; i < encoded.zSearch.length; i++) {
      if (!Number.isFinite(encoded.zSearch[i])) {
        throw new Error(`search latent contains NaN/Inf at index ${i}`);
      }
    }

    // Decoder registry check
    if (this.decoderRegistry.size > 0 && !this.decoderRegistry.has(asset.decoderId)) {
      throw new Error(`decoder_id '${asset.decoderId}' not in registry`);
    }

    // Governance check
    this.checkGovernance("ul_ingest");
  }

  private checkGovernance(tool: string): void {
    const { governance } = this.config;
    if (governance.mode === "restricted") {
      throw new Error(`governance: '${tool}' denied in restricted mode`);
    }
    if (
      governance.deniedTools.includes(tool) ||
      (governance.allowedTools.length > 0 && !governance.allowedTools.includes(tool))
    ) {
      throw new Error(`governance: '${tool}' not in allowed tools`);
    }
  }

  private assetToMetadata(asset: UlAsset): MetadataEntry[] {
    return [
      { fieldId: META_FIELDS.ASSET_ID, value: asset.assetId },
      { fieldId: META_FIELDS.TENANT_ID, value: asset.tenantId },
      { fieldId: META_FIELDS.MODALITY, value: asset.modality },
      { fieldId: META_FIELDS.SOURCE_URI, value: asset.sourceUri },
      { fieldId: META_FIELDS.SHA256, value: asset.sha256 },
      { fieldId: META_FIELDS.ENCODER_ID, value: asset.encoderId },
      { fieldId: META_FIELDS.PRIOR_ID, value: asset.priorId },
      { fieldId: META_FIELDS.DECODER_ID, value: asset.decoderId },
      { fieldId: META_FIELDS.NOISE_SIGMA0, value: asset.noiseSigma0 },
      { fieldId: META_FIELDS.BITRATE_UPPER_BOUND, value: asset.bitrateUpperBound },
      { fieldId: META_FIELDS.SEARCH_QUANT, value: asset.searchQuant },
      { fieldId: META_FIELDS.SAFETY_CLASS, value: asset.safetyClass },
      { fieldId: META_FIELDS.BRANCH_ID, value: asset.branchId },
      { fieldId: META_FIELDS.TAGS, value: asset.tags.join(",") },
    ];
  }
}

// ---------------------------------------------------------------------------
// Governance policy builders
// ---------------------------------------------------------------------------

export function restrictedPolicy(): GovernancePolicy {
  return {
    mode: "restricted",
    allowedTools: [],
    deniedTools: [],
    maxCostMicrodollars: 0,
    maxToolCalls: 0,
  };
}

export function approvedPolicy(): GovernancePolicy {
  return {
    mode: "approved",
    allowedTools: ["ul_ingest", "ul_query", "ul_decode", "ul_branch"],
    deniedTools: [],
    maxCostMicrodollars: 1_000_000,
    maxToolCalls: 1000,
  };
}

export function autonomousPolicy(
  allowedTools: string[],
  maxCostMicrodollars: number,
  maxToolCalls: number,
): GovernancePolicy {
  return {
    mode: "autonomous",
    allowedTools,
    deniedTools: [],
    maxCostMicrodollars,
    maxToolCalls,
  };
}

// ---------------------------------------------------------------------------
// Query option presets (from ADR-058 search policy table)
// ---------------------------------------------------------------------------

export const QUERY_PRESETS = {
  /** Interactive image search — low latency. */
  interactive: (tenantId: string, modality: Modality): UlQueryOptions => ({
    efSearch: 64,
    filters: [
      { field: META_FIELDS.TENANT_ID, op: "eq", value: tenantId },
      { field: META_FIELDS.MODALITY, op: "eq", value: modality },
    ],
    timeoutMs: 500,
    qualityPreference: "auto",
  }),

  /** Compliance audit — prefer quality, witness on. */
  audit: (branchId: string, startTs: number, endTs: number): UlQueryOptions => ({
    efSearch: 128,
    filters: [{ field: META_FIELDS.BRANCH_ID, op: "eq", value: branchId }],
    timeoutMs: 5000,
    qualityPreference: "prefer_quality",
  }),

  /** Reconstruction preview — archive latent required. */
  preview: (decoderId: string, safetyClass: SafetyClass): UlQueryOptions => ({
    efSearch: 96,
    filters: [
      { field: META_FIELDS.DECODER_ID, op: "eq", value: decoderId },
      { field: META_FIELDS.SAFETY_CLASS, op: "eq", value: safetyClass },
    ],
    timeoutMs: 2000,
    qualityPreference: "prefer_quality",
  }),

  /** Edge offline — fast, tenant-only. */
  edge: (tenantId: string): UlQueryOptions => ({
    efSearch: 32,
    filters: [{ field: META_FIELDS.TENANT_ID, op: "eq", value: tenantId }],
    timeoutMs: 200,
    qualityPreference: "prefer_latency",
  }),
} as const;

// ---------------------------------------------------------------------------
// WASM/Browser adapter (uses @ruvector/rvf-wasm for offline search)
// ---------------------------------------------------------------------------

/**
 * Lightweight browser-side search client.
 *
 * In production, import the actual WASM binding:
 *   import { RvfWasmStore } from "@ruvector/rvf-wasm";
 */
export class UlBrowserClient {
  private rvfUrl: string;
  private loaded: boolean;

  constructor(rvfUrl: string) {
    this.rvfUrl = rvfUrl;
    this.loaded = false;
  }

  /** Download and load the RVF file into WASM memory. */
  async load(): Promise<void> {
    // In production:
    //   const response = await fetch(this.rvfUrl);
    //   const bytes = new Uint8Array(await response.arrayBuffer());
    //   this.store = await RvfWasmStore.fromBytes(bytes);
    this.loaded = true;
  }

  /** Progressive boot: load Layer A first for fast approximate search. */
  async loadProgressive(): Promise<void> {
    // In production:
    //   await this.store.loadLayerA();     // ~5ms, ~0.70 recall
    //   await this.store.loadLayerB();     // ~100ms, ~0.85 recall
    //   await this.store.loadLayerC();     // seconds, >=0.95 recall
    this.loaded = true;
  }

  /** Search using the locally loaded RVF index. */
  async search(
    queryVector: Float32Array,
    k: number,
    opts?: Partial<UlQueryOptions>,
  ): Promise<SearchResult[]> {
    if (!this.loaded) {
      throw new Error("RVF not loaded — call load() or loadProgressive() first");
    }
    // In production: return this.store.query(queryVector, k, opts);
    return [];
  }
}

// ---------------------------------------------------------------------------
// Example usage
// ---------------------------------------------------------------------------

async function example(): Promise<void> {
  // 1. Create a mock encoder
  const encoder: UlEncoder = {
    async encode(assetBytes: Uint8Array): Promise<UlEncodeOutput> {
      const dim = 768;
      return {
        z0Archive: new Float32Array(dim).fill(0.01),
        zSearch: new Float32Array(dim).fill(0.01),
        noiseSigma0: 0.01,
        bitrateUpperBound: 3.5,
      };
    },
  };

  // 2. Create the service
  const service = await UlService.create(DEFAULT_STORE_CONFIG, encoder);

  // 3. Register a decoder
  service.registerDecoder("ul-dec-v1", "1.0.0", ["image", "frame"]);

  // 4. Ingest an asset
  const receipt = await service.ingest(new Uint8Array([1, 2, 3]), {
    assetId: "asset-001",
    tenantId: "tenant-a",
    modality: "image",
    sourceUri: "s3://bucket/img.png",
    sha256: "abc123def456",
    encoderId: "ul-enc-v1",
    priorId: "ul-prior-v1",
    decoderId: "ul-dec-v1",
    noiseSigma0: 0.01,
    bitrateUpperBound: 3.5,
    searchQuant: "int8",
    safetyClass: "public",
    branchId: "main",
    tags: ["landscape", "outdoor"],
  });

  console.log("Ingested:", receipt);

  // 5. Query
  const results = await service.query(
    new Uint8Array([4, 5, 6]),
    5,
    QUERY_PRESETS.interactive("tenant-a", "image"),
  );

  console.log("Results:", results);

  // 6. Browser/edge usage
  const browser = new UlBrowserClient("https://cdn.example.com/store.rvf");
  await browser.loadProgressive();
  const edgeResults = await browser.search(new Float32Array(768).fill(0.01), 5);
  console.log("Edge results:", edgeResults);
}
