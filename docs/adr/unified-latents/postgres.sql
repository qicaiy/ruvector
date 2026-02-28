-- ==========================================================================
-- ADR-058 Reference Implementation — PostgreSQL Mirror Schema
--
-- Unified Latents (UL) search-lane mirror for RuVector PostgreSQL.
--
-- This schema provides:
--   1. UL asset table with all ADR-058 metadata fields
--   2. Vector column using ruvector extension (pgvector replacement)
--   3. Edge/graph table for relationship tracking
--   4. Witness/audit log table
--   5. Decoder registry (immutable per branch)
--   6. Branch table for COW experiment tracking
--   7. HNSW index on the search vector
--   8. Metadata indexes for hybrid filtering
--   9. Example queries matching ADR-058 query profiles
--  10. Governance policy enforcement via row-level security
--
-- Prerequisites:
--   CREATE EXTENSION IF NOT EXISTS ruvector;
--   -- or: CREATE EXTENSION IF NOT EXISTS vector;
-- ==========================================================================

-- --------------------------------------------------------------------------
-- 0. Extension
-- --------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS ruvector;

-- --------------------------------------------------------------------------
-- 1. Enums
-- --------------------------------------------------------------------------

CREATE TYPE ul_modality AS ENUM (
    'image', 'frame', 'clip', 'slide', 'diagram'
);

CREATE TYPE ul_safety_class AS ENUM (
    'public', 'internal', 'regulated', 'restricted'
);

CREATE TYPE ul_search_quant AS ENUM (
    'fp16', 'int8', 'int4', 'binary'
);

CREATE TYPE ul_edge_relation AS ENUM (
    'near_duplicate', 'scene_next', 'same_asset',
    'clicked_after', 'user_feedback', 'parent_child'
);

CREATE TYPE ul_governance_mode AS ENUM (
    'restricted', 'approved', 'autonomous'
);

CREATE TYPE ul_witness_action AS ENUM (
    'ingest', 'delete', 'compact', 'query',
    'branch_create', 'branch_merge', 'edge_add',
    'decode', 'generate'
);

-- --------------------------------------------------------------------------
-- 2. Branch table (COW experiment tracking)
-- --------------------------------------------------------------------------

CREATE TABLE ul_branches (
    branch_id       TEXT PRIMARY KEY,
    parent_id       TEXT REFERENCES ul_branches(branch_id),
    encoder_id      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    frozen_at       TIMESTAMPTZ,
    description     TEXT,
    governance_mode ul_governance_mode NOT NULL DEFAULT 'approved'
);

-- Root branch
INSERT INTO ul_branches (branch_id, parent_id, encoder_id, description)
VALUES ('main', NULL, 'ul-enc-v1', 'Root branch');

-- --------------------------------------------------------------------------
-- 3. Decoder registry (immutable per branch)
-- --------------------------------------------------------------------------

CREATE TABLE ul_decoder_registry (
    decoder_id      TEXT NOT NULL,
    branch_id       TEXT NOT NULL REFERENCES ul_branches(branch_id),
    version         TEXT NOT NULL,
    modalities      ul_modality[] NOT NULL,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (decoder_id, branch_id)
);

-- --------------------------------------------------------------------------
-- 4. Core asset table
-- --------------------------------------------------------------------------

CREATE TABLE ul_assets (
    vector_id               BIGSERIAL PRIMARY KEY,
    asset_id                TEXT NOT NULL,
    tenant_id               TEXT NOT NULL,
    modality                ul_modality NOT NULL,
    source_uri              TEXT NOT NULL,
    sha256                  TEXT NOT NULL,

    -- UL model versioning
    encoder_id              TEXT NOT NULL,
    prior_id                TEXT NOT NULL,
    decoder_id              TEXT NOT NULL,
    latent_dim              INTEGER NOT NULL,
    noise_sigma0            REAL NOT NULL,
    bitrate_upper_bound     REAL NOT NULL,

    -- Quantization & safety
    search_quant            ul_search_quant NOT NULL DEFAULT 'int8',
    safety_class            ul_safety_class NOT NULL DEFAULT 'public',

    -- Search-lane vector (quantized latent z_search)
    z_search                vector NOT NULL,

    -- Archive-lane vector (full-precision latent z0)
    z0_archive              vector,

    -- Lineage
    branch_id               TEXT NOT NULL DEFAULT 'main'
                            REFERENCES ul_branches(branch_id),
    proof_receipt           TEXT,
    tags                    TEXT[] DEFAULT '{}',

    -- Timestamps
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Constraints
    CONSTRAINT chk_latent_dim CHECK (latent_dim > 0 AND latent_dim <= 16000),
    CONSTRAINT chk_noise_positive CHECK (noise_sigma0 > 0),
    CONSTRAINT chk_bitrate_positive CHECK (bitrate_upper_bound > 0)
);

-- Unique business key per tenant and branch
CREATE UNIQUE INDEX idx_ul_assets_business_key
    ON ul_assets (tenant_id, asset_id, branch_id);

-- --------------------------------------------------------------------------
-- 5. HNSW index on search vector (ruvector extension)
-- --------------------------------------------------------------------------

-- Cosine distance HNSW index with ruvector parameters
CREATE INDEX idx_ul_assets_z_search_hnsw
    ON ul_assets
    USING hnsw (z_search vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- --------------------------------------------------------------------------
-- 6. Metadata indexes for hybrid filtering
-- --------------------------------------------------------------------------

CREATE INDEX idx_ul_assets_tenant      ON ul_assets (tenant_id);
CREATE INDEX idx_ul_assets_modality    ON ul_assets (modality);
CREATE INDEX idx_ul_assets_encoder     ON ul_assets (encoder_id);
CREATE INDEX idx_ul_assets_decoder     ON ul_assets (decoder_id);
CREATE INDEX idx_ul_assets_safety      ON ul_assets (safety_class);
CREATE INDEX idx_ul_assets_branch      ON ul_assets (branch_id);
CREATE INDEX idx_ul_assets_created     ON ul_assets (created_at);
CREATE INDEX idx_ul_assets_tags        ON ul_assets USING gin (tags);

-- Composite index for common filter patterns
CREATE INDEX idx_ul_assets_tenant_modality
    ON ul_assets (tenant_id, modality);

-- --------------------------------------------------------------------------
-- 7. Edge / graph table
-- --------------------------------------------------------------------------

CREATE TABLE ul_edges (
    edge_id         BIGSERIAL PRIMARY KEY,
    src_vector_id   BIGINT NOT NULL REFERENCES ul_assets(vector_id),
    dst_vector_id   BIGINT NOT NULL REFERENCES ul_assets(vector_id),
    relation        ul_edge_relation NOT NULL,
    weight          REAL NOT NULL DEFAULT 1.0,
    proof_id        TEXT NOT NULL,
    created_by      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_no_self_loop CHECK (src_vector_id <> dst_vector_id),
    CONSTRAINT chk_weight_range CHECK (weight >= 0 AND weight <= 1)
);

CREATE INDEX idx_ul_edges_src      ON ul_edges (src_vector_id);
CREATE INDEX idx_ul_edges_dst      ON ul_edges (dst_vector_id);
CREATE INDEX idx_ul_edges_relation ON ul_edges (relation);
CREATE INDEX idx_ul_edges_proof    ON ul_edges (proof_id);

-- Prevent duplicate edges of the same relation
CREATE UNIQUE INDEX idx_ul_edges_unique_relation
    ON ul_edges (src_vector_id, dst_vector_id, relation);

-- --------------------------------------------------------------------------
-- 8. Witness / audit log
-- --------------------------------------------------------------------------

CREATE TABLE ul_witness_log (
    witness_id      BIGSERIAL PRIMARY KEY,
    action          ul_witness_action NOT NULL,
    actor           TEXT NOT NULL,
    target_id       BIGINT,
    tenant_id       TEXT,
    branch_id       TEXT,
    detail          JSONB,
    receipt_hash    TEXT NOT NULL,
    prev_hash       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_ul_witness_action     ON ul_witness_log (action);
CREATE INDEX idx_ul_witness_tenant     ON ul_witness_log (tenant_id);
CREATE INDEX idx_ul_witness_branch     ON ul_witness_log (branch_id);
CREATE INDEX idx_ul_witness_created    ON ul_witness_log (created_at);
CREATE INDEX idx_ul_witness_target     ON ul_witness_log (target_id);

-- --------------------------------------------------------------------------
-- 9. Governance policy table
-- --------------------------------------------------------------------------

CREATE TABLE ul_governance_policies (
    policy_id               TEXT PRIMARY KEY,
    branch_id               TEXT NOT NULL REFERENCES ul_branches(branch_id),
    mode                    ul_governance_mode NOT NULL,
    allowed_tools           TEXT[] DEFAULT '{}',
    denied_tools            TEXT[] DEFAULT '{}',
    max_cost_microdollars   INTEGER NOT NULL DEFAULT 0,
    max_tool_calls          INTEGER NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------------
-- 10. Hot-set tracking (temperature tiering mirror)
-- --------------------------------------------------------------------------

CREATE TABLE ul_hotset (
    vector_id       BIGINT PRIMARY KEY REFERENCES ul_assets(vector_id),
    read_count      BIGINT NOT NULL DEFAULT 0,
    last_read_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    promoted_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==========================================================================
-- FUNCTIONS
-- ==========================================================================

-- --------------------------------------------------------------------------
-- F1. Ingest asset with witness receipt
-- --------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION ul_ingest_asset(
    p_asset_id          TEXT,
    p_tenant_id         TEXT,
    p_modality          ul_modality,
    p_source_uri        TEXT,
    p_sha256            TEXT,
    p_encoder_id        TEXT,
    p_prior_id          TEXT,
    p_decoder_id        TEXT,
    p_latent_dim        INTEGER,
    p_noise_sigma0      REAL,
    p_bitrate_bound     REAL,
    p_search_quant      ul_search_quant,
    p_safety_class      ul_safety_class,
    p_z_search          vector,
    p_z0_archive        vector DEFAULT NULL,
    p_branch_id         TEXT DEFAULT 'main',
    p_tags              TEXT[] DEFAULT '{}'
)
RETURNS TABLE (vector_id BIGINT, receipt_hash TEXT) AS $$
DECLARE
    v_id        BIGINT;
    v_receipt   TEXT;
    v_prev      TEXT;
BEGIN
    -- Validate dimension
    IF array_length(p_z_search::real[], 1) <> p_latent_dim THEN
        RAISE EXCEPTION 'z_search dimension % does not match latent_dim %',
            array_length(p_z_search::real[], 1), p_latent_dim;
    END IF;

    -- Validate decoder exists in registry
    IF NOT EXISTS (
        SELECT 1 FROM ul_decoder_registry
        WHERE decoder_id = p_decoder_id AND branch_id = p_branch_id
    ) THEN
        RAISE WARNING 'decoder_id % not in registry for branch %',
            p_decoder_id, p_branch_id;
    END IF;

    -- Insert asset
    INSERT INTO ul_assets (
        asset_id, tenant_id, modality, source_uri, sha256,
        encoder_id, prior_id, decoder_id, latent_dim,
        noise_sigma0, bitrate_upper_bound,
        search_quant, safety_class,
        z_search, z0_archive, branch_id, tags
    ) VALUES (
        p_asset_id, p_tenant_id, p_modality, p_source_uri, p_sha256,
        p_encoder_id, p_prior_id, p_decoder_id, p_latent_dim,
        p_noise_sigma0, p_bitrate_bound,
        p_search_quant, p_safety_class,
        p_z_search, p_z0_archive, p_branch_id, p_tags
    )
    RETURNING ul_assets.vector_id INTO v_id;

    -- Get previous witness hash for chain
    SELECT receipt_hash INTO v_prev
    FROM ul_witness_log
    WHERE tenant_id = p_tenant_id
    ORDER BY witness_id DESC
    LIMIT 1;

    -- Compute receipt hash
    v_receipt := encode(
        sha256(convert_to(p_asset_id || ':' || v_id::text, 'UTF8')),
        'hex'
    );

    -- Append witness
    INSERT INTO ul_witness_log (
        action, actor, target_id, tenant_id, branch_id,
        detail, receipt_hash, prev_hash
    ) VALUES (
        'ingest', current_user, v_id, p_tenant_id, p_branch_id,
        jsonb_build_object(
            'asset_id', p_asset_id,
            'encoder_id', p_encoder_id,
            'decoder_id', p_decoder_id,
            'latent_dim', p_latent_dim,
            'noise_sigma0', p_noise_sigma0,
            'bitrate_bound', p_bitrate_bound,
            'search_quant', p_search_quant::text,
            'safety_class', p_safety_class::text
        ),
        v_receipt, v_prev
    );

    RETURN QUERY SELECT v_id, v_receipt;
END;
$$ LANGUAGE plpgsql;

-- --------------------------------------------------------------------------
-- F2. Add proof-gated graph edge
-- --------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION ul_add_edge(
    p_src_vector_id     BIGINT,
    p_dst_vector_id     BIGINT,
    p_relation          ul_edge_relation,
    p_weight            REAL,
    p_created_by        TEXT
)
RETURNS TABLE (edge_id BIGINT, proof_id TEXT) AS $$
DECLARE
    v_edge_id   BIGINT;
    v_proof     TEXT;
    v_prev      TEXT;
BEGIN
    -- Verify both vectors exist
    IF NOT EXISTS (SELECT 1 FROM ul_assets WHERE vector_id = p_src_vector_id) THEN
        RAISE EXCEPTION 'src_vector_id % does not exist', p_src_vector_id;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM ul_assets WHERE vector_id = p_dst_vector_id) THEN
        RAISE EXCEPTION 'dst_vector_id % does not exist', p_dst_vector_id;
    END IF;

    -- Compute proof ID
    v_proof := encode(
        sha256(convert_to(
            p_src_vector_id::text || ':' || p_dst_vector_id::text || ':' || p_relation::text,
            'UTF8'
        )),
        'hex'
    );

    -- Insert edge
    INSERT INTO ul_edges (
        src_vector_id, dst_vector_id, relation, weight, proof_id, created_by
    ) VALUES (
        p_src_vector_id, p_dst_vector_id, p_relation, p_weight, v_proof, p_created_by
    )
    RETURNING ul_edges.edge_id INTO v_edge_id;

    -- Witness
    SELECT receipt_hash INTO v_prev
    FROM ul_witness_log
    ORDER BY witness_id DESC LIMIT 1;

    INSERT INTO ul_witness_log (
        action, actor, target_id,
        detail, receipt_hash, prev_hash
    ) VALUES (
        'edge_add', p_created_by, v_edge_id,
        jsonb_build_object(
            'src', p_src_vector_id,
            'dst', p_dst_vector_id,
            'relation', p_relation::text,
            'weight', p_weight
        ),
        v_proof, v_prev
    );

    RETURN QUERY SELECT v_edge_id, v_proof;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- EXAMPLE QUERIES — matching ADR-058 search policy table
-- ==========================================================================

-- --------------------------------------------------------------------------
-- Q1. Interactive image search (ef_search=64, tenant + modality filter)
-- --------------------------------------------------------------------------

-- Replace :query_vector with the UL-encoded query latent.
/*
SELECT
    a.vector_id,
    a.asset_id,
    a.modality,
    a.z_search <=> :query_vector AS distance,
    a.tags
FROM ul_assets a
WHERE a.tenant_id = :tenant_id
  AND a.modality  = 'image'
ORDER BY a.z_search <=> :query_vector
LIMIT 10;
*/

-- With explicit ef_search (session-level GUC):
-- SET LOCAL hnsw.ef_search = 64;

-- --------------------------------------------------------------------------
-- Q2. Compliance audit (ef_search=128, branch + time range, witness on)
-- --------------------------------------------------------------------------

/*
SET LOCAL hnsw.ef_search = 128;

SELECT
    a.vector_id,
    a.asset_id,
    a.encoder_id,
    a.decoder_id,
    a.z_search <=> :query_vector AS distance,
    a.created_at
FROM ul_assets a
WHERE a.branch_id   = :branch_id
  AND a.created_at >= :start_time
  AND a.created_at <= :end_time
ORDER BY a.z_search <=> :query_vector
LIMIT 20;

-- Log audit witness
INSERT INTO ul_witness_log (
    action, actor, detail, receipt_hash
) VALUES (
    'query', current_user,
    jsonb_build_object('type', 'compliance_audit', 'branch', :branch_id),
    encode(sha256(convert_to('audit:' || :branch_id, 'UTF8')), 'hex')
);
*/

-- --------------------------------------------------------------------------
-- Q3. Reconstruction preview (ef_search=96, decoder_id + safety_class)
-- --------------------------------------------------------------------------

/*
SET LOCAL hnsw.ef_search = 96;

SELECT
    a.vector_id,
    a.asset_id,
    a.z0_archive,       -- Full-precision archive latent for decoder
    a.decoder_id,
    a.z_search <=> :query_vector AS distance
FROM ul_assets a
WHERE a.decoder_id    = :decoder_id
  AND a.safety_class  = 'public'
  AND a.z0_archive IS NOT NULL
ORDER BY a.z_search <=> :query_vector
LIMIT 5;
*/

-- --------------------------------------------------------------------------
-- Q4. Edge offline (ef_search=32, tenant only, prefer latency)
-- --------------------------------------------------------------------------

/*
SET LOCAL hnsw.ef_search = 32;

SELECT
    a.vector_id,
    a.asset_id,
    a.z_search <=> :query_vector AS distance
FROM ul_assets a
WHERE a.tenant_id = :tenant_id
ORDER BY a.z_search <=> :query_vector
LIMIT 5;
*/

-- --------------------------------------------------------------------------
-- Q5. Near-duplicate detection with graph context
-- --------------------------------------------------------------------------

/*
SELECT
    a.vector_id,
    a.asset_id,
    a.z_search <=> :query_vector AS distance,
    e.relation,
    e.weight AS edge_weight
FROM ul_assets a
LEFT JOIN ul_edges e ON e.dst_vector_id = a.vector_id
                    AND e.relation = 'near_duplicate'
WHERE a.tenant_id = :tenant_id
  AND a.z_search <=> :query_vector < 0.15
ORDER BY a.z_search <=> :query_vector
LIMIT 20;
*/

-- --------------------------------------------------------------------------
-- Q6. Witness chain integrity verification
-- --------------------------------------------------------------------------

/*
SELECT
    w1.witness_id,
    w1.action,
    w1.receipt_hash,
    w1.prev_hash,
    w2.receipt_hash AS expected_prev,
    CASE
        WHEN w1.prev_hash IS NULL AND w2.receipt_hash IS NULL THEN 'VALID (genesis)'
        WHEN w1.prev_hash = w2.receipt_hash THEN 'VALID'
        ELSE 'BROKEN CHAIN'
    END AS chain_status
FROM ul_witness_log w1
LEFT JOIN ul_witness_log w2 ON w2.witness_id = w1.witness_id - 1
ORDER BY w1.witness_id;
*/

-- --------------------------------------------------------------------------
-- Q7. Branch lineage query (find all child branches)
-- --------------------------------------------------------------------------

/*
WITH RECURSIVE branch_tree AS (
    SELECT branch_id, parent_id, encoder_id, created_at, 0 AS depth
    FROM ul_branches
    WHERE branch_id = :root_branch_id

    UNION ALL

    SELECT b.branch_id, b.parent_id, b.encoder_id, b.created_at, bt.depth + 1
    FROM ul_branches b
    INNER JOIN branch_tree bt ON b.parent_id = bt.branch_id
)
SELECT * FROM branch_tree
ORDER BY depth, created_at;
*/

-- --------------------------------------------------------------------------
-- Q8. Hot-set promotion (move frequently-read vectors to hot tier)
-- --------------------------------------------------------------------------

/*
INSERT INTO ul_hotset (vector_id, read_count, last_read_at)
SELECT vector_id, read_count, last_read_at
FROM (
    SELECT
        a.vector_id,
        count(*) AS read_count,
        max(w.created_at) AS last_read_at
    FROM ul_witness_log w
    INNER JOIN ul_assets a ON a.vector_id = w.target_id
    WHERE w.action = 'query'
      AND w.created_at > now() - INTERVAL '24 hours'
    GROUP BY a.vector_id
    HAVING count(*) >= 100
) hot
ON CONFLICT (vector_id) DO UPDATE
SET read_count  = EXCLUDED.read_count,
    last_read_at = EXCLUDED.last_read_at;
*/

-- --------------------------------------------------------------------------
-- Q9. Governance policy enforcement view
-- --------------------------------------------------------------------------

CREATE OR REPLACE VIEW ul_active_policies AS
SELECT
    p.policy_id,
    p.branch_id,
    p.mode,
    p.allowed_tools,
    p.denied_tools,
    p.max_cost_microdollars,
    p.max_tool_calls,
    b.encoder_id,
    b.frozen_at IS NOT NULL AS branch_frozen
FROM ul_governance_policies p
INNER JOIN ul_branches b ON b.branch_id = p.branch_id;

-- --------------------------------------------------------------------------
-- Q10. Asset statistics per tenant and branch
-- --------------------------------------------------------------------------

CREATE OR REPLACE VIEW ul_asset_stats AS
SELECT
    tenant_id,
    branch_id,
    modality,
    safety_class,
    encoder_id,
    count(*)                            AS asset_count,
    avg(noise_sigma0)                   AS avg_noise,
    avg(bitrate_upper_bound)            AS avg_bitrate,
    min(created_at)                     AS earliest,
    max(created_at)                     AS latest
FROM ul_assets
GROUP BY tenant_id, branch_id, modality, safety_class, encoder_id;
