-- RuVector-Postgres Initialization Script
-- Creates extension and test tables

-- Create the extension
CREATE EXTENSION IF NOT EXISTS ruvector;

-- Create test schema
CREATE SCHEMA IF NOT EXISTS ruvector_test;

-- Test table for vectors
CREATE TABLE ruvector_test.vectors (
    id SERIAL PRIMARY KEY,
    embedding vector(768),
    sparse_embedding sparsevec(30000),
    category TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Test table for graph nodes
CREATE TABLE ruvector_test.nodes (
    id SERIAL PRIMARY KEY,
    label TEXT NOT NULL,
    embedding vector(256),
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Test table for graph edges
CREATE TABLE ruvector_test.edges (
    id SERIAL PRIMARY KEY,
    src_id INTEGER REFERENCES ruvector_test.nodes(id),
    dst_id INTEGER REFERENCES ruvector_test.nodes(id),
    edge_type TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Test table for learning trajectories
CREATE TABLE ruvector_test.trajectories (
    id SERIAL PRIMARY KEY,
    query_vector vector(768),
    result_ids INTEGER[],
    latency_ms FLOAT,
    recall_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Test table for routing agents
CREATE TABLE ruvector_test.agents (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    agent_type TEXT NOT NULL,
    capabilities TEXT[],
    capability_embedding vector(768),
    cost_per_1k_tokens FLOAT,
    avg_latency_ms FLOAT,
    quality_score FLOAT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes (will be created after extension functions are available)
-- These are placeholder comments for test setup

-- Grant permissions
GRANT ALL ON SCHEMA ruvector_test TO ruvector;
GRANT ALL ON ALL TABLES IN SCHEMA ruvector_test TO ruvector;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ruvector_test TO ruvector;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'RuVector-Postgres initialized successfully';
    RAISE NOTICE 'Extension version: %', (SELECT ruvector_version());
    RAISE NOTICE 'SIMD info: %', (SELECT ruvector_simd_info());
END $$;
