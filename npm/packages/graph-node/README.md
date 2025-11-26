# @ruvector/graph-node

Native Node.js bindings for RuVector Graph Database with hypergraph support.

## Features

- **Native Performance**: 10x faster than WASM with zero-copy buffer sharing
- **Hypergraph Support**: Multi-entity relationships beyond traditional pairwise graphs
- **Cypher-like Queries**: Familiar query syntax for graph traversal
- **Async/Await**: Full async support with thread-safe operations
- **Transaction Support**: ACID transactions with begin/commit/rollback
- **Batch Operations**: Efficient bulk loading of nodes and edges
- **Vector Similarity**: Built-in semantic search capabilities
- **Streaming Results**: AsyncIterator pattern for large result sets

## Installation

```bash
npm install @ruvector/graph-node
```

## Quick Start

```javascript
const { GraphDatabase } = require('@ruvector/graph-node');

// Create a new graph database
const db = new GraphDatabase({
  distanceMetric: 'Cosine',
  dimensions: 384
});

// Create nodes
await db.createNode({
  id: 'alice',
  embedding: new Float32Array([0.1, 0.2, 0.3]),
  properties: { name: 'Alice', age: '30' }
});

await db.createNode({
  id: 'bob',
  embedding: new Float32Array([0.2, 0.3, 0.4]),
  properties: { name: 'Bob', age: '25' }
});

// Create an edge
await db.createEdge({
  from: 'alice',
  to: 'bob',
  description: 'knows',
  embedding: new Float32Array([0.15, 0.25, 0.35]),
  confidence: 0.95
});

// Query the graph
const results = await db.query('MATCH (n) RETURN n');
console.log('Query results:', results);

// Search for similar relationships
const similar = await db.searchHyperedges({
  embedding: new Float32Array([0.1, 0.2, 0.3]),
  k: 10
});
```

## Hypergraph Example

```javascript
// Create a hyperedge connecting multiple entities
await db.createHyperedge({
  nodes: ['alice', 'bob', 'charlie'],
  description: 'collaborated_on_project',
  embedding: new Float32Array([0.3, 0.6, 0.9]),
  confidence: 0.85,
  metadata: { project: 'AI Research' }
});

// Find k-hop neighbors
const neighbors = await db.kHopNeighbors('alice', 2);
console.log('2-hop neighbors:', neighbors);
```

## Transaction Example

```javascript
// Begin a transaction
const txId = await db.begin();

try {
  await db.createNode({
    id: 'node1',
    embedding: new Float32Array([1, 2, 3])
  });

  await db.createEdge({
    from: 'node1',
    to: 'node2',
    description: 'relates_to',
    embedding: new Float32Array([1.5, 2.5, 3.5])
  });

  // Commit the transaction
  await db.commit(txId);
} catch (error) {
  // Rollback on error
  await db.rollback(txId);
  throw error;
}
```

## Batch Operations

```javascript
// Efficient bulk loading
const result = await db.batchInsert({
  nodes: [
    { id: 'n1', embedding: new Float32Array([1, 2]) },
    { id: 'n2', embedding: new Float32Array([3, 4]) },
    { id: 'n3', embedding: new Float32Array([5, 6]) }
  ],
  edges: [
    {
      from: 'n1',
      to: 'n2',
      description: 'connects',
      embedding: new Float32Array([2, 3])
    },
    {
      from: 'n2',
      to: 'n3',
      description: 'links',
      embedding: new Float32Array([4, 5])
    }
  ]
});

console.log('Inserted:', result.nodeIds, result.edgeIds);
```

## Statistics

```javascript
const stats = await db.stats();
console.log(`
  Total Nodes: ${stats.totalNodes}
  Total Edges: ${stats.totalEdges}
  Average Degree: ${stats.avgDegree}
`);
```

## API Reference

See [index.d.ts](./index.d.ts) for complete TypeScript definitions.

## Performance

- **Native Speed**: 10x faster than WASM implementation
- **Zero-Copy**: Direct buffer sharing between Rust and Node.js
- **Thread-Safe**: Concurrent operations with RwLock
- **Async Runtime**: Tokio-powered async execution

## Platform Support

- Linux (x64, ARM64)
- macOS (x64, ARM64 / Apple Silicon)
- Windows (x64)

## License

MIT

## Links

- [Documentation](https://ruv.io/docs)
- [GitHub](https://github.com/ruvnet/ruvector)
- [Issues](https://github.com/ruvnet/ruvector/issues)
