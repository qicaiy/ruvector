# Architecture Documentation

## System Overview

The `ruvector-mincut` crate implements a sophisticated multi-layered architecture for maintaining minimum cuts in dynamic graphs. The design prioritizes both theoretical efficiency (subpolynomial update time) and practical performance (low constants, cache-friendly data structures).

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Public API Layer                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  DynamicMinCut                                       │    │
│  │  - insert_edge() / delete_edge()                     │    │
│  │  - min_cut_value() / min_cut() / partition()        │    │
│  │  - Thread-safe operations via Arc<RwLock<>>         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Decomposition Layer                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  HierarchicalDecomposition                          │    │
│  │  - Balanced binary tree (O(log n) height)           │    │
│  │  - Lazy recomputation (dirty marking)               │    │
│  │  - LCA-based update localization                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                Dynamic Connectivity Layer                    │
│  ┌───────────────────────┐  ┌──────────────────────────┐   │
│  │  LinkCutTree           │  │  EulerTourTree           │   │
│  │  - Splay-based         │  │  - Treap-based           │   │
│  │  - Path aggregates     │  │  - Implicit keys         │   │
│  │  - O(log n) amortized  │  │  - Subtree aggregates    │   │
│  └───────────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Sparsification Layer                        │
│  ┌───────────────────────┐  ┌──────────────────────────┐   │
│  │  Benczúr-Karger        │  │  Nagamochi-Ibaraki       │   │
│  │  - Randomized sampling │  │  - Deterministic         │   │
│  │  - Edge strength       │  │  - k-certificate         │   │
│  │  - (1+ε) guarantee     │  │  - Min-degree ordering   │   │
│  └───────────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  DynamicGraph                                        │    │
│  │  - DashMap for concurrent access                     │    │
│  │  - Adjacency list representation                     │    │
│  │  - Edge index for O(1) lookup                        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Graph Module (`graph/mod.rs`)

**Purpose**: Thread-safe graph storage with efficient operations.

**Key Components**:
- `DynamicGraph`: Main graph structure
- `Edge`: Weighted edge representation
- `GraphStats`: Graph statistics

**Data Structures**:
```rust
pub struct DynamicGraph {
    adjacency: DashMap<VertexId, HashSet<(VertexId, EdgeId)>>,
    edges: DashMap<EdgeId, Edge>,
    edge_index: DashMap<(VertexId, VertexId), EdgeId>,
    next_edge_id: AtomicU64,
    num_vertices: AtomicUsize,
}
```

**Thread Safety**:
- Uses `DashMap` for lock-free concurrent reads
- Atomic counters for IDs
- Canonical edge keys `(min(u,v), max(u,v))` prevent duplicates

**Operations**:
- `insert_edge()`: O(1) average
- `delete_edge()`: O(1) average
- `has_edge()`: O(1) average
- `neighbors()`: O(deg(v))
- `is_connected()`: O(n + m) via BFS

### 2. Algorithm Module (`algorithm/mod.rs`)

**Purpose**: Core minimum cut algorithm implementation.

**Key Components**:
- `DynamicMinCut`: Main algorithm structure
- `MinCutBuilder`: Builder pattern for configuration
- `MinCutResult`: Query results with partition info
- `AlgorithmStats`: Performance metrics

**State Management**:
```rust
pub struct DynamicMinCut {
    graph: Arc<RwLock<DynamicGraph>>,
    decomposition: HierarchicalDecomposition,
    link_cut_tree: LinkCutTree,
    spanning_forest: EulerTourTree,
    current_min_cut: f64,
    config: MinCutConfig,
    stats: Arc<RwLock<AlgorithmStats>>,
    tree_edges: Arc<RwLock<HashSet<(VertexId, VertexId)>>>,
}
```

**Edge Classification**:
- **Tree edges**: In the spanning forest (tracked in `tree_edges`)
- **Non-tree edges**: Create cycles, don't affect connectivity

**Update Strategy**:
1. Classify edge as tree or non-tree
2. Update spanning forest if needed
3. Mark affected decomposition nodes dirty
4. Recompute minimum cut (lazy)

### 3. Link-Cut Tree Module (`linkcut/mod.rs`)

**Purpose**: Dynamic tree operations using Sleator-Tarjan's Link-Cut Trees.

**Algorithm**: Splay trees with path-parent pointers for representing preferred paths.

**Key Operations**:
```rust
impl LinkCutTree {
    pub fn make_tree(&mut self, id: NodeId, value: f64) -> usize;
    pub fn link(&mut self, u: NodeId, v: NodeId) -> Result<()>;
    pub fn cut(&mut self, v: NodeId) -> Result<()>;
    pub fn connected(&mut self, u: NodeId, v: NodeId) -> bool;
    pub fn find_root(&mut self, v: NodeId) -> Result<NodeId>;
    pub fn path_aggregate(&mut self, v: NodeId) -> Result<f64>;
}
```

**Complexity**:
- All operations: **O(log n) amortized**
- Splay tree depth: O(log n) amortized

**Internal Structure**:
```rust
struct SplayNode {
    id: NodeId,
    parent: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    path_parent: Option<usize>,
    size: usize,
    value: f64,
    path_aggregate: f64,
    reversed: bool,
}
```

**Key Techniques**:
- **Access operation**: Makes root-to-node path preferred
- **Splay operation**: Rotates node to root (zig, zig-zig, zig-zag)
- **Path aggregates**: Minimum value on root-to-node path
- **Lazy propagation**: Deferred reversal operations

### 4. Euler Tour Tree Module (`euler/mod.rs`)

**Purpose**: Alternative dynamic tree structure using Euler tours.

**Algorithm**: Treap (randomized BST) storing Euler tour sequence.

**Key Insight**: An Euler tour of a tree visits each edge twice (once entering, once exiting a subtree). Store this tour in a balanced BST with implicit positions.

**Operations**:
```rust
impl EulerTourTree {
    pub fn make_tree(&mut self, v: NodeId) -> Result<()>;
    pub fn link(&mut self, u: NodeId, v: NodeId) -> Result<()>;
    pub fn cut(&mut self, u: NodeId, v: NodeId) -> Result<()>;
    pub fn connected(&self, u: NodeId, v: NodeId) -> bool;
    pub fn tree_size(&self, v: NodeId) -> Result<usize>;
    pub fn reroot(&mut self, v: NodeId) -> Result<()>;
}
```

**Data Structure**:
```rust
struct TreapNode {
    vertex: NodeId,
    priority: u64,
    left: Option<usize>,
    right: Option<usize>,
    parent: Option<usize>,
    size: usize,
    value: f64,
    subtree_aggregate: f64,
}
```

**Key Techniques**:
- **Split**: O(log n) - split tour at position
- **Merge**: O(log n) - merge two tours
- **Reroot**: Rotate tour to make vertex first
- **Link**: Merge tours with edge occurrences
- **Cut**: Split out edge occurrences

### 5. Hierarchical Decomposition Module (`tree/mod.rs`)

**Purpose**: Balanced binary tree over vertices for efficient cut queries.

**Structure**:
```rust
pub struct HierarchicalDecomposition {
    nodes: Vec<DecompositionNode>,
    vertex_to_leaf: HashMap<VertexId, usize>,
    root: Option<usize>,
    min_cut: f64,
    height: usize,
    graph: Arc<DynamicGraph>,
}

pub struct DecompositionNode {
    id: usize,
    level: usize,
    vertices: HashSet<VertexId>,
    parent: Option<usize>,
    children: Vec<usize>,
    cut_value: f64,
    dirty: bool,
}
```

**Key Algorithms**:

**1. Build (O(n log n))**:
```
build_hierarchy():
  1. Create leaf for each vertex
  2. Recursively partition into balanced halves
  3. Create internal nodes for each partition
  4. Height = O(log n)
```

**2. Query (O(1))**:
```
min_cut_value():
  return cached self.min_cut
```

**3. Update (O(log n) nodes × O(m/n) per node)**:
```
insert_edge(u, v):
  1. Find LCA of u and v
  2. Mark LCA and ancestors dirty
  3. Recompute dirty nodes bottom-up
  4. Update global minimum
```

**Cut Computation**:
Each node represents a potential partition:
- S = vertices in this subtree
- T = all other vertices
- Cut value = sum of edge weights crossing S ↔ T

**Limitations**: May not find true minimum cut if it doesn't align with tree partitioning. This is why we also use Link-Cut Trees and maintain a spanning forest.

### 6. Sparsification Module (`sparsify/mod.rs`)

**Purpose**: Graph sparsification for approximate minimum cuts.

#### Benczúr-Karger Algorithm

**Idea**: Sample edges with probability inversely proportional to their "strength".

**Algorithm**:
```
benczur_karger_sparsify(G, ε):
  For each edge e:
    1. Compute strength λ_e (approximate max-flow)
    2. Sample e with probability p_e = min(1, c·log(n) / (ε²·λ_e))
    3. If sampled, scale weight: w'_e = w_e / p_e
  Return sparse graph G'
```

**Result**: G' has O(n log n / ε²) edges and preserves all cuts within (1±ε).

**Implementation**:
```rust
pub struct SparseGraph {
    graph: DynamicGraph,
    edge_weights: HashMap<EdgeId, Weight>,
    epsilon: f64,
    original_edges: usize,
    rng: StdRng,
    strength_calc: EdgeStrength,
}
```

**Edge Strength Approximation**:
```rust
impl EdgeStrength {
    // Approximate strength using local connectivity
    pub fn compute(&mut self, u: VertexId, v: VertexId) -> f64 {
        let weight_u: f64 = neighbors(u).sum_weights();
        let weight_v: f64 = neighbors(v).sum_weights();
        weight_u.min(weight_v).max(1.0)
    }
}
```

#### Nagamochi-Ibaraki Algorithm

**Idea**: Deterministic sparsification using minimum degree ordering.

**Algorithm**:
```
nagamochi_ibaraki(G, k):
  1. Compute minimum degree ordering
  2. For each vertex v in reverse order:
     - Scan connectivity to already-scanned vertices
     - Mark edges with connectivity ≥ k
  3. Return subgraph with marked edges
```

**Result**: O(kn) edges preserve all cuts up to size k.

**Use Case**: When deterministic guarantees are needed or for small k.

### 7. Monitoring Module (`monitoring/mod.rs`)

**Purpose**: Real-time event-driven monitoring of minimum cut changes.

**Architecture**:
```rust
pub struct MinCutMonitor {
    callbacks: RwLock<Vec<CallbackEntry>>,
    thresholds: RwLock<Vec<Threshold>>,
    metrics: RwLock<MonitorMetrics>,
    current_cut: RwLock<f64>,
    config: MonitorConfig,
}
```

**Event Types**:
- `CutIncreased` / `CutDecreased`: Basic cut changes
- `ThresholdCrossedBelow` / `ThresholdCrossedAbove`: Threshold violations
- `Disconnected` / `Connected`: Connectivity changes
- `EdgeInserted` / `EdgeDeleted`: Edge operations

**Callback Mechanism**:
```rust
pub fn notify(&self, old_value: f64, new_value: f64, edge: Option<(u64, u64)>) {
    // 1. Determine event types
    // 2. Check threshold crossings
    // 3. Fire callbacks (non-blocking, panic-safe)
    // 4. Update metrics
}
```

**Thread Safety**:
- Callbacks are `Send + Sync`
- Uses `std::panic::catch_unwind` to prevent callback panics from crashing
- Lock-free reads for current cut value

## Data Flow

### Edge Insertion Flow

```
insert_edge(u, v, weight)
  │
  ├─→ DynamicGraph::insert_edge()
  │   └─→ Update adjacency lists, edge index
  │
  ├─→ Check if u, v in different components
  │   │
  │   ├─→ Yes (bridge edge):
  │   │   └─→ LinkCutTree::link()
  │   │   └─→ EulerTourTree::link()
  │   │   └─→ Add to tree_edges
  │   │
  │   └─→ No (cycle edge):
  │       └─→ Skip connectivity update
  │
  ├─→ HierarchicalDecomposition::insert_edge()
  │   └─→ Find LCA(u, v)
  │   └─→ Mark LCA → root path dirty
  │
  ├─→ Recompute minimum cut
  │   └─→ Recompute dirty subtrees bottom-up
  │   └─→ Find global minimum
  │
  └─→ Return new minimum cut value
```

### Edge Deletion Flow

```
delete_edge(u, v)
  │
  ├─→ Check if tree edge
  │   │
  │   ├─→ Yes:
  │   │   ├─→ LinkCutTree::cut()
  │   │   ├─→ Find replacement edge
  │   │   │   └─→ BFS in one component
  │   │   │   └─→ Check for non-tree edges crossing
  │   │   └─→ If found: link replacement
  │   │       Else: graph disconnected
  │   │
  │   └─→ No: Skip connectivity update
  │
  ├─→ DynamicGraph::delete_edge()
  │   └─→ Update adjacency lists
  │
  ├─→ HierarchicalDecomposition::delete_edge()
  │   └─→ Mark affected path dirty
  │
  └─→ Recompute minimum cut
```

### Query Flow

```
min_cut_value()
  │
  └─→ Return cached current_min_cut (O(1))

min_cut()
  │
  ├─→ Get min_cut_value()
  ├─→ Get partition()
  │   └─→ Find node with minimum cut
  │   └─→ Return (node.vertices, complement)
  ├─→ Get cut_edges()
  │   └─→ For each vertex in S:
  │       └─→ Find neighbors in T
  │       └─→ Collect crossing edges
  │
  └─→ Return MinCutResult {
          value, cut_edges, partition,
          is_exact, approximation_ratio
      }
```

## Thread Safety Guarantees

### Concurrent Read Access

- Multiple threads can query `min_cut_value()` concurrently
- Graph statistics (`num_vertices()`, `num_edges()`) are thread-safe
- Connectivity queries use read locks

### Exclusive Write Access

- Edge insertions/deletions require exclusive access
- Uses `RwLock` for graph and statistics
- Interior mutability via `DashMap` for graph storage

### Lock Ordering

To prevent deadlocks, locks are acquired in this order:
1. `graph` (if needed)
2. `decomposition` (internal)
3. `stats` (short-lived, always last)

### Atomic Operations

- `next_edge_id`: `AtomicU64`
- `num_vertices`: `AtomicUsize`
- No ABA problems due to monotonic IDs

## Memory Management

### Arena Allocation

Both Link-Cut Trees and Euler Tour Trees use arena allocation:

```rust
pub struct LinkCutTree {
    nodes: Vec<SplayNode>,           // Arena
    id_to_index: HashMap<NodeId, usize>,
    index_to_id: Vec<NodeId>,
}

pub struct EulerTourTree {
    nodes: Vec<TreapNode>,           // Arena
    free_list: Vec<usize>,           // Recycled indices
}
```

**Benefits**:
- Cache-friendly: nodes stored contiguously
- No allocation overhead for rotations
- Predictable memory usage

### Graph Storage

```rust
pub struct DynamicGraph {
    adjacency: DashMap<VertexId, HashSet<(VertexId, EdgeId)>>,
    edges: DashMap<EdgeId, Edge>,
}
```

**Memory**: O(n + m) where n = vertices, m = edges

### Decomposition Storage

```rust
pub struct HierarchicalDecomposition {
    nodes: Vec<DecompositionNode>,   // 2n - 1 nodes for n leaves
}
```

**Memory**: O(n) nodes, each storing O(n) vertices in worst case = O(n²) total

**Optimization**: Use compressed bitmaps for large vertex sets.

## Performance Optimizations

### 1. Lazy Recomputation

Only recompute decomposition nodes when queried after being marked dirty.

### 2. Path Compression

Link-Cut Trees use splay operations for amortized O(log n) depth.

### 3. Amortized Analysis

Both Link-Cut Trees and Euler Tour Trees achieve O(log n) amortized via:
- Potential function Φ = sum of node depths
- Splay/treap rotations reduce potential

### 4. Cache Efficiency

- Arena allocation for trees
- Contiguous storage in `Vec`
- Small node sizes (64-128 bytes)

### 5. SIMD (Optional)

When `simd` feature is enabled:
- Parallel edge weight summation
- Vectorized aggregate computation
- Uses `ruvector-core` SIMD primitives

## Complexity Analysis

### Space Complexity

| Component | Space |
|-----------|-------|
| Graph | O(n + m) |
| Link-Cut Tree | O(n) |
| Euler Tour Tree | O(n) |
| Decomposition | O(n log n) |
| **Total** | **O(n log n + m)** |

### Time Complexity

| Operation | Exact Mode | Approximate Mode |
|-----------|-----------|------------------|
| Build | O(m log n) | O(m) |
| Insert | O(n^{o(1)}) | O(log n) |
| Delete | O(n^{o(1)}) | O(log n) |
| Query | O(1) | O(1) |

**Note**: n^{o(1)} = n^{O((log n)^{1/4})} for the hierarchical decomposition approach.

## Design Trade-offs

### Link-Cut Trees vs Euler Tour Trees

| Feature | Link-Cut Trees | Euler Tour Trees |
|---------|---------------|------------------|
| Complexity | O(log n) amortized | O(log n) worst-case |
| Operations | Link, cut, path query | Link, cut, subtree query |
| Implementation | Splay trees (complex) | Treaps (simpler) |
| Constants | Lower | Higher |

**Decision**: Use both - Link-Cut Trees for primary connectivity, Euler Tour Trees as backup/validation.

### Exact vs Approximate

| Feature | Exact | Approximate |
|---------|-------|------------|
| Guarantee | Optimal | (1+ε)-approximate |
| Update time | O(n^{o(1)}) | O(log n) |
| Space | O(n log n + m) | O(n log n / ε²) |
| Use case | Small graphs, exact answer needed | Large graphs, approximate ok |

**Decision**: Support both via builder pattern, let user choose.

### Hierarchical Decomposition Limitations

The balanced binary partitioning may miss the true minimum cut if:
- The minimum cut doesn't align with tree partitions
- The graph structure is adversarial to balanced splits

**Mitigation**: Use spanning forest connectivity as ground truth, decomposition as heuristic.

## Future Optimizations

1. **Parallel Decomposition**: Compute subtree cuts in parallel
2. **Incremental Rebalancing**: Maintain better decomposition over time
3. **Compressed Vertex Sets**: Use roaring bitmaps for large sets
4. **GPU Acceleration**: Offload cut computation to GPU
5. **Persistent Data Structures**: Support versioning and time-travel queries

## Conclusion

The architecture balances theoretical guarantees (subpolynomial updates) with practical performance (cache-friendly, parallel, low constants). The modular design allows using different components independently (e.g., just Link-Cut Trees) while the full system provides state-of-the-art minimum cut maintenance.
