# API Reference

Complete API documentation for `ruvector-mincut` with examples and use cases.

## Table of Contents

1. [Core API](#core-api)
2. [Builder Patterns](#builder-patterns)
3. [Monitoring API](#monitoring-api)
4. [Advanced Features](#advanced-features)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)

---

## Core API

### DynamicMinCut

The main structure for maintaining minimum cuts in dynamic graphs.

#### Creation

```rust
use ruvector_mincut::{MinCutBuilder, MinCutConfig};

// Using builder (recommended)
let mincut = MinCutBuilder::new()
    .exact()
    .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0)])
    .build()?;

// Using config directly
let config = MinCutConfig {
    max_exact_cut_size: 1000,
    epsilon: 0.1,
    approximate: false,
    parallel: true,
    cache_size: 10000,
};
let mincut = DynamicMinCut::new(config);
```

#### Methods

##### `insert_edge`

Insert an edge into the graph.

```rust
pub fn insert_edge(&mut self, u: VertexId, v: VertexId, weight: Weight) -> Result<f64>
```

**Parameters**:
- `u`: Source vertex ID
- `v`: Target vertex ID
- `weight`: Edge weight (positive)

**Returns**: New minimum cut value

**Errors**:
- `MinCutError::EdgeExists(u, v)`: Edge already exists
- `MinCutError::InvalidEdge(u, v)`: Self-loop attempted (u == v)

**Example**:
```rust
let mut mincut = MinCutBuilder::new().build()?;

// Insert edges
let cut_value = mincut.insert_edge(1, 2, 1.0)?;
assert_eq!(cut_value, 1.0);

mincut.insert_edge(2, 3, 1.0)?;
mincut.insert_edge(3, 1, 1.0)?;
assert_eq!(mincut.min_cut_value(), 2.0);

// Error: duplicate edge
assert!(mincut.insert_edge(1, 2, 2.0).is_err());

// Error: self-loop
assert!(mincut.insert_edge(1, 1, 1.0).is_err());
```

**Complexity**: O(n^{o(1)}) amortized

##### `delete_edge`

Delete an edge from the graph.

```rust
pub fn delete_edge(&mut self, u: VertexId, v: VertexId) -> Result<f64>
```

**Parameters**:
- `u`, `v`: Endpoints of edge to delete

**Returns**: New minimum cut value

**Errors**:
- `MinCutError::EdgeNotFound(u, v)`: Edge doesn't exist

**Example**:
```rust
let mut mincut = MinCutBuilder::new()
    .with_edges(vec![
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 1, 1.0),
    ])
    .build()?;

assert_eq!(mincut.min_cut_value(), 2.0);

// Delete edge
let new_cut = mincut.delete_edge(1, 2)?;
assert_eq!(new_cut, 1.0);  // Now a path graph

// Error: edge doesn't exist
assert!(mincut.delete_edge(1, 2).is_err());
```

**Complexity**: O(n^{o(1)}) amortized

##### `min_cut_value`

Get the current minimum cut value (O(1) query).

```rust
pub fn min_cut_value(&self) -> f64
```

**Returns**: Minimum cut value (0.0 if disconnected, ∞ if empty graph)

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .with_edges(vec![(1, 2, 5.0), (2, 3, 3.0), (3, 1, 2.0)])
    .build()?;

let min_cut = mincut.min_cut_value();
assert_eq!(min_cut, 5.0);  // Minimum is separating vertex 1
```

**Complexity**: O(1)

##### `min_cut`

Get detailed minimum cut information.

```rust
pub fn min_cut(&self) -> MinCutResult
```

**Returns**: `MinCutResult` with:
- `value`: Minimum cut value
- `cut_edges`: Edges in the cut (Option)
- `partition`: Vertex partition (S, T) (Option)
- `is_exact`: Whether result is exact
- `approximation_ratio`: 1.0 for exact, 1+ε for approximate

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .exact()
    .with_edges(vec![
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ])
    .build()?;

let result = mincut.min_cut();
assert_eq!(result.value, 1.0);
assert!(result.is_exact);
assert_eq!(result.approximation_ratio, 1.0);

if let Some((s, t)) = result.partition {
    println!("Partition S: {:?}", s);
    println!("Partition T: {:?}", t);
    assert_eq!(s.len() + t.len(), 4);
}

if let Some(edges) = result.cut_edges {
    println!("Cut has {} edges", edges.len());
}
```

**Complexity**: O(n + m) to extract partition and edges

##### `partition`

Get the vertex partition of the minimum cut.

```rust
pub fn partition(&self) -> (Vec<VertexId>, Vec<VertexId>)
```

**Returns**: Tuple (S, T) where S and T partition the vertices

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .with_edges(vec![(1, 2, 1.0), (2, 3, 2.0)])
    .build()?;

let (s, t) = mincut.partition();
assert!(!s.is_empty() && !t.is_empty());
assert_eq!(s.len() + t.len(), 3);

// Verify partition
let s_set: HashSet<_> = s.iter().collect();
let t_set: HashSet<_> = t.iter().collect();
assert!(s_set.is_disjoint(&t_set));
```

**Complexity**: O(n)

##### `cut_edges`

Get the edges in the minimum cut.

```rust
pub fn cut_edges(&self) -> Vec<Edge>
```

**Returns**: Vector of edges crossing the cut

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .with_edges(vec![
        (1, 2, 1.0),
        (2, 3, 1.0),
        (1, 3, 1.0),
    ])
    .build()?;

let edges = mincut.cut_edges();
assert_eq!(edges.len(), 2);  // Triangle has min cut 2

let total_weight: f64 = edges.iter().map(|e| e.weight).sum();
assert_eq!(total_weight, mincut.min_cut_value());
```

**Complexity**: O(n + m)

##### `is_connected`

Check if the graph is connected.

```rust
pub fn is_connected(&self) -> bool
```

**Returns**: `true` if graph is connected

**Example**:
```rust
let mut mincut = MinCutBuilder::new()
    .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0)])
    .build()?;

assert!(mincut.is_connected());

// Create disconnection
mincut.insert_edge(4, 5, 1.0)?;
assert!(!mincut.is_connected());
assert_eq!(mincut.min_cut_value(), 0.0);
```

**Complexity**: O(1) (cached result)

##### `graph`

Get reference to underlying graph.

```rust
pub fn graph(&self) -> Arc<RwLock<DynamicGraph>>
```

**Returns**: Arc-wrapped RwLock to graph

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .with_edges(vec![(1, 2, 1.0)])
    .build()?;

let graph = mincut.graph();
let g = graph.read();
assert_eq!(g.num_vertices(), 2);
assert_eq!(g.num_edges(), 1);
```

##### `num_vertices`, `num_edges`

Get graph size.

```rust
pub fn num_vertices(&self) -> usize
pub fn num_edges(&self) -> usize
```

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .with_edges(vec![(1, 2, 1.0), (2, 3, 1.0)])
    .build()?;

assert_eq!(mincut.num_vertices(), 3);
assert_eq!(mincut.num_edges(), 2);
```

##### `stats`

Get algorithm statistics.

```rust
pub fn stats(&self) -> AlgorithmStats
```

**Returns**: Statistics including operation counts and timings

**Example**:
```rust
let mut mincut = MinCutBuilder::new().build()?;

mincut.insert_edge(1, 2, 1.0)?;
mincut.insert_edge(2, 3, 1.0)?;
mincut.delete_edge(1, 2)?;
let _ = mincut.min_cut_value();

let stats = mincut.stats();
assert_eq!(stats.insertions, 2);
assert_eq!(stats.deletions, 1);
assert_eq!(stats.queries, 1);
println!("Avg update time: {:.2} μs", stats.avg_update_time_us);
println!("Avg query time: {:.2} μs", stats.avg_query_time_us);
```

##### `reset_stats`

Reset statistics counters.

```rust
pub fn reset_stats(&mut self)
```

**Example**:
```rust
let mut mincut = MinCutBuilder::new().build()?;
mincut.insert_edge(1, 2, 1.0)?;
assert_eq!(mincut.stats().insertions, 1);

mincut.reset_stats();
assert_eq!(mincut.stats().insertions, 0);
```

##### `config`

Get configuration.

```rust
pub fn config(&self) -> &MinCutConfig
```

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .approximate(0.1)
    .max_cut_size(500)
    .build()?;

let config = mincut.config();
assert_eq!(config.epsilon, 0.1);
assert_eq!(config.max_exact_cut_size, 500);
assert!(config.approximate);
```

---

## Builder Patterns

### MinCutBuilder

Builder for configuring `DynamicMinCut`.

#### Methods

##### `new`

Create a new builder with default configuration.

```rust
pub fn new() -> Self
```

**Example**:
```rust
let builder = MinCutBuilder::new();
```

##### `exact`

Use exact minimum cut algorithm.

```rust
pub fn exact(mut self) -> Self
```

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .exact()
    .build()?;

assert!(mincut.config().approximate == false);
```

##### `approximate`

Use approximate algorithm with given epsilon.

```rust
pub fn approximate(mut self, epsilon: f64) -> Self
```

**Parameters**:
- `epsilon`: Approximation factor (0 < ε ≤ 1)

**Panics**: If epsilon is not in (0, 1]

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .approximate(0.1)  // 10% approximation
    .build()?;

let result = mincut.min_cut();
assert!(!result.is_exact);
assert_eq!(result.approximation_ratio, 1.1);
```

##### `max_cut_size`

Set maximum cut size for exact algorithm.

```rust
pub fn max_cut_size(mut self, size: usize) -> Self
```

**Parameters**:
- `size`: Maximum cut size to handle exactly

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .exact()
    .max_cut_size(1000)  // Handle cuts up to size 1000
    .build()?;

assert_eq!(mincut.config().max_exact_cut_size, 1000);
```

##### `parallel`

Enable or disable parallel computation.

```rust
pub fn parallel(mut self, enabled: bool) -> Self
```

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .parallel(true)
    .build()?;

assert!(mincut.config().parallel);
```

##### `with_edges`

Initialize with a set of edges.

```rust
pub fn with_edges(mut self, edges: Vec<(VertexId, VertexId, Weight)>) -> Self
```

**Parameters**:
- `edges`: Vector of (u, v, weight) tuples

**Example**:
```rust
let mincut = MinCutBuilder::new()
    .with_edges(vec![
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (4, 1, 1.0),
    ])
    .build()?;

assert_eq!(mincut.num_edges(), 4);
```

##### `build`

Build the `DynamicMinCut` structure.

```rust
pub fn build(self) -> Result<DynamicMinCut>
```

**Returns**: `DynamicMinCut` instance

**Errors**: Construction errors (e.g., invalid edge)

**Example**:
```rust
let result = MinCutBuilder::new()
    .exact()
    .with_edges(vec![(1, 2, 1.0)])
    .build();

assert!(result.is_ok());
```

---

## Monitoring API

Real-time event monitoring for minimum cut changes (requires `monitoring` feature).

### MinCutMonitor

Event-driven monitor for cut value changes.

#### Creation

```rust
use ruvector_mincut::{MonitorBuilder, MonitorConfig, EventType};

// Using builder
let monitor = MonitorBuilder::new()
    .threshold_below(10.0, "critical")
    .threshold_above(100.0, "safe")
    .on_change("logger", |event| {
        println!("Cut changed: {:.2} -> {:.2}", event.old_value, event.new_value);
    })
    .build();

// Using config
let config = MonitorConfig {
    max_callbacks: 100,
    sample_interval: Duration::from_secs(1),
    max_history_size: 1000,
    collect_metrics: true,
};
let monitor = MinCutMonitor::new(config);
```

#### Methods

##### `on_event`

Register callback for all events.

```rust
pub fn on_event<F>(&self, name: &str, callback: F) -> Result<()>
where
    F: Fn(&MinCutEvent) + Send + Sync + 'static
```

**Parameters**:
- `name`: Callback identifier
- `callback`: Event handler function

**Example**:
```rust
monitor.on_event("logger", |event| {
    eprintln!("[{}] Cut: {:.2}", event.timestamp, event.new_value);
})?;
```

##### `on_event_type`

Register callback for specific event type.

```rust
pub fn on_event_type<F>(&self, event_type: EventType, name: &str, callback: F) -> Result<()>
where
    F: Fn(&MinCutEvent) + Send + Sync + 'static
```

**Parameters**:
- `event_type`: Type of event to listen for
- `name`: Callback identifier
- `callback`: Event handler

**Example**:
```rust
monitor.on_event_type(EventType::CutDecreased, "alert", |event| {
    if event.new_value < 5.0 {
        eprintln!("CRITICAL: Cut dropped to {:.2}", event.new_value);
    }
})?;

monitor.on_event_type(EventType::Disconnected, "disconnect_alert", |event| {
    eprintln!("WARNING: Graph disconnected!");
})?;
```

##### `add_threshold`

Add a threshold for monitoring.

```rust
pub fn add_threshold(&self, threshold: Threshold) -> Result<()>
```

**Example**:
```rust
use ruvector_mincut::Threshold;

let threshold = Threshold::new(10.0, "low".to_string(), true);
monitor.add_threshold(threshold)?;
```

##### `remove_threshold`

Remove a threshold by name.

```rust
pub fn remove_threshold(&self, name: &str) -> bool
```

**Returns**: `true` if threshold was removed

**Example**:
```rust
assert!(monitor.remove_threshold("low"));
assert!(!monitor.remove_threshold("low"));  // Already removed
```

##### `remove_callback`

Remove a callback by name.

```rust
pub fn remove_callback(&self, name: &str) -> bool
```

**Example**:
```rust
monitor.remove_callback("logger");
```

##### `notify`

Notify of a cut value change (called internally by `DynamicMinCut`).

```rust
pub fn notify(&self, old_value: f64, new_value: f64, edge: Option<(u64, u64)>)
```

**Example** (internal use):
```rust
// This is called automatically by DynamicMinCut
monitor.notify(10.0, 12.0, Some((1, 2)));
```

##### `metrics`

Get monitoring metrics.

```rust
pub fn metrics(&self) -> MonitorMetrics
```

**Returns**: Metrics including event counts, cut history, etc.

**Example**:
```rust
let metrics = monitor.metrics();
println!("Total events: {}", metrics.total_events);
println!("Min observed: {:.2}", metrics.min_observed);
println!("Max observed: {:.2}", metrics.max_observed);
println!("Average cut: {:.2}", metrics.avg_cut);

for (event_type, count) in &metrics.events_by_type {
    println!("  {}: {}", event_type, count);
}
```

##### `reset_metrics`

Reset all metrics.

```rust
pub fn reset_metrics(&self)
```

##### `current_cut`

Get current monitored cut value.

```rust
pub fn current_cut(&self) -> f64
```

##### `threshold_status`

Get status of all thresholds.

```rust
pub fn threshold_status(&self) -> Vec<(String, bool)>
```

**Returns**: Vector of (threshold_name, is_active) tuples

**Example**:
```rust
for (name, active) in monitor.threshold_status() {
    println!("Threshold '{}': {}", name, if active { "ACTIVE" } else { "inactive" });
}
```

### MonitorBuilder

Builder for `MinCutMonitor`.

#### Methods

##### `threshold_below`

Add threshold that alerts when cut goes below value.

```rust
pub fn threshold_below(mut self, value: f64, name: &str) -> Self
```

**Example**:
```rust
let monitor = MonitorBuilder::new()
    .threshold_below(5.0, "critical")
    .threshold_below(10.0, "warning")
    .build();
```

##### `threshold_above`

Add threshold that alerts when cut goes above value.

```rust
pub fn threshold_above(mut self, value: f64, name: &str) -> Self
```

**Example**:
```rust
let monitor = MonitorBuilder::new()
    .threshold_above(100.0, "safe")
    .build();
```

##### `on_change`

Add callback for all cut changes.

```rust
pub fn on_change<F>(mut self, name: &str, callback: F) -> Self
where
    F: Fn(&MinCutEvent) + Send + Sync + 'static
```

**Example**:
```rust
let monitor = MonitorBuilder::new()
    .on_change("logger", |event| {
        println!("Cut: {:.2}", event.new_value);
    })
    .build();
```

### Event Types

```rust
pub enum EventType {
    CutIncreased,          // Cut value increased
    CutDecreased,          // Cut value decreased
    ThresholdCrossedBelow, // Crossed below threshold
    ThresholdCrossedAbove, // Crossed above threshold
    Disconnected,          // Graph became disconnected
    Connected,             // Graph became connected
    EdgeInserted,          // Edge was inserted
    EdgeDeleted,           // Edge was deleted
}
```

### MinCutEvent

Event notification structure.

```rust
pub struct MinCutEvent {
    pub event_type: EventType,
    pub new_value: f64,
    pub old_value: f64,
    pub timestamp: Instant,
    pub threshold: Option<f64>,
    pub edge: Option<(u64, u64)>,
}
```

**Example**:
```rust
monitor.on_event("handler", |event| {
    match event.event_type {
        EventType::CutDecreased => {
            println!("Cut decreased from {:.2} to {:.2}",
                event.old_value, event.new_value);
        },
        EventType::ThresholdCrossedBelow => {
            if let Some(threshold) = event.threshold {
                println!("Cut {:.2} dropped below threshold {:.2}",
                    event.new_value, threshold);
            }
        },
        EventType::EdgeInserted => {
            if let Some((u, v)) = event.edge {
                println!("Edge ({}, {}) inserted", u, v);
            }
        },
        _ => {}
    }
});
```

---

## Advanced Features

### Graph Sparsification

Create sparse approximations of graphs.

#### SparseGraph

```rust
use ruvector_mincut::{SparseGraph, SparsifyConfig};

// Create sparsified graph
let config = SparsifyConfig::new(0.1)?  // 10% approximation
    .with_seed(42)
    .with_max_edges(1000);

let sparse = SparseGraph::from_graph(&original_graph, config)?;

println!("Original edges: {}", original_graph.num_edges());
println!("Sparse edges: {}", sparse.num_edges());
println!("Reduction: {:.1}%", (1.0 - sparse.sparsification_ratio()) * 100.0);

// Query approximate min cut
let approx_cut = sparse.approximate_min_cut();
println!("Approximate min cut: {:.2}", approx_cut);
```

#### Dynamic Updates

```rust
let mut sparse = SparseGraph::from_graph(&graph, config)?;

// Update sparse graph
sparse.insert_edge(4, 5, 2.0)?;
sparse.delete_edge(1, 2)?;
```

### Link-Cut Trees

Direct access to Link-Cut Tree operations.

```rust
use ruvector_mincut::LinkCutTree;

let mut lct = LinkCutTree::new();

// Create trees
lct.make_tree(1, 1.0);
lct.make_tree(2, 2.0);
lct.make_tree(3, 3.0);

// Link
lct.link(1, 2)?;  // 1 <- 2
lct.link(2, 3)?;  // 2 <- 3

// Connectivity
assert!(lct.connected(1, 3));

// Find root
let root = lct.find_root(1)?;
assert_eq!(root, 3);

// Path aggregate
let min_val = lct.path_aggregate(1)?;
assert_eq!(min_val, 1.0);

// Cut
lct.cut(2)?;  // Cut 2 from its parent (3)
assert!(!lct.connected(1, 3));
```

### Euler Tour Trees

```rust
use ruvector_mincut::EulerTourTree;

let mut ett = EulerTourTree::new();

// Create trees
ett.make_tree(1)?;
ett.make_tree(2)?;
ett.make_tree(3)?;

// Link
ett.link(1, 2)?;
ett.link(2, 3)?;

// Connectivity
assert!(ett.connected(1, 3));

// Tree size
let size = ett.tree_size(1)?;
assert_eq!(size, 3);

// Reroot
ett.reroot(3)?;
assert_eq!(ett.find_root(1)?, 3);
```

### Hierarchical Decomposition

```rust
use ruvector_mincut::{HierarchicalDecomposition, DynamicGraph};
use std::sync::Arc;

let graph = Arc::new(DynamicGraph::new());
graph.insert_edge(1, 2, 1.0)?;
graph.insert_edge(2, 3, 1.0)?;
graph.insert_edge(3, 1, 1.0)?;

// Build decomposition
let decomp = HierarchicalDecomposition::build(graph.clone())?;

println!("Height: {}", decomp.height());
println!("Nodes: {}", decomp.num_nodes());
println!("Min cut: {:.2}", decomp.min_cut_value());

// Get partition
let (s, t) = decomp.min_cut_partition();
println!("Partition: {:?} vs {:?}", s, t);

// Level information
for level_info in decomp.level_info() {
    println!("Level {}: {} nodes, avg cut {:.2}",
        level_info.level, level_info.num_nodes, level_info.avg_cut);
}
```

---

## Error Handling

### Error Types

```rust
pub enum MinCutError {
    EmptyGraph,
    InvalidVertex(u64),
    InvalidEdge(u64, u64),
    EdgeExists(u64, u64),
    EdgeNotFound(u64, u64),
    DisconnectedGraph,
    CutSizeExceeded(usize, usize),
    InvalidEpsilon(f64),
    InvalidParameter(String),
    CallbackError(String),
    InternalError(String),
    ConcurrentModification,
    CapacityExceeded(String),
    SerializationError(String),
}
```

### Error Handling Patterns

#### Pattern 1: Propagate with `?`

```rust
fn process_graph(edges: Vec<(u64, u64, f64)>) -> Result<f64> {
    let mut mincut = MinCutBuilder::new()
        .with_edges(edges)
        .build()?;

    mincut.insert_edge(10, 11, 1.0)?;
    Ok(mincut.min_cut_value())
}
```

#### Pattern 2: Match on Error Type

```rust
match mincut.insert_edge(1, 2, 1.0) {
    Ok(cut_value) => println!("New cut: {:.2}", cut_value),
    Err(MinCutError::EdgeExists(u, v)) => {
        println!("Edge ({}, {}) already exists", u, v);
    },
    Err(e) => eprintln!("Error: {}", e),
}
```

#### Pattern 3: Check Error Category

```rust
if let Err(e) = mincut.insert_edge(1, 2, 1.0) {
    if e.is_recoverable() {
        println!("Recoverable error: {}", e);
        // Continue execution
    } else {
        eprintln!("Fatal error: {}", e);
        return Err(e);
    }
}
```

### Error Methods

```rust
impl MinCutError {
    // Check if error is recoverable
    pub fn is_recoverable(&self) -> bool;

    // Check if error is graph structure related
    pub fn is_graph_structure_error(&self) -> bool;

    // Check if error is resource limit related
    pub fn is_resource_error(&self) -> bool;
}
```

---

## Best Practices

### 1. Choosing Exact vs Approximate

```rust
// Use exact for:
// - Small graphs (< 10,000 vertices)
// - Critical applications requiring optimal solution
// - When cut size is small (< 1,000)
let exact_mincut = MinCutBuilder::new()
    .exact()
    .build()?;

// Use approximate for:
// - Large graphs (> 10,000 vertices)
// - Real-time applications
// - When approximate answer is acceptable
let approx_mincut = MinCutBuilder::new()
    .approximate(0.05)  // 5% approximation
    .build()?;
```

### 2. Batch Operations

```rust
// Good: Batch edge insertions
let edges = vec![
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 4, 1.0),
];
let mincut = MinCutBuilder::new()
    .with_edges(edges)
    .build()?;

// Less efficient: Individual insertions
let mut mincut = MinCutBuilder::new().build()?;
for (u, v, w) in edges {
    mincut.insert_edge(u, v, w)?;
}
```

### 3. Monitoring Setup

```rust
// Set up comprehensive monitoring
let monitor = MonitorBuilder::new()
    // Thresholds
    .threshold_below(10.0, "critical")
    .threshold_below(50.0, "warning")
    .threshold_above(200.0, "safe")

    // Event handlers
    .on_event_type(EventType::Disconnected, "disconnect", |_| {
        // Alert: graph disconnected
    })
    .on_event_type(EventType::ThresholdCrossedBelow, "alert", |event| {
        if event.new_value < 10.0 {
            // Critical alert
        }
    })

    // Logging
    .on_change("logger", |event| {
        log::info!("Cut: {:.2}", event.new_value);
    })
    .build();
```

### 4. Statistics Tracking

```rust
// Enable detailed tracking
let mut mincut = MinCutBuilder::new().build()?;

// Perform operations
for _ in 0..1000 {
    mincut.insert_edge(rand_u(), rand_v(), rand_w())?;
}

// Analyze performance
let stats = mincut.stats();
println!("Operations: {} insertions, {} deletions, {} queries",
    stats.insertions, stats.deletions, stats.queries);
println!("Avg update time: {:.2} μs", stats.avg_update_time_us);
println!("Avg query time: {:.2} μs", stats.avg_query_time_us);
println!("Restructures: {}", stats.restructures);
```

### 5. Thread Safety

```rust
use std::sync::Arc;
use std::thread;

let mincut = Arc::new(Mutex::new(
    MinCutBuilder::new().build()?
));

// Spawn threads for updates
let handles: Vec<_> = (0..10).map(|i| {
    let mincut = Arc::clone(&mincut);
    thread::spawn(move || {
        let mut mc = mincut.lock().unwrap();
        mc.insert_edge(i, i + 1, 1.0)
    })
}).collect();

for handle in handles {
    handle.join().unwrap()?;
}
```

### 6. Resource Management

```rust
// Configure for large graphs
let mincut = MinCutBuilder::new()
    .approximate(0.1)
    .max_cut_size(10000)
    .parallel(true)
    .build()?;

// Monitor memory usage
let graph = mincut.graph();
let stats = graph.read().stats();
println!("Memory estimate: ~{} MB",
    (stats.num_vertices * 64 + stats.num_edges * 32) / 1_000_000);
```

---

## Complete Example

Comprehensive example combining multiple features:

```rust
use ruvector_mincut::prelude::*;

fn main() -> Result<()> {
    // 1. Create mincut structure
    let mut mincut = MinCutBuilder::new()
        .approximate(0.1)
        .parallel(true)
        .with_edges(vec![
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 1, 2.0),
        ])
        .build()?;

    println!("Initial min cut: {:.2}", mincut.min_cut_value());

    // 2. Set up monitoring
    #[cfg(feature = "monitoring")]
    {
        let monitor = MonitorBuilder::new()
            .threshold_below(2.0, "critical")
            .on_change("logger", |event| {
                println!("[{}] Cut: {:.2} -> {:.2}",
                    event.event_type.as_str(),
                    event.old_value,
                    event.new_value);
            })
            .build();
    }

    // 3. Perform updates
    println!("\n--- Inserting edge (1, 3) ---");
    let new_cut = mincut.insert_edge(1, 3, 1.0)?;
    println!("New min cut: {:.2}", new_cut);

    println!("\n--- Deleting edge (4, 1) ---");
    let new_cut = mincut.delete_edge(4, 1)?;
    println!("New min cut: {:.2}", new_cut);

    // 4. Query results
    let result = mincut.min_cut();
    println!("\n=== Final Results ===");
    println!("Min cut value: {:.2}", result.value);
    println!("Exact: {}", result.is_exact);
    println!("Approximation ratio: {:.2}", result.approximation_ratio);

    if let Some((s, t)) = result.partition {
        println!("Partition S: {:?}", s);
        println!("Partition T: {:?}", t);
    }

    if let Some(edges) = result.cut_edges {
        println!("Cut edges: {}", edges.len());
        for edge in edges {
            println!("  ({}, {}): {:.2}", edge.source, edge.target, edge.weight);
        }
    }

    // 5. Statistics
    let stats = mincut.stats();
    println!("\n=== Statistics ===");
    println!("Insertions: {}", stats.insertions);
    println!("Deletions: {}", stats.deletions);
    println!("Queries: {}", stats.queries);
    println!("Avg update time: {:.2} μs", stats.avg_update_time_us);

    Ok(())
}
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md): System architecture and design
- [ALGORITHMS.md](ALGORITHMS.md): Algorithm details and complexity analysis
- [Examples](../examples/): Complete code examples
- [Benchmarks](../benches/): Performance benchmarks
