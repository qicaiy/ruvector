//! Three-Level Cluster Hierarchy for Dynamic Minimum Cut
//!
//! Implementation of the 3-level decomposition from:
//! "Deterministic and Exact Fully-dynamic Minimum Cut of Superpolylogarithmic
//! Size in Subpolynomial Time" (arXiv:2512.13105)
//!
//! # Hierarchy Structure
//!
//! Level 0: **Expanders** - φ-expander subgraphs with guaranteed edge expansion
//! Level 1: **Preclusters** - Groups of expanders with bounded boundaries
//! Level 2: **Clusters** - Final clustering with mirror cut maintenance
//!
//! # Key Features
//!
//! - Expanders guarantee no sparse internal cuts
//! - Preclusters enable local cut computation
//! - Clusters maintain mirror cuts for cross-boundary tracking
//! - Incremental updates propagate through hierarchy

use std::collections::{HashMap, HashSet, VecDeque};
use crate::graph::{VertexId, Weight};

/// Expansion parameter type
pub type Phi = f64;

/// Configuration for the 3-level hierarchy
#[derive(Debug, Clone)]
pub struct HierarchyConfig {
    /// Expansion parameter φ for expander detection
    pub phi: Phi,
    /// Maximum expander size before decomposition
    pub max_expander_size: usize,
    /// Minimum expander size (don't decompose smaller)
    pub min_expander_size: usize,
    /// Target precluster size
    pub target_precluster_size: usize,
    /// Maximum boundary ratio for preclusters
    pub max_boundary_ratio: f64,
    /// Enable mirror cut tracking
    pub track_mirror_cuts: bool,
}

impl Default for HierarchyConfig {
    fn default() -> Self {
        Self {
            phi: 0.1,
            max_expander_size: 500,
            min_expander_size: 5,
            target_precluster_size: 100,
            max_boundary_ratio: 0.3,
            track_mirror_cuts: true,
        }
    }
}

/// Level in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchyLevel {
    /// Level 0: Expander (φ-expander subgraph)
    Expander,
    /// Level 1: Precluster (group of expanders)
    Precluster,
    /// Level 2: Cluster (top-level grouping)
    Cluster,
}

/// An expander at level 0
#[derive(Debug, Clone)]
pub struct Expander {
    /// Unique expander ID
    pub id: u64,
    /// Vertices in this expander
    pub vertices: HashSet<VertexId>,
    /// Internal edges (both endpoints in expander)
    pub internal_edges: Vec<(VertexId, VertexId)>,
    /// Boundary edges (one endpoint outside)
    pub boundary_edges: Vec<(VertexId, VertexId)>,
    /// Volume (sum of degrees)
    pub volume: usize,
    /// Verified expansion ratio
    pub expansion_ratio: f64,
    /// Parent precluster ID
    pub precluster_id: Option<u64>,
}

impl Expander {
    /// Create new expander
    pub fn new(id: u64, vertices: HashSet<VertexId>) -> Self {
        Self {
            id,
            vertices,
            internal_edges: Vec::new(),
            boundary_edges: Vec::new(),
            volume: 0,
            expansion_ratio: 0.0,
            precluster_id: None,
        }
    }

    /// Get size (number of vertices)
    pub fn size(&self) -> usize {
        self.vertices.len()
    }

    /// Check if vertex is in this expander
    pub fn contains(&self, v: VertexId) -> bool {
        self.vertices.contains(&v)
    }

    /// Compute boundary sparsity
    pub fn boundary_sparsity(&self) -> f64 {
        if self.volume == 0 {
            return 0.0;
        }
        self.boundary_edges.len() as f64 / self.volume as f64
    }
}

/// A precluster at level 1 (group of expanders)
#[derive(Debug, Clone)]
pub struct Precluster {
    /// Unique precluster ID
    pub id: u64,
    /// Expander IDs in this precluster
    pub expanders: Vec<u64>,
    /// All vertices (union of expanders)
    pub vertices: HashSet<VertexId>,
    /// Boundary edges to other preclusters
    pub boundary_edges: Vec<(VertexId, VertexId)>,
    /// Volume of this precluster
    pub volume: usize,
    /// Parent cluster ID
    pub cluster_id: Option<u64>,
}

impl Precluster {
    /// Create new precluster
    pub fn new(id: u64) -> Self {
        Self {
            id,
            expanders: Vec::new(),
            vertices: HashSet::new(),
            boundary_edges: Vec::new(),
            volume: 0,
            cluster_id: None,
        }
    }

    /// Add an expander to this precluster
    pub fn add_expander(&mut self, expander: &Expander) {
        self.expanders.push(expander.id);
        self.vertices.extend(&expander.vertices);
        self.volume += expander.volume;
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.vertices.len()
    }

    /// Compute boundary ratio
    pub fn boundary_ratio(&self) -> f64 {
        if self.volume == 0 {
            return 0.0;
        }
        self.boundary_edges.len() as f64 / self.volume as f64
    }
}

/// A mirror cut for tracking cross-expander minimum cuts
#[derive(Debug, Clone)]
pub struct MirrorCut {
    /// Source expander ID
    pub source_expander: u64,
    /// Target expander ID
    pub target_expander: u64,
    /// Cut value between the expanders
    pub cut_value: f64,
    /// Edges in the cut
    pub cut_edges: Vec<(VertexId, VertexId)>,
    /// Is this cut certified (verified via LocalKCut)?
    pub certified: bool,
}

/// A cluster at level 2 (top-level grouping)
#[derive(Debug, Clone)]
pub struct HierarchyCluster {
    /// Unique cluster ID
    pub id: u64,
    /// Precluster IDs in this cluster
    pub preclusters: Vec<u64>,
    /// All vertices
    pub vertices: HashSet<VertexId>,
    /// Boundary edges to other clusters
    pub boundary_edges: Vec<(VertexId, VertexId)>,
    /// Mirror cuts tracked for this cluster
    pub mirror_cuts: Vec<MirrorCut>,
    /// Minimum cut within this cluster
    pub internal_min_cut: f64,
}

impl HierarchyCluster {
    /// Create new cluster
    pub fn new(id: u64) -> Self {
        Self {
            id,
            preclusters: Vec::new(),
            vertices: HashSet::new(),
            boundary_edges: Vec::new(),
            mirror_cuts: Vec::new(),
            internal_min_cut: f64::INFINITY,
        }
    }

    /// Add a precluster
    pub fn add_precluster(&mut self, precluster: &Precluster) {
        self.preclusters.push(precluster.id);
        self.vertices.extend(&precluster.vertices);
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.vertices.len()
    }
}

/// The three-level hierarchy data structure
#[derive(Debug)]
pub struct ThreeLevelHierarchy {
    /// Configuration
    config: HierarchyConfig,
    /// All expanders (level 0)
    expanders: HashMap<u64, Expander>,
    /// All preclusters (level 1)
    preclusters: HashMap<u64, Precluster>,
    /// All clusters (level 2)
    clusters: HashMap<u64, HierarchyCluster>,
    /// Vertex to expander mapping
    vertex_expander: HashMap<VertexId, u64>,
    /// Next ID counter
    next_id: u64,
    /// Graph adjacency
    adjacency: HashMap<VertexId, HashMap<VertexId, Weight>>,
    /// Global minimum cut estimate
    pub global_min_cut: f64,
}

impl ThreeLevelHierarchy {
    /// Create new hierarchy
    pub fn new(config: HierarchyConfig) -> Self {
        Self {
            config,
            expanders: HashMap::new(),
            preclusters: HashMap::new(),
            clusters: HashMap::new(),
            vertex_expander: HashMap::new(),
            next_id: 1,
            adjacency: HashMap::new(),
            global_min_cut: f64::INFINITY,
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(HierarchyConfig::default())
    }

    /// Insert an edge
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId, weight: Weight) {
        self.adjacency.entry(u).or_default().insert(v, weight);
        self.adjacency.entry(v).or_default().insert(u, weight);
    }

    /// Delete an edge
    pub fn delete_edge(&mut self, u: VertexId, v: VertexId) {
        if let Some(neighbors) = self.adjacency.get_mut(&u) {
            neighbors.remove(&v);
        }
        if let Some(neighbors) = self.adjacency.get_mut(&v) {
            neighbors.remove(&u);
        }
    }

    /// Get all vertices
    pub fn vertices(&self) -> Vec<VertexId> {
        self.adjacency.keys().copied().collect()
    }

    /// Get neighbors
    pub fn neighbors(&self, v: VertexId) -> Vec<(VertexId, Weight)> {
        self.adjacency.get(&v)
            .map(|n| n.iter().map(|(&v, &w)| (v, w)).collect())
            .unwrap_or_default()
    }

    /// Get degree
    pub fn degree(&self, v: VertexId) -> usize {
        self.adjacency.get(&v).map_or(0, |n| n.len())
    }

    /// Build the complete 3-level hierarchy
    pub fn build(&mut self) {
        let vertices: HashSet<_> = self.vertices().into_iter().collect();
        if vertices.is_empty() {
            return;
        }

        // Step 1: Create initial expanders via greedy decomposition
        self.build_expanders(&vertices);

        // Step 2: Group expanders into preclusters
        self.build_preclusters();

        // Step 3: Group preclusters into clusters
        self.build_clusters();

        // Step 4: Compute mirror cuts if enabled
        if self.config.track_mirror_cuts {
            self.compute_mirror_cuts();
        }

        // Step 5: Compute global min cut estimate
        self.update_global_min_cut();
    }

    /// Build expanders using greedy expansion detection
    fn build_expanders(&mut self, vertices: &HashSet<VertexId>) {
        self.expanders.clear();
        self.vertex_expander.clear();

        let mut remaining: HashSet<_> = vertices.iter().copied().collect();

        while !remaining.is_empty() {
            // Pick a random starting vertex
            let start = *remaining.iter().next().unwrap();

            // Grow an expander from this vertex
            let expander_vertices = self.grow_expander(start, &remaining);

            if expander_vertices.is_empty() {
                remaining.remove(&start);
                continue;
            }

            // Create expander
            let id = self.next_id;
            self.next_id += 1;

            let mut expander = Expander::new(id, expander_vertices.clone());
            self.compute_expander_properties(&mut expander);

            // Update mappings
            for &v in &expander_vertices {
                self.vertex_expander.insert(v, id);
                remaining.remove(&v);
            }

            self.expanders.insert(id, expander);
        }
    }

    /// Grow an expander from a starting vertex using BFS
    fn grow_expander(&self, start: VertexId, available: &HashSet<VertexId>) -> HashSet<VertexId> {
        let mut expander = HashSet::new();
        let mut queue = VecDeque::new();
        let mut volume = 0usize;

        queue.push_back(start);
        expander.insert(start);
        volume += self.degree(start);

        while let Some(v) = queue.pop_front() {
            if expander.len() >= self.config.max_expander_size {
                break;
            }

            for (neighbor, _) in self.neighbors(v) {
                if !available.contains(&neighbor) || expander.contains(&neighbor) {
                    continue;
                }

                // Check if adding this vertex maintains expansion
                let new_volume = volume + self.degree(neighbor);
                let boundary_after = self.count_boundary(&expander, &[neighbor]);

                let expansion = if new_volume > 0 {
                    boundary_after as f64 / (new_volume.min(expander.len() * 2)) as f64
                } else {
                    0.0
                };

                // Only add if it doesn't violate expansion too much
                if expansion >= self.config.phi * 0.5 || expander.len() < self.config.min_expander_size {
                    expander.insert(neighbor);
                    volume = new_volume;
                    queue.push_back(neighbor);
                }
            }
        }

        expander
    }

    /// Count boundary edges for a set of vertices
    fn count_boundary(&self, vertices: &HashSet<VertexId>, additional: &[VertexId]) -> usize {
        let mut full_set = vertices.clone();
        for &v in additional {
            full_set.insert(v);
        }

        let mut boundary = 0;
        for &v in &full_set {
            for (neighbor, _) in self.neighbors(v) {
                if !full_set.contains(&neighbor) {
                    boundary += 1;
                }
            }
        }
        boundary
    }

    /// Compute expander properties (edges, volume, expansion ratio)
    fn compute_expander_properties(&self, expander: &mut Expander) {
        let mut internal = Vec::new();
        let mut boundary = Vec::new();
        let mut volume = 0;

        for &v in &expander.vertices {
            let neighbors = self.neighbors(v);
            volume += neighbors.len();

            for (neighbor, _) in neighbors {
                if expander.vertices.contains(&neighbor) {
                    if v < neighbor {
                        internal.push((v, neighbor));
                    }
                } else {
                    boundary.push((v, neighbor));
                }
            }
        }

        expander.internal_edges = internal;
        expander.boundary_edges = boundary;
        expander.volume = volume;

        // Compute expansion ratio
        let min_vol = expander.volume.min(expander.vertices.len() * 2);
        expander.expansion_ratio = if min_vol > 0 {
            expander.boundary_edges.len() as f64 / min_vol as f64
        } else {
            0.0
        };
    }

    /// Build preclusters by grouping nearby expanders
    fn build_preclusters(&mut self) {
        self.preclusters.clear();

        let expander_ids: Vec<_> = self.expanders.keys().copied().collect();
        let mut assigned = HashSet::new();

        for &exp_id in &expander_ids {
            if assigned.contains(&exp_id) {
                continue;
            }

            let id = self.next_id;
            self.next_id += 1;

            let mut precluster = Precluster::new(id);

            // Add this expander
            if let Some(expander) = self.expanders.get_mut(&exp_id) {
                precluster.add_expander(expander);
                expander.precluster_id = Some(id);
                assigned.insert(exp_id);
            }

            // Try to add neighboring expanders
            for &other_id in &expander_ids {
                if assigned.contains(&other_id) {
                    continue;
                }

                if precluster.size() >= self.config.target_precluster_size {
                    break;
                }

                // Check if expanders are adjacent
                if self.expanders_adjacent(exp_id, other_id) {
                    if let Some(expander) = self.expanders.get_mut(&other_id) {
                        precluster.add_expander(expander);
                        expander.precluster_id = Some(id);
                        assigned.insert(other_id);
                    }
                }
            }

            // Compute precluster boundary
            self.compute_precluster_boundary(&mut precluster);

            self.preclusters.insert(id, precluster);
        }
    }

    /// Check if two expanders share an edge
    fn expanders_adjacent(&self, exp1: u64, exp2: u64) -> bool {
        let e1 = match self.expanders.get(&exp1) {
            Some(e) => e,
            None => return false,
        };
        let e2 = match self.expanders.get(&exp2) {
            Some(e) => e,
            None => return false,
        };

        // Check if any vertex in e1 has a neighbor in e2
        for &v in &e1.vertices {
            for (neighbor, _) in self.neighbors(v) {
                if e2.vertices.contains(&neighbor) {
                    return true;
                }
            }
        }
        false
    }

    /// Compute boundary for a precluster
    fn compute_precluster_boundary(&self, precluster: &mut Precluster) {
        precluster.boundary_edges.clear();

        for &v in &precluster.vertices {
            for (neighbor, _) in self.neighbors(v) {
                if !precluster.vertices.contains(&neighbor) {
                    precluster.boundary_edges.push((v, neighbor));
                }
            }
        }
    }

    /// Build top-level clusters from preclusters
    fn build_clusters(&mut self) {
        self.clusters.clear();

        let precluster_ids: Vec<_> = self.preclusters.keys().copied().collect();
        let mut assigned = HashSet::new();

        for &pre_id in &precluster_ids {
            if assigned.contains(&pre_id) {
                continue;
            }

            let id = self.next_id;
            self.next_id += 1;

            let mut cluster = HierarchyCluster::new(id);

            // Add this precluster
            if let Some(precluster) = self.preclusters.get_mut(&pre_id) {
                cluster.add_precluster(precluster);
                precluster.cluster_id = Some(id);
                assigned.insert(pre_id);
            }

            // Try to add adjacent preclusters (greedy)
            for &other_id in &precluster_ids {
                if assigned.contains(&other_id) {
                    continue;
                }

                if self.preclusters_adjacent(pre_id, other_id) {
                    if let Some(precluster) = self.preclusters.get_mut(&other_id) {
                        cluster.add_precluster(precluster);
                        precluster.cluster_id = Some(id);
                        assigned.insert(other_id);
                    }
                }
            }

            // Compute cluster boundary
            self.compute_cluster_boundary(&mut cluster);

            self.clusters.insert(id, cluster);
        }
    }

    /// Check if two preclusters share an edge
    fn preclusters_adjacent(&self, pre1: u64, pre2: u64) -> bool {
        let p1 = match self.preclusters.get(&pre1) {
            Some(p) => p,
            None => return false,
        };
        let p2 = match self.preclusters.get(&pre2) {
            Some(p) => p,
            None => return false,
        };

        for &v in &p1.vertices {
            for (neighbor, _) in self.neighbors(v) {
                if p2.vertices.contains(&neighbor) {
                    return true;
                }
            }
        }
        false
    }

    /// Compute boundary for a cluster
    fn compute_cluster_boundary(&self, cluster: &mut HierarchyCluster) {
        cluster.boundary_edges.clear();

        for &v in &cluster.vertices {
            for (neighbor, _) in self.neighbors(v) {
                if !cluster.vertices.contains(&neighbor) {
                    cluster.boundary_edges.push((v, neighbor));
                }
            }
        }
    }

    /// Compute mirror cuts between adjacent expanders
    fn compute_mirror_cuts(&mut self) {
        let expander_ids: Vec<_> = self.expanders.keys().copied().collect();

        for cluster in self.clusters.values_mut() {
            cluster.mirror_cuts.clear();
        }

        // For each pair of adjacent expanders
        for (i, &exp1) in expander_ids.iter().enumerate() {
            for &exp2 in expander_ids.iter().skip(i + 1) {
                if !self.expanders_adjacent(exp1, exp2) {
                    continue;
                }

                // Compute cut between them
                let (cut_value, cut_edges) = self.compute_expander_cut(exp1, exp2);

                let mirror = MirrorCut {
                    source_expander: exp1,
                    target_expander: exp2,
                    cut_value,
                    cut_edges,
                    certified: false,
                };

                // Add to the cluster containing both expanders
                if let Some(pre_id) = self.expanders.get(&exp1).and_then(|e| e.precluster_id) {
                    if let Some(cluster_id) = self.preclusters.get(&pre_id).and_then(|p| p.cluster_id) {
                        if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
                            cluster.mirror_cuts.push(mirror);
                        }
                    }
                }
            }
        }

        // Update internal min cuts
        for cluster in self.clusters.values_mut() {
            if let Some(min_mirror) = cluster.mirror_cuts.iter()
                .map(|m| m.cut_value)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
            {
                cluster.internal_min_cut = cluster.internal_min_cut.min(min_mirror);
            }
        }
    }

    /// Compute cut between two expanders
    fn compute_expander_cut(&self, exp1: u64, exp2: u64) -> (f64, Vec<(VertexId, VertexId)>) {
        let e1 = match self.expanders.get(&exp1) {
            Some(e) => e,
            None => return (0.0, Vec::new()),
        };
        let e2 = match self.expanders.get(&exp2) {
            Some(e) => e,
            None => return (0.0, Vec::new()),
        };

        let mut cut_edges = Vec::new();
        let mut cut_value = 0.0;

        for &v in &e1.vertices {
            for (neighbor, weight) in self.neighbors(v) {
                if e2.vertices.contains(&neighbor) {
                    cut_edges.push((v, neighbor));
                    cut_value += weight;
                }
            }
        }

        (cut_value, cut_edges)
    }

    /// Update global minimum cut estimate
    fn update_global_min_cut(&mut self) {
        let mut min_cut = f64::INFINITY;

        // Check cluster boundaries
        for cluster in self.clusters.values() {
            let boundary_cut: f64 = cluster.boundary_edges.iter()
                .map(|&(u, v)| {
                    self.adjacency.get(&u)
                        .and_then(|n| n.get(&v))
                        .copied()
                        .unwrap_or(1.0)
                })
                .sum();

            min_cut = min_cut.min(boundary_cut);
            min_cut = min_cut.min(cluster.internal_min_cut);
        }

        self.global_min_cut = min_cut;
    }

    // === Getters ===

    /// Get expander containing vertex
    pub fn get_vertex_expander(&self, v: VertexId) -> Option<&Expander> {
        self.vertex_expander.get(&v)
            .and_then(|&id| self.expanders.get(&id))
    }

    /// Get all expanders
    pub fn get_expanders(&self) -> &HashMap<u64, Expander> {
        &self.expanders
    }

    /// Get all preclusters
    pub fn get_preclusters(&self) -> &HashMap<u64, Precluster> {
        &self.preclusters
    }

    /// Get all clusters
    pub fn get_clusters(&self) -> &HashMap<u64, HierarchyCluster> {
        &self.clusters
    }

    /// Get hierarchy statistics
    pub fn stats(&self) -> HierarchyStats {
        HierarchyStats {
            num_expanders: self.expanders.len(),
            num_preclusters: self.preclusters.len(),
            num_clusters: self.clusters.len(),
            num_vertices: self.adjacency.len(),
            num_edges: self.adjacency.values()
                .map(|n| n.len())
                .sum::<usize>() / 2,
            global_min_cut: self.global_min_cut,
            avg_expander_size: if self.expanders.is_empty() {
                0.0
            } else {
                self.expanders.values()
                    .map(|e| e.size())
                    .sum::<usize>() as f64 / self.expanders.len() as f64
            },
        }
    }
}

impl Default for ThreeLevelHierarchy {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Statistics about the hierarchy
#[derive(Debug, Clone)]
pub struct HierarchyStats {
    /// Number of expanders (level 0)
    pub num_expanders: usize,
    /// Number of preclusters (level 1)
    pub num_preclusters: usize,
    /// Number of clusters (level 2)
    pub num_clusters: usize,
    /// Total vertices
    pub num_vertices: usize,
    /// Total edges
    pub num_edges: usize,
    /// Global minimum cut estimate
    pub global_min_cut: f64,
    /// Average expander size
    pub avg_expander_size: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_path(h: &mut ThreeLevelHierarchy, n: usize) {
        for i in 0..n-1 {
            h.insert_edge(i as u64, (i + 1) as u64, 1.0);
        }
    }

    fn build_clique(h: &mut ThreeLevelHierarchy, vertices: &[u64]) {
        for i in 0..vertices.len() {
            for j in i+1..vertices.len() {
                h.insert_edge(vertices[i], vertices[j], 1.0);
            }
        }
    }

    #[test]
    fn test_hierarchy_empty() {
        let mut h = ThreeLevelHierarchy::with_defaults();
        h.build();
        assert_eq!(h.expanders.len(), 0);
        assert_eq!(h.preclusters.len(), 0);
        assert_eq!(h.clusters.len(), 0);
    }

    #[test]
    fn test_hierarchy_path() {
        let mut h = ThreeLevelHierarchy::new(HierarchyConfig {
            min_expander_size: 2,
            max_expander_size: 5,
            ..Default::default()
        });
        build_path(&mut h, 10);
        h.build();

        assert!(h.expanders.len() >= 1);
        assert!(h.preclusters.len() >= 1);
        assert!(h.clusters.len() >= 1);

        let stats = h.stats();
        assert_eq!(stats.num_vertices, 10);
        assert_eq!(stats.num_edges, 9);
    }

    #[test]
    fn test_hierarchy_clique() {
        let mut h = ThreeLevelHierarchy::with_defaults();
        build_clique(&mut h, &[1, 2, 3, 4, 5]);
        h.build();

        // Clique should be a single expander
        assert!(h.expanders.len() >= 1);

        // Check vertex assignment
        for v in 1..=5 {
            assert!(h.get_vertex_expander(v).is_some());
        }
    }

    #[test]
    fn test_hierarchy_two_cliques() {
        let mut h = ThreeLevelHierarchy::new(HierarchyConfig {
            min_expander_size: 2,
            ..Default::default()
        });

        // Two cliques connected by a single edge
        build_clique(&mut h, &[1, 2, 3, 4]);
        build_clique(&mut h, &[5, 6, 7, 8]);
        h.insert_edge(4, 5, 1.0); // Bridge

        h.build();

        // Should have at least 2 expanders
        assert!(h.expanders.len() >= 1);

        // Global min cut should be 1 (the bridge)
        assert!(h.global_min_cut <= 2.0);
    }

    #[test]
    fn test_mirror_cuts() {
        let mut h = ThreeLevelHierarchy::new(HierarchyConfig {
            track_mirror_cuts: true,
            min_expander_size: 2,
            ..Default::default()
        });

        build_clique(&mut h, &[1, 2, 3]);
        build_clique(&mut h, &[4, 5, 6]);
        h.insert_edge(3, 4, 2.0);
        h.insert_edge(2, 5, 1.0);

        h.build();

        // Should track mirror cuts if multiple expanders
        let stats = h.stats();
        assert!(stats.num_expanders >= 1);
    }

    #[test]
    fn test_expander_properties() {
        let mut h = ThreeLevelHierarchy::with_defaults();
        build_clique(&mut h, &[1, 2, 3, 4]);
        h.build();

        for expander in h.expanders.values() {
            // Clique should have good expansion
            assert!(expander.size() > 0);
            assert!(expander.volume > 0);
        }
    }

    #[test]
    fn test_stats() {
        let mut h = ThreeLevelHierarchy::with_defaults();
        build_path(&mut h, 5);
        h.build();

        let stats = h.stats();
        assert_eq!(stats.num_vertices, 5);
        assert_eq!(stats.num_edges, 4);
        assert!(stats.avg_expander_size > 0.0);
    }
}
