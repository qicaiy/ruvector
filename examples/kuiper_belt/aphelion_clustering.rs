//! # Analysis Agent 7: Aphelion Clustering
//!
//! Detects clustering patterns in aphelion distances for distant Kuiper Belt objects.
//! Focuses on objects with Q > 100 AU to identify potential undiscovered planets.
//!
//! ## Features:
//! - Filters Trans-Neptunian Objects (TNOs) with aphelion > 100 AU
//! - Bins aphelion distances in 50 AU intervals
//! - Identifies significant clusters
//! - Estimates planet positions at 60% of clustered aphelion
//! - Reports sheparding effects and orbital architecture

use super::kuiper_cluster::KuiperBeltObject;
use std::collections::HashMap;

/// Aphelion bin representing a cluster of objects
#[derive(Debug, Clone)]
pub struct AphelionBin {
    /// Bin center (AU)
    pub center: f32,
    /// Bin range [min, max)
    pub range: (f32, f32),
    /// Objects in this bin
    pub members: Vec<String>,
    /// Aphelion distances of members
    pub aphelion_distances: Vec<f32>,
    /// Average aphelion in bin
    pub avg_aphelion: f32,
    /// Standard deviation of aphelion
    pub std_aphelion: f32,
    /// Density score (objects per 50 AU)
    pub density: f32,
}

impl AphelionBin {
    /// Calculate statistics for the bin
    fn calculate_stats(&mut self) {
        if self.aphelion_distances.is_empty() {
            self.avg_aphelion = self.center;
            self.std_aphelion = 0.0;
            self.density = 0.0;
            return;
        }

        // Average aphelion
        self.avg_aphelion =
            self.aphelion_distances.iter().sum::<f32>() / self.aphelion_distances.len() as f32;

        // Standard deviation
        let variance = self.aphelion_distances
            .iter()
            .map(|d| (d - self.avg_aphelion).powi(2))
            .sum::<f32>() / self.aphelion_distances.len() as f32;
        self.std_aphelion = variance.sqrt();

        // Density (objects per 50 AU)
        self.density = self.members.len() as f32;
    }

    /// Estimate planet position (60% of clustered aphelion)
    pub fn estimate_planet_distance(&self) -> f32 {
        self.avg_aphelion * 0.6
    }

    /// Check if this is a significant cluster
    pub fn is_significant(&self) -> bool {
        self.members.len() >= 2  // At least 2 objects
    }
}

/// Result from aphelion clustering analysis
#[derive(Debug, Clone)]
pub struct AphelionClusteringResult {
    /// All detected bins
    pub bins: Vec<AphelionBin>,
    /// Significant clusters only
    pub significant_clusters: Vec<AphelionBin>,
    /// Objects with Q > 100 AU
    pub distant_objects: Vec<String>,
    /// Estimated planet positions
    pub estimated_planets: Vec<EstimatedPlanet>,
    /// Total objects analyzed
    pub total_objects: usize,
    /// Objects with Q > 100 AU
    pub distant_object_count: usize,
}

/// Estimated planet from aphelion clustering
#[derive(Debug, Clone)]
pub struct EstimatedPlanet {
    /// Planet designation (e.g., "Planet 9a")
    pub designation: String,
    /// Estimated semi-major axis (AU)
    pub estimated_a: f32,
    /// Based on aphelion clustering
    pub aphelion_cluster: f32,
    /// Number of shepherded objects
    pub shepherded_count: usize,
    /// Confidence score (0-1)
    pub confidence: f32,
}

/// Aphelion clustering analyzer
pub struct AphelionClusterer {
    /// Bin size in AU
    bin_size: f32,
    /// Minimum aphelion threshold (AU)
    min_aphelion: f32,
}

impl AphelionClusterer {
    /// Create a new aphelion clusterer
    pub fn new() -> Self {
        Self {
            bin_size: 50.0,        // 50 AU bins
            min_aphelion: 100.0,   // Only Q > 100 AU
        }
    }

    /// Perform aphelion clustering analysis
    pub fn cluster(&self, objects: &[KuiperBeltObject]) -> AphelionClusteringResult {
        // Filter objects with Q > 100 AU
        let distant_objects: Vec<&KuiperBeltObject> = objects
            .iter()
            .filter(|o| o.ad > self.min_aphelion)
            .collect();

        let distant_names: Vec<String> = distant_objects
            .iter()
            .map(|o| o.name.clone())
            .collect();

        println!("\n[APHELION CLUSTERING] Found {} objects with Q > {} AU",
                 distant_names.len(), self.min_aphelion);

        // Create bins
        let mut bins: HashMap<usize, AphelionBin> = HashMap::new();

        for obj in &distant_objects {
            let bin_index = (obj.ad / self.bin_size).floor() as usize;
            let bin_center = (bin_index as f32 + 0.5) * self.bin_size;
            let range = (
                bin_index as f32 * self.bin_size,
                (bin_index as f32 + 1.0) * self.bin_size,
            );

            let bin = bins
                .entry(bin_index)
                .or_insert_with(|| AphelionBin {
                    center: bin_center,
                    range,
                    members: Vec::new(),
                    aphelion_distances: Vec::new(),
                    avg_aphelion: 0.0,
                    std_aphelion: 0.0,
                    density: 0.0,
                });

            bin.members.push(obj.name.clone());
            bin.aphelion_distances.push(obj.ad);
        }

        // Calculate statistics for each bin
        let mut bins_vec: Vec<AphelionBin> = bins
            .into_values()
            .map(|mut bin| {
                bin.calculate_stats();
                bin
            })
            .collect();

        // Sort by bin center
        bins_vec.sort_by(|a, b| a.center.partial_cmp(&b.center).unwrap());

        // Identify significant clusters
        let significant_clusters: Vec<AphelionBin> = bins_vec
            .iter()
            .filter(|b| b.is_significant())
            .cloned()
            .collect();

        // Estimate planet positions
        let estimated_planets = self.estimate_planets(&significant_clusters);

        AphelionClusteringResult {
            bins: bins_vec,
            significant_clusters,
            distant_objects: distant_names,
            estimated_planets,
            total_objects: objects.len(),
            distant_object_count: distant_objects.len(),
        }
    }

    /// Estimate planet positions from clusters (60% of aphelion)
    fn estimate_planets(&self, clusters: &[AphelionBin]) -> Vec<EstimatedPlanet> {
        let mut planets = Vec::new();

        for (idx, cluster) in clusters.iter().enumerate() {
            let planet_a = cluster.estimate_planet_distance();

            // Calculate confidence based on:
            // 1. Number of objects in cluster
            // 2. Density consistency
            // 3. Orbital coherence
            let object_score = (cluster.members.len() as f32 / 10.0).min(1.0);
            let density_score = if cluster.std_aphelion < 30.0 { 1.0 } else {
                (30.0 / cluster.std_aphelion).min(1.0)
            };
            let confidence = (object_score * 0.6 + density_score * 0.4).min(1.0);

            planets.push(EstimatedPlanet {
                designation: format!("Planet {}", idx + 9),  // Planet 9, 10, 11, etc.
                estimated_a: planet_a,
                aphelion_cluster: cluster.avg_aphelion,
                shepherded_count: cluster.members.len(),
                confidence,
            });
        }

        planets
    }
}

impl Default for AphelionClusterer {
    fn default() -> Self {
        Self::new()
    }
}

impl AphelionClusteringResult {
    /// Generate analysis report
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("         APHELION CLUSTERING ANALYSIS - PLANET DETECTION         \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str("📊 DISTANT OBJECT CENSUS\n");
        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str(&format!("  Total objects analyzed:        {}\n", self.total_objects));
        report.push_str(&format!("  Objects with Q > 100 AU:       {}\n", self.distant_object_count));
        report.push_str(&format!("  Percentage of distant:         {:.1}%\n",
                                 (self.distant_object_count as f32 / self.total_objects as f32) * 100.0));
        report.push_str("\n");

        if self.bins.is_empty() {
            report.push_str("  No objects with Q > 100 AU found in dataset.\n");
            return report;
        }

        // Aphelion range
        let min_aphelion = self.distant_objects.iter()
            .map(|_| 0.0)  // Placeholder - we'd need access to objects
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        report.push_str("📈 APHELION DISTRIBUTION (50 AU BINS)\n");
        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str(&format!("{:<15} {:<15} {:<15} {:<10}\n",
                                 "Bin Range (AU)", "Center (AU)", "Objects", "Density"));
        report.push_str("───────────────────────────────────────────────────────────────\n");

        for bin in &self.bins {
            let density_marker = if bin.is_significant() { "▓▓▓" } else { "░░░" };
            report.push_str(&format!("{:<7.0}-{:<7.0} {:<15.1} {:<15} {}\n",
                                     bin.range.0, bin.range.1,
                                     bin.center, bin.members.len(), density_marker));
        }
        report.push_str("\n");

        // Significant clusters
        if !self.significant_clusters.is_empty() {
            report.push_str("🎯 SIGNIFICANT APHELION CLUSTERS (2+ objects)\n");
            report.push_str("───────────────────────────────────────────────────────────────\n");

            for (idx, cluster) in self.significant_clusters.iter().enumerate() {
                report.push_str(&format!("  Cluster {}:\n", idx + 1));
                report.push_str(&format!("    Bin range:        {:.0} - {:.0} AU\n",
                                        cluster.range.0, cluster.range.1));
                report.push_str(&format!("    Mean aphelion:    {:.1} AU (σ = {:.1} AU)\n",
                                        cluster.avg_aphelion, cluster.std_aphelion));
                report.push_str(&format!("    Object count:     {}\n", cluster.members.len()));
                report.push_str(&format!("    Density:          {:.1} objects per 50 AU\n", cluster.density));
                report.push_str(&format!("    Members:          ", ));

                let display_count = 5.min(cluster.members.len());
                for (i, member) in cluster.members.iter().take(display_count).enumerate() {
                    if i > 0 { report.push_str(", "); }
                    report.push_str(member);
                }
                if cluster.members.len() > display_count {
                    report.push_str(&format!(", ... and {} more", cluster.members.len() - display_count));
                }
                report.push_str("\n\n");
            }
        }

        // Estimated planets
        if !self.estimated_planets.is_empty() {
            report.push_str("🪐 ESTIMATED PLANET POSITIONS (60% of Aphelion)\n");
            report.push_str("───────────────────────────────────────────────────────────────\n");

            for planet in &self.estimated_planets {
                let confidence_bar = (planet.confidence * 10.0) as usize;
                let bar = "█".repeat(confidence_bar) + &"░".repeat(10 - confidence_bar);

                report.push_str(&format!("  {}: ~{:.1} AU\n", planet.designation, planet.estimated_a));
                report.push_str(&format!("    Based on: {:.1} AU aphelion cluster\n",
                                        planet.aphelion_cluster));
                report.push_str(&format!("    Shepherded objects: {}\n", planet.shepherded_count));
                report.push_str(&format!("    Confidence: [{}] {:.1}%\n", bar, planet.confidence * 100.0));
                report.push_str("\n");
            }
        }

        // Orbital architecture
        report.push_str("🌌 ORBITAL ARCHITECTURE IMPLICATIONS\n");
        report.push_str("───────────────────────────────────────────────────────────────\n");

        if self.estimated_planets.len() >= 2 {
            report.push_str("  Multiple planet candidates detected:\n");
            for (i, p1) in self.estimated_planets.iter().enumerate() {
                for p2 in self.estimated_planets.iter().skip(i + 1) {
                    let ratio = p2.estimated_a / p1.estimated_a;
                    let resonance_type = match ratio {
                        r if (r - 2.0).abs() < 0.2 => "2:1 resonance zone",
                        r if (r - 1.5).abs() < 0.2 => "3:2 resonance zone",
                        r if (r - 1.67).abs() < 0.2 => "5:3 resonance zone",
                        _ => "separated regime",
                    };
                    report.push_str(&format!("    {}/{}: {:.2} ({})\n",
                                           p2.designation, p1.designation, ratio, resonance_type));
                }
            }
        } else if self.estimated_planets.len() == 1 {
            let p = &self.estimated_planets[0];
            report.push_str(&format!("  Single planet candidate at {:.1} AU\n", p.estimated_a));
            report.push_str(&format!("  Shepherding {} objects with aphelion ~{:.1} AU\n",
                                    p.shepherded_count, p.aphelion_cluster));
        }

        report.push_str("\n");
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("  Data source: NASA/JPL Small-Body Database                    \n");
        report.push_str("  Analysis: RuVector Aphelion Clustering (Agent 7)            \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_objects() -> Vec<KuiperBeltObject> {
        vec![
            // Sedna-like (aphelion ~1022 AU)
            KuiperBeltObject {
                name: "90377 Sedna".to_string(),
                a: 549.5, e: 0.8613, i: 11.93, q: 76.223, ad: 1022.86,
                period: 4710000.0, omega: 144.48, w: 311.01, h: Some(1.49), class: "TNO".to_string(),
            },
            // 2000 CR105 (aphelion ~413 AU)
            KuiperBeltObject {
                name: "148209 (2000 CR105)".to_string(),
                a: 228.7, e: 0.8071, i: 22.71, q: 44.117, ad: 413.29,
                period: 1260000.0, omega: 128.21, w: 316.92, h: Some(6.14), class: "TNO".to_string(),
            },
            // Low-aphelion object
            KuiperBeltObject {
                name: "50000 Quaoar".to_string(),
                a: 43.15, e: 0.0358, i: 7.99, q: 41.601, ad: 44.69,
                period: 104000.0, omega: 188.96, w: 163.92, h: Some(2.41), class: "TNO".to_string(),
            },
        ]
    }

    #[test]
    fn test_aphelion_filter() {
        let objects = create_test_objects();
        let clusterer = AphelionClusterer::new();
        let result = clusterer.cluster(&objects);

        // Should find 2 objects with Q > 100 AU
        assert_eq!(result.distant_object_count, 2);
    }

    #[test]
    fn test_significant_clusters() {
        let objects = create_test_objects();
        let clusterer = AphelionClusterer::new();
        let result = clusterer.cluster(&objects);

        // Should have at least one significant cluster
        assert!(result.significant_clusters.len() >= 1);
    }

    #[test]
    fn test_planet_estimation() {
        let objects = create_test_objects();
        let clusterer = AphelionClusterer::new();
        let result = clusterer.cluster(&objects);

        // Should estimate planets from significant clusters
        assert!(!result.estimated_planets.is_empty());

        // Each planet should be at 60% of cluster aphelion
        for planet in &result.estimated_planets {
            let expected = planet.aphelion_cluster * 0.6;
            assert!((planet.estimated_a - expected).abs() < 0.1);
        }
    }

    #[test]
    fn test_bin_statistics() {
        let objects = create_test_objects();
        let clusterer = AphelionClusterer::new();
        let result = clusterer.cluster(&objects);

        for bin in &result.bins {
            if !bin.members.is_empty() {
                assert!(bin.avg_aphelion > 0.0);
                assert!(bin.density > 0.0);
            }
        }
    }
}
