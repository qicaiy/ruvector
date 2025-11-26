//! # Longitude of Ascending Node (Ω) Clustering Analysis
//!
//! Analysis Agent 3: Specialized clustering study of Ω for distant objects
//! with a > 100 AU. Calculates circular statistics and reports planet
//! longitude candidates.
//!
//! ## Key Analysis:
//! - Filters objects with semi-major axis > 100 AU
//! - Calculates circular variance (R) of Ω values
//! - Performs statistical significance testing
//! - Estimates potential perturbing planet longitude
//! - Identifies dynamically coherent sub-populations
//!
//! ## Theory:
//! A distant planet will create correlations in orbital elements of affected
//! objects, including clustering in Ω (longitude of ascending node). The
//! clustering strength (R value) indicates the significance of the signal.
//!
//! Run with:
//! ```bash
//! cargo run -p ruvector-core --example kuiper_longitude_node_analysis --features storage
//! ```

use std::f64::consts::PI;
use std::collections::BTreeMap;

/// Orbital data structure for analysis
#[derive(Debug, Clone)]
pub struct OrbitalObject {
    pub name: String,
    pub a: f64,      // Semi-major axis (AU)
    pub e: f64,      // Eccentricity
    pub i: f64,      // Inclination (degrees)
    pub omega: f64,  // Longitude of ascending node (degrees)
    pub w: f64,      // Argument of perihelion (degrees)
    pub q: f64,      // Perihelion distance (AU)
    pub ad: f64,     // Aphelion distance (AU)
}

/// Circular statistics result
#[derive(Debug, Clone)]
pub struct CircularStats {
    /// Mean resultant length (0 to 1, measures concentration)
    pub r: f64,
    /// Mean angle (degrees)
    pub mean_angle: f64,
    /// Circular variance (1 - R)
    pub circular_variance: f64,
    /// Circular standard deviation (degrees)
    pub circular_std_dev: f64,
    /// Number of objects in sample
    pub n: usize,
}

impl CircularStats {
    /// Calculate circular statistics for angles (in degrees)
    pub fn from_angles(angles: &[f64]) -> Self {
        let n = angles.len();
        if n == 0 {
            return Self {
                r: 0.0,
                mean_angle: 0.0,
                circular_variance: 1.0,
                circular_std_dev: 0.0,
                n: 0,
            };
        }

        // Convert degrees to radians
        let rad_angles: Vec<f64> = angles.iter().map(|a| a * PI / 180.0).collect();

        // Calculate sin and cos sums
        let sin_sum: f64 = rad_angles.iter().map(|a| a.sin()).sum();
        let cos_sum: f64 = rad_angles.iter().map(|a| a.cos()).sum();

        // Calculate resultant vector length
        let r = ((sin_sum.powi(2) + cos_sum.powi(2)).sqrt()) / n as f64;

        // Calculate mean angle
        let mean_rad = sin_sum.atan2(cos_sum);
        let mean_angle = if mean_rad < 0.0 {
            (mean_rad + 2.0 * PI) * 180.0 / PI
        } else {
            mean_rad * 180.0 / PI
        };

        // Circular variance
        let circular_variance = 1.0 - r;

        // Circular standard deviation (in degrees)
        let circular_std_dev = if r < 0.53 {
            ((2.0 * (1.0 - r)).sqrt()) * 180.0 / PI
        } else {
            ((-2.0 * ((1.0 - r) / (2.0 * r * r)).ln()).sqrt()) * 180.0 / PI
        };

        Self {
            r,
            mean_angle,
            circular_variance,
            circular_std_dev,
            n,
        }
    }

    /// Statistical significance test (Rayleigh test p-value approximation)
    /// Returns approximate p-value (1 = significant, 0 = not significant)
    pub fn rayleigh_significance(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }

        let z = self.n as f64 * self.r * self.r;

        // Approximation of Rayleigh test p-value
        // For large n and small p-values, p ≈ exp(-z) * (1 + (2*z - z^2)/(4*n) - ...)
        if z < 3.0 {
            (-z).exp() * (1.0 + (2.0 * z - z * z) / (4.0 * self.n as f64))
        } else {
            0.0 // Highly significant
        }
    }

    /// Confidence level for clustering (0 = no clustering, 1 = perfect clustering)
    pub fn clustering_confidence(&self) -> f64 {
        // Threshold for significance: R > 0.5 indicates probable clustering
        if self.r < 0.3 {
            0.0
        } else if self.r > 0.8 {
            1.0
        } else {
            (self.r - 0.3) / 0.5 // Linear interpolation
        }
    }
}

/// Results of longitude node clustering analysis
#[derive(Debug, Clone)]
pub struct LongitudeNodeAnalysis {
    /// Objects with a > 100 AU
    pub distant_objects: Vec<OrbitalObject>,

    /// Circular statistics for all distant objects
    pub overall_stats: CircularStats,

    /// Sub-population analyses
    pub subpopulations: Vec<SubpopulationAnalysis>,

    /// Identified clusters (longitude ranges)
    pub clusters: Vec<LongitudeCluster>,

    /// Estimated planet longitude
    pub estimated_planet_longitude: Option<PlanetLongitudeEstimate>,
}

#[derive(Debug, Clone)]
pub struct SubpopulationAnalysis {
    pub name: String,
    pub filter_description: String,
    pub objects: Vec<String>,
    pub stats: CircularStats,
}

#[derive(Debug, Clone)]
pub struct LongitudeCluster {
    pub center_longitude: f64,
    pub width: f64,
    pub objects: Vec<String>,
    pub significance: f64,
}

#[derive(Debug, Clone)]
pub struct PlanetLongitudeEstimate {
    /// Primary estimate of planet longitude
    pub primary_longitude: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Alternative estimates if multi-modal distribution
    pub alternatives: Vec<f64>,
    /// Evidence strength
    pub evidence_strength: String,
}

/// Longitude of Ascending Node Analyzer
pub struct LongitudeNodeAnalyzer;

impl LongitudeNodeAnalyzer {
    /// Run complete analysis on objects
    pub fn analyze(objects: &[OrbitalObject]) -> LongitudeNodeAnalysis {
        // Step 1: Filter distant objects (a > 100 AU)
        let distant_objects: Vec<_> = objects
            .iter()
            .filter(|o| o.a > 100.0)
            .cloned()
            .collect();

        println!("📊 Longitude of Ascending Node (Ω) Clustering Analysis");
        println!("═══════════════════════════════════════════════════════");
        println!("   Objects analyzed: {}", distant_objects.len());
        println!();

        // Step 2: Calculate overall circular statistics
        let omegas: Vec<f64> = distant_objects.iter().map(|o| o.omega).collect();
        let overall_stats = CircularStats::from_angles(&omegas);

        // Step 3: Analyze sub-populations
        let subpopulations = Self::analyze_subpopulations(&distant_objects);

        // Step 4: Identify clusters
        let clusters = Self::identify_clusters(&distant_objects, &overall_stats);

        // Step 5: Estimate planet longitude
        let estimated_planet_longitude = Self::estimate_planet_longitude(
            &distant_objects,
            &overall_stats,
            &clusters,
        );

        LongitudeNodeAnalysis {
            distant_objects,
            overall_stats,
            subpopulations,
            clusters,
            estimated_planet_longitude,
        }
    }

    /// Analyze sub-populations by various criteria
    fn analyze_subpopulations(objects: &[OrbitalObject]) -> Vec<SubpopulationAnalysis> {
        let mut subpops = Vec::new();

        // Sub-population 1: Extreme TNOs (a > 250 AU)
        let etnos: Vec<_> = objects
            .iter()
            .filter(|o| o.a > 250.0)
            .cloned()
            .collect();
        if !etnos.is_empty() {
            let omegas: Vec<f64> = etnos.iter().map(|o| o.omega).collect();
            let names: Vec<String> = etnos.iter().map(|o| o.name.clone()).collect();
            subpops.push(SubpopulationAnalysis {
                name: "Extreme TNOs (a > 250 AU)".to_string(),
                filter_description: "Objects beyond 250 AU".to_string(),
                objects: names,
                stats: CircularStats::from_angles(&omegas),
            });
        }

        // Sub-population 2: High eccentricity (e > 0.7)
        let high_e: Vec<_> = objects
            .iter()
            .filter(|o| o.e > 0.7)
            .cloned()
            .collect();
        if !high_e.is_empty() {
            let omegas: Vec<f64> = high_e.iter().map(|o| o.omega).collect();
            let names: Vec<String> = high_e.iter().map(|o| o.name.clone()).collect();
            subpops.push(SubpopulationAnalysis {
                name: "High Eccentricity (e > 0.7)".to_string(),
                filter_description: "Very elliptical orbits".to_string(),
                objects: names,
                stats: CircularStats::from_angles(&omegas),
            });
        }

        // Sub-population 3: High inclination (i > 20°)
        let high_i: Vec<_> = objects
            .iter()
            .filter(|o| o.i > 20.0)
            .cloned()
            .collect();
        if !high_i.is_empty() {
            let omegas: Vec<f64> = high_i.iter().map(|o| o.omega).collect();
            let names: Vec<String> = high_i.iter().map(|o| o.name.clone()).collect();
            subpops.push(SubpopulationAnalysis {
                name: "High Inclination (i > 20°)".to_string(),
                filter_description: "Highly inclined orbits".to_string(),
                objects: names,
                stats: CircularStats::from_angles(&omegas),
            });
        }

        // Sub-population 4: Low perihelion (q > 40 AU) - detached objects
        let detached: Vec<_> = objects
            .iter()
            .filter(|o| o.q > 40.0)
            .cloned()
            .collect();
        if !detached.is_empty() {
            let omegas: Vec<f64> = detached.iter().map(|o| o.omega).collect();
            let names: Vec<String> = detached.iter().map(|o| o.name.clone()).collect();
            subpops.push(SubpopulationAnalysis {
                name: "Detached Objects (q > 40 AU)".to_string(),
                filter_description: "Objects with high perihelion".to_string(),
                objects: names,
                stats: CircularStats::from_angles(&omegas),
            });
        }

        subpops
    }

    /// Identify clusters in the longitude distribution
    fn identify_clusters(
        objects: &[OrbitalObject],
        stats: &CircularStats,
    ) -> Vec<LongitudeCluster> {
        let mut clusters = Vec::new();

        if objects.is_empty() || stats.r < 0.3 {
            return clusters;
        }

        // Use histogram binning to identify cluster centers
        let mut bins: BTreeMap<u32, Vec<String>> = BTreeMap::new();

        for obj in objects {
            let bin = (obj.omega / 10.0).floor() as u32; // 10-degree bins
            bins.entry(bin).or_insert_with(Vec::new).push(obj.name.clone());
        }

        // Find peak bins (potential clusters)
        let max_count = bins.values().map(|v| v.len()).max().unwrap_or(0);
        let threshold = (max_count as f64 * 0.5).ceil() as usize;

        for (bin, names) in bins {
            if names.len() >= threshold.max(2) {
                let center = (bin as f64 + 0.5) * 10.0;
                clusters.push(LongitudeCluster {
                    center_longitude: center,
                    width: 20.0, // ±10 degrees
                    objects: names,
                    significance: stats.r,
                });
            }
        }

        clusters
    }

    /// Estimate the longitude of the perturbing planet
    fn estimate_planet_longitude(
        objects: &[OrbitalObject],
        stats: &CircularStats,
        clusters: &[LongitudeCluster],
    ) -> Option<PlanetLongitudeEstimate> {
        if stats.r < 0.3 {
            return None; // Clustering not significant
        }

        let mut evidence_strength = "Weak".to_string();
        let mut confidence = stats.r;

        if stats.r > 0.5 && !clusters.is_empty() {
            evidence_strength = "Moderate".to_string();
        }

        if stats.r > 0.7 {
            evidence_strength = "Strong".to_string();
        }

        let mut alternatives = Vec::new();

        // Primary estimate: mean longitude
        let primary_longitude = stats.mean_angle;

        // Alternative 1: Anti-aligned (180° opposite)
        alternatives.push((primary_longitude + 180.0) % 360.0);

        // Alternative 2: Cluster centers if available
        for cluster in clusters {
            let alt_long = (cluster.center_longitude + 180.0) % 360.0;
            if !alternatives.contains(&alt_long) {
                alternatives.push(alt_long);
            }
        }

        Some(PlanetLongitudeEstimate {
            primary_longitude,
            confidence,
            alternatives,
            evidence_strength,
        })
    }

    /// Generate comprehensive report
    pub fn generate_report(analysis: &LongitudeNodeAnalysis) -> String {
        let mut report = String::new();

        report.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        report.push_str("║   LONGITUDE OF ASCENDING NODE (Ω) CLUSTERING ANALYSIS      ║\n");
        report.push_str("║        Analysis Agent 3: Distant Object Distribution        ║\n");
        report.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

        // Main Results
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("                    MAIN RESULTS                              \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str(&format!("📊 Objects Analyzed (a > 100 AU): {}\n", analysis.distant_objects.len()));
        report.push_str(&format!("📈 Mean Resultant Length (R):      {:.4}\n", analysis.overall_stats.r));
        report.push_str(&format!("📉 Circular Variance:              {:.4}\n", analysis.overall_stats.circular_variance));
        report.push_str(&format!("🔄 Mean Ω:                         {:.2}°\n", analysis.overall_stats.mean_angle));
        report.push_str(&format!("🎲 Circular Std Dev:               {:.2}°\n", analysis.overall_stats.circular_std_dev));
        report.push_str("\n");

        // Significance interpretation
        let significance = 1.0 - analysis.overall_stats.rayleigh_significance();
        let clustering_conf = analysis.overall_stats.clustering_confidence();

        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str("CLUSTERING SIGNIFICANCE\n");
        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str(&format!("Rayleigh Test (p-value):           {:.6}\n", 1.0 - significance));
        report.push_str(&format!("Clustering Confidence:             {:.2}%\n", clustering_conf * 100.0));

        if analysis.overall_stats.r < 0.3 {
            report.push_str("Assessment: ⚪ Random distribution - no significant clustering\n");
        } else if analysis.overall_stats.r < 0.5 {
            report.push_str("Assessment: 🟡 Weak clustering - possible but uncertain signal\n");
        } else if analysis.overall_stats.r < 0.7 {
            report.push_str("Assessment: 🟠 Moderate clustering - significant deviation from random\n");
        } else {
            report.push_str("Assessment: 🔴 Strong clustering - highly significant signal\n");
        }
        report.push_str("\n");

        // Sub-populations
        if !analysis.subpopulations.is_empty() {
            report.push_str("═══════════════════════════════════════════════════════════════\n");
            report.push_str("                SUB-POPULATION ANALYSIS                        \n");
            report.push_str("═══════════════════════════════════════════════════════════════\n\n");

            for sub in &analysis.subpopulations {
                report.push_str(&format!("🔬 {}\n", sub.name));
                report.push_str(&format!("   Objects: {}\n", sub.objects.len()));
                report.push_str(&format!("   R-value: {:.4}\n", sub.stats.r));
                report.push_str(&format!("   Mean Ω: {:.2}°\n", sub.stats.mean_angle));

                let conf = sub.stats.clustering_confidence();
                if conf > 0.5 {
                    report.push_str(&format!("   ✓ Clustering detected ({:.0}% confidence)\n", conf * 100.0));
                } else {
                    report.push_str(&format!("   ○ No significant clustering\n"));
                }
                report.push_str("\n");
            }
        }

        // Identified clusters
        if !analysis.clusters.is_empty() {
            report.push_str("═══════════════════════════════════════════════════════════════\n");
            report.push_str("                  IDENTIFIED CLUSTERS                         \n");
            report.push_str("═══════════════════════════════════════════════════════════════\n\n");

            for (i, cluster) in analysis.clusters.iter().enumerate() {
                report.push_str(&format!("Cluster {}: Ω ≈ {:.1}° (±{:.1}°)\n", i + 1,
                    cluster.center_longitude, cluster.width / 2.0));
                report.push_str(&format!("   Objects: {}\n", cluster.objects.len()));
                report.push_str(&format!("   Significance: {:.4}\n", cluster.significance));
                report.push_str("\n");
            }
        }

        // Planet longitude estimate
        if let Some(estimate) = &analysis.estimated_planet_longitude {
            report.push_str("═══════════════════════════════════════════════════════════════\n");
            report.push_str("              ESTIMATED PERTURBING PLANET LONGITUDE            \n");
            report.push_str("═══════════════════════════════════════════════════════════════\n\n");

            report.push_str(&format!("💫 Primary Estimate:  {:.1}°\n", estimate.primary_longitude));
            report.push_str(&format!("📊 Confidence Level:  {:.1}%\n", estimate.confidence * 100.0));
            report.push_str(&format!("💪 Evidence Strength: {}\n", estimate.evidence_strength));

            if !estimate.alternatives.is_empty() {
                report.push_str("\n   Alternative Longitudes:\n");
                for (i, alt) in estimate.alternatives.iter().enumerate() {
                    report.push_str(&format!("     • {:.1}° (anti-aligned offset {})\n", alt,
                        if i == 0 { "180°".to_string() } else { format!("{}°", (i as f64) * 90.0) }));
                }
            }
            report.push_str("\n");

            report.push_str("⚠️  INTERPRETATION:\n");
            if estimate.confidence < 0.4 {
                report.push_str("   • Weak clustering suggests either:\n");
                report.push_str("     - Random distribution (no distant planet)\n");
                report.push_str("     - Object sample too small or incomplete\n");
            } else if estimate.confidence < 0.6 {
                report.push_str("   • Moderate clustering may indicate:\n");
                report.push_str("     - Weak perturbation from distant planet\n");
                report.push_str("     - Insufficient data for confirmation\n");
            } else {
                report.push_str("   • Strong clustering suggests:\n");
                report.push_str("     - Significant perturbing body present\n");
                report.push_str("     - Orbital parameters show correlation\n");
                report.push_str("     - Further investigation recommended\n");
            }
            report.push_str("\n");
        }

        // Objects list
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("                   ANALYZED OBJECTS                           \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");
        report.push_str(&format!("{:<35} {:>8} {:>8} {:>8}\n", "Name", "a (AU)", "e", "Ω (°)"));
        report.push_str("───────────────────────────────────────────────────────────────\n");

        for obj in &analysis.distant_objects {
            report.push_str(&format!("{:<35} {:>8.1} {:>8.3} {:>8.1}\n",
                obj.name, obj.a, obj.e, obj.omega));
        }
        report.push_str("\n");

        report
    }
}

/// Load sample KBO data for testing
pub fn get_distant_kbo_data() -> Vec<OrbitalObject> {
    vec![
        // Extreme TNOs
        OrbitalObject {
            name: "Sedna".to_string(),
            a: 506.0,
            e: 0.855,
            i: 11.93,
            omega: 144.48,
            w: 311.01,
            q: 76.223,
            ad: 1022.86,
        },
        OrbitalObject {
            name: "2012 VP113".to_string(),
            a: 256.0,
            e: 0.69,
            i: 24.1,
            omega: 90.8,
            w: 293.8,
            q: 80.5,
            ad: 431.5,
        },
        OrbitalObject {
            name: "Leleakuhonua".to_string(),
            a: 1085.0,
            e: 0.94,
            i: 11.7,
            omega: 300.8,
            w: 118.0,
            q: 65.0,
            ad: 2105.0,
        },
        OrbitalObject {
            name: "2013 SY99".to_string(),
            a: 735.0,
            e: 0.93,
            i: 4.2,
            omega: 32.3,
            w: 32.1,
            q: 50.0,
            ad: 1420.0,
        },
        OrbitalObject {
            name: "2015 TG387".to_string(),
            a: 1094.0,
            e: 0.94,
            i: 11.7,
            omega: 300.8,
            w: 118.2,
            q: 65.0,
            ad: 2123.0,
        },
        OrbitalObject {
            name: "2007 TG422".to_string(),
            a: 501.0,
            e: 0.93,
            i: 18.6,
            omega: 112.9,
            w: 285.7,
            q: 35.6,
            ad: 966.0,
        },
        OrbitalObject {
            name: "2013 RF98".to_string(),
            a: 350.0,
            e: 0.89,
            i: 29.6,
            omega: 67.6,
            w: 316.5,
            q: 36.1,
            ad: 664.0,
        },
        OrbitalObject {
            name: "2014 SR349".to_string(),
            a: 298.0,
            e: 0.84,
            i: 18.0,
            omega: 34.8,
            w: 341.4,
            q: 47.6,
            ad: 548.4,
        },
        OrbitalObject {
            name: "2010 GB174".to_string(),
            a: 370.0,
            e: 0.87,
            i: 21.5,
            omega: 130.6,
            w: 347.8,
            q: 48.8,
            ad: 691.2,
        },
        OrbitalObject {
            name: "2004 VN112".to_string(),
            a: 327.0,
            e: 0.85,
            i: 25.6,
            omega: 66.0,
            w: 327.1,
            q: 47.3,
            ad: 606.7,
        },
        OrbitalObject {
            name: "2000 CR105".to_string(),
            a: 228.7,
            e: 0.80,
            i: 22.71,
            omega: 128.21,
            w: 316.92,
            q: 44.117,
            ad: 413.29,
        },
        OrbitalObject {
            name: "2012 GB174".to_string(),
            a: 677.0,
            e: 0.93,
            i: 21.5,
            omega: 130.6,
            w: 347.0,
            q: 48.7,
            ad: 1305.0,
        },
        OrbitalObject {
            name: "2015 GT50".to_string(),
            a: 312.0,
            e: 0.88,
            i: 8.8,
            omega: 46.1,
            w: 129.0,
            q: 38.4,
            ad: 585.6,
        },
        OrbitalObject {
            name: "2015 RX245".to_string(),
            a: 430.0,
            e: 0.89,
            i: 12.1,
            omega: 8.6,
            w: 65.2,
            q: 45.5,
            ad: 814.5,
        },
        OrbitalObject {
            name: "2013 FT28".to_string(),
            a: 310.0,
            e: 0.86,
            i: 17.3,
            omega: 217.8,
            w: 40.2,
            q: 43.5,
            ad: 576.5,
        },
        OrbitalObject {
            name: "2014 FE72".to_string(),
            a: 2155.0,
            e: 0.98,
            i: 20.6,
            omega: 336.8,
            w: 134.0,
            q: 36.3,
            ad: 4274.0,
        },
        OrbitalObject {
            name: "2005 RH52".to_string(),
            a: 153.0,
            e: 0.74,
            i: 20.5,
            omega: 306.1,
            w: 32.4,
            q: 39.0,
            ad: 267.0,
        },
        OrbitalObject {
            name: "Gonggong".to_string(),
            a: 66.89,
            e: 0.5032,
            i: 30.87,
            omega: 336.84,
            w: 206.64,
            q: 33.235,
            ad: 100.55,
        },
        OrbitalObject {
            name: "Eris".to_string(),
            a: 68.0,
            e: 0.4370,
            i: 43.87,
            omega: 36.03,
            w: 150.73,
            q: 38.284,
            ad: 97.71,
        },
        OrbitalObject {
            name: "1999 TD10".to_string(),
            a: 98.47,
            e: 0.8743,
            i: 5.96,
            omega: 184.61,
            w: 173.03,
            q: 12.374,
            ad: 184.56,
        },
        OrbitalObject {
            name: "2002 TC302".to_string(),
            a: 55.84,
            e: 0.2995,
            i: 35.01,
            omega: 23.83,
            w: 86.07,
            q: 39.113,
            ad: 72.56,
        },
        OrbitalObject {
            name: "2005 TB190".to_string(),
            a: 75.93,
            e: 0.3912,
            i: 26.48,
            omega: 180.46,
            w: 171.99,
            q: 46.227,
            ad: 105.64,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_stats_random() {
        let angles = vec![10.0, 50.0, 100.0, 200.0, 300.0];
        let stats = CircularStats::from_angles(&angles);

        // Random distribution should have low R
        assert!(stats.r < 0.6);
    }

    #[test]
    fn test_circular_stats_clustered() {
        let angles = vec![10.0, 15.0, 20.0, 25.0, 30.0];
        let stats = CircularStats::from_angles(&angles);

        // Clustered distribution should have high R
        assert!(stats.r > 0.9);
        assert!(stats.circular_variance < 0.1);
    }

    #[test]
    fn test_analysis_runs() {
        let data = get_distant_kbo_data();
        let analysis = LongitudeNodeAnalyzer::analyze(&data);

        assert!(!analysis.distant_objects.is_empty());
        assert!(analysis.overall_stats.n > 0);
    }
}
