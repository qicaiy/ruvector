//! # Semi-Major Axis Gap Analysis for Planet Discovery
//!
//! Analysis Agent 5: Identifies significant gaps in the semi-major axis distribution
//! of Kuiper Belt Objects, with focus on regions beyond 50 AU.
//!
//! ## Methodology
//! - Sort all KBOs by semi-major axis (a)
//! - Identify gaps > 20 AU with lower bound > 50 AU
//! - Calculate planet location at gap midpoint
//! - Estimate planet mass from gap size and object distribution
//! - Cross-reference with known resonances and orbital families
//!
//! ## Expected Results
//! Gaps in orbital distribution suggest dynamical clearing by massive perturbers
//! (e.g., undiscovered planets). The gap center represents the most likely orbital
//! location of the perturbing body.

use std::collections::HashMap;
use crate::kuiper_cluster::KuiperBeltObject;

/// Gap analysis result from semi-major axis distribution
#[derive(Debug, Clone)]
pub struct SMAGap {
    /// Lower semi-major axis bound of gap (AU)
    pub lower_bound: f64,
    /// Upper semi-major axis bound of gap (AU)
    pub upper_bound: f64,
    /// Gap width (AU)
    pub gap_size: f64,
    /// Estimated planet semi-major axis (center of gap)
    pub estimated_planet_a: f64,
    /// Objects just inside (lower) boundary
    pub lower_objects: Vec<String>,
    /// Objects just outside (upper) boundary
    pub upper_objects: Vec<String>,
    /// Number of objects in lower bin
    pub lower_count: usize,
    /// Number of objects in upper bin
    pub upper_count: usize,
    /// Dynamical significance score (0-1)
    pub significance: f64,
}

/// Comprehensive SMA gap analysis result
#[derive(Debug, Clone)]
pub struct SMAGapAnalysisResult {
    /// Total number of objects analyzed
    pub total_objects: usize,
    /// Number of gaps detected
    pub gap_count: usize,
    /// Significant gaps (> 20 AU beyond 50 AU)
    pub significant_gaps: Vec<SMAGap>,
    /// All gaps found (for completeness)
    pub all_gaps: Vec<SMAGap>,
    /// Sorted semi-major axis values
    pub sorted_a_values: Vec<f64>,
    /// Gap statistics summary
    pub stats: GapStatistics,
}

/// Statistical summary of gap analysis
#[derive(Debug, Clone)]
pub struct GapStatistics {
    /// Mean gap size across all objects
    pub mean_gap_size: f64,
    /// Standard deviation of gaps
    pub std_dev_gap: f64,
    /// Largest gap overall
    pub largest_gap: f64,
    /// Largest gap beyond 50 AU
    pub largest_gap_beyond_50au: f64,
    /// Median gap size
    pub median_gap_size: f64,
    /// Number of gaps > 5 AU
    pub gaps_gt_5au: usize,
    /// Number of gaps > 10 AU
    pub gaps_gt_10au: usize,
    /// Number of gaps > 20 AU
    pub gaps_gt_20au: usize,
}

impl SMAGapAnalysisResult {
    /// Generate formatted analysis report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        report.push_str("║          SEMI-MAJOR AXIS GAP ANALYSIS - AGENT 5              ║\n");
        report.push_str("║        Potential Planet Discovery via Orbital Clearing       ║\n");
        report.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

        // Summary section
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("                     ANALYSIS SUMMARY                         \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str(&format!("📊 Objects Analyzed:                    {}\n", self.total_objects));
        report.push_str(&format!("🔍 Total Gaps Detected:                 {}\n", self.gap_count));
        report.push_str(&format!("⚠️  Significant Gaps (>20 AU, >50 AU):  {}\n\n", self.significant_gaps.len()));

        // Statistics section
        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str("               GAP STATISTICS\n");
        report.push_str("───────────────────────────────────────────────────────────────\n\n");

        report.push_str(&format!("Mean gap size:                          {:.2} AU\n", self.stats.mean_gap_size));
        report.push_str(&format!("Standard deviation:                     {:.2} AU\n", self.stats.std_dev_gap));
        report.push_str(&format!("Median gap size:                        {:.2} AU\n", self.stats.median_gap_size));
        report.push_str(&format!("Largest gap overall:                    {:.2} AU\n", self.stats.largest_gap));
        report.push_str(&format!("Largest gap (a > 50 AU):                {:.2} AU\n\n", self.stats.largest_gap_beyond_50au));

        report.push_str("Gap Distribution:\n");
        report.push_str(&format!("  • Gaps > 5 AU:                        {}\n", self.stats.gaps_gt_5au));
        report.push_str(&format!("  • Gaps > 10 AU:                       {}\n", self.stats.gaps_gt_10au));
        report.push_str(&format!("  • Gaps > 20 AU:                       {}\n\n", self.stats.gaps_gt_20au));

        // Significant gaps section
        if !self.significant_gaps.is_empty() {
            report.push_str("═══════════════════════════════════════════════════════════════\n");
            report.push_str("              SIGNIFICANT GAPS (> 20 AU, > 50 AU)              \n");
            report.push_str("═══════════════════════════════════════════════════════════════\n\n");

            for (idx, gap) in self.significant_gaps.iter().enumerate() {
                report.push_str(&format!("Gap #{}\n", idx + 1));
                report.push_str("───────────────────────────────────────────────────────────────\n");
                report.push_str(&format!("  Orbital Region:     {:.2} AU - {:.2} AU\n", gap.lower_bound, gap.upper_bound));
                report.push_str(&format!("  Gap Width:          {:.2} AU\n", gap.gap_size));
                report.push_str(&format!("  ★ PLANET LOCATION:  {:.2} AU (estimated)\n", gap.estimated_planet_a));
                report.push_str(&format!("  Significance:       {:.2} (score 0-1)\n\n", gap.significance));

                report.push_str("  Lower Boundary Objects (just inside gap):\n");
                if gap.lower_objects.is_empty() {
                    report.push_str("    (none)\n");
                } else {
                    for obj in gap.lower_objects.iter().take(5) {
                        report.push_str(&format!("    • {}\n", obj));
                    }
                    if gap.lower_objects.len() > 5 {
                        report.push_str(&format!("    ... and {} more\n", gap.lower_objects.len() - 5));
                    }
                }

                report.push_str(&format!("\n  Upper Boundary Objects (just outside gap):\n"));
                if gap.upper_objects.is_empty() {
                    report.push_str("    (none)\n");
                } else {
                    for obj in gap.upper_objects.iter().take(5) {
                        report.push_str(&format!("    • {}\n", obj));
                    }
                    if gap.upper_objects.len() > 5 {
                        report.push_str(&format!("    ... and {} more\n", gap.upper_objects.len() - 5));
                    }
                }

                report.push_str(&format!("\n  Population Change: {} objects → {} objects\n\n",
                    gap.lower_count, gap.upper_count));
            }
        } else {
            report.push_str("\n⚠️  No significant gaps (>20 AU) found beyond 50 AU\n\n");
        }

        // Semi-major axis distribution
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("          SEMI-MAJOR AXIS DISTRIBUTION (ALL OBJECTS)          \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        // Create histogram of a values
        report.push_str("Semi-major axis (AU)    Count\n");
        report.push_str("───────────────────────────────\n");

        let bins = vec![
            (30.0, 40.0, "30-40 AU (Inner)"),
            (40.0, 50.0, "40-50 AU (Classical)"),
            (50.0, 100.0, "50-100 AU (Scattered)"),
            (100.0, 200.0, "100-200 AU (Distant)"),
            (200.0, f64::INFINITY, "200+ AU (Extreme)"),
        ];

        for (min, max, label) in bins {
            let count = self.sorted_a_values.iter()
                .filter(|&&a| a >= min && a < max)
                .count();
            let bar = "█".repeat(count.min(50));
            report.push_str(&format!("{:<25} {} ({})\n", label, bar, count));
        }

        report.push_str("\n");

        // Detailed gap analysis
        if self.significant_gaps.len() > 0 {
            report.push_str("═══════════════════════════════════════════════════════════════\n");
            report.push_str("                  INTERPRETATION & PHYSICS                    \n");
            report.push_str("═══════════════════════════════════════════════════════════════\n\n");

            report.push_str("ORBITAL CLEARING MECHANISM:\n");
            report.push_str("When a massive object orbits at semi-major axis 'a', its gravitational\n");
            report.push_str("influence creates a 'clearing' region where smaller objects are scattered\n");
            report.push_str("outward or inward. This produces observable gaps in the orbital distribution.\n\n");

            report.push_str("EVIDENCE STRENGTH:\n");
            for (idx, gap) in self.significant_gaps.iter().enumerate() {
                let strength = if gap.gap_size > 50.0 {
                    "VERY STRONG"
                } else if gap.gap_size > 30.0 {
                    "STRONG"
                } else if gap.gap_size > 20.0 {
                    "MODERATE"
                } else {
                    "WEAK"
                };
                report.push_str(&format!("  Gap #{}: {} ({:.1} AU wide)\n", idx + 1, strength, gap.gap_size));
            }

            report.push_str("\nESTIMATED PLANET PARAMETERS:\n");
            for (idx, gap) in self.significant_gaps.iter().enumerate() {
                let mass_est = estimate_mass_from_gap(gap.gap_size);
                report.push_str(&format!("  Gap #{}: a ≈ {:.0} AU, M ≈ {:.1} Earth masses (estimated)\n",
                    idx + 1, gap.estimated_planet_a, mass_est));
            }
        }

        report.push_str("\n═══════════════════════════════════════════════════════════════\n");
        report.push_str("                    NEXT STEPS FOR VERIFICATION                 \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str("1. ORBITAL INTEGRATION\n");
        report.push_str("   • Integrate orbits of near-gap objects backward in time\n");
        report.push_str("   • Check for consistent scattering from planet location\n\n");

        report.push_str("2. DYNAMICAL RESONANCE ANALYSIS\n");
        report.push_str("   • Calculate mean-motion resonances with proposed planet\n");
        report.push_str("   • Look for 1:1, 2:1, 3:2 capture signatures\n\n");

        report.push_str("3. GRAVITATIONAL SCULPTING MODELS\n");
        report.push_str("   • Simulate N-body dynamics with hypothetical planet\n");
        report.push_str("   • Verify gap structure matches observations\n\n");

        report.push_str("4. OBSERVATIONAL CAMPAIGNS\n");
        report.push_str("   • Direct imaging at estimated orbital region\n");
        report.push_str("   • Infrared detection (thermal signature)\n");
        report.push_str("   • Occultation surveys\n\n");

        report
    }
}

/// Estimate planet mass from gap size (simplified empirical relation)
fn estimate_mass_from_gap(gap_size: f64) -> f64 {
    // Empirical scaling: larger gaps suggest more massive planets
    // Gap width ~ sqrt(M) for clearing zones
    // Assume baseline: 20 AU gap ≈ 5 Earth masses
    let baseline_gap = 20.0;
    let baseline_mass = 5.0;

    baseline_mass * (gap_size / baseline_gap).sqrt()
}

/// Analyze semi-major axis distribution for gaps
pub fn analyze_sma_gaps(objects: &[KuiperBeltObject]) -> SMAGapAnalysisResult {
    // Convert to f64 for precision
    let mut a_values: Vec<(f64, String)> = objects.iter()
        .map(|o| (o.a as f64, o.name.clone()))
        .collect();

    // Sort by semi-major axis
    a_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let sorted_a: Vec<f64> = a_values.iter().map(|a| a.0).collect();
    let sorted_names: Vec<String> = a_values.iter().map(|a| a.1.clone()).collect();

    // Calculate all gaps
    let mut all_gaps = Vec::new();
    let mut gap_sizes = Vec::new();

    for i in 1..sorted_a.len() {
        let gap = sorted_a[i] - sorted_a[i-1];
        let lower_a = sorted_a[i-1];
        let upper_a = sorted_a[i];

        // Collect nearby objects for context
        let lower_obj = sorted_names[i-1].clone();
        let upper_obj = sorted_names[i].clone();

        all_gaps.push((lower_a, upper_a, gap, lower_obj.clone(), upper_obj.clone()));
        gap_sizes.push(gap);
    }

    // Find significant gaps (> 20 AU beyond 50 AU)
    let significant_gaps: Vec<SMAGap> = all_gaps.iter()
        .filter(|(lower, _, gap, _, _)| *gap > 20.0 && *lower > 50.0)
        .map(|(lower_a, upper_a, gap_size, lower_obj, upper_obj)| {
            // Find objects near gap boundaries
            let lower_objects: Vec<String> = sorted_names.iter()
                .zip(sorted_a.iter())
                .filter(|(_, &a)| a >= lower_a - 5.0 && a <= *lower_a)
                .map(|(name, _)| name.clone())
                .collect();

            let upper_objects: Vec<String> = sorted_names.iter()
                .zip(sorted_a.iter())
                .filter(|(_, &a)| a >= *upper_a && a <= upper_a + 5.0)
                .map(|(name, _)| name.clone())
                .collect();

            SMAGap {
                lower_bound: *lower_a,
                upper_bound: *upper_a,
                gap_size: *gap_size,
                estimated_planet_a: (lower_a + upper_a) / 2.0,
                lower_count: lower_objects.len(),
                upper_count: upper_objects.len(),
                lower_objects,
                upper_objects,
                significance: (*gap_size / 50.0).min(1.0), // Normalize to 0-1
            }
        })
        .collect();

    // Calculate statistics
    let mean_gap = gap_sizes.iter().sum::<f64>() / gap_sizes.len() as f64;
    let variance = gap_sizes.iter()
        .map(|g| (g - mean_gap).powi(2))
        .sum::<f64>() / gap_sizes.len() as f64;
    let std_dev = variance.sqrt();

    let mut sorted_gaps = gap_sizes.clone();
    sorted_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_gap = sorted_gaps[sorted_gaps.len() / 2];

    let largest_gap = gap_sizes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let largest_gap_beyond_50 = all_gaps.iter()
        .filter(|(lower, _, _, _, _)| *lower > 50.0)
        .map(|(_, _, gap, _, _)| gap)
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let gaps_gt_5au = gap_sizes.iter().filter(|&&g| g > 5.0).count();
    let gaps_gt_10au = gap_sizes.iter().filter(|&&g| g > 10.0).count();
    let gaps_gt_20au = gap_sizes.iter().filter(|&&g| g > 20.0).count();

    let stats = GapStatistics {
        mean_gap_size: mean_gap,
        std_dev_gap: std_dev,
        largest_gap,
        largest_gap_beyond_50au: if largest_gap_beyond_50 == f64::NEG_INFINITY { 0.0 } else { largest_gap_beyond_50 },
        median_gap_size: median_gap,
        gaps_gt_5au,
        gaps_gt_10au,
        gaps_gt_20au,
    };

    SMAGapAnalysisResult {
        total_objects: objects.len(),
        gap_count: all_gaps.len(),
        significant_gaps,
        all_gaps: all_gaps.into_iter()
            .map(|(lower, upper, gap, lower_obj, upper_obj)| {
                let lower_objects = vec![lower_obj];
                let upper_objects = vec![upper_obj];
                SMAGap {
                    lower_bound: lower,
                    upper_bound: upper,
                    gap_size: gap,
                    estimated_planet_a: (lower + upper) / 2.0,
                    lower_count: 1,
                    upper_count: 1,
                    lower_objects,
                    upper_objects,
                    significance: (gap / 50.0).min(1.0),
                }
            })
            .collect(),
        sorted_a_values: sorted_a,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_analysis_basic() {
        // Create mock objects with known gaps
        let mut objects = Vec::new();

        // Cluster 1: 40-45 AU
        for i in 0..5 {
            let obj = KuiperBeltObject {
                name: format!("Obj1-{}", i),
                a: 40.0 + i as f32 * 0.8,
                e: 0.1,
                i: 5.0,
                q: 35.0,
                ad: 45.0,
                period: 100000.0,
                omega: 0.0,
                w: 0.0,
                h: None,
                class: "TNO".to_string(),
            };
            objects.push(obj);
        }

        // Large gap: 45-70 AU
        // Cluster 2: 70-75 AU
        for i in 0..3 {
            let obj = KuiperBeltObject {
                name: format!("Obj2-{}", i),
                a: 70.0 + i as f32 * 1.2,
                e: 0.2,
                i: 10.0,
                q: 55.0,
                ad: 85.0,
                period: 300000.0,
                omega: 45.0,
                w: 45.0,
                h: None,
                class: "TNO".to_string(),
            };
            objects.push(obj);
        }

        let result = analyze_sma_gaps(&objects);

        assert_eq!(result.total_objects, 8);
        assert!(result.significant_gaps.len() > 0);
        assert!(result.stats.largest_gap > 20.0);
    }
}
