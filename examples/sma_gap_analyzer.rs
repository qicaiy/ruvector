//! # Semi-Major Axis Gap Analysis - Agent 5
//!
//! Specialized analysis for detecting gaps in the semi-major axis distribution
//! of Kuiper Belt Objects. These gaps indicate dynamical clearing by massive
//! perturbing bodies (potential undiscovered planets).
//!
//! ## Run with:
//! ```bash
//! cargo run --example sma_gap_analyzer
//! ```

use std::collections::HashMap;

/// Kuiper Belt Object structure
#[derive(Debug, Clone)]
struct KBObject {
    name: String,
    a: f64,  // Semi-major axis in AU
}

/// Gap analysis result from semi-major axis distribution
#[derive(Debug, Clone)]
struct SMAGap {
    lower_bound: f64,
    upper_bound: f64,
    gap_size: f64,
    estimated_planet_a: f64,
    lower_objects: Vec<String>,
    upper_objects: Vec<String>,
    lower_count: usize,
    upper_count: usize,
    significance: f64,
}

/// Main analysis result
#[derive(Debug, Clone)]
struct SMAGapAnalysisResult {
    total_objects: usize,
    gap_count: usize,
    significant_gaps: Vec<SMAGap>,
    sorted_a_values: Vec<f64>,
    stats: GapStatistics,
}

#[derive(Debug, Clone)]
struct GapStatistics {
    mean_gap_size: f64,
    std_dev_gap: f64,
    largest_gap: f64,
    largest_gap_beyond_50au: f64,
    median_gap_size: f64,
    gaps_gt_5au: usize,
    gaps_gt_10au: usize,
    gaps_gt_20au: usize,
}

/// Estimate planet mass from gap size
fn estimate_mass_from_gap(gap_size: f64) -> f64 {
    // Empirical scaling: larger gaps suggest more massive planets
    // Gap width ~ sqrt(M) for clearing zones
    let baseline_gap = 20.0;
    let baseline_mass = 5.0;
    baseline_mass * (gap_size / baseline_gap).sqrt()
}

/// Get KBO data for Kuiper Belt analysis
fn get_kbo_data_subset() -> Vec<KBObject> {
    vec![
        // Plutinos (3:2 resonance ~39.4 AU)
        KBObject { name: "Pluto".to_string(), a: 39.59 },
        KBObject { name: "Orcus".to_string(), a: 39.34 },
        KBObject { name: "Ixion".to_string(), a: 39.35 },
        KBObject { name: "Huya".to_string(), a: 39.21 },

        // Classical KBOs (42-48 AU, low e)
        KBObject { name: "Quaoar".to_string(), a: 43.15 },
        KBObject { name: "Varuna".to_string(), a: 43.18 },
        KBObject { name: "Chaos".to_string(), a: 46.11 },
        KBObject { name: "Albion".to_string(), a: 44.2 },
        KBObject { name: "Makemake".to_string(), a: 45.51 },
        KBObject { name: "Haumea".to_string(), a: 43.01 },

        // Twotinos (2:1 resonance ~47.8 AU)
        KBObject { name: "1996 TR66".to_string(), a: 47.98 },
        KBObject { name: "2002 WC19".to_string(), a: 48.28 },

        // Scattered Disk Objects
        KBObject { name: "Eris".to_string(), a: 68.0 },
        KBObject { name: "1999 DE9".to_string(), a: 55.5 },
        KBObject { name: "2002 TC302".to_string(), a: 55.84 },
        KBObject { name: "2000 YW134".to_string(), a: 58.23 },
        KBObject { name: "1996 TL66".to_string(), a: 84.89 },
        KBObject { name: "1996 GQ21".to_string(), a: 92.48 },
        KBObject { name: "Gonggong".to_string(), a: 66.89 },

        // Extreme/Detached Objects
        KBObject { name: "Sedna".to_string(), a: 549.5 },
        KBObject { name: "2012 VP113".to_string(), a: 256.0 },
        KBObject { name: "2000 CR105".to_string(), a: 228.7 },
        KBObject { name: "2001 FP185".to_string(), a: 213.4 },
        KBObject { name: "2000 OO67".to_string(), a: 617.9 },
        KBObject { name: "2006 SQ372".to_string(), a: 839.3 },
        KBObject { name: "2010 VZ98".to_string(), a: 159.8 },
        KBObject { name: "2005 QU182".to_string(), a: 112.2 },
        KBObject { name: "2013 TV158".to_string(), a: 114.1 },
    ]
}

/// Analyze semi-major axis distribution for gaps
fn analyze_sma_gaps(objects: &[KBObject]) -> SMAGapAnalysisResult {
    // Sort by semi-major axis
    let mut a_values: Vec<(f64, String)> = objects.iter()
        .map(|o| (o.a, o.name.clone()))
        .collect();

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

        all_gaps.push((lower_a, upper_a, gap));
        gap_sizes.push(gap);
    }

    // Find significant gaps (> 20 AU beyond 50 AU)
    let significant_gaps: Vec<SMAGap> = all_gaps.iter()
        .filter(|(lower, _, gap)| *gap > 20.0 && *lower > 50.0)
        .map(|(lower_a, upper_a, gap_size)| {
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
                significance: (*gap_size / 50.0).min(1.0),
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
        .filter(|(lower, _, _)| *lower > 50.0)
        .map(|(_, _, gap)| gap)
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
        sorted_a_values: sorted_a,
        stats,
    }
}

fn generate_report(result: &SMAGapAnalysisResult) -> String {
    let mut report = String::new();

    report.push_str("╔══════════════════════════════════════════════════════════════╗\n");
    report.push_str("║          SEMI-MAJOR AXIS GAP ANALYSIS - AGENT 5              ║\n");
    report.push_str("║        Potential Planet Discovery via Orbital Clearing       ║\n");
    report.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

    // Summary section
    report.push_str("═══════════════════════════════════════════════════════════════\n");
    report.push_str("                     ANALYSIS SUMMARY                         \n");
    report.push_str("═══════════════════════════════════════════════════════════════\n\n");

    report.push_str(&format!("📊 Objects Analyzed:                    {}\n", result.total_objects));
    report.push_str(&format!("🔍 Total Gaps Detected:                 {}\n", result.gap_count));
    report.push_str(&format!("⚠️  Significant Gaps (>20 AU, >50 AU):  {}\n\n", result.significant_gaps.len()));

    // Statistics section
    report.push_str("───────────────────────────────────────────────────────────────\n");
    report.push_str("               GAP STATISTICS\n");
    report.push_str("───────────────────────────────────────────────────────────────\n\n");

    report.push_str(&format!("Mean gap size:                          {:.2} AU\n", result.stats.mean_gap_size));
    report.push_str(&format!("Standard deviation:                     {:.2} AU\n", result.stats.std_dev_gap));
    report.push_str(&format!("Median gap size:                        {:.2} AU\n", result.stats.median_gap_size));
    report.push_str(&format!("Largest gap overall:                    {:.2} AU\n", result.stats.largest_gap));
    report.push_str(&format!("Largest gap (a > 50 AU):                {:.2} AU\n\n", result.stats.largest_gap_beyond_50au));

    report.push_str("Gap Distribution:\n");
    report.push_str(&format!("  • Gaps > 5 AU:                        {}\n", result.stats.gaps_gt_5au));
    report.push_str(&format!("  • Gaps > 10 AU:                       {}\n", result.stats.gaps_gt_10au));
    report.push_str(&format!("  • Gaps > 20 AU:                       {}\n\n", result.stats.gaps_gt_20au));

    // Significant gaps section
    if !result.significant_gaps.is_empty() {
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("              SIGNIFICANT GAPS (> 20 AU, > 50 AU)              \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        for (idx, gap) in result.significant_gaps.iter().enumerate() {
            report.push_str(&format!("Gap #{}\n", idx + 1));
            report.push_str("───────────────────────────────────────────────────────────────\n");
            report.push_str(&format!("  Orbital Region:     {:.2} AU - {:.2} AU\n", gap.lower_bound, gap.upper_bound));
            report.push_str(&format!("  Gap Width:          {:.2} AU\n", gap.gap_size));
            report.push_str(&format!("  ★ PLANET LOCATION:  {:.2} AU (estimated)\n", gap.estimated_planet_a));
            report.push_str(&format!("  Significance:       {:.2} (score 0-1)\n\n", gap.significance));

            let mass_est = estimate_mass_from_gap(gap.gap_size);
            report.push_str(&format!("  Estimated Mass:     {:.1} Earth masses\n\n", mass_est));

            report.push_str("  Lower Boundary Objects (just inside gap):\n");
            if gap.lower_objects.is_empty() {
                report.push_str("    (none)\n");
            } else {
                for obj in gap.lower_objects.iter().take(10) {
                    report.push_str(&format!("    • {}\n", obj));
                }
                if gap.lower_objects.len() > 10 {
                    report.push_str(&format!("    ... and {} more\n", gap.lower_objects.len() - 10));
                }
            }

            report.push_str(&format!("\n  Upper Boundary Objects (just outside gap):\n"));
            if gap.upper_objects.is_empty() {
                report.push_str("    (none)\n");
            } else {
                for obj in gap.upper_objects.iter().take(10) {
                    report.push_str(&format!("    • {}\n", obj));
                }
                if gap.upper_objects.len() > 10 {
                    report.push_str(&format!("    ... and {} more\n", gap.upper_objects.len() - 10));
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

    report.push_str("All objects sorted by semi-major axis:\n");
    report.push_str("─────────────────────────────────────────\n");
    report.push_str("|  #  | Object Name              | a (AU)   |\n");
    report.push_str("|─────┼──────────────────────────┼──────────|\n");

    for (idx, (name, a)) in result.sorted_a_values.iter().zip(
        (0..result.sorted_a_values.len())
            .map(|i| {
                // Find original object name
                let mut result_name = format!("Object {}", i);
                for kbo in get_kbo_data_subset() {
                    if (kbo.a - result.sorted_a_values[i]).abs() < 0.01 {
                        result_name = kbo.name.clone();
                        break;
                    }
                }
                result_name
            })
    ).enumerate() {
        report.push_str(&format!("|{:4} | {:<24} | {:8.2} |\n", idx + 1, name, a));
    }
    report.push_str("└─────┴──────────────────────────┴──────────┘\n\n");

    // Histogram of a values
    report.push_str("Semi-major axis histogram:\n");
    report.push_str("───────────────────────────\n");

    let bins = vec![
        (30.0, 40.0, "30-40 AU"),
        (40.0, 50.0, "40-50 AU"),
        (50.0, 100.0, "50-100 AU"),
        (100.0, 200.0, "100-200 AU"),
        (200.0, f64::INFINITY, "200+ AU"),
    ];

    for (min, max, label) in bins {
        let count = result.sorted_a_values.iter()
            .filter(|&&a| a >= min && a < max)
            .count();
        let bar = "█".repeat(count.min(40));
        report.push_str(&format!("{:<12} {} ({})\n", label, bar, count));
    }

    report.push_str("\n");

    // Detailed gap analysis
    if result.significant_gaps.len() > 0 {
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("                  INTERPRETATION & PHYSICS                    \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str("ORBITAL CLEARING MECHANISM:\n");
        report.push_str("When a massive object orbits at semi-major axis 'a', its gravitational\n");
        report.push_str("influence creates a 'clearing' region where smaller objects are scattered\n");
        report.push_str("outward or inward. This produces observable gaps in the orbital distribution.\n\n");

        report.push_str("EVIDENCE STRENGTH:\n");
        for (idx, gap) in result.significant_gaps.iter().enumerate() {
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
        for (idx, gap) in result.significant_gaps.iter().enumerate() {
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

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     SEMI-MAJOR AXIS GAP ANALYSIS - ANALYSIS AGENT 5         ║");
    println!("║              Planet Discovery via Orbital Gaps              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load Kuiper Belt Objects
    println!("📥 Loading KBO data...\n");
    let objects = get_kbo_data_subset();
    println!("   Loaded {} Trans-Neptunian Objects\n", objects.len());

    // Run analysis
    println!("🔬 Running semi-major axis gap analysis...\n");
    let result = analyze_sma_gaps(&objects);

    // Display report
    let report = generate_report(&result);
    println!("{}", report);

    println!("═══════════════════════════════════════════════════════════════");
    println!("                   ANALYSIS COMPLETE");
    println!("═══════════════════════════════════════════════════════════════\n");
}
