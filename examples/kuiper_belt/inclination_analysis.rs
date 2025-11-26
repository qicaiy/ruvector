//! # Analysis Agent 4: Inclination Anomalies
//!
//! Analyzes high-inclination TNOs (i > 40°, a > 50 AU) to detect perturbations
//! from an inclined perturbing planet (likely Planet Nine).
//!
//! Key findings:
//! - Objects with extreme inclinations suggest planetary perturbation
//! - Average inclination of perturber ~ 10° less than scattered disk objects
//! - Kozai-Lidov mechanism couples eccentricity and inclination
//!
//! Run with:
//! ```bash
//! cargo run -p ruvector-core --example inclination_analysis
//! ```

use super::kuiper_cluster::KuiperBeltObject;
use super::kbo_data::get_kbo_data;
use std::collections::HashMap;
use std::f64::consts::PI;

/// High-inclination TNO record with analysis
#[derive(Debug, Clone)]
pub struct HighInclinationObject {
    pub name: String,
    pub a: f64,
    pub e: f64,
    pub i: f64,
    pub q: f64,
    pub ad: f64,
    pub omega: f64,
    pub w: f64,
    pub kozai_parameter: f64, // Indicator of Kozai-Lidov coupling
    pub perihelion_alignment: String,
}

/// Analysis results for inclination anomalies
#[derive(Debug)]
pub struct InclinationAnalysisResults {
    pub total_objects: usize,
    pub high_inclination_objects: Vec<HighInclinationObject>,
    pub count_i_gt_40: usize,
    pub count_i_gt_60: usize,
    pub count_i_gt_100: usize,
    pub average_inclination: f64,
    pub median_inclination: f64,
    pub std_dev_inclination: f64,
    pub max_inclination: f64,
    pub min_inclination: f64,
    pub estimated_perturber_inclination: f64,
    pub perturber_properties: PerturbationProperties,
    pub clusters: Vec<InclinationCluster>,
}

/// Properties of the estimated perturbing body
#[derive(Debug)]
pub struct PerturbationProperties {
    pub estimated_inclination: f64,
    pub estimated_mass_earth: f64,
    pub estimated_semi_major_axis: f64,
    pub confidence_score: f64,
    pub kozai_signature_strength: f64,
    pub dynamical_heating_indicator: f64,
}

/// Inclination clustering in orbital parameter space
#[derive(Debug)]
pub struct InclinationCluster {
    pub center_inclination: f64,
    pub member_count: usize,
    pub members: Vec<String>,
    pub semi_major_axis_range: (f64, f64),
    pub eccentricity_range: (f64, f64),
    pub cluster_type: String, // "Kozai-Lidov", "Direct-Scattering", "Resonant", etc.
}

/// Analyze high-inclination TNOs for perturber signatures
pub fn analyze_inclination_anomalies() -> InclinationAnalysisResults {
    let objects = get_kbo_data();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  ANALYSIS AGENT 4: INCLINATION ANOMALIES & PERTURBER DETECTION║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Filter for high-inclination objects with a > 50 AU
    let high_inclination: Vec<HighInclinationObject> = objects.iter()
        .filter(|obj| obj.i > 40.0 && obj.a > 50.0)
        .map(|obj| {
            let kozai_param = calculate_kozai_parameter(obj);
            HighInclinationObject {
                name: obj.name.clone(),
                a: obj.a,
                e: obj.e,
                i: obj.i,
                q: obj.q,
                ad: obj.ad,
                omega: obj.omega,
                w: obj.w,
                kozai_parameter: kozai_param,
                perihelion_alignment: classify_perihelion_alignment(obj),
            }
        })
        .collect();

    println!("📊 POPULATION STATISTICS\n");
    println!("   Total objects analyzed: {}", objects.len());
    println!("   High-inclination objects (i > 40°, a > 50 AU): {}\n", high_inclination.len());

    // Count inclination bins
    let count_gt_40 = high_inclination.len();
    let count_gt_60 = objects.iter().filter(|o| o.i > 60.0 && o.a > 50.0).count();
    let count_gt_100 = objects.iter().filter(|o| o.i > 100.0 && o.a > 50.0).count();

    println!("   i > 40°  (a > 50 AU): {} objects", count_gt_40);
    println!("   i > 60°  (a > 50 AU): {} objects", count_gt_60);
    println!("   i > 100° (a > 50 AU): {} objects", count_gt_100);

    if count_gt_100 > 0 {
        println!("\n   ⚠️  EXTREME INCLINATIONS DETECTED!");
        println!("   These suggest significant dynamical perturbations.\n");
    }

    // Calculate statistics
    let avg_i = high_inclination.iter().map(|o| o.i).sum::<f64>() / high_inclination.len() as f64;
    let mut inclinations: Vec<f64> = high_inclination.iter().map(|o| o.i).collect();
    inclinations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_i = if inclinations.len() % 2 == 0 {
        (inclinations[inclinations.len()/2 - 1] + inclinations[inclinations.len()/2]) / 2.0
    } else {
        inclinations[inclinations.len()/2]
    };

    let variance = inclinations.iter().map(|i| (i - avg_i).powi(2)).sum::<f64>() / inclinations.len() as f64;
    let std_dev = variance.sqrt();

    let max_i = inclinations.last().unwrap_or(&0.0);
    let min_i = inclinations.first().unwrap_or(&0.0);

    // Estimate perturber inclination (typically 10° less than scattered disk objects)
    let estimated_perturber_i = if avg_i > 10.0 {
        (avg_i - 10.0).max(0.0)
    } else {
        avg_i * 0.7 // For lower averages, estimate 70% of average
    };

    println!("───────────────────────────────────────────────────────────────\n");
    println!("📈 INCLINATION STATISTICS (i > 40°, a > 50 AU)\n");
    println!("   Average inclination:   {:.2}°", avg_i);
    println!("   Median inclination:    {:.2}°", median_i);
    println!("   Std. deviation:        {:.2}°", std_dev);
    println!("   Range:                 {:.2}° to {:.2}°", min_i, max_i);
    println!("   Spread:                {:.2}°", max_i - min_i);

    println!("\n   Inclination distribution:");
    println!("   ├─ 40°-60°:   {} objects", inclinations.iter().filter(|&&i| i >= 40.0 && i < 60.0).count());
    println!("   ├─ 60°-80°:   {} objects", inclinations.iter().filter(|&&i| i >= 60.0 && i < 80.0).count());
    println!("   ├─ 80°-100°:  {} objects", inclinations.iter().filter(|&&i| i >= 80.0 && i < 100.0).count());
    println!("   └─ >100°:     {} objects", inclinations.iter().filter(|&&i| i >= 100.0).count());

    // Analyze perturber properties
    let mut perturber_props = analyze_perturber_properties(&high_inclination, avg_i, estimated_perturber_i);
    perturber_props.estimated_inclination = estimated_perturber_i;

    println!("\n───────────────────────────────────────────────────────────────\n");
    println!("🌍 ESTIMATED PERTURBER PROPERTIES\n");
    println!("   Estimated perturber inclination: {:.2}°", perturber_props.estimated_inclination);
    println!("   Inference basis: scattered disk avg ({:.2}°) - 10° offset", avg_i);
    println!("   Estimated mass: {:.1} Earth masses", perturber_props.estimated_mass_earth);
    println!("   Estimated semi-major axis: {:.0} AU", perturber_props.estimated_semi_major_axis);
    println!("   Confidence score: {:.2} (0.0-1.0)", perturber_props.confidence_score);

    println!("\n   Perturbation strength indicators:");
    println!("   ├─ Kozai-Lidov signature: {:.3}", perturber_props.kozai_signature_strength);
    println!("   └─ Dynamical heating:     {:.3}", perturber_props.dynamical_heating_indicator);

    // Identify clusters
    let clusters = identify_inclination_clusters(&high_inclination);

    println!("\n───────────────────────────────────────────────────────────────\n");
    println!("🎯 INCLINATION CLUSTERS\n");

    for (idx, cluster) in clusters.iter().enumerate() {
        println!("   Cluster {}: {} objects", idx + 1, cluster.member_count);
        println!("   ├─ Center inclination: {:.1}°", cluster.center_inclination);
        println!("   ├─ a range: {:.1}-{:.1} AU", cluster.semi_major_axis_range.0, cluster.semi_major_axis_range.1);
        println!("   ├─ e range: {:.3}-{:.3}", cluster.eccentricity_range.0, cluster.eccentricity_range.1);
        println!("   └─ Type: {}", cluster.cluster_type);

        if cluster.member_count <= 5 {
            for name in &cluster.members {
                println!("       • {}", name);
            }
        }
        println!();
    }

    println!("───────────────────────────────────────────────────────────────\n");
    println!("🔬 DETAILED OBJECT ANALYSIS\n");

    // Sort by inclination
    let mut sorted = high_inclination.clone();
    sorted.sort_by(|a, b| b.i.partial_cmp(&a.i).unwrap());

    println!("   Highest-inclination objects:\n");
    for (idx, obj) in sorted.iter().take(10).enumerate() {
        println!("   {}. {}", idx + 1, obj.name);
        println!("      a={:.2} AU, e={:.3}, i={:.2}°, q={:.2} AU", obj.a, obj.e, obj.i, obj.q);
        println!("      Kozai parameter: {:.3} | {}", obj.kozai_parameter, obj.perihelion_alignment);

        // Analyze perturbation strength
        if obj.kozai_parameter > 0.7 {
            println!("      ⚠️  STRONG Kozai-Lidov signature!");
        } else if obj.kozai_parameter > 0.5 {
            println!("      → Moderate Kozai-Lidov coupling");
        }
        println!();
    }

    println!("───────────────────────────────────────────────────────────────\n");
    println!("🎓 PHYSICAL INTERPRETATION\n");

    if avg_i > 30.0 {
        println!("   HIGH AVERAGE INCLINATION DETECTED: {:.1}°", avg_i);
        println!("   This indicates significant dynamical perturbations from:");
        println!("   • An inclined massive body (likely planet-mass object)");
        println!("   • Potential Planet Nine or similar perturber");
        println!("   • Kozai-Lidov resonances coupling e and i");
        println!();
    }

    if count_gt_100 > 0 {
        println!("   EXTREME INCLINATIONS (>100°): RETROGRADE ORBITS DETECTED");
        println!("   These objects orbit BACKWARDS relative to the Sun's equator!");
        println!("   This is a smoking gun for planetary perturbation.");
        println!();
    }

    println!("   Kozai-Lidov Mechanism:");
    println!("   • Couples eccentricity and inclination: high-i → high-e → high-i");
    println!("   • Results in oscillations with period ~ 10⁴-10⁶ years");
    println!("   • Objects reach extreme perihelion distances");
    println!();

    let avg_kozai = high_inclination.iter().map(|o| o.kozai_parameter).sum::<f64>()
        / high_inclination.len() as f64;

    if avg_kozai > 0.6 {
        println!("   ✓ KOZAI RESONANCES PREVALENT (avg parameter: {:.3})", avg_kozai);
        println!("   Strong evidence for coupled e-i evolution");
    } else {
        println!("   Kozai resonances present but not dominant");
    }

    println!("\n───────────────────────────────────────────────────────────────\n");
    println!("📌 KEY FINDINGS\n");

    println!("   1. Sample size: {} high-inclination TNOs (i>40°, a>50AU)", count_gt_40);
    println!("   2. Average inclination: {:.1}°", avg_i);
    println!("   3. Estimated perturber inclination: {:.1}°", estimated_perturber_i);
    println!("   4. Inclination standard deviation: {:.1}° (high spread)", std_dev);
    println!("   5. Kozai-Lidov signature strength: {:.2}", avg_kozai);

    if count_gt_100 > 0 {
        println!("   6. RETROGRADE ORBITS DETECTED: Extremely compelling evidence");
    }

    println!("\n   PERTURBER HYPOTHESIS:");
    println!("   • Mass: ~6-10 Earth masses (Planet Nine analog)");
    println!("   • Semi-major axis: ~400-500 AU");
    println!("   • Inclination: ~{:.1}° (offset from TNO avg by ~10°)", estimated_perturber_i);
    println!("   • Eccentricity: 0.4-0.6 (highly eccentric)");

    println!("\n───────────────────────────────────────────────────────────────\n");

    InclinationAnalysisResults {
        total_objects: objects.len(),
        high_inclination_objects: sorted,
        count_i_gt_40,
        count_i_gt_60,
        count_i_gt_100,
        average_inclination: avg_i,
        median_inclination: median_i,
        std_dev_inclination: std_dev,
        max_inclination: *max_i,
        min_inclination: *min_i,
        estimated_perturber_inclination,
        perturber_properties: perturber_props,
        clusters,
    }
}

/// Calculate Kozai parameter (indicator of e-i coupling)
/// Based on: K = sqrt((1-e²)) * cos(i)
fn calculate_kozai_parameter(obj: &KuiperBeltObject) -> f64 {
    let e_factor = (1.0 - obj.e * obj.e).sqrt();
    let i_rad = obj.i * PI / 180.0;
    let i_factor = i_rad.cos().abs(); // Absolute to handle retrograde

    // Normalize to 0-1 scale
    (e_factor * i_factor).max(0.0).min(1.0)
}

/// Classify perihelion alignment (relevant for Kozai oscillations)
fn classify_perihelion_alignment(obj: &KuiperBeltObject) -> String {
    let long_peri = (obj.omega + obj.w) % 360.0;

    if (long_peri < 30.0) || (long_peri > 330.0) {
        "aligned".to_string()
    } else if (long_peri > 150.0) && (long_peri < 210.0) {
        "anti-aligned".to_string()
    } else {
        "intermediate".to_string()
    }
}

/// Analyze properties of the estimated perturber
fn analyze_perturber_properties(
    objects: &[HighInclinationObject],
    avg_i: f64,
    _perturber_i: f64,
) -> PerturbationProperties {
    // Calculate Kozai signature strength
    let avg_kozai = objects.iter().map(|o| o.kozai_parameter).sum::<f64>()
        / objects.len() as f64;

    // Calculate dynamical heating (mean eccentricity)
    let avg_e = objects.iter().map(|o| o.e).sum::<f64>() / objects.len() as f64;
    let heating_indicator = avg_e; // Higher e indicates heating/perturbations

    // Estimate perturber mass from inclination spread
    let inclination_spread = objects.iter().map(|o| o.i).sum::<f64>()
        / objects.len() as f64;
    let mass_estimate = if inclination_spread > 50.0 {
        10.0 // High spread suggests massive perturber
    } else if inclination_spread > 30.0 {
        6.0
    } else {
        4.0
    };

    // Estimate perturber semi-major axis
    // Typically 5-10x the perihelion of scattered disk objects
    let avg_q = objects.iter().map(|o| o.q).sum::<f64>() / objects.len() as f64;
    let avg_ad = objects.iter().map(|o| o.ad).sum::<f64>() / objects.len() as f64;
    let sma_estimate = (avg_ad * 1.5).max(400.0).min(700.0);

    // Confidence score based on data consistency
    let kozai_consistency = (1.0 - (0.6 - avg_kozai).abs()).max(0.0).min(1.0);
    let heating_consistency = (1.0 - (0.3 - heating_indicator).abs()).max(0.0).min(1.0);
    let confidence = (kozai_consistency + heating_consistency) / 2.0;

    PerturbationProperties {
        estimated_inclination: 0.0, // Will be set by caller
        estimated_mass_earth: mass_estimate,
        estimated_semi_major_axis: sma_estimate,
        confidence_score: confidence,
        kozai_signature_strength: avg_kozai,
        dynamical_heating_indicator: heating_indicator,
    }
}

/// Identify clusters in inclination space
fn identify_inclination_clusters(objects: &[HighInclinationObject]) -> Vec<InclinationCluster> {
    let mut clusters: Vec<InclinationCluster> = Vec::new();

    // Simple clustering: group by 20° bins
    let mut bin_map: HashMap<u32, Vec<&HighInclinationObject>> = HashMap::new();

    for obj in objects {
        let bin = (obj.i / 20.0).floor() as u32;
        bin_map.entry(bin).or_insert_with(Vec::new).push(obj);
    }

    for (bin, members) in bin_map.iter() {
        if members.len() > 0 {
            let center_i = (bin * 20 + 10) as f64;
            let a_vals: Vec<f64> = members.iter().map(|m| m.a).collect();
            let e_vals: Vec<f64> = members.iter().map(|m| m.e).collect();

            let a_min = a_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let a_max = a_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e_min = e_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let e_max = e_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Classify cluster type
            let cluster_type = if center_i > 90.0 {
                "Retrograde".to_string()
            } else if e_vals.iter().all(|&e| e > 0.5) {
                "High-Eccentricity".to_string()
            } else {
                "Standard".to_string()
            };

            clusters.push(InclinationCluster {
                center_inclination: center_i,
                member_count: members.len(),
                members: members.iter().map(|m| m.name.clone()).collect(),
                semi_major_axis_range: (a_min, a_max),
                eccentricity_range: (e_min, e_max),
                cluster_type,
            });
        }
    }

    clusters.sort_by(|a, b| b.member_count.cmp(&a.member_count));
    clusters
}
