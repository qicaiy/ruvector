//! # Argument of Perihelion Clustering Analysis for High-q Objects
//!
//! This module performs focused analysis on the clustering of argument of perihelion (ω)
//! for high-perihelion Kuiper Belt Objects (q > 37 AU, a > 50 AU).
//!
//! ## Objective
//! - Detect clustering around 0° or 180° (Kozai resonance signature)
//! - Identify potential planet perturbation evidence
//! - Analyze statistical significance of clustering patterns
//!
//! ## Kozai-Lidov Mechanism
//! The Kozai mechanism causes oscillations in eccentricity and inclination while conserving
//! a specific combination of orbital elements (Kozai integral). Objects typically cluster
//! around ω = 0° or 180° depending on the perturbing body's orbital parameters.

use super::kbo_data::get_kbo_data;
use super::kuiper_cluster::KuiperBeltObject;
use std::f32::consts::PI as PI_F32;

/// High-perihelion object with analysis metrics
#[derive(Debug, Clone)]
pub struct HighQObject {
    pub name: String,
    pub a: f32,
    pub e: f32,
    pub i: f32,
    pub q: f32,
    pub ad: f32,
    pub omega: f32,
    pub w: f32,
    pub class: String,
}

/// Clustering analysis result
#[derive(Debug, Clone)]
pub struct ClusteringAnalysis {
    pub objects: Vec<HighQObject>,
    pub mean_w: f32,
    pub std_dev_w: f32,
    pub min_w: f32,
    pub max_w: f32,
    pub cluster_0: Vec<HighQObject>,
    pub cluster_180: Vec<HighQObject>,
    pub scattered: Vec<HighQObject>,
    pub kozai_score: f32,
    pub planet_perturbation_evidence: bool,
}

/// Perform comprehensive argument of perihelion analysis
pub fn analyze_argument_of_perihelion() -> ClusteringAnalysis {
    let all_objects = get_kbo_data();

    // Filter for high-q objects: q > 37 AU and a > 50 AU
    let high_q_objects: Vec<HighQObject> = all_objects
        .iter()
        .filter(|o| o.q > 37.0 && o.a > 50.0)
        .map(|o| HighQObject {
            name: o.name.clone(),
            a: o.a,
            e: o.e,
            i: o.i,
            q: o.q,
            ad: o.ad,
            omega: o.omega,
            w: o.w,
            class: o.class.clone(),
        })
        .collect();

    // Calculate statistics on argument of perihelion (w)
    if high_q_objects.is_empty() {
        return ClusteringAnalysis {
            objects: vec![],
            mean_w: 0.0,
            std_dev_w: 0.0,
            min_w: 0.0,
            max_w: 0.0,
            cluster_0: vec![],
            cluster_180: vec![],
            scattered: vec![],
            kozai_score: 0.0,
            planet_perturbation_evidence: false,
        };
    }

    let w_values: Vec<f32> = high_q_objects.iter().map(|o| o.w).collect();

    // Circular statistics for angle data
    let mean_w = calculate_circular_mean(&w_values);
    let std_dev_w = calculate_circular_std_dev(&w_values, mean_w);
    let min_w = w_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_w = w_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Cluster around 0° (perihelion aligned)
    let cluster_0: Vec<HighQObject> = high_q_objects
        .iter()
        .filter(|o| o.w < 45.0 || o.w > 315.0)
        .cloned()
        .collect();

    // Cluster around 180° (perihelion anti-aligned)
    let cluster_180: Vec<HighQObject> = high_q_objects
        .iter()
        .filter(|o| o.w > 135.0 && o.w < 225.0)
        .cloned()
        .collect();

    // Objects scattered elsewhere
    let scattered: Vec<HighQObject> = high_q_objects
        .iter()
        .filter(|o| !((o.w < 45.0 || o.w > 315.0) || (o.w > 135.0 && o.w < 225.0)))
        .cloned()
        .collect();

    // Calculate Kozai score (clustering strength)
    let kozai_score = calculate_kozai_score(&cluster_0, &cluster_180, high_q_objects.len());

    // Determine if clustering suggests planet perturbation
    let planet_perturbation_evidence = kozai_score > 0.4;

    ClusteringAnalysis {
        objects: high_q_objects,
        mean_w,
        std_dev_w,
        min_w,
        max_w,
        cluster_0,
        cluster_180,
        scattered,
        kozai_score,
        planet_perturbation_evidence,
    }
}

/// Calculate circular mean for angle data (accounts for periodicity)
fn calculate_circular_mean(angles: &[f32]) -> f32 {
    if angles.is_empty() {
        return 0.0;
    }

    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;

    for &angle in angles {
        let rad = angle * PI_F32 / 180.0;
        sin_sum += rad.sin();
        cos_sum += rad.cos();
    }

    let mean_rad = sin_sum.atan2(cos_sum);
    let mean_deg = (mean_rad * 180.0 / PI_F32 + 360.0) % 360.0;
    mean_deg
}

/// Calculate circular standard deviation
fn calculate_circular_std_dev(angles: &[f32], mean: f32) -> f32 {
    if angles.len() < 2 {
        return 0.0;
    }

    let mean_rad = mean * PI_F32 / 180.0;
    let mut cos_sum = 0.0;

    for &angle in angles {
        let rad = angle * PI_F32 / 180.0;
        cos_sum += (rad - mean_rad).cos();
    }

    let r = cos_sum / angles.len() as f32;
    let variance = 2.0 * (1.0 - r);
    let std_dev_rad = variance.sqrt();
    let std_dev_deg = std_dev_rad * 180.0 / PI_F32;
    std_dev_deg
}

/// Calculate Kozai score based on clustering strength
/// Score ranges 0-1, where 1 indicates strong clustering around 0° or 180°
fn calculate_kozai_score(cluster_0: &[HighQObject], cluster_180: &[HighQObject], total: usize) -> f32 {
    let cluster_count = cluster_0.len() + cluster_180.len();
    let clustering_ratio = cluster_count as f32 / total.max(1) as f32;

    // Bonus if objects are clearly in one cluster or the other
    let dominance = if cluster_0.len() > cluster_180.len() {
        cluster_0.len() as f32 / cluster_count.max(1) as f32
    } else {
        cluster_180.len() as f32 / cluster_count.max(1) as f32
    };

    // Score: 50% clustering ratio + 50% dominance
    (clustering_ratio + dominance) / 2.0
}

/// Generate detailed report on perihelion clustering
pub fn generate_report(analysis: &ClusteringAnalysis) -> String {
    let mut report = String::new();

    report.push_str("╔══════════════════════════════════════════════════════════════╗\n");
    report.push_str("║     ARGUMENT OF PERIHELION CLUSTERING ANALYSIS              ║\n");
    report.push_str("║         HIGH-q KUIPER BELT OBJECTS (q > 37 AU, a > 50 AU)  ║\n");
    report.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

    report.push_str("📊 SAMPLE STATISTICS\n");
    report.push_str("───────────────────────────────────────────────────────────────\n");
    report.push_str(&format!("  Total high-q objects: {}\n", analysis.objects.len()));
    report.push_str(&format!("  Mean ω (circular):    {:.1}°\n", analysis.mean_w));
    report.push_str(&format!("  Std dev ω:            {:.1}°\n", analysis.std_dev_w));
    report.push_str(&format!("  Range:                {:.1}° - {:.1}°\n\n", analysis.min_w, analysis.max_w));

    report.push_str("🎯 CLUSTERING ANALYSIS\n");
    report.push_str("───────────────────────────────────────────────────────────────\n");
    report.push_str(&format!("  Objects near 0° (±45°):       {:3} ({:5.1}%)\n",
        analysis.cluster_0.len(),
        (analysis.cluster_0.len() as f32 / analysis.objects.len().max(1) as f32) * 100.0));
    report.push_str(&format!("  Objects near 180° (±45°):     {:3} ({:5.1}%)\n",
        analysis.cluster_180.len(),
        (analysis.cluster_180.len() as f32 / analysis.objects.len().max(1) as f32) * 100.0));
    report.push_str(&format!("  Objects scattered (45°-135°): {:3} ({:5.1}%)\n\n",
        analysis.scattered.len(),
        (analysis.scattered.len() as f32 / analysis.objects.len().max(1) as f32) * 100.0));

    report.push_str("🔍 KOZAI RESONANCE ANALYSIS\n");
    report.push_str("───────────────────────────────────────────────────────────────\n");
    report.push_str(&format!("  Kozai Score: {:.3}\n", analysis.kozai_score));
    report.push_str("  Interpretation:\n");

    if analysis.kozai_score > 0.6 {
        report.push_str("    ⭐ STRONG CLUSTERING - Consistent with Kozai mechanism\n");
        report.push_str("    This indicates a perturbing body with significant inclination.\n");
    } else if analysis.kozai_score > 0.4 {
        report.push_str("    ✓ MODERATE CLUSTERING - Possible Kozai influence\n");
        report.push_str("    May indicate a distant perturbing body.\n");
    } else if analysis.kozai_score > 0.2 {
        report.push_str("    ○ WEAK CLUSTERING - Insufficient evidence for Kozai\n");
        report.push_str("    Objects distributed more randomly in ω.\n");
    } else {
        report.push_str("    ✗ NO CLUSTERING - No Kozai signature detected\n");
        report.push_str("    Objects are randomly distributed in argument of perihelion.\n");
    }
    report.push_str("\n");

    report.push_str("🪐 PLANET PERTURBATION EVIDENCE\n");
    report.push_str("───────────────────────────────────────────────────────────────\n");

    if analysis.planet_perturbation_evidence {
        report.push_str("  ✓ EVIDENCE DETECTED\n\n");
        report.push_str("  Clustering of ω around 0° or 180° is a hallmark of the Kozai-Lidov\n");
        report.push_str("  mechanism, which occurs when an external perturber (planet) excites\n");
        report.push_str("  oscillations in object eccentricity and inclination.\n\n");

        if analysis.cluster_0.len() > analysis.cluster_180.len() {
            report.push_str("  Dominant cluster: 0° (aligned perihelion)\n");
            report.push_str("  Interpretation: Perturbing planet likely has LOW inclination\n");
        } else if analysis.cluster_180.len() > analysis.cluster_0.len() {
            report.push_str("  Dominant cluster: 180° (anti-aligned perihelion)\n");
            report.push_str("  Interpretation: Perturbing planet likely has HIGH inclination\n");
        } else {
            report.push_str("  Dual clusters: Both 0° and 180° significantly populated\n");
            report.push_str("  Interpretation: Possible resonance with two distinct perturbers\n");
        }
        report.push_str("\n");
    } else {
        report.push_str("  ✗ NO SIGNIFICANT EVIDENCE\n");
        report.push_str("  Kozai score too low to suggest planet perturbation.\n");
        report.push_str("  Current ω distribution is consistent with random variation.\n\n");
    }

    report.push_str("📋 OBJECT LIST (q > 37 AU, a > 50 AU)\n");
    report.push_str("───────────────────────────────────────────────────────────────\n");
    report.push_str(&format!("{:<35} {:<8} {:<8} {:<8} {:<8}\n",
        "Object", "a (AU)", "q (AU)", "ω (°)", "Cluster"));
    report.push_str("───────────────────────────────────────────────────────────────\n");

    let mut sorted_objects = analysis.objects.clone();
    sorted_objects.sort_by(|a, b| a.w.partial_cmp(&b.w).unwrap());

    for obj in &sorted_objects {
        let cluster_label = if obj.w < 45.0 || obj.w > 315.0 {
            "0°"
        } else if obj.w > 135.0 && obj.w < 225.0 {
            "180°"
        } else {
            "Scattered"
        };

        report.push_str(&format!("{:<35} {:<8.1} {:<8.1} {:<8.1} {:<8}\n",
            &obj.name[..obj.name.len().min(34)],
            obj.a, obj.q, obj.w, cluster_label));
    }

    report.push_str("\n");
    report.push_str("🔬 CLUSTER 0° (Aligned Perihelion)\n");
    report.push_str("───────────────────────────────────────────────────────────────\n");

    if analysis.cluster_0.is_empty() {
        report.push_str("  No objects in this cluster.\n");
    } else {
        report.push_str(&format!("  Objects: {}\n\n", analysis.cluster_0.len()));
        for obj in &analysis.cluster_0 {
            report.push_str(&format!("    • {} (ω={:.1}°, a={:.1} AU, e={:.2})\n",
                obj.name, obj.w, obj.a, obj.e));
        }
    }
    report.push_str("\n");

    report.push_str("🔬 CLUSTER 180° (Anti-aligned Perihelion)\n");
    report.push_str("───────────────────────────────────────────────────────────────\n");

    if analysis.cluster_180.is_empty() {
        report.push_str("  No objects in this cluster.\n");
    } else {
        report.push_str(&format!("  Objects: {}\n\n", analysis.cluster_180.len()));
        for obj in &analysis.cluster_180 {
            report.push_str(&format!("    • {} (ω={:.1}°, a={:.1} AU, e={:.2})\n",
                obj.name, obj.w, obj.a, obj.e));
        }
    }
    report.push_str("\n");

    report.push_str("📊 ORBITAL PARAMETER ANALYSIS\n");
    report.push_str("───────────────────────────────────────────────────────────────\n");

    if !analysis.objects.is_empty() {
        let avg_a = analysis.objects.iter().map(|o| o.a).sum::<f32>() / analysis.objects.len() as f32;
        let avg_e = analysis.objects.iter().map(|o| o.e).sum::<f32>() / analysis.objects.len() as f32;
        let avg_i = analysis.objects.iter().map(|o| o.i).sum::<f32>() / analysis.objects.len() as f32;
        let avg_q = analysis.objects.iter().map(|o| o.q).sum::<f32>() / analysis.objects.len() as f32;

        report.push_str(&format!("  Average a: {:.1} AU\n", avg_a));
        report.push_str(&format!("  Average e: {:.3}\n", avg_e));
        report.push_str(&format!("  Average i: {:.1}°\n", avg_i));
        report.push_str(&format!("  Average q: {:.1} AU\n\n", avg_q));
    }

    report.push_str("⚙️ KOZAI MECHANISM PHYSICS\n");
    report.push_str("───────────────────────────────────────────────────────────────\n");
    report.push_str("  The Kozai-Lidov mechanism describes oscillations of\n");
    report.push_str("  eccentricity and inclination caused by an external perturber.\n\n");
    report.push_str("  Key features:\n");
    report.push_str("  • Oscillation period: ~100,000-1,000,000 years\n");
    report.push_str("  • Conserved quantity: Kozai integral K = a(1-e²)cos²i\n");
    report.push_str("  • Peak eccentricity: e_max can reach near 1 at perihelion\n");
    report.push_str("  • ω oscillates between ~0° and ~180° phases\n\n");
    report.push_str("  Objects cluster near extreme phases due to:\n");
    report.push_str("  • Dynamical equilibrium points in (e,ω) phase space\n");
    report.push_str("  • Resonant coupling with perturber's orbit\n");
    report.push_str("  • Long-term orbital evolution patterns\n");

    report.push_str("\n═══════════════════════════════════════════════════════════════\n");
    report.push_str("  Analysis Source: RuVector Kuiper Belt Analysis\n");
    report.push_str("  Data Source: NASA/JPL Small-Body Database\n");
    report.push_str("═══════════════════════════════════════════════════════════════\n");

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_mean() {
        let angles = vec![10.0, 20.0, 30.0];
        let mean = calculate_circular_mean(&angles);
        assert!(mean > 15.0 && mean < 25.0);
    }

    #[test]
    fn test_kozai_analysis() {
        let analysis = analyze_argument_of_perihelion();
        assert!(analysis.std_dev_w >= 0.0);
        assert!(analysis.kozai_score >= 0.0 && analysis.kozai_score <= 1.0);
    }
}
