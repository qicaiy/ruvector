//! # Eccentricity Pumping Analysis
//!
//! Analysis Agent 6: Analyzes eccentricity pumping from distant planets
//! - Identifies objects with e > 0.7 and a > 50 AU
//! - Calculates average semi-major axis
//! - Estimates perturber distance (typically 3x average a)

use super::kbo_data::get_kbo_data;
use super::kuiper_cluster::KuiperBeltObject;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of eccentricity pumping analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EccentricityAnalysis {
    /// Objects matching criteria (e > 0.7, a > 50 AU)
    pub high_e_objects: Vec<HighEccentricityObject>,

    /// Statistical summary
    pub summary: EccentricityStats,

    /// Perturber analysis
    pub perturber_analysis: PerturberEstimate,
}

/// Object with high eccentricity pumping signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighEccentricityObject {
    pub name: String,
    pub a: f32,
    pub e: f32,
    pub i: f32,
    pub q: f32,
    pub ad: f32,
    /// Eccentricity pumping strength indicator
    pub pumping_strength: f32,
    /// Perturber-induced orbital heating estimate
    pub heating_factor: f32,
}

/// Statistical summary of high-e population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EccentricityStats {
    /// Number of objects with e > 0.7 and a > 50 AU
    pub count: usize,

    /// Average semi-major axis of high-e objects
    pub avg_a: f32,

    /// Median semi-major axis
    pub median_a: f32,

    /// Average eccentricity
    pub avg_e: f32,

    /// Average perihelion distance
    pub avg_q: f32,

    /// Average aphelion distance
    pub avg_ad: f32,

    /// Min and max semi-major axes
    pub a_range: (f32, f32),

    /// Min and max eccentricities
    pub e_range: (f32, f32),

    /// Min and max perihelion distances
    pub q_range: (f32, f32),

    /// Distribution by object class
    pub class_distribution: HashMap<String, usize>,
}

/// Perturber distance estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturberEstimate {
    /// Estimated perturber distance (3x average a)
    pub estimated_distance: f32,

    /// Confidence in estimate
    pub confidence: f32,

    /// Expected perturber mass range for induced eccentricities
    pub expected_mass_range: String,

    /// Known bodies at estimated distance
    pub candidate_perturbers: Vec<String>,

    /// Analysis notes
    pub analysis_notes: Vec<String>,
}

/// Perform complete eccentricity pumping analysis
pub fn analyze_eccentricity_pumping() -> EccentricityAnalysis {
    let all_objects = get_kbo_data();

    // Filter for e > 0.7 AND a > 50 AU
    let mut high_e_objects: Vec<HighEccentricityObject> = all_objects
        .iter()
        .filter(|obj| obj.e > 0.7 && obj.a > 50.0)
        .map(|obj| {
            let pumping_strength = calculate_pumping_strength(obj);
            let heating_factor = calculate_heating_factor(obj);
            HighEccentricityObject {
                name: obj.name.clone(),
                a: obj.a,
                e: obj.e,
                i: obj.i,
                q: obj.q,
                ad: obj.ad,
                pumping_strength,
                heating_factor,
            }
        })
        .collect();

    // Sort by eccentricity descending
    high_e_objects.sort_by(|a, b| b.e.partial_cmp(&a.e).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate statistics
    let summary = calculate_statistics(&high_e_objects, &all_objects);

    // Estimate perturber distance
    let perturber_analysis = estimate_perturber_distance(&summary);

    EccentricityAnalysis {
        high_e_objects,
        summary,
        perturber_analysis,
    }
}

/// Calculate eccentricity pumping strength (based on e deviation from classical values)
fn calculate_pumping_strength(obj: &KuiperBeltObject) -> f32 {
    // Classical belt objects typically have e < 0.2
    // Each 0.1 increase in e from baseline represents stronger pumping
    let baseline_e = 0.2;
    let deviation = (obj.e - baseline_e).max(0.0);

    // Normalize to 0-1 scale where 1.0 = e >= 0.9
    (deviation / 0.7).min(1.0)
}

/// Calculate orbital heating factor
/// Higher eccentricity and larger aphelion indicate more heating from perturbers
fn calculate_heating_factor(obj: &KuiperBeltObject) -> f32 {
    // Heating factor combines:
    // 1. Eccentricity deviation from classical
    // 2. Aphelion distance (interaction range)
    // 3. Semi-major axis (orbital energy)

    let e_component = obj.e * 0.4;
    let a_component = (obj.a / 100.0) * 0.3;
    let ad_component = (obj.ad / 1000.0) * 0.3;

    e_component + a_component + ad_component
}

/// Calculate detailed statistics
fn calculate_statistics(
    high_e_objects: &[HighEccentricityObject],
    all_objects: &[KuiperBeltObject],
) -> EccentricityStats {
    if high_e_objects.is_empty() {
        return EccentricityStats {
            count: 0,
            avg_a: 0.0,
            median_a: 0.0,
            avg_e: 0.0,
            avg_q: 0.0,
            avg_ad: 0.0,
            a_range: (0.0, 0.0),
            e_range: (0.0, 0.0),
            q_range: (0.0, 0.0),
            class_distribution: HashMap::new(),
        };
    }

    let count = high_e_objects.len();

    let sum_a: f32 = high_e_objects.iter().map(|o| o.a).sum();
    let avg_a = sum_a / count as f32;

    let sum_e: f32 = high_e_objects.iter().map(|o| o.e).sum();
    let avg_e = sum_e / count as f32;

    let sum_q: f32 = high_e_objects.iter().map(|o| o.q).sum();
    let avg_q = sum_q / count as f32;

    let sum_ad: f32 = high_e_objects.iter().map(|o| o.ad).sum();
    let avg_ad = sum_ad / count as f32;

    // Calculate median
    let mut a_values: Vec<f32> = high_e_objects.iter().map(|o| o.a).collect();
    a_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_a = if a_values.len() % 2 == 0 {
        (a_values[a_values.len() / 2 - 1] + a_values[a_values.len() / 2]) / 2.0
    } else {
        a_values[a_values.len() / 2]
    };

    let min_a = high_e_objects.iter().map(|o| o.a).fold(f32::INFINITY, f32::min);
    let max_a = high_e_objects.iter().map(|o| o.a).fold(0.0, f32::max);

    let min_e = high_e_objects.iter().map(|o| o.e).fold(f32::INFINITY, f32::min);
    let max_e = high_e_objects.iter().map(|o| o.e).fold(0.0, f32::max);

    let min_q = high_e_objects.iter().map(|o| o.q).fold(f32::INFINITY, f32::min);
    let max_q = high_e_objects.iter().map(|o| o.q).fold(0.0, f32::max);

    // Class distribution
    let mut class_dist = HashMap::new();
    for obj_name in &high_e_objects.iter().map(|o| &o.name).collect::<Vec<_>>() {
        if let Some(original) = all_objects.iter().find(|o| &o.name == *obj_name) {
            *class_dist.entry(original.class.clone()).or_insert(0) += 1;
        }
    }

    EccentricityStats {
        count,
        avg_a,
        median_a,
        avg_e,
        avg_q,
        avg_ad,
        a_range: (min_a, max_a),
        e_range: (min_e, max_e),
        q_range: (min_q, max_q),
        class_distribution: class_dist,
    }
}

/// Estimate perturber distance based on semi-major axes of high-e population
fn estimate_perturber_distance(stats: &EccentricityStats) -> PerturberEstimate {
    let estimated_distance = stats.avg_a * 3.0;

    let confidence = if stats.count > 5 { 0.85 } else { 0.60 };

    // Expected mass range based on perturber-induced eccentricity pumping
    // Typical Earth-mass perturber at 100+ AU can pump e to > 0.7
    let mass_range = if estimated_distance > 150.0 {
        "Earth to Super-Earth mass (1-5 Earth masses)".to_string()
    } else if estimated_distance > 100.0 {
        "Mars to Earth mass (0.1-2 Earth masses)".to_string()
    } else {
        "Neptune to Jupiter mass (10-1000 Earth masses)".to_string()
    };

    let mut candidate_perturbers = vec![];
    let mut notes = vec![];

    // Add analysis notes
    notes.push(format!(
        "Detected {} objects with e > 0.7 and a > 50 AU",
        stats.count
    ));
    notes.push(format!(
        "Average semi-major axis: {:.2} AU",
        stats.avg_a
    ));
    notes.push(format!(
        "Eccentricity range: {:.3} - {:.3}",
        stats.e_range.0, stats.e_range.1
    ));
    notes.push(format!(
        "Estimated perturber distance: {:.1} AU (3x avg a)",
        estimated_distance
    ));

    // Identify candidate perturbers
    if estimated_distance > 200.0 {
        candidate_perturbers.push("Unknown Planet 9 (predicted)".to_string());
        notes.push("Distance consistent with 'Planet Nine' hypothetical perturber".to_string());
    }

    if stats.avg_a > 100.0 {
        candidate_perturbers.push("Distant stellar companion".to_string());
        notes.push("May indicate perturbation by distant stellar object".to_string());
    }

    notes.push(format!(
        "Object class distribution: {:?}",
        stats.class_distribution
    ));

    PerturberEstimate {
        estimated_distance,
        confidence,
        expected_mass_range: mass_range,
        candidate_perturbers,
        analysis_notes: notes,
    }
}

/// Get detailed summary for display
pub fn get_analysis_summary(analysis: &EccentricityAnalysis) -> String {
    let mut output = String::new();

    output.push_str("═════════════════════════════════════════════════════════════\n");
    output.push_str("ECCENTRICITY PUMPING ANALYSIS - KUIPER BELT OBJECTS\n");
    output.push_str("Analysis Agent 6: High Eccentricity Population Study\n");
    output.push_str("═════════════════════════════════════════════════════════════\n\n");

    output.push_str("SELECTION CRITERIA: e > 0.7 AND a > 50 AU\n");
    output.push_str("─────────────────────────────────────────────────────────────\n\n");

    let stats = &analysis.summary;
    output.push_str(&format!("IDENTIFIED OBJECTS: {}\n", stats.count));
    output.push_str(&format!("Average Semi-Major Axis: {:.2} AU\n", stats.avg_a));
    output.push_str(&format!("Median Semi-Major Axis: {:.2} AU\n", stats.median_a));
    output.push_str(&format!("Average Eccentricity: {:.4}\n", stats.avg_e));
    output.push_str(&format!("Average Perihelion: {:.2} AU\n", stats.avg_q));
    output.push_str(&format!("Average Aphelion: {:.2} AU\n\n", stats.avg_ad));

    output.push_str("ORBITAL PARAMETER RANGES:\n");
    output.push_str(&format!("  Semi-major axis:  {:.2} - {:.2} AU\n", stats.a_range.0, stats.a_range.1));
    output.push_str(&format!("  Eccentricity:     {:.3} - {:.3}\n", stats.e_range.0, stats.e_range.1));
    output.push_str(&format!("  Perihelion:       {:.2} - {:.2} AU\n\n", stats.q_range.0, stats.q_range.1));

    output.push_str("OBJECT CLASSIFICATION:\n");
    for (class, count) in &stats.class_distribution {
        output.push_str(&format!("  {}: {}\n", class, count));
    }
    output.push_str("\n");

    let perturber = &analysis.perturber_analysis;
    output.push_str("PERTURBER DISTANCE ESTIMATION:\n");
    output.push_str("─────────────────────────────────────────────────────────────\n");
    output.push_str(&format!("Estimated Distance: {:.1} AU\n", perturber.estimated_distance));
    output.push_str(&format!("Confidence Level: {:.0}%\n", perturber.confidence * 100.0));
    output.push_str(&format!("Expected Mass Range: {}\n\n", perturber.expected_mass_range));

    if !perturber.candidate_perturbers.is_empty() {
        output.push_str("CANDIDATE PERTURBERS:\n");
        for candidate in &perturber.candidate_perturbers {
            output.push_str(&format!("  • {}\n", candidate));
        }
        output.push_str("\n");
    }

    output.push_str("DETAILED ANALYSIS NOTES:\n");
    for (i, note) in perturber.analysis_notes.iter().enumerate() {
        output.push_str(&format!("  {}. {}\n", i + 1, note));
    }
    output.push_str("\n");

    output.push_str("HIGH ECCENTRICITY OBJECTS (Sorted by Eccentricity):\n");
    output.push_str("─────────────────────────────────────────────────────────────\n");

    for (idx, obj) in analysis.high_e_objects.iter().enumerate() {
        output.push_str(&format!(
            "{:2}. {} | e={:.4} | a={:.2} AU | q={:.2} AU | ad={:.2} AU\n",
            idx + 1, obj.name, obj.e, obj.a, obj.q, obj.ad
        ));
        output.push_str(&format!(
            "      Pumping Strength: {:.3} | Heating Factor: {:.4}\n",
            obj.pumping_strength, obj.heating_factor
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eccentricity_analysis() {
        let analysis = analyze_eccentricity_pumping();
        assert!(!analysis.high_e_objects.is_empty());
        assert!(analysis.summary.count > 0);
    }

    #[test]
    fn test_perturber_estimation() {
        let analysis = analyze_eccentricity_pumping();
        let estimated = analysis.perturber_analysis.estimated_distance;

        // Should be reasonable (> 50 AU, < 2000 AU)
        assert!(estimated > 50.0 && estimated < 2000.0);
    }

    #[test]
    fn test_high_e_criteria() {
        let analysis = analyze_eccentricity_pumping();

        for obj in &analysis.high_e_objects {
            assert!(obj.e > 0.7, "Object {} has e={}, expected > 0.7", obj.name, obj.e);
            assert!(obj.a > 50.0, "Object {} has a={}, expected > 50.0", obj.name, obj.a);
        }
    }
}
