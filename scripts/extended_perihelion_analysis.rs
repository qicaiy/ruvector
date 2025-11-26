#!/usr/bin/env rust-script

/// Extended Perihelion Clustering Analysis
/// Examines broader range of objects: a > 100 AU and q > 30 AU

use std::f64::consts::PI;

#[derive(Debug, Clone)]
struct TNO {
    name: String,
    a: f64,      // Semi-major axis (AU)
    e: f64,      // Eccentricity
    q: f64,      // Perihelion distance (AU)
    omega: f64,  // Longitude of ascending node (degrees)
    w: f64,      // Argument of perihelion (degrees)
}

impl TNO {
    fn longitude_of_perihelion(&self) -> f64 {
        (self.omega + self.w) % 360.0
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║         EXTENDED PERIHELION CLUSTERING ANALYSIS               ║");
    println!("║     Sensitivity Analysis with Broader Selection Criteria      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let objects = get_extended_etno_data();
    println!("📦 Loaded {} Trans-Neptunian Objects from NASA/JPL SBDB\n", objects.len());

    // Analysis 1: Strict criteria (a > 150 AU, q > 30 AU)
    run_analysis(&objects, 150.0, "STRICT CRITERIA (a > 150 AU, q > 30 AU)");

    // Analysis 2: Extended criteria (a > 100 AU, q > 30 AU)
    run_analysis(&objects, 100.0, "EXTENDED CRITERIA (a > 100 AU, q > 30 AU)");

    // Analysis 3: Very broad criteria (a > 80 AU, q > 30 AU)
    run_analysis(&objects, 80.0, "BROAD CRITERIA (a > 80 AU, q > 30 AU)");

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    COMPARATIVE SUMMARY                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Compare results across criteria
    let strict = filter_and_analyze(&objects, 150.0);
    let extended = filter_and_analyze(&objects, 100.0);
    let broad = filter_and_analyze(&objects, 80.0);

    println!("Criteria Set        | n  | Mean (°) | σ (°) | r     | p-value");
    println!("────────────────────┼────┼──────────┼───────┼───────┼────────");
    println!("a > 150, q > 30    | {:2} | {:8.2} | {:5.2} | {:.3} | {:.3}",
        strict.count, strict.mean, strict.std_dev, strict.r, strict.p_value);
    println!("a > 100, q > 30    | {:2} | {:8.2} | {:5.2} | {:.3} | {:.3}",
        extended.count, extended.mean, extended.std_dev, extended.r, extended.p_value);
    println!("a >  80, q > 30    | {:2} | {:8.2} | {:5.2} | {:.3} | {:.3}",
        broad.count, broad.mean, broad.std_dev, broad.r, broad.p_value);

    println!("\n📊 INTERPRETATION:");
    println!("──────────────────────────────────────────────────────────────");

    let strict_clustered = strict.std_dev < 90.0;
    let extended_clustered = extended.std_dev < 90.0;
    let broad_clustered = broad.std_dev < 90.0;

    println!("Strict (n=4):      {} clustering", if strict_clustered { "✓" } else { "✗" });
    println!("Extended (n={}):    {} clustering", extended.count, if extended_clustered { "✓" } else { "✗" });
    println!("Broad (n={}):      {} clustering", broad.count, if broad_clustered { "✓" } else { "✗" });

    println!("\n🔬 STATISTICAL ROBUSTNESS:");
    println!("──────────────────────────────────────────────────────────────");

    if extended.p_value < 0.05 {
        println!("✓ Extended dataset shows SIGNIFICANT clustering (p < 0.05)");
        println!("  This strengthens the evidence for external perturbation.");
    } else if extended.p_value < 0.20 {
        println!("◐ Extended dataset shows BORDERLINE clustering (0.05 < p < 0.20)");
        println!("  Clustering detected but not conclusive at conventional levels.");
    } else {
        println!("✗ Extended dataset does not show significant clustering (p ≥ 0.20)");
        println!("  Larger sample suggests randomness in broader population.");
    }

    println!("\n💡 PLANET NINE ASSESSMENT:");
    println!("──────────────────────────────────────────────────────────────");

    if extended.std_dev < 70.0 && extended.r > 0.6 {
        println!("STRONG EVIDENCE: Perihelion clustering supports Planet Nine hypothesis");
        println!("Recommendation: Prioritize search in region opposite perihelion cluster");
    } else if extended.std_dev < 90.0 && extended.r > 0.5 {
        println!("MODERATE EVIDENCE: Some perihelion clustering detected");
        println!("Recommendation: Continue detailed analysis with new discoveries");
    } else {
        println!("WEAK EVIDENCE: Insufficient clustering for definitive conclusion");
        println!("Recommendation: Expand sample; alternative mechanisms may dominate");
    }

    println!("\n════════════════════════════════════════════════════════════════\n");
}

fn run_analysis(objects: &[TNO], a_min: f64, criteria_name: &str) {
    println!("🎯 {}", criteria_name);
    println!("──────────────────────────────────────────────────────────────\n");

    let filtered: Vec<_> = objects.iter()
        .filter(|o| o.a > a_min && o.q > 30.0)
        .collect();

    println!("   Objects matching criteria: {}\n", filtered.len());

    if filtered.len() < 2 {
        println!("   (Insufficient for analysis)\n");
        return;
    }

    // Display objects
    println!("   {:30} {:>8} {:>8} {:>8}", "Name", "a (AU)", "q (AU)", "ω+w (°)");
    println!("   ──────────────────────────────────────────────────────────");

    for obj in &filtered {
        let varpi = obj.longitude_of_perihelion();
        println!("   {:30} {:>8.1} {:>8.1} {:>8.1}",
            &obj.name[..obj.name.len().min(28)], obj.a, obj.q, varpi);
    }

    // Calculate statistics
    let perihelia: Vec<f64> = filtered.iter()
        .map(|o| o.longitude_of_perihelion())
        .collect();

    let sin_sum: f64 = perihelia.iter().map(|p| (p * PI / 180.0).sin()).sum();
    let cos_sum: f64 = perihelia.iter().map(|p| (p * PI / 180.0).cos()).sum();

    let mean_rad = sin_sum.atan2(cos_sum);
    let mean_deg = mean_rad * 180.0 / PI;
    let mean = if mean_deg < 0.0 { mean_deg + 360.0 } else { mean_deg };

    let r = ((sin_sum.powi(2) + cos_sum.powi(2)).sqrt()) / perihelia.len() as f64;

    let deviations: Vec<f64> = perihelia.iter()
        .map(|p| {
            let diff = (p - mean).abs();
            if diff > 180.0 { 360.0 - diff } else { diff }
        })
        .collect();

    let variance = deviations.iter().map(|d| d.powi(2)).sum::<f64>() / deviations.len() as f64;
    let std_dev = variance.sqrt();

    let z = perihelia.len() as f64 * r.powi(2);
    let p_value = (-z).exp();

    println!("\n   Mean longitude:        {:.2}°", mean);
    println!("   Std deviation (σ):     {:.2}°", std_dev);
    println!("   Concentration (r):     {:.4}", r);
    println!("   Rayleigh p-value:      {:.4}", p_value);

    if std_dev < 90.0 {
        println!("   Result:                ✓ CLUSTERING DETECTED\n");
    } else {
        println!("   Result:                ✗ No significant clustering\n");
    }
}

struct AnalysisResult {
    count: usize,
    mean: f64,
    std_dev: f64,
    r: f64,
    p_value: f64,
}

fn filter_and_analyze(objects: &[TNO], a_min: f64) -> AnalysisResult {
    let filtered: Vec<_> = objects.iter()
        .filter(|o| o.a > a_min && o.q > 30.0)
        .collect();

    if filtered.len() < 2 {
        return AnalysisResult {
            count: 0,
            mean: 0.0,
            std_dev: 0.0,
            r: 0.0,
            p_value: 1.0,
        };
    }

    let perihelia: Vec<f64> = filtered.iter()
        .map(|o| o.longitude_of_perihelion())
        .collect();

    let sin_sum: f64 = perihelia.iter().map(|p| (p * PI / 180.0).sin()).sum();
    let cos_sum: f64 = perihelia.iter().map(|p| (p * PI / 180.0).cos()).sum();

    let mean_rad = sin_sum.atan2(cos_sum);
    let mean_deg = mean_rad * 180.0 / PI;
    let mean = if mean_deg < 0.0 { mean_deg + 360.0 } else { mean_deg };

    let r = ((sin_sum.powi(2) + cos_sum.powi(2)).sqrt()) / perihelia.len() as f64;

    let deviations: Vec<f64> = perihelia.iter()
        .map(|p| {
            let diff = (p - mean).abs();
            if diff > 180.0 { 360.0 - diff } else { diff }
        })
        .collect();

    let variance = deviations.iter().map(|d| d.powi(2)).sum::<f64>() / deviations.len() as f64;
    let std_dev = variance.sqrt();

    let z = perihelia.len() as f64 * r.powi(2);
    let p_value = (-z).exp();

    AnalysisResult { count: filtered.len(), mean, std_dev, r, p_value }
}

fn get_extended_etno_data() -> Vec<TNO> {
    vec![
        // Ultra-extreme (a > 300 AU)
        TNO { name: "90377 Sedna".to_string(), a: 549.5, e: 0.8613, q: 76.223, omega: 144.48, w: 311.01 },
        TNO { name: "336756 (2010 NV1)".to_string(), a: 305.2, e: 0.9690, q: 9.457, omega: 136.32, w: 133.20 },
        TNO { name: "418993 (2009 MS9)".to_string(), a: 375.7, e: 0.9706, q: 11.046, omega: 220.18, w: 128.90 },
        TNO { name: "87269 (2000 OO67)".to_string(), a: 617.9, e: 0.9663, q: 20.850, omega: 142.38, w: 212.72 },
        TNO { name: "308933 (2006 SQ372)".to_string(), a: 839.3, e: 0.9711, q: 24.226, omega: 197.37, w: 122.65 },

        // Very distant (200-300 AU)
        TNO { name: "148209 (2000 CR105)".to_string(), a: 228.7, e: 0.8071, q: 44.117, omega: 128.21, w: 316.92 },
        TNO { name: "82158 (2001 FP185)".to_string(), a: 213.4, e: 0.8398, q: 34.190, omega: 179.36, w: 6.62 },

        // Distant (150-200 AU)
        TNO { name: "445473 (2010 VZ98)".to_string(), a: 159.8, e: 0.7851, q: 34.356, omega: 117.44, w: 313.74 },

        // Moderately distant (100-150 AU)
        TNO { name: "353222 (2009 YD7)".to_string(), a: 125.7, e: 0.8936, q: 13.379, omega: 125.99, w: 326.91 },
        TNO { name: "54520 (2000 PJ30)".to_string(), a: 121.9, e: 0.7654, q: 28.603, omega: 293.41, w: 303.43 },
        TNO { name: "303775 (2005 QU182)".to_string(), a: 112.2, e: 0.6696, q: 37.059, omega: 78.54, w: 224.26 },
        TNO { name: "437360 (2013 TV158)".to_string(), a: 114.1, e: 0.6801, q: 36.482, omega: 181.07, w: 232.30 },

        // Extended outer belt (80-100 AU, still relevant)
        TNO { name: "82155 (2001 FZ173)".to_string(), a: 84.62, e: 0.6176, q: 32.362, omega: 2.40, w: 199.01 },
        TNO { name: "65489 Ceto".to_string(), a: 100.5, e: 0.8238, q: 17.709, omega: 171.95, w: 319.46 },
        TNO { name: "29981 (1999 TD10)".to_string(), a: 98.47, e: 0.8743, q: 12.374, omega: 184.61, w: 173.03 },
        TNO { name: "15874 (1996 TL66)".to_string(), a: 84.89, e: 0.5866, q: 35.094, omega: 217.70, w: 185.14 },
        TNO { name: "26181 (1996 GQ21)".to_string(), a: 92.48, e: 0.5874, q: 38.152, omega: 194.22, w: 356.02 },
    ]
}
