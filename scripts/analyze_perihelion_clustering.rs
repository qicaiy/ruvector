#!/usr/bin/env rust-script

/// Perihelion Clustering Analysis for Planet Nine
/// Script to analyze ETNO perihelion longitude distribution

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
    println!("║          PERIHELION CLUSTERING ANALYSIS - PLANET NINE          ║");
    println!("║       Analysis of Extreme Trans-Neptunian Objects (ETNOs)      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Load Kuiper Belt data
    let objects = get_etno_data();
    println!("📦 Loaded {} Trans-Neptunian Objects from NASA/JPL SBDB\n", objects.len());

    // Filter for extreme TNOs: a > 150 AU and q > 30 AU
    let extreme_tnos: Vec<_> = objects.iter()
        .filter(|o| o.a > 150.0 && o.q > 30.0)
        .collect();

    println!("🎯 ETNO SELECTION CRITERIA: a > 150 AU AND q > 30 AU");
    println!("──────────────────────────────────────────────────────────────\n");
    println!("   Objects matching criteria: {}\n", extreme_tnos.len());

    if extreme_tnos.is_empty() {
        println!("   ⚠️  No extreme TNOs found with specified criteria.");
        println!("   Analysis cannot proceed.\n");
        return;
    }

    // Display the identified ETNOs
    println!("📋 IDENTIFIED EXTREME TNOs:");
    println!("──────────────────────────────────────────────────────────────");
    println!("{:<30} {:>8} {:>8} {:>8} {:>8}", "Name", "a (AU)", "e", "q (AU)", "ω+w (°)");
    println!("──────────────────────────────────────────────────────────────");

    for obj in &extreme_tnos {
        let varpi = obj.longitude_of_perihelion();
        println!("{:<30} {:>8.1} {:>8.3} {:>8.1} {:>8.1}",
            &obj.name[..obj.name.len().min(28)],
            obj.a, obj.e, obj.q, varpi);
    }
    println!();

    // Calculate perihelion longitudes
    let perihelia: Vec<f64> = extreme_tnos.iter()
        .map(|o| o.longitude_of_perihelion())
        .collect();

    // Circular statistics: calculate mean direction
    let sin_sum: f64 = perihelia.iter()
        .map(|p| (p * PI / 180.0).sin())
        .sum();
    let cos_sum: f64 = perihelia.iter()
        .map(|p| (p * PI / 180.0).cos())
        .sum();

    let mean_rad = sin_sum.atan2(cos_sum);
    let mean_deg = mean_rad * 180.0 / PI;
    let mean_longitude = if mean_deg < 0.0 { mean_deg + 360.0 } else { mean_deg };

    // Resultant vector length (concentration/clustering indicator)
    let r = ((sin_sum.powi(2) + cos_sum.powi(2)).sqrt()) / perihelia.len() as f64;

    // Linear standard deviation
    let deviations: Vec<f64> = perihelia.iter()
        .map(|p| {
            let diff = (p - mean_longitude).abs();
            if diff > 180.0 { 360.0 - diff } else { diff }
        })
        .collect();

    let variance = deviations.iter()
        .map(|d| d.powi(2))
        .sum::<f64>() / deviations.len() as f64;
    let std_dev = variance.sqrt();

    // Rayleigh test for non-uniformity
    let z = perihelia.len() as f64 * r.powi(2);
    let p_value = (-z).exp();

    // Clustering detection
    let is_clustered = std_dev < 90.0;

    println!("📊 PERIHELION LONGITUDE DISTRIBUTION ANALYSIS");
    println!("──────────────────────────────────────────────────────────────\n");

    println!("   Sample size (n):                     {}", extreme_tnos.len());
    println!("   Mean perihelion longitude:           {:.2}°", mean_longitude);
    println!("   Linear standard deviation (σ):       {:.2}°", std_dev);
    println!("   Linear variance:                     {:.2}°²\n", variance);

    println!("   Circular concentration (r):          {:.4}", r);
    println!("   Rayleigh test statistic (Z):         {:.4}", z);
    println!("   P-value (uniformity test):           {:.6}", p_value);
    println!();

    // Statistical significance
    println!("🔬 STATISTICAL SIGNIFICANCE");
    println!("──────────────────────────────────────────────────────────────");

    if p_value < 0.001 {
        println!("   ★★★ HIGHLY SIGNIFICANT (p < 0.001)");
        println!("   Perihelion distribution is highly non-uniform");
        println!("   Strong evidence for clustering\n");
    } else if p_value < 0.01 {
        println!("   ★★ VERY SIGNIFICANT (p < 0.01)");
        println!("   Perihelion distribution shows clustering\n");
    } else if p_value < 0.05 {
        println!("   ★ SIGNIFICANT (p < 0.05)");
        println!("   Perihelion distribution deviates from uniform\n");
    } else {
        println!("   ○ NOT SIGNIFICANT (p ≥ 0.05)");
        println!("   Perihelion distribution consistent with random\n");
    }

    // Clustering detection
    println!("🎯 CLUSTERING ASSESSMENT");
    println!("──────────────────────────────────────────────────────────────");

    if is_clustered {
        println!("   ✓ CLUSTERING DETECTED");
        println!("   Standard deviation ({:.2}°) < 90° threshold", std_dev);
        println!("   ETNOs show non-random perihelion longitude clustering\n");
    } else {
        println!("   ✗ NO SIGNIFICANT CLUSTERING");
        println!("   Standard deviation ({:.2}°) ≥ 90° threshold", std_dev);
        println!("   Distribution suggests randomness\n");
    }

    // Planet Nine implications
    println!("🪐 PLANET NINE IMPLICATIONS");
    println!("──────────────────────────────────────────────────────────────");

    if is_clustered && p_value < 0.05 {
        let perturber_longitude = (mean_longitude + 180.0) % 360.0;
        println!("   ✓ Evidence consistent with Planet Nine hypothesis");
        println!("   Observed clustering suggests external perturber");
        println!("   ");
        println!("   Estimated perturber location (anti-aligned): {:.1}°", perturber_longitude);
        println!("   Confidence level: MODERATE TO HIGH");
        println!("   ");
        println!("   Interpretation:");
        println!("   • ETNOs cluster at their perihelion due to perturbation");
        println!("   • A massive body ~400-600 AU could cause this pattern");
        println!("   • Anti-aligned configuration maintains ETNO clustering");
    } else if r > 0.4 {
        println!("   ◐ WEAK EVIDENCE for external perturbation");
        println!("   Concentration parameter r = {:.3}", r);
        println!("   Some perihelion clustering detected but not statistically significant");
    } else {
        println!("   ✗ No significant evidence for Planet Nine");
        println!("   Perihelion distribution appears random");
    }
    println!();

    // Clustering breakdown
    println!("📈 PERIHELION LONGITUDE CLUSTERING BREAKDOWN");
    println!("──────────────────────────────────────────────────────────────");

    let mut sorted_perihelia = perihelia.clone();
    sorted_perihelia.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("   Sorted perihelion longitudes (degrees):");
    for (i, &p) in sorted_perihelia.iter().enumerate() {
        if i % 4 == 0 && i > 0 {
            println!();
        }
        print!("   {:>7.1}° ", p);
    }
    println!("\n");

    // Find natural clusters
    let mut clusters: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut current_cluster = vec![(0, sorted_perihelia[0])];

    for i in 1..sorted_perihelia.len() {
        let prev = sorted_perihelia[i-1];
        let curr = sorted_perihelia[i];
        let gap = curr - prev;

        // Consider gap > 30° as cluster boundary
        if gap > 30.0 {
            clusters.push(current_cluster);
            current_cluster = vec![(i, curr)];
        } else {
            current_cluster.push((i, curr));
        }
    }
    clusters.push(current_cluster);

    println!("   Identified {} sub-clusters (gap threshold: 30°)", clusters.len());
    for (idx, cluster) in clusters.iter().enumerate() {
        let min = cluster.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let max = cluster.iter().map(|(_, v)| *v).fold(0.0, f64::max);
        let count = cluster.len();
        println!("     Cluster {}: [{:>6.1}° - {:>6.1}°] - {} objects",
            idx + 1, min, max, count);
    }
    println!();

    // Summary statistics
    println!("📑 SUMMARY");
    println!("══════════════════════════════════════════════════════════════");
    println!("   Data Source: NASA/JPL Small-Body Database");
    println!("   Analysis Date: 2025-11-26");
    println!("   Method: Circular statistics (Rayleigh test)");
    println!();
    println!("   Key Finding: {}", if is_clustered {
        "PERIHELION CLUSTERING DETECTED - Consistent with Planet Nine hypothesis"
    } else {
        "No statistically significant clustering - Random distribution expected"
    });
    println!("══════════════════════════════════════════════════════════════\n");
}

fn get_etno_data() -> Vec<TNO> {
    vec![
        // Extreme/Detached Objects - these are the focus of Planet Nine analysis
        TNO {
            name: "90377 Sedna".to_string(),
            a: 549.5, e: 0.8613, q: 76.223, omega: 144.48, w: 311.01,
        },
        TNO {
            name: "148209 (2000 CR105)".to_string(),
            a: 228.7, e: 0.8071, q: 44.117, omega: 128.21, w: 316.92,
        },
        TNO {
            name: "82158 (2001 FP185)".to_string(),
            a: 213.4, e: 0.8398, q: 34.190, omega: 179.36, w: 6.62,
        },
        TNO {
            name: "87269 (2000 OO67)".to_string(),
            a: 617.9, e: 0.9663, q: 20.850, omega: 142.38, w: 212.72,
        },
        TNO {
            name: "308933 (2006 SQ372)".to_string(),
            a: 839.3, e: 0.9711, q: 24.226, omega: 197.37, w: 122.65,
        },
        TNO {
            name: "445473 (2010 VZ98)".to_string(),
            a: 159.8, e: 0.7851, q: 34.356, omega: 117.44, w: 313.74,
        },
        TNO {
            name: "303775 (2005 QU182)".to_string(),
            a: 112.2, e: 0.6696, q: 37.059, omega: 78.54, w: 224.26,
        },
        TNO {
            name: "437360 (2013 TV158)".to_string(),
            a: 114.1, e: 0.6801, q: 36.482, omega: 181.07, w: 232.30,
        },
        TNO {
            name: "336756 (2010 NV1)".to_string(),
            a: 305.2, e: 0.9690, q: 9.457, omega: 136.32, w: 133.20,
        },
        TNO {
            name: "418993 (2009 MS9)".to_string(),
            a: 375.7, e: 0.9706, q: 11.046, omega: 220.18, w: 128.90,
        },
        TNO {
            name: "353222 (2009 YD7)".to_string(),
            a: 125.7, e: 0.8936, q: 13.379, omega: 125.99, w: 326.91,
        },
        TNO {
            name: "54520 (2000 PJ30)".to_string(),
            a: 121.9, e: 0.7654, q: 28.603, omega: 293.41, w: 303.43,
        },
        TNO {
            name: "82155 (2001 FZ173)".to_string(),
            a: 84.62, e: 0.6176, q: 32.362, omega: 2.40, w: 199.01,
        },
        TNO {
            name: "65489 Ceto".to_string(),
            a: 100.5, e: 0.8238, q: 17.709, omega: 171.95, w: 319.46,
        },
        TNO {
            name: "29981 (1999 TD10)".to_string(),
            a: 98.47, e: 0.8743, q: 12.374, omega: 184.61, w: 173.03,
        },
    ]
}
