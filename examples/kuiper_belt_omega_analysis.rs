//! # Standalone Argument of Perihelion (ω) Clustering Analysis
//!
//! Analyzes clustering patterns in argument of perihelion for high-perihelion KBOs
//! without dependency on other example modules.

use std::f32::consts::PI;

/// KBO data structure
#[derive(Debug, Clone)]
pub struct KBO {
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

/// Load pre-configured KBO dataset
fn load_kbo_data() -> Vec<KBO> {
    vec![
        // DWARF PLANETS AND MAJOR TNOs
        KBO { name: "134340 Pluto".to_string(), a: 39.59, e: 0.2518, i: 17.15, q: 29.619, ad: 49.56, omega: 110.29, w: 113.71, class: "TNO".to_string() },
        KBO { name: "136199 Eris".to_string(), a: 68.0, e: 0.4370, i: 43.87, q: 38.284, ad: 97.71, omega: 36.03, w: 150.73, class: "TNO".to_string() },
        KBO { name: "136108 Haumea".to_string(), a: 43.01, e: 0.1958, i: 28.21, q: 34.586, ad: 51.42, omega: 121.80, w: 240.89, class: "TNO".to_string() },
        KBO { name: "136472 Makemake".to_string(), a: 45.51, e: 0.1604, i: 29.03, q: 38.210, ad: 52.81, omega: 79.27, w: 297.08, class: "TNO".to_string() },
        KBO { name: "225088 Gonggong".to_string(), a: 66.89, e: 0.5032, i: 30.87, q: 33.235, ad: 100.55, omega: 336.84, w: 206.64, class: "TNO".to_string() },
        KBO { name: "90377 Sedna".to_string(), a: 549.5, e: 0.8613, i: 11.93, q: 76.223, ad: 1022.86, omega: 144.48, w: 311.01, class: "TNO".to_string() },
        KBO { name: "50000 Quaoar".to_string(), a: 43.15, e: 0.0358, i: 7.99, q: 41.601, ad: 44.69, omega: 188.96, w: 163.92, class: "TNO".to_string() },
        KBO { name: "90482 Orcus".to_string(), a: 39.34, e: 0.2217, i: 20.56, q: 30.614, ad: 48.06, omega: 268.39, w: 73.72, class: "TNO".to_string() },

        // SCATTERED DISK OBJECTS
        KBO { name: "15874 (1996 TL66)".to_string(), a: 84.89, e: 0.5866, i: 23.96, q: 35.094, ad: 134.69, omega: 217.70, w: 185.14, class: "TNO".to_string() },
        KBO { name: "26181 (1996 GQ21)".to_string(), a: 92.48, e: 0.5874, i: 13.36, q: 38.152, ad: 146.81, omega: 194.22, w: 356.02, class: "TNO".to_string() },
        KBO { name: "26375 (1999 DE9)".to_string(), a: 55.5, e: 0.4201, i: 7.61, q: 32.184, ad: 78.81, omega: 322.88, w: 159.37, class: "TNO".to_string() },
        KBO { name: "82075 (2000 YW134)".to_string(), a: 58.23, e: 0.2936, i: 19.77, q: 41.128, ad: 75.32, omega: 126.91, w: 316.59, class: "TNO".to_string() },
        KBO { name: "84522 (2002 TC302)".to_string(), a: 55.84, e: 0.2995, i: 35.01, q: 39.113, ad: 72.56, omega: 23.83, w: 86.07, class: "TNO".to_string() },
        KBO { name: "145480 (2005 TB190)".to_string(), a: 75.93, e: 0.3912, i: 26.48, q: 46.227, ad: 105.64, omega: 180.46, w: 171.99, class: "TNO".to_string() },
        KBO { name: "229762 G!kun||'homdima".to_string(), a: 74.59, e: 0.4961, i: 23.33, q: 37.585, ad: 111.59, omega: 131.24, w: 345.94, class: "TNO".to_string() },
        KBO { name: "145451 Rumina".to_string(), a: 92.27, e: 0.6190, i: 28.70, q: 35.160, ad: 149.39, omega: 84.63, w: 318.73, class: "TNO".to_string() },

        // EXTREME/DETACHED OBJECTS
        KBO { name: "148209 (2000 CR105)".to_string(), a: 228.7, e: 0.8071, i: 22.71, q: 44.117, ad: 413.29, omega: 128.21, w: 316.92, class: "TNO".to_string() },
        KBO { name: "82158 (2001 FP185)".to_string(), a: 213.4, e: 0.8398, i: 30.80, q: 34.190, ad: 392.66, omega: 179.36, w: 6.62, class: "TNO".to_string() },
        KBO { name: "87269 (2000 OO67)".to_string(), a: 617.9, e: 0.9663, i: 20.05, q: 20.850, ad: 1215.04, omega: 142.38, w: 212.72, class: "TNO".to_string() },
        KBO { name: "308933 (2006 SQ372)".to_string(), a: 839.3, e: 0.9711, i: 19.46, q: 24.226, ad: 1654.33, omega: 197.37, w: 122.65, class: "TNO".to_string() },
        KBO { name: "445473 (2010 VZ98)".to_string(), a: 159.8, e: 0.7851, i: 4.51, q: 34.356, ad: 285.32, omega: 117.44, w: 313.74, class: "TNO".to_string() },
        KBO { name: "303775 (2005 QU182)".to_string(), a: 112.2, e: 0.6696, i: 14.01, q: 37.059, ad: 187.28, omega: 78.54, w: 224.26, class: "TNO".to_string() },
        KBO { name: "437360 (2013 TV158)".to_string(), a: 114.1, e: 0.6801, i: 31.14, q: 36.482, ad: 191.62, omega: 181.07, w: 232.30, class: "TNO".to_string() },
    ]
}

/// Calculate circular mean for angle data
fn circular_mean(angles: &[f32]) -> f32 {
    if angles.is_empty() {
        return 0.0;
    }
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    for &angle in angles {
        let rad = angle * PI / 180.0;
        sin_sum += rad.sin();
        cos_sum += rad.cos();
    }
    let mean_rad = sin_sum.atan2(cos_sum);
    ((mean_rad * 180.0 / PI + 360.0) % 360.0).abs()
}

/// Calculate circular standard deviation
fn circular_std_dev(angles: &[f32], mean: f32) -> f32 {
    if angles.len() < 2 {
        return 0.0;
    }
    let mean_rad = mean * PI / 180.0;
    let mut cos_sum = 0.0;
    for &angle in angles {
        let rad = angle * PI / 180.0;
        cos_sum += (rad - mean_rad).cos();
    }
    let r = cos_sum / angles.len() as f32;
    let variance = 2.0 * (1.0 - r);
    let std_dev_rad = variance.sqrt();
    std_dev_rad * 180.0 / PI
}

/// Calculate Kozai score
fn kozai_score(cluster_0: usize, cluster_180: usize, total: usize) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let cluster_count = (cluster_0 + cluster_180) as f32;
    let clustering_ratio = cluster_count / total as f32;
    let dominance = if cluster_0 > cluster_180 {
        cluster_0 as f32 / cluster_count.max(1.0)
    } else {
        cluster_180 as f32 / cluster_count.max(1.0)
    };
    (clustering_ratio + dominance) / 2.0
}

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     ARGUMENT OF PERIHELION CLUSTERING ANALYSIS              ║");
    println!("║  HIGH-q KUIPER BELT OBJECTS (q > 37 AU, a > 50 AU)         ║");
    println!("║           Analysis Agent 2: Argument of Perihelion          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load KBO data
    let all_kbos = load_kbo_data();

    // Filter for high-q objects: q > 37 AU and a > 50 AU
    let high_q_objects: Vec<&KBO> = all_kbos.iter()
        .filter(|o| o.q > 37.0 && o.a > 50.0)
        .collect();

    if high_q_objects.is_empty() {
        println!("No high-q objects found in dataset (q > 37 AU, a > 50 AU)");
        return;
    }

    println!("📊 SAMPLE STATISTICS");
    println!("───────────────────────────────────────────────────────────────");
    println!("Total high-q objects found: {}\n", high_q_objects.len());

    // Calculate ω statistics
    let w_values: Vec<f32> = high_q_objects.iter().map(|o| o.w).collect();
    let mean_w = circular_mean(&w_values);
    let std_dev_w = circular_std_dev(&w_values, mean_w);
    let min_w = w_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_w = w_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("Mean ω (circular):    {:.1}°", mean_w);
    println!("Std dev ω:            {:.1}°", std_dev_w);
    println!("Range:                {:.1}° - {:.1}°\n", min_w, max_w);

    // Cluster analysis
    let cluster_0: Vec<&KBO> = high_q_objects.iter()
        .filter(|o| o.w < 45.0 || o.w > 315.0)
        .copied()
        .collect();

    let cluster_180: Vec<&KBO> = high_q_objects.iter()
        .filter(|o| o.w > 135.0 && o.w < 225.0)
        .copied()
        .collect();

    let scattered: Vec<&KBO> = high_q_objects.iter()
        .filter(|o| !((o.w < 45.0 || o.w > 315.0) || (o.w > 135.0 && o.w < 225.0)))
        .copied()
        .collect();

    println!("🎯 CLUSTERING ANALYSIS");
    println!("───────────────────────────────────────────────────────────────");
    println!("Objects near 0° (±45°):       {:3} ({:5.1}%)",
        cluster_0.len(),
        (cluster_0.len() as f32 / high_q_objects.len() as f32) * 100.0);
    println!("Objects near 180° (±45°):     {:3} ({:5.1}%)",
        cluster_180.len(),
        (cluster_180.len() as f32 / high_q_objects.len() as f32) * 100.0);
    println!("Objects scattered (45°-135°): {:3} ({:5.1}%)\n",
        scattered.len(),
        (scattered.len() as f32 / high_q_objects.len() as f32) * 100.0);

    let kozai = kozai_score(cluster_0.len(), cluster_180.len(), high_q_objects.len());
    println!("🔍 KOZAI RESONANCE ANALYSIS");
    println!("───────────────────────────────────────────────────────────────");
    println!("Kozai Score: {:.3}\n", kozai);

    if kozai > 0.6 {
        println!("⭐ STRONG CLUSTERING - Consistent with Kozai mechanism");
    } else if kozai > 0.4 {
        println!("✓ MODERATE CLUSTERING - Possible Kozai influence");
    } else if kozai > 0.2 {
        println!("○ WEAK CLUSTERING - Insufficient evidence for Kozai");
    } else {
        println!("✗ NO CLUSTERING - No Kozai signature detected");
    }

    println!("\n🪐 PLANET PERTURBATION EVIDENCE");
    println!("───────────────────────────────────────────────────────────────");

    if kozai > 0.4 {
        println!("✓ EVIDENCE DETECTED\n");
        if cluster_0.len() > cluster_180.len() {
            println!("Dominant cluster: 0° (aligned perihelion)");
            println!("→ Perturbing planet likely has LOW inclination\n");
        } else if cluster_180.len() > cluster_0.len() {
            println!("Dominant cluster: 180° (anti-aligned perihelion)");
            println!("→ Perturbing planet likely has HIGH inclination\n");
        }
    } else {
        println!("✗ NO SIGNIFICANT EVIDENCE\n");
    }

    println!("📋 HIGH-q OBJECT LIST (q > 37 AU, a > 50 AU)");
    println!("───────────────────────────────────────────────────────────────");
    println!("{:<35} {:<8} {:<8} {:<8} {:<12}", "Object", "a (AU)", "q (AU)", "ω (°)", "Cluster");
    println!("───────────────────────────────────────────────────────────────");

    let mut sorted = high_q_objects.clone();
    sorted.sort_by(|a, b| a.w.partial_cmp(&b.w).unwrap());

    for obj in &sorted {
        let cluster_label = if obj.w < 45.0 || obj.w > 315.0 {
            "0° cluster"
        } else if obj.w > 135.0 && obj.w < 225.0 {
            "180° cluster"
        } else {
            "Scattered"
        };

        println!("{:<35} {:<8.1} {:<8.1} {:<8.1} {:<12}",
            &obj.name[..obj.name.len().min(34)],
            obj.a, obj.q, obj.w, cluster_label);
    }

    println!("\n🔬 0° CLUSTER (Aligned Perihelion)");
    println!("───────────────────────────────────────────────────────────────");
    for obj in &cluster_0 {
        println!("  • {} (ω={:.1}°, a={:.1} AU, e={:.2})", obj.name, obj.w, obj.a, obj.e);
    }

    println!("\n🔬 180° CLUSTER (Anti-aligned Perihelion)");
    println!("───────────────────────────────────────────────────────────────");
    for obj in &cluster_180 {
        println!("  • {} (ω={:.1}°, a={:.1} AU, e={:.2})", obj.name, obj.w, obj.a, obj.e);
    }

    println!("\n📊 ORBITAL PARAMETER ANALYSIS");
    println!("───────────────────────────────────────────────────────────────");
    let avg_a: f32 = high_q_objects.iter().map(|o| o.a).sum::<f32>() / high_q_objects.len() as f32;
    let avg_e: f32 = high_q_objects.iter().map(|o| o.e).sum::<f32>() / high_q_objects.len() as f32;
    let avg_i: f32 = high_q_objects.iter().map(|o| o.i).sum::<f32>() / high_q_objects.len() as f32;
    let avg_q: f32 = high_q_objects.iter().map(|o| o.q).sum::<f32>() / high_q_objects.len() as f32;

    println!("Average a: {:.1} AU", avg_a);
    println!("Average e: {:.3}", avg_e);
    println!("Average i: {:.1}°", avg_i);
    println!("Average q: {:.1} AU\n", avg_q);

    println!("═══════════════════════════════════════════════════════════════\n");
}
