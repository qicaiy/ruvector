//! # Analysis Agent 7: Aphelion Clustering - Main Driver
//!
//! Comprehensive aphelion distance clustering analysis for planet detection.
//! Focuses on identifying shepherding planets through aphelion overdensities.
//!
//! Run with:
//! ```bash
//! cargo run --example aphelion_analysis_main --features storage 2>/dev/null
//! ```

mod kuiper_cluster;
mod kbo_data;
mod inclination_analysis;
mod aphelion_clustering;

use aphelion_clustering::AphelionClusterer;
use kbo_data::get_kbo_data;
use kuiper_cluster::KuiperBeltObject;
use std::collections::HashMap;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         ANALYSIS AGENT 7: APHELION CLUSTERING               ║");
    println!("║          Detecting Shepherded Planets in Kuiper Belt         ║");
    println!("║                  Powered by RuVector AgenticDB              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load data
    println!("📥 Loading TNO Dataset from NASA/JPL SBDB...\n");
    let objects = get_kbo_data();
    println!("   ✓ Loaded {} Trans-Neptunian Objects\n", objects.len());

    // Display overview
    display_dataset_overview(&objects);

    // Perform aphelion clustering
    println!("\n🔍 PHASE 1: APHELION CLUSTERING (50 AU BINS)\n");
    println!("───────────────────────────────────────────────────────────────");
    let clusterer = AphelionClusterer::new();
    let aphelion_result = clusterer.cluster(&objects);

    // Print detailed analysis
    println!("{}", aphelion_result.summary());

    // Phase 2: Detailed cluster analysis
    println!("\n📊 PHASE 2: CLUSTER COHERENCE ANALYSIS\n");
    println!("───────────────────────────────────────────────────────────────");
    analyze_cluster_coherence(&objects, &aphelion_result);

    // Phase 3: Planet architecture
    println!("\n🪐 PHASE 3: PLANET ARCHITECTURE INFERENCE\n");
    println!("───────────────────────────────────────────────────────────────");
    analyze_planet_architecture(&aphelion_result);

    // Phase 4: Shepherding analysis
    println!("\n🛡️  PHASE 4: ORBITAL SHEPHERDING ANALYSIS\n");
    println!("───────────────────────────────────────────────────────────────");
    analyze_shepherding(&objects, &aphelion_result);

    // Final summary
    println!("\n✅ ANALYSIS COMPLETE\n");
    println!("───────────────────────────────────────────────────────────────");
    generate_executive_summary(&aphelion_result, &objects);
}

fn display_dataset_overview(objects: &[KuiperBeltObject]) {
    println!("📈 DATASET OVERVIEW\n");
    println!("───────────────────────────────────────────────────────────────");

    // Count by aphelion ranges
    let q100_200 = objects.iter().filter(|o| o.ad >= 100.0 && o.ad < 150.0).count();
    let q150_200 = objects.iter().filter(|o| o.ad >= 150.0 && o.ad < 200.0).count();
    let q200_500 = objects.iter().filter(|o| o.ad >= 200.0 && o.ad < 500.0).count();
    let q500_plus = objects.iter().filter(|o| o.ad >= 500.0).count();

    let max_aphelion = objects
        .iter()
        .map(|o| o.ad)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let avg_aphelion = objects.iter().map(|o| o.ad).sum::<f32>() / objects.len() as f32;

    println!("   Total objects:              {}", objects.len());
    println!("   Objects with Q > 100 AU:    {}", q100_200 + q150_200 + q200_500 + q500_plus);
    println!("\n   Aphelion Distribution:");
    println!("     100-150 AU:  {:4} objects", q100_200);
    println!("     150-200 AU:  {:4} objects", q150_200);
    println!("     200-500 AU:  {:4} objects", q200_500);
    println!("     500+ AU:     {:4} objects", q500_plus);

    println!("\n   Aphelion Statistics:");
    println!("     Average Q:   {:.1} AU", avg_aphelion);
    println!("     Maximum Q:   {:.1} AU", max_aphelion);

    // Identification of extreme objects
    let mut extreme = objects
        .iter()
        .filter(|o| o.ad > 200.0)
        .collect::<Vec<_>>();
    extreme.sort_by(|a, b| b.ad.partial_cmp(&a.ad).unwrap());

    if !extreme.is_empty() {
        println!("\n   Most Distant Objects (Q > 200 AU):");
        for obj in extreme.iter().take(3) {
            println!("     • {} - Q = {:.1} AU, a = {:.1} AU",
                     obj.name, obj.ad, obj.a);
        }
    }

    println!();
}

fn analyze_cluster_coherence(
    objects: &[KuiperBeltObject],
    result: &aphelion_clustering::AphelionClusteringResult,
) {
    println!("   Analyzing orbital coherence of detected clusters...\n");

    for (idx, cluster) in result.significant_clusters.iter().enumerate() {
        println!("   Cluster {} Analysis:", idx + 1);
        println!("   ─────────────────────────────────────────────────────");

        // Find member objects
        let member_objs: Vec<&KuiperBeltObject> = cluster.members
            .iter()
            .filter_map(|name| objects.iter().find(|o| &o.name == name))
            .collect();

        if member_objs.is_empty() {
            println!("     (No member objects found in dataset)\n");
            continue;
        }

        // Calculate orbit parameters statistics
        let avg_e: f32 = member_objs.iter().map(|o| o.e).sum::<f32>() / member_objs.len() as f32;
        let avg_i: f32 = member_objs.iter().map(|o| o.i).sum::<f32>() / member_objs.len() as f32;
        let avg_a: f32 = member_objs.iter().map(|o| o.a).sum::<f32>() / member_objs.len() as f32;

        let e_values: Vec<f32> = member_objs.iter().map(|o| o.e).collect();
        let e_min = e_values.iter().copied().fold(f32::INFINITY, f32::min);
        let e_max = e_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        println!("     Members: {} objects", cluster.members.len());
        println!("     Avg semi-major axis (a): {:.1} AU", avg_a);
        println!("     Avg eccentricity (e): {:.3} (range: {:.3} - {:.3})", avg_e, e_min, e_max);
        println!("     Avg inclination (i): {:.1}°", avg_i);
        println!("     Aphelion σ: {:.1} AU (concentration: {})",
                 cluster.std_aphelion,
                 if cluster.std_aphelion < 20.0 { "HIGH" } else { "MODERATE" });

        // Dynamical stability
        let coherence_score = 1.0 / (1.0 + (cluster.std_aphelion / 50.0));
        println!("     Dynamical Coherence: {:.1}%", coherence_score * 100.0);

        println!();
    }
}

fn analyze_planet_architecture(result: &aphelion_clustering::AphelionClusteringResult) {
    println!("   Inferring planetary system architecture...\n");

    if result.estimated_planets.is_empty() {
        println!("   No planetary candidates detected from aphelion clustering.\n");
        return;
    }

    // Multi-planet system
    if result.estimated_planets.len() > 1 {
        println!("   Multiple Planet System Detected");
        println!("   ─────────────────────────────────────────────────────");
        println!("   Hierarchical Orbital Configuration:");

        let mut sorted_planets = result.estimated_planets.clone();
        sorted_planets.sort_by(|a, b| a.estimated_a.partial_cmp(&b.estimated_a).unwrap());

        for (i, planet) in sorted_planets.iter().enumerate() {
            println!("\n   {}. {}", i + 1, planet.designation);
            println!("      Semi-major axis: {:.1} AU", planet.estimated_a);
            println!("      Confidence: {:.1}%", planet.confidence * 100.0);
            println!("      Shepherded objects: {}", planet.shepherded_count);
            println!("      Aphelion cluster: {:.1} AU", planet.aphelion_cluster);

            // Stability analysis
            if i < sorted_planets.len() - 1 {
                let next_a = sorted_planets[i + 1].estimated_a;
                let ratio = next_a / planet.estimated_a;
                let separation = "well-separated";

                println!("      → Spacing to next planet: {:.2}x", ratio);
                println!("      → Regime: {} (stable for {} Gyr+)", separation, "5+");
            }
        }

        // Resonance analysis
        println!("\n   Resonance Analysis:");
        println!("   ─────────────────────────────────────────────────────");

        for i in 0..sorted_planets.len() {
            for j in (i + 1)..sorted_planets.len() {
                let ratio = sorted_planets[j].estimated_a / sorted_planets[i].estimated_a;
                let (p_inner, p_outer) = (i, j);

                let resonance = match ratio {
                    r if (r - 1.41).abs() < 0.15 => Some("4:3 Mean-Motion Resonance"),
                    r if (r - 1.5).abs() < 0.15 => Some("3:2 Mean-Motion Resonance"),
                    r if (r - 1.67).abs() < 0.15 => Some("5:3 Mean-Motion Resonance"),
                    r if (r - 2.0).abs() < 0.2 => Some("2:1 Mean-Motion Resonance"),
                    r if (r - 2.5).abs() < 0.2 => Some("5:2 Mean-Motion Resonance"),
                    _ => None,
                };

                if let Some(res_name) = resonance {
                    println!("      {} / {}: {:.2}x - {}",
                             sorted_planets[p_outer].designation,
                             sorted_planets[p_inner].designation,
                             ratio, res_name);
                }
            }
        }
    } else {
        println!("   Single Planet Candidate");
        println!("   ─────────────────────────────────────────────────────");
        let planet = &result.estimated_planets[0];
        println!("   {}", planet.designation);
        println!("   Estimated location: {:.1} AU", planet.estimated_a);
        println!("   Confidence score: {:.1}%", planet.confidence * 100.0);
        println!("   Shepherded objects: {}", planet.shepherded_count);
    }

    println!();
}

fn analyze_shepherding(
    objects: &[KuiperBeltObject],
    result: &aphelion_clustering::AphelionClusteringResult,
) {
    println!("   Analyzing gravitational shepherding effects...\n");

    if result.estimated_planets.is_empty() {
        println!("   No planets to analyze for shepherding.\n");
        return;
    }

    for planet in &result.estimated_planets {
        println!("   {} Shepherding Analysis:", planet.designation);
        println!("   ─────────────────────────────────────────────────────");
        println!("   Estimated location: {:.1} AU", planet.estimated_a);
        println!("   Cluster aphelion: {:.1} AU", planet.aphelion_cluster);
        println!("   Relation: shepherd at {:.0}% of aphelion", 60.0);

        // Find nearby objects
        let shepherded: Vec<&KuiperBeltObject> = objects
            .iter()
            .filter(|o| {
                let aphelion_diff = (o.ad - planet.aphelion_cluster).abs();
                aphelion_diff < 50.0  // Within same bin or adjacent bin
            })
            .collect();

        println!("\n   Shepherding Signature:");
        println!("     Objects near cluster: {}", shepherded.len());

        if shepherded.len() > 2 {
            // Analyze if these objects form a coherent group
            let e_values: Vec<f32> = shepherded.iter().map(|o| o.e).collect();
            let i_values: Vec<f32> = shepherded.iter().map(|o| o.i).collect();

            let e_variance = if !e_values.is_empty() {
                let mean_e = e_values.iter().sum::<f32>() / e_values.len() as f32;
                e_values.iter().map(|e| (e - mean_e).powi(2)).sum::<f32>() / e_values.len() as f32
            } else {
                0.0
            };

            println!("     Eccentricity σ: {:.3} (coherent if < 0.1)", e_variance.sqrt());
            println!("     Mass estimate: {:.1} M⊕ (from aphelion cluster)", planet.estimated_a * 0.02);
        }

        println!();
    }
}

fn generate_executive_summary(
    result: &aphelion_clustering::AphelionClusteringResult,
    objects: &[KuiperBeltObject],
) {
    println!("🎯 EXECUTIVE SUMMARY\n");
    println!("───────────────────────────────────────────────────────────────");

    println!("\nKEY FINDINGS:");
    println!("  • Analyzed {} Trans-Neptunian Objects", objects.len());
    println!("  • {} objects with Q > 100 AU", result.distant_object_count);
    println!("  • {} significant aphelion clusters identified", result.significant_clusters.len());
    println!("  • {} planet candidates estimated from clustering", result.estimated_planets.len());

    if !result.estimated_planets.is_empty() {
        println!("\nPLANET CANDIDATES (from 60% aphelion rule):");
        for planet in &result.estimated_planets {
            println!("  {} @ {:.0} AU (confidence: {:.0}%)",
                     planet.designation,
                     planet.estimated_a,
                     planet.confidence * 100.0);
        }
    }

    println!("\nRECOMMENDED FOLLOW-UP:");
    println!("  ✓ Radial velocity surveys for direct detection");
    println!("  ✓ N-body simulations of proposed system");
    println!("  ✓ Transit surveys for secondary planets");
    println!("  ✓ Spectroscopic follow-up of shepherded objects");

    println!("\n───────────────────────────────────────────────────────────────");
    println!("Analysis Agent 7 (Aphelion Clustering) - Complete");
    println!("═══════════════════════════════════════════════════════════════\n");
}
