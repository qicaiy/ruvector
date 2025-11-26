//! # Argument of Perihelion Clustering Analysis
//!
//! Analyzes clustering patterns in argument of perihelion (ω) for high-perihelion
//! Kuiper Belt Objects (q > 37 AU, a > 50 AU).
//!
//! ## Objective
//! - Detect Kozai-Lidov mechanism signatures
//! - Identify potential planet perturbation evidence
//! - Report statistical significance of clustering patterns
//!
//! Run with:
//! ```bash
//! cargo run --example perihelion_clustering
//! ```

mod kuiper_belt;
use kuiper_belt::perihelion_analysis::{analyze_argument_of_perihelion, generate_report};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  KUIPER BELT ANALYSIS: ARGUMENT OF PERIHELION CLUSTERING    ║");
    println!("║              Analysis Agent 2: Argument of Perihelion         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("🔬 Initializing analysis...\n");

    // Run the perihelion clustering analysis
    let analysis = analyze_argument_of_perihelion();

    // Generate and print the report
    let report = generate_report(&analysis);
    println!("{}", report);

    // Additional summary
    println!("\n📌 KEY FINDINGS\n");
    println!("───────────────────────────────────────────────────────────────");

    if analysis.objects.is_empty() {
        println!("\n⚠️  No high-q objects found in the dataset (q > 37 AU, a > 50 AU)");
    } else {
        println!("\n✓ Analysis completed for {} high-q objects\n", analysis.objects.len());

        // Summarize findings
        println!("1. CLUSTERING PATTERN:");
        let cluster_ratio = (analysis.cluster_0.len() + analysis.cluster_180.len()) as f32 / analysis.objects.len() as f32;
        if cluster_ratio > 0.6 {
            println!("   Strong clustering around ω = 0° or 180°");
        } else if cluster_ratio > 0.4 {
            println!("   Moderate clustering around ω = 0° or 180°");
        } else {
            println!("   Weak or no clustering - random distribution");
        }

        println!("\n2. KOZAI RESONANCE SIGNATURE:");
        if analysis.kozai_score > 0.6 {
            println!("   ⭐ STRONG evidence of Kozai-Lidov mechanism");
        } else if analysis.kozai_score > 0.4 {
            println!("   ✓ MODERATE evidence of Kozai-Lidov mechanism");
        } else {
            println!("   ✗ WEAK or no evidence of Kozai-Lidov mechanism");
        }

        println!("\n3. PLANET PERTURBATION EVIDENCE:");
        if analysis.planet_perturbation_evidence {
            println!("   ✓ YES - Clustering suggests an external perturber");

            if analysis.cluster_0.len() > analysis.cluster_180.len() {
                println!("   Dominant 0° cluster suggests low-inclination perturber");
            } else if analysis.cluster_180.len() > analysis.cluster_0.len() {
                println!("   Dominant 180° cluster suggests high-inclination perturber");
            }
        } else {
            println!("   ✗ NO - No strong evidence for planet perturbation from ω clustering");
        }

        println!("\n4. STATISTICAL METRICS:");
        println!("   Mean ω: {:.1}° (circular mean)", analysis.mean_w);
        println!("   Std dev: {:.1}° (circular dispersion)", analysis.std_dev_w);
        println!("   Range: {:.1}° - {:.1}°", analysis.min_w, analysis.max_w);
        println!("   Kozai Score: {:.3}", analysis.kozai_score);

        println!("\n5. CLUSTER COMPOSITION:");
        println!("   Cluster 0° (aligned):     {} objects ({:.1}%)",
            analysis.cluster_0.len(),
            (analysis.cluster_0.len() as f32 / analysis.objects.len() as f32) * 100.0);
        println!("   Cluster 180° (anti-aligned): {} objects ({:.1}%)",
            analysis.cluster_180.len(),
            (analysis.cluster_180.len() as f32 / analysis.objects.len() as f32) * 100.0);
        println!("   Scattered: {} objects ({:.1}%)",
            analysis.scattered.len(),
            (analysis.scattered.len() as f32 / analysis.objects.len() as f32) * 100.0);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Full report generated above. See complete object list and details.");
    println!("═══════════════════════════════════════════════════════════════\n");
}
