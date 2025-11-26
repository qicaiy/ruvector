//! # Executable for Longitude of Ascending Node Analysis
//!
//! Analysis Agent 3: Runs comprehensive clustering analysis

mod longitude_node_analysis;
use longitude_node_analysis::{LongitudeNodeAnalyzer, get_distant_kbo_data};

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     ANALYSIS AGENT 3: LONGITUDE OF ASCENDING NODE (Ω)       ║");
    println!("║          Clustering Analysis for Distant Objects            ║");
    println!("║              (Semi-major axis a > 100 AU)                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load data
    let objects = get_distant_kbo_data();

    // Filter to show data source
    let distant: Vec<_> = objects.iter().filter(|o| o.a > 100.0).collect();
    println!("📥 Loaded {} objects from NASA/JPL database", objects.len());
    println!("✓ Filtered to {} objects with a > 100 AU\n", distant.len());

    // Run analysis
    println!("🔍 Running comprehensive circular statistics analysis...\n");
    let analysis = LongitudeNodeAnalyzer::analyze(&objects);

    // Generate and display report
    let report = LongitudeNodeAnalyzer::generate_report(&analysis);
    println!("{}", report);

    // Additional detailed statistics
    println!("═══════════════════════════════════════════════════════════════");
    println!("                  DETAILED STATISTICS                          ");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Calculate some additional metrics
    let omegas: Vec<f64> = analysis.distant_objects.iter().map(|o| o.omega).collect();

    // Min/max longitude
    let min_omega = omegas.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_omega = omegas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Longitude Range:");
    println!("  Minimum Ω: {:.2}°", min_omega);
    println!("  Maximum Ω: {:.2}°", max_omega);
    println!("  Span: {:.2}°\n", max_omega - min_omega);

    // Quartile analysis
    let mut sorted = omegas.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1_idx = sorted.len() / 4;
    let q2_idx = sorted.len() / 2;
    let q3_idx = 3 * sorted.len() / 4;

    println!("Quartile Distribution:");
    println!("  Q1 (25%): {:.2}°", sorted[q1_idx]);
    println!("  Q2 (50%): {:.2}°", sorted[q2_idx]);
    println!("  Q3 (75%): {:.2}°", sorted[q3_idx]);
    println!("  IQR: {:.2}°\n", sorted[q3_idx] - sorted[q1_idx]);

    // Standard deviation (linear - for comparison)
    let mean = omegas.iter().sum::<f64>() / omegas.len() as f64;
    let variance = omegas.iter()
        .map(|o| (o - mean).powi(2))
        .sum::<f64>() / omegas.len() as f64;
    let std_dev = variance.sqrt();

    println!("Linear Statistics (for reference):");
    println!("  Mean: {:.2}°", mean);
    println!("  Linear Std Dev: {:.2}°", std_dev);
    println!("  Variance: {:.2}°²\n", variance);

    // Interpretation guide
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    INTERPRETATION GUIDE                       ");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Mean Resultant Length (R):");
    println!("  R = 0.0  → Perfectly random distribution");
    println!("  R < 0.3  → Likely random distribution");
    println!("  R = 0.3-0.5 → Possible weak clustering");
    println!("  R = 0.5-0.7 → Moderate clustering (significant)");
    println!("  R > 0.7  → Strong clustering (highly significant)\n");

    println!("Circular Variance:");
    println!("  CV = 1 - R  (complementary to R)");
    println!("  Lower values indicate tighter clustering\n");

    println!("Significance:");
    println!("  Rayleigh test determines if clustering is statistically");
    println!("  different from a random uniform distribution.\n");

    // Planet longitude implications
    println!("═══════════════════════════════════════════════════════════════");
    println!("              PLANETARY PERTURBATION IMPLICATIONS               ");
    println!("═══════════════════════════════════════════════════════════════\n");

    if let Some(estimate) = &analysis.estimated_planet_longitude {
        println!("If a distant planet exists:");
        println!();
        println!("1. CURRENT EVIDENCE:");
        match estimate.evidence_strength.as_str() {
            "Strong" => {
                println!("   ✓ Strong clustering detected");
                println!("   ✓ Multiple independent methods confirm signal");
                println!("   ✓ Statistical significance is high (p < 0.01)");
                println!("   → Further investigation strongly recommended");
            }
            "Moderate" => {
                println!("   ◐ Moderate clustering detected");
                println!("   ◐ Signal above random but not overwhelming");
                println!("   ◐ Additional data would help confirm");
                println!("   → Continue monitoring and analysis");
            }
            _ => {
                println!("   ○ Weak or no significant clustering");
                println!("   ○ Results consistent with random distribution");
                println!("   ○ No strong evidence for planet perturbation");
                println!("   → More data needed for confirmation");
            }
        }
        println!();

        println!("2. LONGITUDE ESTIMATES:");
        println!("   Primary estimate (planet longitude):  {:.1}°", estimate.primary_longitude);
        println!("   Anti-aligned direction (180° offset): {:.1}°",
            (estimate.primary_longitude + 180.0) % 360.0);
        println!();

        println!("3. WHAT THIS MEANS:");
        println!("   • Objects showing Ω clustering suggest perturbation");
        println!("   • A distant massive body affects orbital elements");
        println!("   • Clustering in Ω indicates orbital plane alignment");
        println!("   • Planet Nine candidate region: 400-800 AU");
        println!();

        println!("4. NEXT STEPS:");
        println!("   □ Cross-reference with other orbital elements");
        println!("   □ Check for clustering in argument of perihelion (ω)");
        println!("   □ Analyze longitude of perihelion (ϖ = ω + Ω)");
        println!("   □ Examine Tisserand parameters");
        println!("   □ Perform dynamical simulations");
        println!("   □ Search for additional extreme TNOs\n");
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                    REFERENCE INFORMATION                      ");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Orbital Element Definitions:");
    println!("  a   = Semi-major axis (orbital size)");
    println!("  e   = Eccentricity (orbital shape)");
    println!("  i   = Inclination (orbital tilt)");
    println!("  Ω   = Longitude of ascending node (orbital pole direction)");
    println!("  ω   = Argument of perihelion (perihelion location)");
    println!("  ϖ   = Longitude of perihelion (ω + Ω)\n");

    println!("Data Source:");
    println!("  NASA/JPL Small-Body Database");
    println!("  https://ssd-api.jpl.nasa.gov/sbdb_query.api\n");

    println!("References:");
    println!("  • Batygin & Brown (2016): Planet Nine evidence");
    println!("  • Brown et al. (2004+): TNO orbital characteristics");
    println!("  • Mardia & Jupp (1999): Circular statistics\n");

    println!("═══════════════════════════════════════════════════════════════");
    println!("                  Analysis Complete ✓");
    println!("═══════════════════════════════════════════════════════════════\n");
}
