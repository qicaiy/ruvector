// Direct eccentricity analysis without dependencies on external modules
// This file contains a standalone analysis that can be compiled and run

mod kbo_data;
mod eccentricity_analysis;

use kbo_data::get_kbo_data;
use eccentricity_analysis::analyze_eccentricity_pumping;
use eccentricity_analysis::get_analysis_summary;

fn main() {
    // Run the analysis
    let analysis = analyze_eccentricity_pumping();

    // Display results
    let summary = get_analysis_summary(&analysis);
    println!("{}", summary);

    // Print JSON for processing
    println!("\n\n═══════════════════════════════════════════════════════════════");
    println!("RAW DATA (JSON FORMAT):");
    println!("═══════════════════════════════════════════════════════════════\n");

    match serde_json::to_string_pretty(&analysis) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error serializing: {}", e),
    }
}
