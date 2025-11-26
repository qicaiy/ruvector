/// Test script to run and display eccentricity pumping analysis
/// Focuses on identifying distant objects with high eccentricity

use std::path::Path;
use std::fs;

fn main() {
    // Import the modules
    use ruvector_core::examples::kuiper_belt::{
        analyze_eccentricity_pumping,
        get_analysis_summary,
    };

    println!("Starting Eccentricity Pumping Analysis...\n");

    // Run the analysis
    let analysis = analyze_eccentricity_pumping();

    // Get summary
    let summary = get_analysis_summary(&analysis);

    // Print summary
    println!("{}", summary);

    // Save results to file
    let output_path = "/tmp/eccentricity_analysis_results.txt";
    match fs::write(output_path, summary) {
        Ok(_) => println!("Results saved to: {}", output_path),
        Err(e) => eprintln!("Error saving results: {}", e),
    }

    // Additional JSON output for data processing
    match serde_json::to_string_pretty(&analysis) {
        Ok(json) => {
            let json_path = "/tmp/eccentricity_analysis_results.json";
            match fs::write(json_path, json) {
                Ok(_) => println!("JSON results saved to: {}", json_path),
                Err(e) => eprintln!("Error saving JSON: {}", e),
            }
        }
        Err(e) => eprintln!("Error serializing JSON: {}", e),
    }
}
