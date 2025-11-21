//! Example demonstrating Prometheus metrics collection with Tiny Dancer
//!
//! This example shows how to:
//! - Collect routing metrics
//! - Export metrics in Prometheus format
//! - Monitor circuit breaker state
//! - Track routing decisions and latencies
//!
//! Run with: cargo run --example metrics_example

use ruvector_tiny_dancer_core::{Candidate, Router, RouterConfig, RoutingRequest};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tiny Dancer Metrics Example ===\n");

    // Create router with metrics enabled
    let config = RouterConfig {
        model_path: "./models/fastgrnn.safetensors".to_string(),
        confidence_threshold: 0.85,
        max_uncertainty: 0.15,
        enable_circuit_breaker: true,
        circuit_breaker_threshold: 5,
        ..Default::default()
    };

    let router = Router::new(config)?;

    // Process multiple routing requests
    println!("Processing routing requests...\n");

    for i in 0..10 {
        let candidates = vec![
            Candidate {
                id: format!("candidate-{}-1", i),
                embedding: vec![0.5 + (i as f32 * 0.01); 384],
                metadata: HashMap::new(),
                created_at: chrono::Utc::now().timestamp(),
                access_count: 10 + i as u64,
                success_rate: 0.95 - (i as f32 * 0.01),
            },
            Candidate {
                id: format!("candidate-{}-2", i),
                embedding: vec![0.3 + (i as f32 * 0.01); 384],
                metadata: HashMap::new(),
                created_at: chrono::Utc::now().timestamp(),
                access_count: 5 + i as u64,
                success_rate: 0.85 - (i as f32 * 0.01),
            },
            Candidate {
                id: format!("candidate-{}-3", i),
                embedding: vec![0.7 + (i as f32 * 0.01); 384],
                metadata: HashMap::new(),
                created_at: chrono::Utc::now().timestamp(),
                access_count: 15 + i as u64,
                success_rate: 0.98 - (i as f32 * 0.01),
            },
        ];

        let request = RoutingRequest {
            query_embedding: vec![0.5; 384],
            candidates,
            metadata: None,
        };

        match router.route(request) {
            Ok(response) => {
                println!(
                    "Request {}: Processed {} candidates in {}μs",
                    i + 1,
                    response.candidates_processed,
                    response.inference_time_us
                );
                println!("  Top decision: {:?}", response.decisions.first());
            }
            Err(e) => {
                eprintln!("Error processing request {}: {}", i + 1, e);
            }
        }
    }

    // Export metrics
    println!("\n=== Prometheus Metrics ===\n");
    let metrics = router.export_metrics()?;
    println!("{}", metrics);

    // Parse and display key metrics
    println!("\n=== Key Metrics Summary ===\n");

    for line in metrics.lines() {
        if line.starts_with("tiny_dancer_routing_requests_total") {
            println!("{}", line);
        } else if line.starts_with("tiny_dancer_routing_decisions_total") {
            println!("{}", line);
        } else if line.starts_with("tiny_dancer_circuit_breaker_state") {
            println!("{}", line);
        } else if line.starts_with("tiny_dancer_candidates_processed_total") {
            println!("{}", line);
        }
    }

    println!("\n=== Metrics Collection Complete ===");
    println!("\nTo visualize these metrics:");
    println!("1. Set up a Prometheus server");
    println!("2. Configure scraping from your application");
    println!("3. Use Grafana to create dashboards");
    println!("\nExample Prometheus configuration:");
    println!("  scrape_configs:");
    println!("    - job_name: 'tiny-dancer'");
    println!("      static_configs:");
    println!("        - targets: ['localhost:9090']");

    Ok(())
}
