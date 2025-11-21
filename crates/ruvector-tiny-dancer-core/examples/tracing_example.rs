//! Example demonstrating distributed tracing with OpenTelemetry and Jaeger
//!
//! This example shows how to:
//! - Initialize OpenTelemetry tracing
//! - Create spans for routing operations
//! - Propagate trace context
//! - Export traces to Jaeger
//!
//! Prerequisites:
//! - Run Jaeger: docker run -d -p6831:6831/udp -p16686:16686 jaegertracing/all-in-one:latest
//!
//! Run with: cargo run --example tracing_example

use ruvector_tiny_dancer_core::{
    Candidate, Router, RouterConfig, RoutingRequest, TraceContext, TracingConfig, TracingSystem,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tiny Dancer Distributed Tracing Example ===\n");

    // Initialize tracing with stdout exporter (for demonstration)
    // In production, use Jaeger endpoint
    let tracing_config = TracingConfig {
        service_name: "tiny-dancer-example".to_string(),
        service_version: "1.0.0".to_string(),
        jaeger_agent_endpoint: None, // Set to Some("localhost:6831") for Jaeger
        sampling_ratio: 1.0,
        enable_stdout: true, // Set to false when using Jaeger
    };

    let tracing_system = TracingSystem::new(tracing_config);
    tracing_system.init()?;

    println!("Tracing initialized (stdout mode for demonstration)\n");
    println!("To use Jaeger:");
    println!("1. Start Jaeger: docker run -d -p6831:6831/udp -p16686:16686 jaegertracing/all-in-one:latest");
    println!("2. Set jaeger_agent_endpoint to Some(\"localhost:6831\")");
    println!("3. Set enable_stdout to false");
    println!("4. Visit http://localhost:16686 to view traces\n");

    // Create router
    let config = RouterConfig {
        model_path: "./models/fastgrnn.safetensors".to_string(),
        confidence_threshold: 0.85,
        max_uncertainty: 0.15,
        enable_circuit_breaker: true,
        circuit_breaker_threshold: 5,
        ..Default::default()
    };

    let router = Router::new(config)?;

    // Process requests with tracing
    println!("Processing requests with distributed tracing...\n");

    for i in 0..3 {
        println!("Request {} - Processing", i + 1);

        // Get trace context for propagation (requires OpenTelemetry to be initialized)
        if let Some(trace_ctx) = TraceContext::from_current() {
            println!("  Trace ID: {}", trace_ctx.trace_id);
            println!("  Span ID: {}", trace_ctx.span_id);
            println!("  W3C Traceparent: {}", trace_ctx.to_w3c_traceparent());
        }

        // Create candidates
        let candidates = vec![
            Candidate {
                id: format!("candidate-{}-1", i),
                embedding: vec![0.5; 384],
                metadata: HashMap::new(),
                created_at: chrono::Utc::now().timestamp(),
                access_count: 10,
                success_rate: 0.95,
            },
            Candidate {
                id: format!("candidate-{}-2", i),
                embedding: vec![0.3; 384],
                metadata: HashMap::new(),
                created_at: chrono::Utc::now().timestamp(),
                access_count: 5,
                success_rate: 0.85,
            },
        ];

        let request = RoutingRequest {
            query_embedding: vec![0.5; 384],
            candidates: candidates.clone(),
            metadata: None,
        };

        // Route with automatic span creation
        match router.route(request) {
            Ok(response) => {
                println!(
                    "\nRequest {}: Processed {} candidates in {}μs",
                    i + 1,
                    response.candidates_processed,
                    response.inference_time_us
                );

                for decision in response.decisions.iter().take(2) {
                    println!(
                        "  - {} (confidence: {:.2}, lightweight: {})",
                        decision.candidate_id, decision.confidence, decision.use_lightweight
                    );
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }

        println!();
    }

    // Shutdown tracing to flush remaining spans
    println!("\n=== Flushing traces ===");
    tracing_system.shutdown();

    println!("\n=== Tracing Example Complete ===");
    println!("\nSpans created during execution:");
    println!("- routing_request (Router::route)");
    println!("- circuit_breaker_check");
    println!("- feature_engineering");
    println!("- model_inference (per candidate)");
    println!("- uncertainty_estimation (per candidate)");

    Ok(())
}
