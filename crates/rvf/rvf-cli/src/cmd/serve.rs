//! `rvf serve` -- Start HTTP server (stub, requires 'serve' feature).

use clap::Args;

#[derive(Args)]
pub struct ServeArgs {
    /// Path to the RVF store
    #[allow(dead_code)]
    path: String,
    /// Server port
    #[arg(short, long, default_value = "8080")]
    #[allow(dead_code)]
    port: u16,
}

pub fn run(_args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "serve")]
    {
        eprintln!("HTTP server not yet implemented in CLI");
        return Ok(());
    }
    #[cfg(not(feature = "serve"))]
    {
        eprintln!(
            "The 'serve' feature is not enabled. Rebuild with: cargo build -p rvf-cli --features serve"
        );
        Ok(())
    }
}
