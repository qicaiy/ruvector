use clap::{Parser, Subcommand};
use std::process;

mod cmd;
mod output;

#[derive(Parser)]
#[command(name = "rvf", version, about = "RuVector Format CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new empty RVF store
    Create(cmd::create::CreateArgs),
    /// Ingest vectors from a JSON file
    Ingest(cmd::ingest::IngestArgs),
    /// Query nearest neighbors
    Query(cmd::query::QueryArgs),
    /// Delete vectors by ID or filter
    Delete(cmd::delete::DeleteArgs),
    /// Show store status
    Status(cmd::status::StatusArgs),
    /// Inspect segments and lineage
    Inspect(cmd::inspect::InspectArgs),
    /// Compact to reclaim dead space
    Compact(cmd::compact::CompactArgs),
    /// Derive a child store from a parent
    Derive(cmd::derive::DeriveArgs),
    /// Start HTTP server (requires 'serve' feature)
    Serve(cmd::serve::ServeArgs),
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Create(args) => cmd::create::run(args),
        Commands::Ingest(args) => cmd::ingest::run(args),
        Commands::Query(args) => cmd::query::run(args),
        Commands::Delete(args) => cmd::delete::run(args),
        Commands::Status(args) => cmd::status::run(args),
        Commands::Inspect(args) => cmd::inspect::run(args),
        Commands::Compact(args) => cmd::compact::run(args),
        Commands::Derive(args) => cmd::derive::run(args),
        Commands::Serve(args) => cmd::serve::run(args),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
