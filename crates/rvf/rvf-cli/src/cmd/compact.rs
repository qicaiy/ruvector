//! `rvf compact` -- Compact store to reclaim dead space.

use clap::Args;
use std::path::Path;

use rvf_runtime::RvfStore;

use super::map_rvf_err;

#[derive(Args)]
pub struct CompactArgs {
    /// Path to the RVF store
    path: String,
    /// Output as JSON
    #[arg(long)]
    json: bool,
}

pub fn run(args: CompactArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = RvfStore::open(Path::new(&args.path)).map_err(map_rvf_err)?;

    let status_before = store.status();
    let result = store.compact().map_err(map_rvf_err)?;
    let status_after = store.status();

    store.close().map_err(map_rvf_err)?;

    if args.json {
        crate::output::print_json(&serde_json::json!({
            "segments_compacted": result.segments_compacted,
            "bytes_reclaimed": result.bytes_reclaimed,
            "epoch": result.epoch,
            "vectors_before": status_before.total_vectors,
            "vectors_after": status_after.total_vectors,
            "file_size_before": status_before.file_size,
            "file_size_after": status_after.file_size,
        }));
    } else {
        println!("Compaction complete:");
        crate::output::print_kv("Segments compacted:", &result.segments_compacted.to_string());
        crate::output::print_kv("Bytes reclaimed:", &result.bytes_reclaimed.to_string());
        crate::output::print_kv("Epoch:", &result.epoch.to_string());
        crate::output::print_kv("Vectors before:", &status_before.total_vectors.to_string());
        crate::output::print_kv("Vectors after:", &status_after.total_vectors.to_string());
        crate::output::print_kv(
            "File size before:",
            &format!("{} bytes", status_before.file_size),
        );
        crate::output::print_kv(
            "File size after:",
            &format!("{} bytes", status_after.file_size),
        );
    }
    Ok(())
}
