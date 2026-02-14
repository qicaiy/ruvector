//! Core types for the RuVector Format (RVF).
//!
//! This crate provides the foundational types shared across all RVF crates:
//! segment headers, type enums, flags, error codes, and format constants.
//!
//! All types are `no_std` compatible by default.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(test)]
extern crate alloc;

pub mod checksum;
pub mod compression;
pub mod constants;
pub mod data_type;
pub mod ebpf;
pub mod error;
pub mod filter;
pub mod flags;
pub mod kernel;
pub mod manifest;
pub mod profile;
pub mod quant_type;
pub mod segment;
pub mod segment_type;
pub mod signature;
pub mod attestation;
pub mod lineage;

pub use attestation::{AttestationHeader, AttestationWitnessType, TeePlatform, KEY_TYPE_TEE_BOUND};
pub use ebpf::{
    EbpfAttachType, EbpfHeader, EbpfProgramType, EBPF_MAGIC,
};
pub use kernel::{
    ApiTransport, KernelArch, KernelHeader, KernelType, KERNEL_MAGIC,
    KERNEL_FLAG_SIGNED, KERNEL_FLAG_COMPRESSED, KERNEL_FLAG_REQUIRES_TEE,
    KERNEL_FLAG_MEASURED, KERNEL_FLAG_REQUIRES_KVM, KERNEL_FLAG_REQUIRES_UEFI,
    KERNEL_FLAG_HAS_NETWORKING, KERNEL_FLAG_HAS_QUERY_API, KERNEL_FLAG_HAS_INGEST_API,
    KERNEL_FLAG_HAS_ADMIN_API, KERNEL_FLAG_ATTESTATION_READY, KERNEL_FLAG_RELOCATABLE,
    KERNEL_FLAG_HAS_VIRTIO_NET, KERNEL_FLAG_HAS_VIRTIO_BLK, KERNEL_FLAG_HAS_VSOCK,
};
pub use lineage::{
    DerivationType, FileIdentity, LineageRecord, LINEAGE_RECORD_SIZE,
    WITNESS_DERIVATION, WITNESS_LINEAGE_MERGE, WITNESS_LINEAGE_SNAPSHOT,
    WITNESS_LINEAGE_TRANSFORM, WITNESS_LINEAGE_VERIFY,
};
pub use checksum::ChecksumAlgo;
pub use compression::CompressionAlgo;
pub use constants::*;
pub use data_type::DataType;
pub use error::{ErrorCode, RvfError};
pub use filter::FilterOp;
pub use flags::SegmentFlags;
pub use manifest::{
    CentroidPtr, EntrypointPtr, HotCachePtr, Level0Root, PrefetchMapPtr, QuantDictPtr, TopLayerPtr,
};
pub use profile::{DomainProfile, ProfileId};
pub use quant_type::QuantType;
pub use segment::SegmentHeader;
pub use segment_type::SegmentType;
pub use signature::{SignatureAlgo, SignatureFooter};
