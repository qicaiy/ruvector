//! AGI Cognitive Container types (ADR-036).
//!
//! An AGI container is a single RVF file that packages the complete intelligence
//! runtime: micro Linux kernel, Claude Code orchestrator config, Claude Flow
//! swarm manager, RuVector world model, evaluation harness, witness chains,
//! and tool adapters.
//!
//! Wire format: 64-byte `AgiContainerHeader` + TLV manifest sections.
//! The header is stored as a META segment (SegmentType::Meta) in the RVF file,
//! alongside the KERNEL_SEG, WASM_SEG, VEC_SEG, INDEX_SEG, WITNESS_SEG, and
//! CRYPTO_SEG that hold the actual payload data.

/// Magic bytes for AGI container manifest: "RVAG" (RuVector AGI).
pub const AGI_MAGIC: u32 = 0x5256_4147;

/// Size of the AGI container header in bytes.
pub const AGI_HEADER_SIZE: usize = 64;

// --- Flags ---

/// Container includes a KERNEL_SEG with micro Linux kernel.
pub const AGI_HAS_KERNEL: u16 = 1 << 0;
/// Container includes WASM_SEG modules.
pub const AGI_HAS_WASM: u16 = 1 << 1;
/// Container includes Claude Code + Claude Flow orchestrator config.
pub const AGI_HAS_ORCHESTRATOR: u16 = 1 << 2;
/// Container includes VEC_SEG + INDEX_SEG world model data.
pub const AGI_HAS_WORLD_MODEL: u16 = 1 << 3;
/// Container includes evaluation harness (task suite + graders).
pub const AGI_HAS_EVAL: u16 = 1 << 4;
/// Container includes promoted skill library.
pub const AGI_HAS_SKILLS: u16 = 1 << 5;
/// Container includes ADR-035 witness chain.
pub const AGI_HAS_WITNESS: u16 = 1 << 6;
/// Container is cryptographically signed (HMAC-SHA256 or Ed25519).
pub const AGI_SIGNED: u16 = 1 << 7;
/// All tool outputs stored — container supports replay mode.
pub const AGI_REPLAY_CAPABLE: u16 = 1 << 8;
/// Container can run without network (offline-first).
pub const AGI_OFFLINE_CAPABLE: u16 = 1 << 9;
/// Container includes MCP tool adapter registry.
pub const AGI_HAS_TOOLS: u16 = 1 << 10;
/// Container includes coherence gate configuration.
pub const AGI_HAS_COHERENCE_GATES: u16 = 1 << 11;

// --- TLV tags for the manifest payload ---

/// Container UUID.
pub const AGI_TAG_CONTAINER_ID: u16 = 0x0100;
/// Build UUID.
pub const AGI_TAG_BUILD_ID: u16 = 0x0101;
/// Pinned model identifier (UTF-8 string, e.g. "claude-opus-4-6").
pub const AGI_TAG_MODEL_ID: u16 = 0x0102;
/// Serialized governance policy (binary, per ADR-035).
pub const AGI_TAG_POLICY: u16 = 0x0103;
/// Claude Code + Claude Flow orchestrator config (JSON or TOML).
pub const AGI_TAG_ORCHESTRATOR: u16 = 0x0104;
/// MCP tool adapter registry (JSON array of tool schemas).
pub const AGI_TAG_TOOL_REGISTRY: u16 = 0x0105;
/// Agent role prompts (one per agent type).
pub const AGI_TAG_AGENT_PROMPTS: u16 = 0x0106;
/// Evaluation task suite (JSON array of task specs).
pub const AGI_TAG_EVAL_TASKS: u16 = 0x0107;
/// Grading rules (JSON or binary grader config).
pub const AGI_TAG_EVAL_GRADERS: u16 = 0x0108;
/// Promoted skill library (serialized skill nodes).
pub const AGI_TAG_SKILL_LIBRARY: u16 = 0x0109;
/// Replay automation script.
pub const AGI_TAG_REPLAY_SCRIPT: u16 = 0x010A;
/// Kernel boot parameters (command line, initrd config).
pub const AGI_TAG_KERNEL_CONFIG: u16 = 0x010B;
/// Network configuration (ports, endpoints, TLS).
pub const AGI_TAG_NETWORK_CONFIG: u16 = 0x010C;
/// Coherence gate thresholds and rules.
pub const AGI_TAG_COHERENCE_CONFIG: u16 = 0x010D;
/// Claude.md project instructions.
pub const AGI_TAG_PROJECT_INSTRUCTIONS: u16 = 0x010E;
/// Dependency snapshot hashes (pinned repos, packages).
pub const AGI_TAG_DEPENDENCY_SNAPSHOT: u16 = 0x010F;

// --- Execution mode ---

/// Container execution mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum ExecutionMode {
    /// Replay: no external tool calls, use stored receipts.
    /// All graders must match exactly. Witness chain must match.
    Replay = 0,
    /// Verify: live tool calls, outputs stored and hashed.
    /// Outputs must pass same tests. Costs within expected bounds.
    Verify = 1,
    /// Live: full autonomous operation with governance controls.
    Live = 2,
}

impl TryFrom<u8> for ExecutionMode {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Replay),
            1 => Ok(Self::Verify),
            2 => Ok(Self::Live),
            other => Err(other),
        }
    }
}

/// Wire-format AGI container header (exactly 64 bytes, `repr(C)`).
///
/// ```text
/// Offset  Type        Field
/// 0x00    u32         magic (0x52564147 "RVAG")
/// 0x04    u16         version
/// 0x06    u16         flags
/// 0x08    [u8; 16]    container_id (UUID)
/// 0x18    [u8; 16]    build_id (UUID)
/// 0x28    u64         created_ns (UNIX epoch nanoseconds)
/// 0x30    [u8; 8]     model_id_hash (SHA-256 truncated)
/// 0x38    [u8; 8]     policy_hash (SHA-256 truncated)
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct AgiContainerHeader {
    /// Magic bytes: AGI_MAGIC.
    pub magic: u32,
    /// Format version (currently 1).
    pub version: u16,
    /// Bitfield flags indicating which segments are present.
    pub flags: u16,
    /// Unique container identifier (UUID).
    pub container_id: [u8; 16],
    /// Build identifier (UUID, changes on each repackaging).
    pub build_id: [u8; 16],
    /// Creation timestamp (nanoseconds since UNIX epoch).
    pub created_ns: u64,
    /// SHA-256 of the pinned model identifier, truncated to 8 bytes.
    pub model_id_hash: [u8; 8],
    /// SHA-256 of the governance policy, truncated to 8 bytes.
    pub policy_hash: [u8; 8],
}

// Compile-time size assertion.
const _: () = assert!(core::mem::size_of::<AgiContainerHeader>() == 64);

impl AgiContainerHeader {
    /// Check magic bytes.
    pub const fn is_valid_magic(&self) -> bool {
        self.magic == AGI_MAGIC
    }

    /// Check if the container is signed.
    pub const fn is_signed(&self) -> bool {
        self.flags & AGI_SIGNED != 0
    }

    /// Check if the container has a micro Linux kernel.
    pub const fn has_kernel(&self) -> bool {
        self.flags & AGI_HAS_KERNEL != 0
    }

    /// Check if the container has an orchestrator config.
    pub const fn has_orchestrator(&self) -> bool {
        self.flags & AGI_HAS_ORCHESTRATOR != 0
    }

    /// Check if the container supports replay mode.
    pub const fn is_replay_capable(&self) -> bool {
        self.flags & AGI_REPLAY_CAPABLE != 0
    }

    /// Check if the container can run offline.
    pub const fn is_offline_capable(&self) -> bool {
        self.flags & AGI_OFFLINE_CAPABLE != 0
    }

    /// Serialize header to a 64-byte array.
    pub fn to_bytes(&self) -> [u8; AGI_HEADER_SIZE] {
        let mut buf = [0u8; AGI_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..24].copy_from_slice(&self.container_id);
        buf[24..40].copy_from_slice(&self.build_id);
        buf[40..48].copy_from_slice(&self.created_ns.to_le_bytes());
        buf[48..56].copy_from_slice(&self.model_id_hash);
        buf[56..64].copy_from_slice(&self.policy_hash);
        buf
    }

    /// Deserialize header from a byte slice (>= 64 bytes).
    pub fn from_bytes(data: &[u8]) -> Result<Self, crate::RvfError> {
        if data.len() < AGI_HEADER_SIZE {
            return Err(crate::RvfError::SizeMismatch {
                expected: AGI_HEADER_SIZE,
                got: data.len(),
            });
        }
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != AGI_MAGIC {
            return Err(crate::RvfError::BadMagic {
                expected: AGI_MAGIC,
                got: magic,
            });
        }
        let mut container_id = [0u8; 16];
        container_id.copy_from_slice(&data[8..24]);
        let mut build_id = [0u8; 16];
        build_id.copy_from_slice(&data[24..40]);
        let mut model_id_hash = [0u8; 8];
        model_id_hash.copy_from_slice(&data[48..56]);
        let mut policy_hash = [0u8; 8];
        policy_hash.copy_from_slice(&data[56..64]);

        Ok(Self {
            magic,
            version: u16::from_le_bytes([data[4], data[5]]),
            flags: u16::from_le_bytes([data[6], data[7]]),
            container_id,
            build_id,
            created_ns: u64::from_le_bytes([
                data[40], data[41], data[42], data[43],
                data[44], data[45], data[46], data[47],
            ]),
            model_id_hash,
            policy_hash,
        })
    }
}

/// Required segments for a valid AGI container.
///
/// Used by the container builder/validator to ensure completeness.
#[derive(Clone, Debug, Default)]
pub struct ContainerSegments {
    /// KERNEL_SEG: micro Linux kernel (e.g. Firecracker-compatible vmlinux).
    pub kernel_present: bool,
    /// KERNEL_SEG size in bytes.
    pub kernel_size: u64,
    /// WASM_SEG: interpreter + microkernel modules.
    pub wasm_count: u16,
    /// Total WASM_SEG size in bytes.
    pub wasm_total_size: u64,
    /// VEC_SEG: world model vector count.
    pub vec_segment_count: u16,
    /// INDEX_SEG: HNSW index count.
    pub index_segment_count: u16,
    /// WITNESS_SEG: witness bundle count.
    pub witness_count: u32,
    /// CRYPTO_SEG: present.
    pub crypto_present: bool,
    /// META segment with AGI manifest: present.
    pub manifest_present: bool,
    /// Total container size in bytes.
    pub total_size: u64,
}

impl ContainerSegments {
    /// Validate that the container has all required segments for a given
    /// execution mode.
    pub fn validate(&self, mode: ExecutionMode) -> Result<(), ContainerError> {
        // All modes require the manifest.
        if !self.manifest_present {
            return Err(ContainerError::MissingSegment("AGI manifest"));
        }

        match mode {
            ExecutionMode::Replay => {
                // Replay needs witness chains.
                if self.witness_count == 0 {
                    return Err(ContainerError::MissingSegment("witness chain"));
                }
            }
            ExecutionMode::Verify | ExecutionMode::Live => {
                // Verify/Live need at least kernel or WASM.
                if !self.kernel_present && self.wasm_count == 0 {
                    return Err(ContainerError::MissingSegment(
                        "kernel or WASM runtime",
                    ));
                }
            }
        }

        Ok(())
    }

    /// Compute the flags bitfield from present segments.
    pub fn to_flags(&self) -> u16 {
        let mut flags: u16 = 0;
        if self.kernel_present {
            flags |= AGI_HAS_KERNEL;
        }
        if self.wasm_count > 0 {
            flags |= AGI_HAS_WASM;
        }
        if self.witness_count > 0 {
            flags |= AGI_HAS_WITNESS;
        }
        if self.crypto_present {
            flags |= AGI_SIGNED;
        }
        flags
    }
}

/// Error type for AGI container operations.
#[derive(Debug, PartialEq, Eq)]
pub enum ContainerError {
    /// A required segment is missing.
    MissingSegment(&'static str),
    /// Container exceeds size limit.
    TooLarge { size: u64 },
    /// Invalid segment configuration.
    InvalidConfig(&'static str),
    /// Signature verification failed.
    SignatureInvalid,
}

impl core::fmt::Display for ContainerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ContainerError::MissingSegment(s) => write!(f, "missing segment: {s}"),
            ContainerError::TooLarge { size } => write!(f, "container too large: {size} bytes"),
            ContainerError::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            ContainerError::SignatureInvalid => write!(f, "signature verification failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agi_header_size() {
        assert_eq!(core::mem::size_of::<AgiContainerHeader>(), 64);
    }

    #[test]
    fn agi_header_round_trip() {
        let hdr = AgiContainerHeader {
            magic: AGI_MAGIC,
            version: 1,
            flags: AGI_HAS_KERNEL | AGI_HAS_ORCHESTRATOR | AGI_HAS_WORLD_MODEL
                | AGI_HAS_EVAL | AGI_SIGNED | AGI_REPLAY_CAPABLE,
            container_id: [0x42; 16],
            build_id: [0x43; 16],
            created_ns: 1_700_000_000_000_000_000,
            model_id_hash: [0xAA; 8],
            policy_hash: [0xBB; 8],
        };
        let bytes = hdr.to_bytes();
        assert_eq!(bytes.len(), AGI_HEADER_SIZE);
        let decoded = AgiContainerHeader::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn agi_header_bad_magic() {
        let mut bytes = [0u8; 64];
        bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        assert!(AgiContainerHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn agi_header_too_short() {
        assert!(AgiContainerHeader::from_bytes(&[0u8; 32]).is_err());
    }

    #[test]
    fn agi_flags() {
        let hdr = AgiContainerHeader {
            magic: AGI_MAGIC,
            version: 1,
            flags: AGI_HAS_KERNEL | AGI_HAS_ORCHESTRATOR | AGI_SIGNED,
            container_id: [0; 16],
            build_id: [0; 16],
            created_ns: 0,
            model_id_hash: [0; 8],
            policy_hash: [0; 8],
        };
        assert!(hdr.has_kernel());
        assert!(hdr.has_orchestrator());
        assert!(hdr.is_signed());
        assert!(!hdr.is_replay_capable());
        assert!(!hdr.is_offline_capable());
    }

    #[test]
    fn execution_mode_round_trip() {
        for raw in 0..=2u8 {
            let m = ExecutionMode::try_from(raw).unwrap();
            assert_eq!(m as u8, raw);
        }
        assert!(ExecutionMode::try_from(3).is_err());
    }

    #[test]
    fn segments_validate_replay_needs_witness() {
        let segs = ContainerSegments {
            manifest_present: true,
            witness_count: 0,
            ..Default::default()
        };
        assert_eq!(
            segs.validate(ExecutionMode::Replay),
            Err(ContainerError::MissingSegment("witness chain"))
        );
    }

    #[test]
    fn segments_validate_live_needs_runtime() {
        let segs = ContainerSegments {
            manifest_present: true,
            kernel_present: false,
            wasm_count: 0,
            ..Default::default()
        };
        assert_eq!(
            segs.validate(ExecutionMode::Live),
            Err(ContainerError::MissingSegment("kernel or WASM runtime"))
        );
    }

    #[test]
    fn segments_validate_live_with_kernel() {
        let segs = ContainerSegments {
            manifest_present: true,
            kernel_present: true,
            ..Default::default()
        };
        assert!(segs.validate(ExecutionMode::Live).is_ok());
    }

    #[test]
    fn segments_validate_live_with_wasm() {
        let segs = ContainerSegments {
            manifest_present: true,
            wasm_count: 2,
            ..Default::default()
        };
        assert!(segs.validate(ExecutionMode::Live).is_ok());
    }

    #[test]
    fn segments_validate_replay_with_witness() {
        let segs = ContainerSegments {
            manifest_present: true,
            witness_count: 10,
            ..Default::default()
        };
        assert!(segs.validate(ExecutionMode::Replay).is_ok());
    }

    #[test]
    fn segments_to_flags() {
        let segs = ContainerSegments {
            kernel_present: true,
            wasm_count: 1,
            witness_count: 5,
            crypto_present: true,
            ..Default::default()
        };
        let flags = segs.to_flags();
        assert_ne!(flags & AGI_HAS_KERNEL, 0);
        assert_ne!(flags & AGI_HAS_WASM, 0);
        assert_ne!(flags & AGI_HAS_WITNESS, 0);
        assert_ne!(flags & AGI_SIGNED, 0);
    }

    #[test]
    fn container_error_display() {
        let e = ContainerError::MissingSegment("kernel");
        assert!(format!("{e}").contains("kernel"));
        let e2 = ContainerError::TooLarge { size: 999 };
        assert!(format!("{e2}").contains("999"));
    }
}
