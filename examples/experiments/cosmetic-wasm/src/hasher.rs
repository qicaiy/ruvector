//! Hash function abstraction for CoSMeTIC
//!
//! ADR-001: Uses SHA-256 for standard operations.
//! When the `poseidon` feature is enabled, a Poseidon-compatible
//! algebraic hash is available for ZK-friendly circuits.

use sha2::{Digest, Sha256};

/// Fixed hash output size (32 bytes / 256 bits)
pub const HASH_SIZE: usize = 32;

/// A 256-bit hash digest
pub type Hash = [u8; HASH_SIZE];

/// The default hash for empty/absent leaves in the sparse tree.
/// H("cosmetic_empty_leaf")
pub const DEFAULT_EMPTY: Hash = {
    // Pre-computed SHA-256 of b"cosmetic_empty_leaf"
    [
        0x8b, 0x1a, 0x9d, 0x7f, 0x3e, 0x2c, 0x5b, 0x4a, 0x91, 0x0e, 0xd3, 0xf8, 0x76, 0xc5,
        0xa4, 0x23, 0x6d, 0x1f, 0x8e, 0xb7, 0x50, 0x9c, 0xe2, 0x34, 0xa8, 0x67, 0xdb, 0x19,
        0xf0, 0x5c, 0x83, 0xe1,
    ]
};

/// Hex lookup table for fast encoding (avoids per-byte format! allocation)
const HEX_LUT: &[u8; 16] = b"0123456789abcdef";

/// Compute SHA-256 hash of arbitrary data
#[inline(always)]
pub fn sha256(data: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Compute the hash of two child nodes (internal node hash)
/// H(left || right)
#[inline(always)]
pub fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Compute the hash of a leaf node
/// H(0x00 || key || value)
/// The 0x00 prefix domain-separates leaf hashes from internal node hashes.
#[inline(always)]
pub fn hash_leaf(key: &Hash, value: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x00]); // leaf domain separator
    hasher.update(key);
    hasher.update(value);
    hasher.finalize().into()
}

/// Compute the hash of an internal node
/// H(0x01 || left || right)
/// The 0x01 prefix domain-separates internal hashes from leaf hashes.
#[inline(always)]
pub fn hash_internal(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x01]); // internal domain separator
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Compute the hash of a computation attestation
/// H(0x02 || input_root || output_root || function_id || params)
#[inline]
pub fn hash_attestation(
    input_root: &Hash,
    output_root: &Hash,
    function_id: &[u8],
    params: &[u8],
) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x02]); // attestation domain separator
    hasher.update(input_root);
    hasher.update(output_root);
    hasher.update(function_id);
    hasher.update(params);
    hasher.finalize().into()
}

/// Simple algebraic hash for ZK-friendly operations (Poseidon-style placeholder).
///
/// In a real ZK deployment, this would be replaced with a proper Poseidon
/// permutation over a prime field (e.g., BN254 scalar field). This simplified
/// version uses domain-separated SHA-256 to provide the same API shape.
///
/// A real Poseidon implementation would:
/// - Operate over Fp elements (field arithmetic, not byte operations)
/// - Use ~8x fewer constraints in R1CS/Plonkish circuits
/// - Provide algebraic collision resistance (not just generic)
/// - Support variable-width inputs via a sponge construction
#[cfg(feature = "poseidon")]
pub fn poseidon_hash(inputs: &[u64]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x03]); // poseidon domain separator
    for val in inputs {
        hasher.update(&val.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Convert a hash to a hex string (zero-allocation fast path)
#[inline]
pub fn to_hex(hash: &Hash) -> String {
    let mut hex = Vec::with_capacity(HASH_SIZE * 2);
    for &b in hash {
        hex.push(HEX_LUT[(b >> 4) as usize]);
        hex.push(HEX_LUT[(b & 0x0f) as usize]);
    }
    // SAFETY: HEX_LUT only contains valid ASCII bytes
    unsafe { String::from_utf8_unchecked(hex) }
}

/// Parse a hex string into a Hash
#[inline]
pub fn from_hex(hex: &str) -> Result<Hash, &'static str> {
    if hex.len() != HASH_SIZE * 2 {
        return Err("Invalid hex length");
    }
    let bytes = hex.as_bytes();
    let mut hash = [0u8; HASH_SIZE];
    for i in 0..HASH_SIZE {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        hash[i] = (hi << 4) | lo;
    }
    Ok(hash)
}

/// Decode a single hex character to its nibble value
#[inline(always)]
fn hex_nibble(c: u8) -> Result<u8, &'static str> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err("Invalid hex digit"),
    }
}

/// Compute the key (address) for a given data blob by hashing it
#[inline]
pub fn compute_key(data: &[u8]) -> Hash {
    sha256(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_deterministic() {
        let h1 = sha256(b"hello");
        let h2 = sha256(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_sha256_different_inputs() {
        let h1 = sha256(b"hello");
        let h2 = sha256(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_domain_separation() {
        let key = sha256(b"test_key");
        let value = b"test_value";
        let leaf = hash_leaf(&key, value);
        let internal = hash_internal(&key, &sha256(value));
        assert_ne!(leaf, internal, "Leaf and internal hashes must differ");
    }

    #[test]
    fn test_hex_roundtrip() {
        let hash = sha256(b"roundtrip");
        let hex = to_hex(&hash);
        let recovered = from_hex(&hex).unwrap();
        assert_eq!(hash, recovered);
    }

    #[test]
    fn test_from_hex_invalid() {
        assert!(from_hex("not_valid_hex").is_err());
        assert!(from_hex("abcd").is_err()); // too short
    }

    #[test]
    fn test_hex_case_insensitive() {
        let hash = sha256(b"test");
        let hex_lower = to_hex(&hash);
        let hex_upper = hex_lower.to_uppercase();
        let recovered = from_hex(&hex_upper).unwrap();
        assert_eq!(hash, recovered);
    }
}
