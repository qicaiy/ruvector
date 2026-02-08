//! Spacetime tile system for organizing Gaussians into quantized blocks.
//!
//! The world is partitioned into a regular 3D spatial grid with temporal bucketing.
//! Each [`Tile`] holds a [`PrimitiveBlock`] containing encoded Gaussian data at a
//! particular [`QuantTier`]. Tiles are addressed by [`TileCoord`] which includes
//! spatial coordinates, a time bucket, and a level-of-detail index.

use crate::gaussian::Gaussian4D;

/// Tile coordinate in spacetime grid.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct TileCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub time_bucket: u32,
    pub lod: u8,
}

/// A spacetime tile containing a block of Gaussians.
#[derive(Clone, Debug)]
pub struct Tile {
    pub coord: TileCoord,
    pub primitive_block: PrimitiveBlock,
    /// Entity IDs contained in this tile.
    pub entity_refs: Vec<u64>,
    /// Coherence score from the last evaluation (0.0 = incoherent, 1.0 = fully coherent).
    pub coherence_score: f32,
    /// Epoch of the last update.
    pub last_update_epoch: u64,
}

/// Packed, quantized block of Gaussian primitives.
///
/// In the current implementation, encoding stores raw `f32` bytes regardless of
/// the quantization tier. Actual quantized packing (8/7/5/3-bit) is deferred to
/// a future iteration; the [`QuantTier`] enum is stored to tag the intended
/// compression level.
#[derive(Clone, Debug)]
pub struct PrimitiveBlock {
    /// Raw encoded data.
    pub data: Vec<u8>,
    /// Number of Gaussians in this block.
    pub count: u32,
    /// Quantization tier tag.
    pub quant_tier: QuantTier,
    /// Checksum over `data`.
    pub checksum: u32,
    /// Descriptor for decoding fields from the packed data.
    pub decode_descriptor: DecodeDescriptor,
}

/// Quantization tier controlling compression ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantTier {
    /// 8-bit quantization, ~4x compression.
    Hot8,
    /// 7-bit quantization, ~4.57x compression.
    Warm7,
    /// 5-bit quantization, ~6.4x compression.
    Warm5,
    /// 3-bit quantization, ~10.67x compression.
    Cold3,
}

/// Descriptor that tells the decoder how to interpret a [`PrimitiveBlock`].
#[derive(Clone, Debug)]
pub struct DecodeDescriptor {
    /// Total bytes per Gaussian in the packed format.
    pub bytes_per_gaussian: u16,
    /// Byte offsets of each field within a Gaussian record.
    pub field_offsets: FieldOffsets,
    /// Per-field rescaling factors applied after dequantization.
    pub scale_factors: [f32; 4],
}

/// Byte offsets of each field within a packed Gaussian record.
#[derive(Clone, Debug)]
pub struct FieldOffsets {
    pub center: u16,
    pub covariance: u16,
    pub color: u16,
    pub opacity: u16,
    pub scale: u16,
    pub rotation: u16,
    pub temporal: u16,
}

// Size of a single Gaussian when stored as raw f32 bytes:
//   center(3) + covariance(6) + sh_coeffs(3) + opacity(1) + scale(3) + rotation(4)
//   + time_range(2) + velocity(3) + id(1 as u32→f32 reinterpret, stored separately)
// We store the id as 4 bytes (u32) at the end.
// Total floats: 3+6+3+1+3+4+2+3 = 25 floats = 100 bytes + 4 bytes id = 104 bytes
const RAW_GAUSSIAN_BYTES: u16 = 104;

impl PrimitiveBlock {
    /// Encode a slice of Gaussians into a primitive block.
    ///
    /// Currently stores raw `f32` bytes regardless of the chosen `tier`.
    /// The tier is recorded in the block for future quantized implementations.
    pub fn encode(gaussians: &[Gaussian4D], tier: QuantTier) -> Self {
        let count = gaussians.len() as u32;
        let mut data = Vec::with_capacity(gaussians.len() * RAW_GAUSSIAN_BYTES as usize);

        for g in gaussians {
            // center: 3 floats (offset 0)
            for &v in &g.center {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // covariance: 6 floats (offset 12)
            for &v in &g.covariance {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // sh_coeffs: 3 floats (offset 36)
            for &v in &g.sh_coeffs {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // opacity: 1 float (offset 48)
            data.extend_from_slice(&g.opacity.to_le_bytes());
            // scale: 3 floats (offset 52)
            for &v in &g.scale {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // rotation: 4 floats (offset 64)
            for &v in &g.rotation {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // time_range: 2 floats (offset 80)
            for &v in &g.time_range {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // velocity: 3 floats (offset 88)
            for &v in &g.velocity {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // id: u32 (offset 100)
            data.extend_from_slice(&g.id.to_le_bytes());
        }

        let checksum = compute_checksum(&data);

        let decode_descriptor = DecodeDescriptor {
            bytes_per_gaussian: RAW_GAUSSIAN_BYTES,
            field_offsets: FieldOffsets {
                center: 0,
                covariance: 12,
                color: 36,
                opacity: 48,
                scale: 52,
                rotation: 64,
                temporal: 80,
            },
            scale_factors: [1.0, 1.0, 1.0, 1.0],
        };

        Self {
            data,
            count,
            quant_tier: tier,
            checksum,
            decode_descriptor,
        }
    }

    /// Decode the primitive block back into Gaussians.
    pub fn decode(&self) -> Vec<Gaussian4D> {
        let stride = self.decode_descriptor.bytes_per_gaussian as usize;
        let mut gaussians = Vec::with_capacity(self.count as usize);

        for i in 0..self.count as usize {
            let base = i * stride;
            if base + stride > self.data.len() {
                break;
            }

            let read_f32 = |offset: usize| -> f32 {
                let o = base + offset;
                f32::from_le_bytes([
                    self.data[o],
                    self.data[o + 1],
                    self.data[o + 2],
                    self.data[o + 3],
                ])
            };

            let read_f32_array = |offset: usize, count: usize| -> Vec<f32> {
                (0..count).map(|j| read_f32(offset + j * 4)).collect()
            };

            let center_v = read_f32_array(0, 3);
            let cov_v = read_f32_array(12, 6);
            let sh_v = read_f32_array(36, 3);
            let opacity = read_f32(48);
            let scale_v = read_f32_array(52, 3);
            let rot_v = read_f32_array(64, 4);
            let time_v = read_f32_array(80, 2);
            let vel_v = read_f32_array(88, 3);

            let id_offset = base + 100;
            let id = u32::from_le_bytes([
                self.data[id_offset],
                self.data[id_offset + 1],
                self.data[id_offset + 2],
                self.data[id_offset + 3],
            ]);

            gaussians.push(Gaussian4D {
                center: [center_v[0], center_v[1], center_v[2]],
                covariance: [cov_v[0], cov_v[1], cov_v[2], cov_v[3], cov_v[4], cov_v[5]],
                sh_coeffs: [sh_v[0], sh_v[1], sh_v[2]],
                opacity,
                scale: [scale_v[0], scale_v[1], scale_v[2]],
                rotation: [rot_v[0], rot_v[1], rot_v[2], rot_v[3]],
                time_range: [time_v[0], time_v[1]],
                velocity: [vel_v[0], vel_v[1], vel_v[2]],
                id,
            });
        }

        gaussians
    }

    /// Recompute and return the checksum over the data.
    pub fn compute_checksum(&self) -> u32 {
        compute_checksum(&self.data)
    }

    /// Verify that the stored checksum matches the data.
    pub fn verify_checksum(&self) -> bool {
        self.checksum == compute_checksum(&self.data)
    }
}

/// Simple additive hash checksum (not cryptographic).
///
/// Processes data in 4-byte chunks, treating each as a little-endian u32
/// and summing with wrapping arithmetic. Remaining bytes are incorporated
/// by shifting into a final u32.
fn compute_checksum(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5; // FNV offset basis
    for &byte in data {
        hash = hash.wrapping_mul(0x0100_0193); // FNV prime
        hash ^= byte as u32;
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::Gaussian4D;

    #[test]
    fn test_encode_decode_roundtrip() {
        let gaussians = vec![
            Gaussian4D::new([1.0, 2.0, 3.0], 10),
            Gaussian4D::new([4.0, 5.0, 6.0], 20),
        ];
        let block = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
        assert_eq!(block.count, 2);
        assert!(block.verify_checksum());

        let decoded = block.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].center, [1.0, 2.0, 3.0]);
        assert_eq!(decoded[0].id, 10);
        assert_eq!(decoded[1].center, [4.0, 5.0, 6.0]);
        assert_eq!(decoded[1].id, 20);
    }

    #[test]
    fn test_encode_decode_preserves_all_fields() {
        let mut g = Gaussian4D::new([1.0, 2.0, 3.0], 99);
        g.covariance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        g.sh_coeffs = [0.7, 0.8, 0.9];
        g.opacity = 0.75;
        g.scale = [1.5, 2.5, 3.5];
        g.rotation = [0.5, 0.5, 0.5, 0.5];
        g.time_range = [0.0, 10.0];
        g.velocity = [0.1, 0.2, 0.3];

        let block = PrimitiveBlock::encode(&[g.clone()], QuantTier::Warm5);
        let decoded = block.decode();
        assert_eq!(decoded.len(), 1);
        let d = &decoded[0];
        assert_eq!(d.center, g.center);
        assert_eq!(d.covariance, g.covariance);
        assert_eq!(d.sh_coeffs, g.sh_coeffs);
        assert_eq!(d.opacity, g.opacity);
        assert_eq!(d.scale, g.scale);
        assert_eq!(d.rotation, g.rotation);
        assert_eq!(d.time_range, g.time_range);
        assert_eq!(d.velocity, g.velocity);
        assert_eq!(d.id, g.id);
    }

    #[test]
    fn test_empty_encode() {
        let block = PrimitiveBlock::encode(&[], QuantTier::Cold3);
        assert_eq!(block.count, 0);
        assert!(block.data.is_empty());
        let decoded = block.decode();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_checksum_changes_with_data() {
        let g1 = Gaussian4D::new([1.0, 0.0, 0.0], 1);
        let g2 = Gaussian4D::new([2.0, 0.0, 0.0], 2);
        let block1 = PrimitiveBlock::encode(&[g1], QuantTier::Hot8);
        let block2 = PrimitiveBlock::encode(&[g2], QuantTier::Hot8);
        assert_ne!(block1.checksum, block2.checksum);
    }

    #[test]
    fn test_tile_coord_equality() {
        let c1 = TileCoord {
            x: 1,
            y: 2,
            z: 3,
            time_bucket: 0,
            lod: 0,
        };
        let c2 = TileCoord {
            x: 1,
            y: 2,
            z: 3,
            time_bucket: 0,
            lod: 0,
        };
        assert_eq!(c1, c2);
    }
}
