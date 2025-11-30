//! Delta compression algorithms for CANTOR.

use cantor_core::{CantorError, Result};

/// Compression method selection.
#[derive(Clone, Copy, Debug, Default)]
pub enum CompressionMethod {
    #[default]
    Lz4,
    Varint,
    RunLength,
}

/// Delta encoder with multiple compression strategies.
pub struct DeltaEncoder {
    method: CompressionMethod,
}

impl DeltaEncoder {
    pub fn new(method: CompressionMethod) -> Self {
        Self { method }
    }

    pub fn encode(&self, delta: &[f32]) -> Result<Vec<u8>> {
        match self.method {
            CompressionMethod::Lz4 => self.encode_lz4(delta),
            CompressionMethod::Varint => self.encode_varint(delta),
            CompressionMethod::RunLength => self.encode_rle(delta),
        }
    }

    pub fn decode(&self, data: &[u8]) -> Result<Vec<f32>> {
        match self.method {
            CompressionMethod::Lz4 => self.decode_lz4(data),
            CompressionMethod::Varint => self.decode_varint(data),
            CompressionMethod::RunLength => self.decode_rle(data),
        }
    }

    fn encode_lz4(&self, delta: &[f32]) -> Result<Vec<u8>> {
        let bytes: Vec<u8> = delta.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        
        lz4::block::compress(&bytes, None, false)
            .map_err(|e| CantorError::CompressionFailed(e.to_string()))
    }

    fn decode_lz4(&self, data: &[u8]) -> Result<Vec<f32>> {
        let decompressed = lz4::block::decompress(data, None)
            .map_err(|e| CantorError::DecompressionFailed(e.to_string()))?;
        
        if decompressed.len() % 4 != 0 {
            return Err(CantorError::InvalidDeltaEncoding);
        }
        
        Ok(decompressed
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn encode_varint(&self, delta: &[f32]) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(delta.len() * 2);
        
        for &val in delta {
            let quantized = (val * 1000.0).round() as i32;
            let zigzag = Self::zigzag_encode(quantized);
            Self::write_varint(&mut result, zigzag);
        }
        
        Ok(result)
    }

    fn decode_varint(&self, data: &[u8]) -> Result<Vec<f32>> {
        let mut result = Vec::new();
        let mut pos = 0;
        
        while pos < data.len() {
            let (value, consumed) = Self::read_varint(&data[pos..])
                .ok_or(CantorError::InvalidDeltaEncoding)?;
            let decoded = Self::zigzag_decode(value);
            result.push(decoded as f32 / 1000.0);
            pos += consumed;
        }
        
        Ok(result)
    }

    fn encode_rle(&self, delta: &[f32]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < delta.len() {
            if delta[i].abs() < 1e-6 {
                let mut count = 0u8;
                while i < delta.len() && delta[i].abs() < 1e-6 && count < 255 {
                    count += 1;
                    i += 1;
                }
                result.push(0);
                result.push(count);
            } else {
                result.extend_from_slice(&delta[i].to_le_bytes());
                i += 1;
            }
        }
        
        Ok(result)
    }

    fn decode_rle(&self, data: &[u8]) -> Result<Vec<f32>> {
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < data.len() {
            if data[i] == 0 && i + 1 < data.len() {
                let count = data[i + 1] as usize;
                result.extend(std::iter::repeat(0.0).take(count));
                i += 2;
            } else if i + 4 <= data.len() {
                let bytes: [u8; 4] = data[i..i+4].try_into().unwrap();
                result.push(f32::from_le_bytes(bytes));
                i += 4;
            } else {
                return Err(CantorError::InvalidDeltaEncoding);
            }
        }
        
        Ok(result)
    }

    fn zigzag_encode(n: i32) -> u32 {
        ((n << 1) ^ (n >> 31)) as u32
    }

    fn zigzag_decode(n: u32) -> i32 {
        ((n >> 1) as i32) ^ -((n & 1) as i32)
    }

    fn write_varint(buf: &mut Vec<u8>, mut n: u32) {
        while n >= 0x80 {
            buf.push((n as u8) | 0x80);
            n >>= 7;
        }
        buf.push(n as u8);
    }

    fn read_varint(data: &[u8]) -> Option<(u32, usize)> {
        let mut result = 0u32;
        let mut shift = 0;
        
        for (i, &byte) in data.iter().enumerate() {
            result |= ((byte & 0x7F) as u32) << shift;
            if byte & 0x80 == 0 {
                return Some((result, i + 1));
            }
            shift += 7;
            if shift >= 32 {
                return None;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_roundtrip() {
        let encoder = DeltaEncoder::new(CompressionMethod::Lz4);
        let delta = vec![0.1, 0.2, 0.0, 0.0, 0.3];
        let encoded = encoder.encode(&delta).unwrap();
        let decoded = encoder.decode(&encoded).unwrap();
        assert_eq!(delta.len(), decoded.len());
    }

    #[test]
    fn test_varint_roundtrip() {
        let encoder = DeltaEncoder::new(CompressionMethod::Varint);
        let delta = vec![0.001, -0.002, 0.0, 0.5];
        let encoded = encoder.encode(&delta).unwrap();
        let decoded = encoder.decode(&encoded).unwrap();
        for (a, b) in delta.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_zigzag() {
        assert_eq!(DeltaEncoder::zigzag_encode(0), 0);
        assert_eq!(DeltaEncoder::zigzag_encode(-1), 1);
        assert_eq!(DeltaEncoder::zigzag_encode(1), 2);
        assert_eq!(DeltaEncoder::zigzag_decode(0), 0);
        assert_eq!(DeltaEncoder::zigzag_decode(1), -1);
        assert_eq!(DeltaEncoder::zigzag_decode(2), 1);
    }
}

