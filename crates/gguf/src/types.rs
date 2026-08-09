//! ggml tensor types and their block geometry.
//!
//! Every quantised ggml type packs `block_size` consecutive elements of a row into
//! `type_size` bytes. A row is therefore `ne0 / block_size * type_size` bytes, and rows
//! are stored back to back with no padding. That is the whole of the byte arithmetic we
//! need to locate an individual expert inside a fused `*_exps` tensor.

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlType {
    pub id: u32,
    pub name: &'static str,
    /// Elements per quantisation block (1 for unquantised types).
    pub block_size: u64,
    /// Bytes per quantisation block.
    pub type_size: u64,
}

impl GgmlType {
    /// Bytes occupied by a row of `ne0` elements.
    pub fn row_bytes(&self, ne0: u64) -> Result<u64> {
        if !ne0.is_multiple_of(self.block_size) {
            bail!(
                "row length {} is not a multiple of {} block size {}",
                ne0,
                self.name,
                self.block_size
            );
        }
        Ok(ne0 / self.block_size * self.type_size)
    }
}

macro_rules! types {
    ($(($id:expr, $name:expr, $blk:expr, $sz:expr)),* $(,)?) => {
        const TABLE: &[GgmlType] = &[
            $(GgmlType { id: $id, name: $name, block_size: $blk, type_size: $sz }),*
        ];
    };
}

// Mirrors ggml.c `type_traits[]`. Only the types that actually appear in GGUF files on
// disk are listed; anything else is reported as unsupported rather than guessed at.
types![
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (34, "TQ1_0", 256, 54),
    (35, "TQ2_0", 256, 66),
    (39, "MXFP4", 32, 17),
];

pub fn lookup(id: u32) -> Result<GgmlType> {
    TABLE
        .iter()
        .copied()
        .find(|t| t.id == id)
        .ok_or_else(|| anyhow::anyhow!("unsupported ggml type id {id}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_row_geometry() {
        // OLMoE ffn_gate_exps is Q4_K with ne0 = 2048: 8 blocks of 144 bytes.
        let t = lookup(12).unwrap();
        assert_eq!(t.row_bytes(2048).unwrap(), 1152);
    }

    #[test]
    fn q6_k_row_geometry() {
        // OLMoE ffn_down_exps is Q6_K with ne0 = 1024: 4 blocks of 210 bytes.
        let t = lookup(14).unwrap();
        assert_eq!(t.row_bytes(1024).unwrap(), 840);
    }

    #[test]
    fn ragged_row_is_rejected() {
        let t = lookup(12).unwrap();
        assert!(t.row_bytes(2000).is_err());
    }
}
