//! A dense square distance matrix on disk, and the two ways we build one.
//!
//! Format: `MATX` | u32 version | u32 n | u32 reserved | u32 reserved | n*n f32, row-major.
//! Deliberately plain — this is the *uncompressed* baseline that a codec would have to beat,
//! so it should have no cleverness of its own.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"MATX";

#[derive(Clone)]
pub struct Matrix {
    pub n: usize,
    pub d: Vec<f32>,
}

impl Matrix {
    pub fn zeros(n: usize) -> Self {
        Matrix {
            n,
            d: vec![0.0; n * n],
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.d[i * self.n + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f32) {
        let n = self.n;
        self.d[i * n + j] = v;
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let mut w = BufWriter::with_capacity(1 << 22, File::create(path)?);
        w.write_all(MAGIC)?;
        for v in [1u32, self.n as u32, 0, 0] {
            w.write_all(&v.to_le_bytes())?;
        }
        for v in &self.d {
            w.write_all(&v.to_le_bytes())?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut r = BufReader::with_capacity(
            1 << 22,
            File::open(path).with_context(|| format!("opening {}", path.display()))?,
        );
        let mut hdr = [0u8; 20];
        r.read_exact(&mut hdr)?;
        if &hdr[0..4] != MAGIC {
            bail!("{} is not a MATX matrix", path.display());
        }
        let n = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let mut raw = vec![0u8; n * n * 4];
        r.read_exact(&mut raw)?;
        let d = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Matrix { n, d })
    }

    /// Symmetrises in place, which several of the probes assume.
    pub fn symmetrise(&mut self) {
        for i in 0..self.n {
            for j in i + 1..self.n {
                let m = 0.5 * (self.get(i, j) + self.get(j, i));
                self.set(i, j, m);
                self.set(j, i, m);
            }
        }
    }
}

/// Angular distance between rows of an `[n, dim]` embedding block.
///
/// `arccos(cosine similarity)` rather than `1 - cosine`: the former is a true metric on the
/// sphere and satisfies the triangle inequality, which is precisely what matcodec's rank-1
/// base and `cell_bounds` are built on. Using `1 - cos` would quietly break the assumption
/// the whole probe is testing.
pub fn angular_from_embeddings(emb: &[f32], n: usize, dim: usize) -> Matrix {
    let mut norms = vec![0.0f32; n];
    for i in 0..n {
        let row = &emb[i * dim..(i + 1) * dim];
        norms[i] = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    }
    let mut m = Matrix::zeros(n);
    for i in 0..n {
        let a = &emb[i * dim..(i + 1) * dim];
        for j in i + 1..n {
            let b = &emb[j * dim..(j + 1) * dim];
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let cos = (dot / (norms[i] * norms[j])).clamp(-1.0, 1.0);
            let d = cos.acos();
            m.set(i, j, d);
            m.set(j, i, d);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angular_distance_is_zero_on_the_diagonal_and_pi_for_opposites() {
        let emb = vec![1.0f32, 0.0, -1.0, 0.0];
        let m = angular_from_embeddings(&emb, 2, 2);
        assert!(m.get(0, 0).abs() < 1e-6);
        assert!((m.get(0, 1) - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn orthogonal_vectors_are_a_right_angle_apart() {
        let emb = vec![1.0f32, 0.0, 0.0, 1.0];
        let m = angular_from_embeddings(&emb, 2, 2);
        assert!((m.get(0, 1) - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = std::env::temp_dir().join("matstruct-test");
        std::fs::create_dir_all(&dir).ok();
        let p = dir.join("m.matx");
        let mut m = Matrix::zeros(3);
        m.set(0, 1, 1.5);
        m.set(2, 0, -0.25);
        m.save(&p).unwrap();
        let back = Matrix::load(&p).unwrap();
        assert_eq!(back.n, 3);
        assert_eq!(back.get(0, 1), 1.5);
        assert_eq!(back.get(2, 0), -0.25);
        std::fs::remove_file(&p).ok();
    }
}
