//! Minimal GGUF v2/v3 reader, specialised for locating individual MoE experts on disk.
//!
//! We deliberately do not decode tensor *data* — the whole point of this project is that
//! expert reordering is a byte-level block permutation that never has to interpret a
//! single quantised weight.

pub mod moe;
pub mod types;

use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub use types::GgmlType;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ne[0..n_dims], ggml order (fastest-varying first).
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the tensor data section.
    pub rel_offset: u64,
    /// Absolute byte offset in the file.
    pub offset: u64,
    /// Total bytes occupied by this tensor.
    pub nbytes: u64,
}

impl TensorInfo {
    /// Number of elements along the fastest-varying axis.
    pub fn ne0(&self) -> u64 {
        self.dims[0]
    }

    /// Extent of the slowest-varying axis — the expert axis for fused `*_exps` tensors.
    pub fn last_dim(&self) -> u64 {
        *self.dims.last().expect("tensor has at least one dimension")
    }

    /// Bytes occupied by one slice along the last axis.
    ///
    /// For `[n_embd, n_ff, n_expert]` this is one whole expert's matrix, and because ggml
    /// stores dimensions in row-major-from-the-back order those bytes are contiguous.
    pub fn last_axis_stride(&self) -> Result<u64> {
        let n = self.last_dim();
        if n == 0 {
            bail!("tensor {} has a zero-length last dimension", self.name);
        }
        if !self.nbytes.is_multiple_of(n) {
            bail!(
                "tensor {} size {} is not divisible by last dim {}",
                self.name,
                self.nbytes,
                n
            );
        }
        Ok(self.nbytes / n)
    }

    /// Absolute `(offset, len)` of one slice along the last axis.
    pub fn slice_range(&self, index: u64) -> Result<(u64, u64)> {
        let stride = self.last_axis_stride()?;
        if index >= self.last_dim() {
            bail!(
                "index {} out of range for tensor {} (last dim {})",
                index,
                self.name,
                self.last_dim()
            );
        }
        Ok((self.offset + index * stride, stride))
    }
}

#[derive(Debug, Clone)]
pub struct Gguf {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: u64,
    pub kv: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute offset where the tensor data section begins.
    pub data_offset: u64,
    pub file_size: u64,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let file_size = file.metadata()?.len();
        let mut r = BufReader::with_capacity(1 << 20, file);
        Self::parse(path, &mut r, file_size)
    }

    /// Parses a GGUF from just its leading bytes, given the true size of the whole file.
    ///
    /// The header and tensor index live at the front, so an HTTP range request of a few
    /// megabytes is enough to learn a model's complete expert geometry. That makes it
    /// possible to reason about a 400 GB model without downloading it.
    pub fn from_header(bytes: &[u8], file_size: u64, name: &str) -> Result<Self> {
        let mut cur = std::io::Cursor::new(bytes);
        Self::parse(PathBuf::from(name), &mut cur, file_size)
    }

    fn parse<R: Read + Seek>(path: PathBuf, r: &mut R, file_size: u64) -> Result<Self> {
        let magic = read_u32(r)?;
        if magic != MAGIC {
            bail!("not a GGUF file: magic {magic:#x}");
        }
        let version = read_u32(r)?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let n_tensors = read_u64(r)?;
        let n_kv = read_u64(r)?;

        let mut kv = BTreeMap::new();
        for _ in 0..n_kv {
            let key = read_string(r)?;
            let ty = read_u32(r)?;
            let val = read_value(r, ty)?;
            kv.insert(key, val);
        }

        let alignment = kv
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32);
        if !alignment.is_power_of_two() {
            bail!("general.alignment {alignment} is not a power of two");
        }

        let mut raw = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(r)?;
            let n_dims = read_u32(r)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(r)?);
            }
            let ty = types::lookup(read_u32(r)?).with_context(|| format!("tensor {name}"))?;
            let rel_offset = read_u64(r)?;
            raw.push((name, dims, ty, rel_offset));
        }

        let pos = r.stream_position()?;
        let data_offset = pos.div_ceil(alignment) * alignment;

        let mut tensors = Vec::with_capacity(raw.len());
        for (name, dims, ty, rel_offset) in raw {
            let row = ty
                .row_bytes(dims[0])
                .with_context(|| format!("tensor {name}"))?;
            let rows: u64 = dims[1..].iter().product::<u64>().max(1);
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                rel_offset,
                offset: data_offset + rel_offset,
                nbytes: row * rows,
            });
        }

        let g = Gguf {
            path,
            version,
            alignment,
            kv,
            tensors,
            data_offset,
            file_size,
        };
        g.validate()?;
        Ok(g)
    }

    /// Cross-checks our block arithmetic against the offsets the file itself declares.
    ///
    /// If our `type_size` table were wrong for any type present in the file, consecutive
    /// tensors would overlap or leave impossible gaps. This turns a silent corruption bug
    /// into a load-time error.
    fn validate(&self) -> Result<()> {
        let mut by_offset: Vec<&TensorInfo> = self.tensors.iter().collect();
        by_offset.sort_by_key(|t| t.rel_offset);
        for w in by_offset.windows(2) {
            let (a, b) = (w[0], w[1]);
            let end = a.rel_offset + a.nbytes;
            if end > b.rel_offset {
                bail!(
                    "tensor {} (rel {}..{}) overlaps {} at rel {} — block size table is wrong",
                    a.name,
                    a.rel_offset,
                    end,
                    b.name,
                    b.rel_offset
                );
            }
            let gap = b.rel_offset - end;
            if gap >= self.alignment {
                bail!(
                    "unexpected {}-byte gap between {} and {} (alignment {})",
                    gap,
                    a.name,
                    b.name,
                    self.alignment
                );
            }
        }
        if let Some(last) = by_offset.last() {
            let end = self.data_offset + last.rel_offset + last.nbytes;
            if end > self.file_size {
                bail!(
                    "tensor {} ends at {} but file is only {} bytes",
                    last.name,
                    end,
                    self.file_size
                );
            }
        }
        Ok(())
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("no tensor named {name}"))
    }

    pub fn architecture(&self) -> Result<&str> {
        self.kv
            .get("general.architecture")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("missing general.architecture"))
    }

    /// Reads an architecture-scoped key, e.g. `expert_count` -> `olmoe.expert_count`.
    pub fn arch_u64(&self, suffix: &str) -> Result<u64> {
        let key = format!("{}.{}", self.architecture()?, suffix);
        self.kv
            .get(&key)
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("missing or non-integer key {key}"))
    }
}

fn read_exact<R: Read, const N: usize>(r: &mut R) -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact::<_, 4>(r)?))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact::<_, 8>(r)?))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    // Guard against a corrupt length turning into a multi-gigabyte allocation.
    if len > 1 << 30 {
        bail!("implausible string length {len}");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_value<R: Read + Seek>(r: &mut R, ty: u32) -> Result<Value> {
    Ok(match ty {
        0 => Value::U8(read_exact::<_, 1>(r)?[0]),
        1 => Value::I8(read_exact::<_, 1>(r)?[0] as i8),
        2 => Value::U16(u16::from_le_bytes(read_exact::<_, 2>(r)?)),
        3 => Value::I16(i16::from_le_bytes(read_exact::<_, 2>(r)?)),
        4 => Value::U32(read_u32(r)?),
        5 => Value::I32(i32::from_le_bytes(read_exact::<_, 4>(r)?)),
        6 => Value::F32(f32::from_le_bytes(read_exact::<_, 4>(r)?)),
        7 => Value::Bool(read_exact::<_, 1>(r)?[0] != 0),
        8 => Value::String(read_string(r)?),
        9 => {
            let elem_ty = read_u32(r)?;
            let n = read_u64(r)?;
            // Token vocabularies run to hundreds of thousands of strings; we never need
            // their contents, so skip past them instead of materialising the Vec.
            if n > 4096 {
                skip_array(r, elem_ty, n)?;
                Value::Array(Vec::new())
            } else {
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(read_value(r, elem_ty)?);
                }
                Value::Array(v)
            }
        }
        10 => Value::U64(read_u64(r)?),
        11 => Value::I64(i64::from_le_bytes(read_exact::<_, 8>(r)?)),
        12 => Value::F64(f64::from_le_bytes(read_exact::<_, 8>(r)?)),
        other => bail!("unknown GGUF value type {other}"),
    })
}

fn skip_array<R: Read + Seek>(r: &mut R, elem_ty: u32, n: u64) -> Result<()> {
    let fixed = match elem_ty {
        0 | 1 | 7 => Some(1u64),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        8 => None,
        9 => bail!("nested arrays are not supported"),
        other => bail!("unknown GGUF value type {other} in array"),
    };
    match fixed {
        Some(sz) => {
            r.seek(SeekFrom::Current((sz * n) as i64))?;
        }
        None => {
            for _ in 0..n {
                let len = read_u64(r)? as i64;
                r.seek(SeekFrom::Current(len))?;
            }
        }
    }
    Ok(())
}
