//! Occupy RAM so the page cache cannot hold the whole model.
//!
//! On a 36 GB machine a 4 GB model is entirely cached after the first run, and disk layout
//! becomes unmeasurable end to end. This pins N GB of resident memory so llama.cpp's mmap
//! has to fault pages back in from the SSD — the regime where layout is the thing that
//! decides throughput.
//!
//! It deliberately does not touch swap settings or system state: kill it and the machine is
//! exactly as it was.

use anyhow::{bail, Result};
use clap::Parser;
use std::io::Read;

const CHUNK: usize = 64 << 20;

#[derive(Parser)]
#[command(about = "Hold N GiB of RAM resident to squeeze the page cache")]
struct Args {
    /// Gibibytes to occupy.
    #[arg(short, long)]
    gib: f64,
    /// Re-touch every page at this interval (seconds) so the pages stay hot. 0 disables.
    #[arg(long, default_value_t = 5)]
    refresh: u64,
    /// mlock the region. Needs a raised memlock limit and can wedge a busy machine.
    #[arg(long)]
    lock: bool,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.gib <= 0.0 {
        bail!("--gib must be positive");
    }
    let total = (a.gib * (1u64 << 30) as f64) as usize;
    let n_chunks = total.div_ceil(CHUNK);

    eprintln!("ballast: reserving {:.1} GiB in {n_chunks} chunks", a.gib);
    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(n_chunks);
    for i in 0..n_chunks {
        let len = CHUNK.min(total - i * CHUNK);
        // Fill every byte with an incompressible stream. Touching one byte per page is not
        // enough: macOS compresses anonymous memory, and a page that is 4095 zeros collapses
        // to nothing, so the ballast weighs far less than it claims and the page cache never
        // feels it. This cost one round of flat, meaningless pressure measurements.
        let mut v = vec![0u8; len];
        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        for chunk in v.chunks_mut(8) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let b = x.to_le_bytes();
            chunk.copy_from_slice(&b[..chunk.len()]);
        }
        if a.lock {
            let rc = unsafe { libc::mlock(v.as_ptr() as *const libc::c_void, len) };
            if rc != 0 {
                eprintln!(
                    "ballast: mlock failed ({}); continuing without it",
                    std::io::Error::last_os_error()
                );
            }
        }
        chunks.push(v);
        if i % 8 == 0 {
            eprint!(
                "\rballast: {:.1} GiB resident",
                (i + 1) as f64 * CHUNK as f64 / (1u64 << 30) as f64
            );
        }
    }
    eprintln!(
        "\rballast: {:.1} GiB resident — press Ctrl-C or send EOF on stdin to release",
        a.gib
    );

    if a.refresh == 0 {
        // Block until stdin closes so the parent can control the lifetime.
        let mut sink = Vec::new();
        std::io::stdin().read_to_end(&mut sink).ok();
        return Ok(());
    }

    let mut tick = 0u64;
    loop {
        std::thread::sleep(std::time::Duration::from_secs(a.refresh));
        tick = tick.wrapping_add(1);
        for (i, c) in chunks.iter_mut().enumerate() {
            let len = c.len();
            for p in (0..len).step_by(4096) {
                c[p] ^= ((i as u64 ^ tick) as u8) | 1;
            }
        }
    }
}
