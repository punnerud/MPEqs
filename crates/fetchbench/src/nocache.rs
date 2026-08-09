//! Uncached reads on macOS.
//!
//! `fcntl(fd, F_NOCACHE, 1)` tells the kernel to keep this descriptor's data out of the
//! unified buffer cache, so every `pread` reaches the SSD. That is what makes these numbers
//! reproducible without `sudo purge` and without waiting for cache state to decay — and it
//! is the only way to measure disk layout on a machine with far more RAM than model.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// Reads are aligned to this boundary; a real fetcher would do the same.
pub const ALIGN: u64 = 4096;

pub struct NoCacheFile {
    file: File,
    pub len: u64,
}

// `pread` takes its offset as an argument and never touches the descriptor's own file
// position, so concurrent reads on one fd are safe and need no lock.
unsafe impl Send for NoCacheFile {}
unsafe impl Sync for NoCacheFile {}

impl NoCacheFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let len = file.metadata()?.len();

        #[cfg(target_os = "macos")]
        {
            let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
            if rc == -1 {
                bail!(
                    "fcntl(F_NOCACHE) failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            // Also disable readahead: we are measuring the cost of the fetch pattern the
            // layout produces, and kernel readahead would quietly paper over a bad one.
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0) };
        }
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("warning: uncached reads are only implemented for macOS; numbers will be cache hits");
        }

        Ok(NoCacheFile { file, len })
    }

    /// Reads `[offset, offset+len)` into `buf`, expanding to alignment boundaries.
    ///
    /// Returns the number of bytes actually transferred, which is what the device paid for.
    pub fn read_range(&self, offset: u64, len: u64, buf: &mut AlignedBuf) -> Result<u64> {
        let start = offset / ALIGN * ALIGN;
        let end = ((offset + len).min(self.len)).div_ceil(ALIGN) * ALIGN;
        let end = end.min(self.len.div_ceil(ALIGN) * ALIGN);
        let mut pos = start;
        let mut total = 0u64;
        while pos < end {
            let want = (end - pos).min(buf.capacity() as u64) as usize;
            let n = unsafe {
                libc::pread(
                    self.file.as_raw_fd(),
                    buf.as_mut_ptr() as *mut libc::c_void,
                    want,
                    pos as libc::off_t,
                )
            };
            if n < 0 {
                bail!("pread at {pos} failed: {}", std::io::Error::last_os_error());
            }
            if n == 0 {
                break;
            }
            total += n as u64;
            pos += n as u64;
        }
        Ok(total)
    }
}

/// Page-aligned scratch buffer for uncached I/O.
pub struct AlignedBuf {
    ptr: *mut u8,
    cap: usize,
    layout: std::alloc::Layout,
}

impl AlignedBuf {
    pub fn new(cap: usize) -> Self {
        let layout =
            std::alloc::Layout::from_size_align(cap, ALIGN as usize).expect("valid buffer layout");
        // SAFETY: cap > 0 and the alignment is a power of two.
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "failed to allocate {cap} bytes");
        AlignedBuf { ptr, cap, layout }
    }
    pub fn capacity(&self) -> usize {
        self.cap
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated by us with this exact layout, and never freed elsewhere.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}
