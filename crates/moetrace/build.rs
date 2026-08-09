use std::path::{Path, PathBuf};
use std::process::Command;

fn pkg_config(args: &[&str]) -> Vec<String> {
    let out = Command::new("pkg-config")
        .args(args)
        .output()
        .expect("pkg-config not found — install llama.cpp via Homebrew");
    if !out.status.success() {
        panic!(
            "pkg-config {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// llama.pc advertises only llama.cpp's own keg, but the shim needs ggml's headers and
/// both kegs' libraries. Homebrew installs them as siblings under `Cellar/`.
fn ggml_keg(llama_include: &Path) -> Option<PathBuf> {
    let cellar = llama_include.ancestors().nth(3)?; // <keg>/include -> <keg> -> llama.cpp -> Cellar
    let entries = std::fs::read_dir(cellar.join("ggml")).ok()?;
    entries
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("include/ggml.h").exists())
}

fn main() {
    println!("cargo:rerun-if-changed=src/moetrace.c");

    let mut build = cc::Build::new();
    build.file("src/moetrace.c").std("c11").warnings(true);

    let mut lib_dirs: Vec<PathBuf> = Vec::new();
    for flag in pkg_config(&["--cflags", "llama"]) {
        if let Some(dir) = flag.strip_prefix("-I") {
            let dir = PathBuf::from(dir);
            build.include(&dir);
            if let Some(keg) = ggml_keg(&dir) {
                build.include(keg.join("include"));
                lib_dirs.push(keg.join("lib"));
            }
        } else {
            build.flag(&flag);
        }
    }

    build.compile("moetrace_shim");

    for flag in pkg_config(&["--libs", "llama"]) {
        if let Some(dir) = flag.strip_prefix("-L") {
            lib_dirs.push(PathBuf::from(dir));
        } else if let Some(lib) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
    // Homebrew's opt-prefix link farm, as a last resort for either keg.
    lib_dirs.push(PathBuf::from("/opt/homebrew/lib"));

    for d in lib_dirs {
        if d.is_dir() {
            println!("cargo:rustc-link-search=native={}", d.display());
        }
    }
}
