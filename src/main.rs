//! Tiny C Compiler (subset of _C17_).

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(rust_2018_idioms)]
#![warn(missing_debug_implementations)]
#![allow(clippy::use_self)]
#![allow(clippy::redundant_else)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::struct_excessive_bools)]

pub mod args;
pub mod compiler;
pub mod error;

fn main() {
    let args = args::Args::parse();

    if let Err(err) = compiler::driver::run_compiler(&args) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
