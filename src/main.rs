//! Tiny C Compiler (cc2).

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

pub mod compiler;

use std::fs;

fn print_usage(program: &str) -> ! {
    eprintln!("usage: \n\t{} <file_path>", program);
    std::process::exit(1);
}

fn main() {
    let mut args = std::env::args();
    let program = args.next().expect("missing program name");

    let file_path = args.next().unwrap_or_else(|| {
        print_usage(&program);
    });

    let input = fs::read_to_string(file_path).unwrap_or_else(|err| {
        eprintln!("error: {err}");
        print_usage(&program);
    });

    let _ = compiler::Lexer::new(input.as_bytes()).unwrap_or_else(|err| {
        eprintln!("error: {err}");
        std::process::exit(1);
    });
}
