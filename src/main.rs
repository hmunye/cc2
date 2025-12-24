//! Tiny C Compiler (cc2).

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

// TODO: Add command line options to indicate what stage of compilation to run
// until ("lex", "parse", "codegen").
//
// TODO: Organize error handling.

pub mod compiler;

use std::fs;
use std::path::Path;

fn print_usage(program: &str) -> ! {
    eprintln!("\x1b[1;1musage:\x1b[0m\n\t{} <file_path>", program);
    std::process::exit(1);
}

fn main() {
    let mut args = std::env::args();
    let program = args.next().expect("missing program name");

    let file = args.next().unwrap_or_else(|| {
        print_usage(&program);
    });

    let file_path = Path::new(&file);
    let Some(file_name) = file_path.file_name() else {
        print_usage(&program);
    };

    let input = fs::read_to_string(file_path).unwrap_or_else(|err| {
        eprintln!("\x1b[1;31merror:\x1b[0m {err}");
        print_usage(&program);
    });

    let mut lexer = compiler::Lexer::new();
    lexer
        .lex(file_name, input.as_bytes())
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        });

    println!("{lexer:#?}");

    let ast = compiler::parser::parse_program(file_name, &mut lexer).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    let ir = compiler::ir::generate_ir(&ast).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    println!("AST: {ast:#?}");
    println!("IR: {ir:#?}");

    let output_path = file_path.with_extension("s");

    compiler::emit::emit_assembly(file_path, output_path, &ir).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });
}
