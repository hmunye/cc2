//! Tiny C Compiler (cc2).

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

// TODO: Organize error handling.

pub mod args;
pub mod compiler;
pub mod error;

use std::io::Read;
use std::process;

fn main() {
    let mut args = args::Args::parse();
    let Ok(metadata) = args.in_file.metadata() else {
        print_err!(
            &args.program,
            "failed to query file: '{}'",
            args.in_path.display()
        );
        process::exit(1);
    };

    let mut src = Vec::with_capacity(metadata.len() as usize);

    let Ok(_) = args.in_file.read_to_end(&mut src) else {
        print_err!(
            &args.program,
            "failed to read from file: '{}'",
            args.in_path.display()
        );
        process::exit(1);
    };

    let mut lexer = compiler::Lexer::new();
    lexer.lex(args.in_path, &src).unwrap_or_else(|err| {
        eprintln!("{err}");
        process::exit(1);
    });

    match args.stage.as_str() {
        "lex" => {
            println!("{lexer:#?}");
        }
        "parse" => {
            let ast =
                compiler::parser::parse_program(args.in_path, &mut lexer).unwrap_or_else(|err| {
                    eprintln!("{err}");
                    process::exit(1);
                });

            println!("AST: {ast:#?}");
        }
        "codegen" => {
            let ast =
                compiler::parser::parse_program(args.in_path, &mut lexer).unwrap_or_else(|err| {
                    eprintln!("{err}");
                    process::exit(1);
                });

            let ir = compiler::ir::generate_ir(&ast).unwrap_or_else(|err| {
                eprintln!("{err}");
                process::exit(1);
            });

            println!("IR: {ir:#?}");
        }
        _ => {
            let ast =
                compiler::parser::parse_program(args.in_path, &mut lexer).unwrap_or_else(|err| {
                    eprintln!("{err}");
                    process::exit(1);
                });

            let ir = compiler::ir::generate_ir(&ast).unwrap_or_else(|err| {
                eprintln!("{err}");
                process::exit(1);
            });

            compiler::emit::emit_assembly(args.in_path, args.out_path, &ir).unwrap_or_else(|err| {
                eprintln!("{err}");
                process::exit(1);
            });
        }
    }
}
