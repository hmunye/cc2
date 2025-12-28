//! Tiny C Compiler (cc2)

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

pub mod args;
pub mod compiler;
pub mod error;

use std::io::{self, Read, Write};
use std::path::Path;
use std::{fs, process};

/// Context for each compilation phase.
#[derive(Debug)]
pub struct Context<'a> {
    /// Name of the program.
    pub program: &'a str,
    /// Path of the input file.
    pub in_path: &'static Path,
    /// Path of the output file.
    pub out_path: &'a Path,
}

fn main() {
    let mut args = args::Args::parse();
    let metadata = args.in_file.metadata().unwrap_or_else(|err| {
        report_err!(
            &args.program,
            "failed to query input file '{}': {err}",
            args.in_path.display()
        );
        process::exit(1);
    });

    let mut src = Vec::with_capacity(metadata.len() as usize);

    args.in_file.read_to_end(&mut src).unwrap_or_else(|err| {
        report_err!(
            &args.program,
            "failed to read from input file '{}': {err}",
            args.in_path.display()
        );
        process::exit(1);
    });

    let ctx = Context {
        program: &args.program,
        in_path: args.in_path,
        out_path: args.out_path.as_path(),
    };

    let mut lexer = compiler::lexer::Lexer::new(&src);
    lexer.lex(&ctx);

    match args.stage.as_str() {
        "lex" => {
            print!("{lexer}");
        }
        "parse" => {
            let ast = compiler::parser::parse_program(&ctx, &mut lexer);
            print!("{ast}");
        }
        "ir" => {
            let ast = compiler::parser::parse_program(&ctx, &mut lexer);
            let ir = compiler::ir::generate_ir(&ast);
            println!("IR: {ir:#?}");
        }
        "mir" => {
            let ast = compiler::parser::parse_program(&ctx, &mut lexer);
            let ir = compiler::ir::generate_ir(&ast);
            let mir = compiler::mir::generate_mir(&ir);
            println!("MIR: {mir:#?}");
        }
        stage => {
            let ast = compiler::parser::parse_program(&ctx, &mut lexer);
            let ir = compiler::ir::generate_ir(&ast);
            let mir = compiler::mir::generate_mir(&ir);

            let output: Box<dyn Write> = if stage == "asm" {
                Box::new(io::stdout().lock())
            } else {
                Box::new(fs::File::create(ctx.out_path).unwrap_or_else(|err| {
                    report_err!(
                        &ctx.program,
                        "failed to create output file '{}': {err}",
                        ctx.in_path.display()
                    );
                    process::exit(1);
                }))
            };

            compiler::emit::emit_asm(&ctx, &mir, output);
        }
    }
}
