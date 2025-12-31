//! Tiny C Compiler (cc2)

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

pub mod args;
pub mod compiler;
pub mod error;

use std::io::{self, Read, Write};
use std::path::Path;
use std::{env, fs, process};

/// Information about the current program.
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
    let args = args::Args::parse();
    let mut f = preprocess_input(&args);

    let metadata = f.metadata().unwrap_or_else(|err| {
        report_err!(&args.program, "failed to query preprocessed file: {err}");
        process::exit(1);
    });

    let mut src = Vec::with_capacity(metadata.len() as usize);
    f.read_to_end(&mut src).unwrap_or_else(|err| {
        report_err!(
            &args.program,
            "failed to read from preprocessed file: {err}"
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
            println!("{ast:#?}");
        }
        "ir" => {
            let ast = compiler::parser::parse_program(&ctx, &mut lexer);
            let ir = compiler::ir::generate_ir(&ast);
            print!("{ir}");
        }
        "mir" => {
            let ast = compiler::parser::parse_program(&ctx, &mut lexer);
            let ir = compiler::ir::generate_ir(&ast);
            let mir = compiler::mir::generate_mir(&ir);
            print!("{mir}");
        }
        stage => {
            let ast = compiler::parser::parse_program(&ctx, &mut lexer);
            let ir = compiler::ir::generate_ir(&ast);
            let mir = compiler::mir::generate_mir(&ir);

            let output: Box<dyn Write> = if stage == "asm" {
                // Print to `stdout`, the assembly that would have been emitted.
                Box::new(io::stdout().lock())
            } else {
                let f = fs::File::create(ctx.out_path).unwrap_or_else(|err| {
                    report_err!(
                        &ctx.program,
                        "failed to create output file '{}': {err}",
                        ctx.in_path.display()
                    );
                    process::exit(1);
                });
                Box::new(f)
            };

            compiler::emit::emit_asm(&ctx, &mir, output);
        }
    }
}

/// Perform preprocessing on the input _C_ source code, expanding macros,
/// handling include directives, removing comments, etc., returning the file
/// handle of the preprocessed file. [Exits] on error with non-zero status.
///
/// [Exits]: std::process::exit
fn preprocess_input(args: &args::Args) -> fs::File {
    let tmp_dir = env::temp_dir();
    let tmp_path = tmp_dir.join("input.i");

    let tmp_file = fs::File::options()
        .read(true)
        .write(true)
        .truncate(true)
        .create(true)
        .open(&tmp_path)
        .unwrap_or_else(|err| {
            report_err!(&args.program, "failed to create preprocessed file: {err}");
            process::exit(1);
        });

    let output = process::Command::new("gcc")
        .arg("-E") // Run only the preprocessor (cpp).
        .arg("-P") // Omit linemarkers.
        .arg(args.in_path)
        .arg("-o")
        .arg(tmp_path)
        .output()
        .unwrap_or_else(|err| {
            report_err!(
                &args.program,
                "failed to preprocess input file '{}': {err}",
                args.in_path.display()
            );
            process::exit(1);
        });

    if !output.status.success() {
        io::stderr()
            .write_all(&output.stderr)
            .expect("failed to write gcc 'stderr' contents");
        process::exit(output.status.code().unwrap_or(1));
    }

    tmp_file
}
