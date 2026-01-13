//! Tiny C Compiler (cc2)

#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

pub mod args;
pub mod compiler;
pub mod error;

use std::io::{self, Read, Write};
use std::ops::Range;
use std::path::Path;
use std::{env, fs, process};

/// Information about the current program context.
#[derive(Debug)]
pub struct Context<'a> {
    /// Name of the program.
    pub program: &'a str,
    /// Path of the input file.
    pub in_path: &'static Path,
    /// Reference to input file bytes.
    pub src: &'a [u8],
}

impl<'a> Context<'a> {
    /// Returns the UTF-8 representation for the given byte range from `src`.
    #[inline]
    pub fn src_slice(&self, range: Range<usize>) -> &str {
        std::str::from_utf8(&self.src[range])
            .expect("any range of ASCII bytes should be valid UTF-8")
    }
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
        src: &src,
    };

    let lexer = compiler::lexer::Lexer::new(&ctx);

    match args.stage.as_str() {
        "lex" => {
            print!("{lexer}");
        }
        "parse" => {
            let ast = compiler::parser::parse_program(&ctx, lexer.peekable());
            println!("{ast:#?}");
        }
        "ir" => {
            let ast = compiler::parser::parse_program(&ctx, lexer.peekable());
            let ir = compiler::ir::generate_ir(&ast);
            print!("{ir}");
        }
        "mir" => {
            let ast = compiler::parser::parse_program(&ctx, lexer.peekable());
            let ir = compiler::ir::generate_ir(&ast);
            let mir = compiler::mir::generate_mir(&ir);
            print!("{mir}");
        }
        stage => {
            let ast = compiler::parser::parse_program(&ctx, lexer.peekable());
            let ir = compiler::ir::generate_ir(&ast);
            let mir = compiler::mir::generate_mir(&ir);

            let output: Box<dyn Write> = if stage == "asm" {
                Box::new(io::stdout().lock())
            } else {
                let f = fs::File::create(&args.out_path).unwrap_or_else(|err| {
                    report_err!(
                        &ctx.program,
                        "failed to create output file '{}': {err}",
                        ctx.in_path.display()
                    );
                    process::exit(1);
                });
                Box::new(f)
            };

            compiler::emit::emit_gas_x86_64_linux(&ctx, &mir, output);
        }
    }
}

/// Perform preprocessing on the input _C_ translation unit (expanding macros,
/// handling include directives, removing comments, etc.), returning a file
/// handle. [Exits] on error with non-zero status.
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
            report_err!(
                &args.program,
                "failed to create preprocessed file '{}': {err}",
                tmp_path.display()
            );
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
        eprintln!("{:?}", &output.stderr);
        process::exit(output.status.code().unwrap_or(1));
    }

    tmp_file
}
