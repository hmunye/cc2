//! Tiny C Compiler (subset of _C17_).

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::use_self)]
#![allow(clippy::redundant_else)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![warn(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod args;
pub mod compiler;
pub mod error;

pub type Result<T> = std::result::Result<T, String>;

use std::io::{self, Read, Write};
use std::ops::Range;
use std::path::Path;
use std::{env, fs, process};

/// Information about the current program context.
#[derive(Debug)]
pub struct Context<'a> {
    /// Name of the program.
    pub program: &'a str,
    /// Path of the input _C_ file.
    pub in_path: &'static Path,
    /// Input file bytes.
    pub src: &'a [u8],
}

impl Context<'_> {
    /// Returns the UTF-8 representation for the given `range` from source.
    ///
    /// # Panics
    ///
    /// Panics if `range` is not valid UTF-8.
    #[inline]
    #[must_use]
    pub fn src_slice(&self, range: Range<usize>) -> &str {
        std::str::from_utf8(&self.src[range]).expect("source should only contain ASCII bytes")
    }
}

fn main() {
    let args = args::Args::parse();

    let mut f = if args.preprocess {
        preprocess_input(&args).unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        })
    } else {
        fs::File::open(args.in_path).unwrap_or_else(|err| {
            report_err!(
                &args.program,
                "failed to open input file '{}': {err}",
                args.in_path.display()
            );
            process::exit(1);
        })
    };

    let metadata = f.metadata().unwrap_or_else(|err| {
        report_err!(&args.program, "failed to query input file: {err}");
        process::exit(1);
    });

    let mut src = Vec::with_capacity(metadata.len() as usize);
    f.read_to_end(&mut src).unwrap_or_else(|err| {
        report_err!(&args.program, "failed to read input file: {err}");
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
            let ast = compiler::parser::parse_ast(&ctx, lexer.peekable()).unwrap_or_else(|err| {
                eprintln!("{err}");
                process::exit(1);
            });

            print!("{ast}");
        }
        "ir" => {
            let ast = compiler::parser::parse_ast(&ctx, lexer.peekable()).unwrap_or_else(|err| {
                eprintln!("{err}");
                process::exit(1);
            });

            let ir = compiler::ir::generate_ir(&ast);

            print!("{ir}");
        }
        "mir" => {
            let ast = compiler::parser::parse_ast(&ctx, lexer.peekable()).unwrap_or_else(|err| {
                eprintln!("{err}");
                process::exit(1);
            });

            let ir = compiler::ir::generate_ir(&ast);
            let mir = compiler::mir::generate_x86_64_mir(&ir);

            print!("{mir}");
        }
        stage => {
            let ast = compiler::parser::parse_ast(&ctx, lexer.peekable()).unwrap_or_else(|err| {
                eprintln!("{err}");
                process::exit(1);
            });

            let ir = compiler::ir::generate_ir(&ast);
            let mir = compiler::mir::generate_x86_64_mir(&ir);

            let writer: Box<dyn Write> = if stage == "asm" {
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

            compiler::emit::emit_gas_x86_64_linux(&ctx, &mir, writer).unwrap_or_else(|err| {
                report_err!(args.program, "failed to emit assembly: {err}");
                process::exit(1);
            });
        }
    }
}

/// Perform preprocessing on a _C_ source file (e.g., expanding macros, handling
/// include directives, removing comments), returning a file handle.
///
/// # Errors
///
/// Returns an error if a temporary file cannot be created/opened or
/// preprocessing fails.
fn preprocess_input(args: &args::Args) -> Result<fs::File> {
    let tmp_dir = env::temp_dir();
    let tmp_path = tmp_dir
        .join(
            args.in_path
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("input.c")),
        )
        .with_extension("i");

    let tmp_file = fs::File::options()
        .read(true)
        .write(true)
        .truncate(true)
        .create(true)
        .open(&tmp_path)
        .map_err(|err| {
            fmt_err!(
                args.program,
                "failed to create/open preprocessed file '{}': {err}",
                tmp_path.display()
            )
        })?;

    let output = process::Command::new("cpp")
        .arg(args.in_path)
        .arg("-o")
        .arg(tmp_path)
        .output()
        .map_err(|err| {
            fmt_err!(
                args.program,
                "failed to preprocess input file '{}': {err}",
                args.in_path.display()
            )
        })?;

    if !output.status.success() {
        return Err(fmt_err!(args.program, "{:?}", &output.stderr));
    }

    Ok(tmp_file)
}
