//! Module for parsing command-line arguments passed to the compiler.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process;

use crate::print_err;

/// Compiler command-line arguments.
#[derive(Debug)]
pub struct Args {
    /// Name of the program.
    pub program: String,
    /// Compilation phase to terminate at (lexical analysis, parsing, code
    /// generation).
    ///
    /// Defaults to invoking full compilation process.
    pub stage: String,
    /// Input file containing C source code (required).
    pub in_file: File,
    /// Path to input file (used in error reporting).
    pub in_path: &'static Path,
    /// Output path for assembly code emission (defaults to input file path with
    /// `.s` extension).
    pub out_path: PathBuf,
}

impl Args {
    /// Parses command-line arguments from `std::env::args()`, [exiting] on
    /// error.
    ///
    /// [exiting]: std::process::exit
    pub fn parse() -> Self {
        let mut args = std::env::args().peekable();
        let program = args.next().unwrap_or("cc2".into());

        let mut stage = String::new();
        let mut out_path = PathBuf::new();

        while let Some(arg) = args.peek() {
            if arg.starts_with("-") {
                // Already peeked the next argument.
                let flag_name = args.next().expect("next argument should be present");

                if let Some(flag) = FLAG_REGISTRY
                    .iter()
                    .find(|flag| flag.names.contains(&flag_name.as_str()))
                {
                    match flag.names {
                        ["-s", "--stage"] => match args.peek().map(|s| &**s) {
                            Some("lex") | Some("parse") | Some("codegen") => {
                                // Already peeked the next argument.
                                stage = args.next().expect("next argument should be present");
                            }
                            Some(s) => {
                                // Already peeked the next argument.
                                print_err!(&program, "invalid stage: '{s}'");
                                print_usage(&program);
                            }
                            None => {
                                print_err!(&program, "missing stage name after '-s'|'--stage'");
                                print_usage(&program);
                            }
                        },
                        ["-o", "--output"] => match args.next() {
                            Some(path) => out_path = PathBuf::from(&path),
                            None => {
                                print_err!(&program, "missing file name after '-o'|'--output'");
                                print_usage(&program);
                            }
                        },
                        _ => {
                            if let Some(run) = flag.run {
                                run(&program);
                            }
                        }
                    }
                } else {
                    print_err!(&program, "invalid flag '{flag_name}'");
                    print_usage(&program);
                }
            } else {
                // No remaining flags to process.
                break;
            }
        }

        // Input file should come after all flags have been processed.
        let Some(file_path) = args.next() else {
            print_err!(&program, "no input file");
            print_usage(&program);
        };

        // NOTE: Leaking `file_path` to ensure the input path is available for
        // error reporting throughout the runtime. Could use PathBuf instead.
        let in_path = Path::new(file_path.leak());

        let in_file = File::open(in_path).unwrap_or_else(|err| {
            print_err!(&program, "failed to open file: {err}");
            process::exit(1);
        });

        // Indicates no output path was provided.
        if out_path.capacity() == 0 {
            out_path = in_path.with_extension("s");
        }

        Self {
            program,
            stage,
            in_file,
            in_path,
            out_path,
        }
    }
}

struct Flag {
    names: [&'static str; 2],
    description: &'static str,
    run: Option<fn(&str) -> !>,
}

const FLAG_REGISTRY: &[Flag] = &[
    Flag {
        names: ["-s", "--stage"],
        description: "          stop after the specified compilation phase: 'lex', 'parse', or 'codegen'.",
        run: None,
    },
    Flag {
        names: ["-o", "--output"],
        description: "         specify the output file. defaults to input path with '.s' extension",
        run: None,
    },
    Flag {
        names: ["-h", "--help"],
        description: "           print this summary.",
        run: Some(print_usage),
    },
    Flag {
        names: ["-v", "--version"],
        description: "        show version.",
        run: Some(print_version),
    },
];

/// Prints the usage information for the program, exiting with a non-zero
/// status.
pub fn print_usage(program: &str) -> ! {
    eprintln!("\x1b[1;1musage:\x1b[0m");
    eprintln!("      {program} [options] <infile>");
    eprintln!("\x1b[1;1moptions:\x1b[0m");

    for flag in FLAG_REGISTRY {
        eprintln!("   {}  {}", flag.names.join(", "), flag.description);
    }

    process::exit(1);
}

fn print_version(program: &str) -> ! {
    println!(
        "\x1b[1;1m{} - {}\x1b[0m",
        program,
        env!("CARGO_PKG_VERSION")
    );
    process::exit(0);
}
