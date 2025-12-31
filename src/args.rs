//! Parsing for the compiler's command-line arguments.

use std::path::{Path, PathBuf};
use std::process;

use crate::report_err;

/// Compiler command-line arguments.
#[derive(Debug)]
pub struct Args {
    /// Name of the program.
    pub program: String,
    /// Compilation phase to terminate at.
    pub stage: String,
    /// Path to input file (required).
    pub in_path: &'static Path,
    /// Output path for assembly code emission.
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
        let mut in_path = String::new();

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
                            Some("lex") | Some("parse") | Some("ir") | Some("mir")
                            | Some("asm") => {
                                // Already peeked the next argument.
                                stage = args.next().expect("next argument should be present");
                            }
                            Some(s) => {
                                // Already peeked the next argument.
                                report_err!(&program, "invalid stage: '{s}'");
                                print_usage(&program);
                            }
                            None => {
                                report_err!(&program, "missing stage name after '-s'|'--stage'");
                                print_usage(&program);
                            }
                        },
                        ["-o", "--output"] => match args.next() {
                            Some(path) => out_path = PathBuf::from(&path),
                            None => {
                                report_err!(&program, "missing file path after '-o'|'--output'");
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
                    report_err!(&program, "invalid flag: '{flag_name}'");
                    print_usage(&program);
                }
            } else if in_path.is_empty() {
                // Input file can come before or after option flags.
                in_path = args.next().expect("next argument should be present");
            } else {
                report_err!(&program, "invalid argument: '{arg}'");
                print_usage(&program);
            }
        }

        // NOTE: Leaking `in_path` to ensure the input path is available for
        // error reporting during runtime. Could use `PathBuf` instead but the
        // path will not be mutated.
        let path = Path::new(in_path.leak());
        if !path.exists() {
            report_err!(&program, "'{}': no such file or directory", path.display());
            process::exit(1);
        }

        if path == out_path.as_path() {
            report_err!(
                &program,
                "input file '{}' is the same as output file",
                path.display()
            );
            process::exit(1);
        }

        // No output path was provided - use default output path name.
        if out_path.capacity() == 0 {
            out_path = path.with_extension("s");
        }

        Self {
            program,
            stage,
            in_path: path,
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
        description: "          stop after the specified compilation phase: 'lex', 'parse', 'ir', 'mir', or 'asm'.",
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
