//! Module for parsing command-line arguments.

use std::path::{Path, PathBuf};
use std::process;

use crate::report_err;

/// Compiler command-line arguments.
#[derive(Debug)]
pub struct Args {
    /// Name of the program.
    pub program: String,
    /// Compilation phase to terminate at (optional).
    pub stage: String,
    /// Indicates whether the input file should be preprocessed before compiling
    /// (optional).
    pub preprocess: bool,
    /// Input file path (required).
    pub in_path: &'static Path,
    /// Output file path (optional).
    pub out_path: PathBuf,
}

impl Args {
    /// Parses command-line arguments from [`std::env::args()`]. [Exits] on
    /// error with non-zero status.
    ///
    /// [Exits]: std::process::exit
    ///
    /// # Panics
    ///
    /// Will _panic_ if peeked arguments could not be consumed.
    #[must_use]
    pub fn parse() -> Self {
        let mut args = std::env::args().peekable();
        let program = args.next().unwrap_or("cc2".into());

        let mut stage = String::new();
        let mut preprocess = false;
        let mut in_path = String::new();
        let mut out_path = PathBuf::new();

        while let Some(arg) = args.peek() {
            if arg.starts_with('-') {
                let flag_name = args
                    .next()
                    .expect("already peeked the next argument, should be present");

                if let Some(flag) = PROGRAM_FLAGS
                    .iter()
                    .find(|flag| flag.names.contains(&flag_name.as_str()))
                {
                    match flag.names {
                        ["-s", "--stage"] => match args.peek().map(|s| &**s) {
                            Some("lex" | "parse" | "ir" | "mir" | "asm") => {
                                stage = args
                                    .next()
                                    .expect("already peeked the next argument, should be present");
                            }
                            Some(s) => {
                                report_err!(&program, "invalid stage: '{s}'");
                                print_usage(&program);
                            }
                            None => {
                                report_err!(&program, "missing stage name after '-s'|'--stage'");
                                print_usage(&program);
                            }
                        },
                        ["-p", "--preprocess"] => preprocess = true,
                        ["-o", "--output"] => {
                            if let Some(path) = args.next() {
                                out_path = PathBuf::from(&path);
                            } else {
                                report_err!(&program, "missing file path after '-o'|'--output'");
                                print_usage(&program);
                            }
                        }
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
                in_path = args
                    .next()
                    .expect("already peeked the next argument, should be present");
            } else {
                report_err!(&program, "invalid argument: '{arg}'");
                print_usage(&program);
            }
        }

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

        if out_path.capacity() == 0 {
            out_path = path.with_extension("s");
        }

        Self {
            program,
            stage,
            preprocess,
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

const PROGRAM_FLAGS: &[Flag] = &[
    Flag {
        names: ["-s", "--stage"],
        description: "          stop after the specified compilation phase and display its output: 'lex', 'parse', 'ir', 'mir', or 'asm'.",
        run: None,
    },
    Flag {
        names: ["-o", "--output"],
        description: "         specify the output file. defaults to the input filename with '.s' extension.",
        run: None,
    },
    Flag {
        names: ["-p", "--preprocess"],
        description: "     run the GCC preprocessor (cpp) on the input file prior to compiling.",
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
        run: Some(|program: &str| -> ! {
            println!(
                "\x1b[1;1m{} - {}\x1b[0m",
                program,
                env!("CARGO_PKG_VERSION")
            );
            process::exit(0);
        }),
    },
];

fn print_usage(program: &str) -> ! {
    eprintln!("\x1b[1;1musage:\x1b[0m");
    eprintln!("      {program} [options] <infile>");
    eprintln!("\x1b[1;1moptions:\x1b[0m");

    for flag in PROGRAM_FLAGS {
        eprintln!("   {}{}", flag.names.join(", "), flag.description);
    }

    process::exit(1);
}
