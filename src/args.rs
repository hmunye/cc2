//! Module for parsing command-line arguments.

use std::path::{Path, PathBuf};
use std::process;

use crate::report_err;

/// Optimization flags.
#[derive(Debug, Default)]
pub struct Opts {
    /// Constant folding optimization (can be enabled explicitly, overrides
    /// `opt_level` preset).
    pub fold: bool,
    /// Copy propagation optimization (can be enabled explicitly, overrides
    /// `opt_level` preset).
    pub copy_prop: bool,
    /// Unreachable code elimination optimization (can be enabled explicitly,
    /// overrides `opt_level` preset).
    pub uce: bool,
    /// Dead-store elimination optimization (can be enabled explicitly,
    /// overrides `opt_level` preset).
    pub dse: bool,
}

impl Opts {
    /// Returns `true` if any machine-independent optimization passes are
    /// enabled.
    #[inline]
    #[must_use]
    pub const fn any_passes_enabled(&self) -> bool {
        self.fold || self.copy_prop || self.uce || self.dse
    }
}

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
    /// Optimizations available to the compiler.
    pub opts: Opts,
    /// Input file path (required).
    pub in_path: &'static Path,
    /// Output file path (optional).
    pub out_path: PathBuf,
}

impl Args {
    /// Parses command-line arguments from `std::env::args()`. [Exits] on error
    /// with non-zero status.
    ///
    /// [Exits]: std::process::exit
    ///
    /// # Panics
    ///
    /// Panics if a peeked argument could not be consumed.
    #[must_use]
    pub fn parse() -> Self {
        let mut args = std::env::args().peekable();
        let program = args.next().unwrap_or_else(|| "cc2".into());

        let mut stage = String::new();
        let mut preprocess = false;
        let mut opts = Opts::default();
        let mut in_path = String::new();
        let mut out_path = PathBuf::new();

        while let Some(arg) = args.peek() {
            if arg.starts_with('-') {
                let flag_name = args
                    .next()
                    .expect("already peeked the next argument, should be present");

                if let Some(level_str) = flag_name.strip_prefix("-O") {
                    // '-O' implies '-O1': enable all optimizations.
                    if level_str.is_empty() {
                        opts.fold = true;
                        opts.copy_prop = true;
                        opts.uce = true;
                        opts.dse = true;
                    } else {
                        match level_str.parse::<u8>() {
                            Ok(0) => {
                                opts.fold = false;
                                opts.copy_prop = false;
                                opts.uce = false;
                                opts.dse = false;
                            }
                            Ok(1) => {
                                opts.fold = true;
                                opts.copy_prop = true;
                                opts.uce = true;
                                opts.dse = true;
                            }
                            _ => {
                                report_err!(
                                    &program,
                                    "invalid optimization level: expected '-O0' or '-O1'"
                                );
                                print_usage(&program);
                            }
                        }
                    }

                    continue;
                }

                if let Some(flag) = PROGRAM_FLAGS
                    .iter()
                    .find(|flag| !flag_name.is_empty() && flag.names.contains(&flag_name.as_str()))
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
                        ["", "--fold"] => opts.fold = true,
                        ["", "--copy-prop"] => opts.copy_prop = true,
                        ["", "--uce"] => opts.uce = true,
                        ["", "--dse"] => opts.dse = true,
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
            opts,
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
        description: "         specify the output file. default is the input filename with '.s' extension.",
        run: None,
    },
    Flag {
        names: ["-p", "--preprocess"],
        description: "     run the GCC preprocessor (cpp) on the input file prior to compiling.",
        run: None,
    },
    Flag {
        names: ["", "-O"],
        description: "                   set optimization level preset ('-O0' = none, '-O' or '-O1' = all).",
        run: None,
    },
    Flag {
        names: ["", "--fold"],
        description: "               enable constant folding optimization.",
        run: None,
    },
    Flag {
        names: ["", "--copy-prop"],
        description: "          enable copy propagation optimization.",
        run: None,
    },
    Flag {
        names: ["", "--uce"],
        description: "                enable unreachable code elimination optimization.",
        run: None,
    },
    Flag {
        names: ["", "--dse"],
        description: "                enable dead-store elimination optimization.",
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
        if flag.names[0].is_empty() {
            eprintln!("   {}{}", flag.names[1], flag.description);
        } else {
            eprintln!("   {}{}", flag.names.join(", "), flag.description);
        }
    }

    process::exit(1);
}
