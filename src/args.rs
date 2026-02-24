//! Module for parsing command-line arguments.

use std::path::{Path, PathBuf};
use std::process;

use crate::report_err;

/// Compiler optimization flags.
#[derive(Debug, Default)]
pub struct Opts {
    /// Constant folding optimization.
    pub fold: bool,
    /// Copy propagation optimization.
    pub copy_prop: bool,
    /// Unreachable code elimination optimization.
    pub uce: bool,
    /// Dead-store elimination optimization.
    pub dse: bool,
    /// Register allocation optimization.
    pub reg_alloc: bool,
    /// Register coalescing optimization.
    pub coalesce: bool,
}

impl Opts {
    /// Returns `true` if any machine-independent optimization passes are
    /// enabled.
    #[inline]
    #[must_use]
    pub const fn any_passes_enabled(&self) -> bool {
        self.fold || self.copy_prop || self.uce || self.dse
    }

    /// Returns `true` if any machine-dependent optimization passes are enabled.
    #[inline]
    #[must_use]
    pub const fn any_target_passes_enabled(&self) -> bool {
        self.reg_alloc || self.coalesce
    }

    /// Enables all optimizations flags.
    #[inline]
    const fn enable(&mut self) {
        self.fold = true;
        self.copy_prop = true;
        self.uce = true;
        self.dse = true;
        self.reg_alloc = true;
        self.coalesce = true;
    }

    /// Disables all optimizations flags.
    #[inline]
    const fn disable(&mut self) {
        self.fold = false;
        self.copy_prop = false;
        self.uce = false;
        self.dse = false;
        self.reg_alloc = false;
        self.coalesce = false;
    }
}

/// Compiler command-line arguments.
#[derive(Debug)]
pub struct Args {
    /// Name of the program.
    pub program: String,
    /// Specified compilation phase to terminate at.
    pub stage: String,
    /// If the input file should be preprocessed before compiling.
    pub preprocess: bool,
    /// Optimizations available to the compiler.
    pub opts: Opts,
    /// Input file path (required).
    pub in_path: &'static Path,
    /// Output file path.
    pub out_path: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            program: String::default(),
            stage: String::default(),
            preprocess: Default::default(),
            opts: Opts::default(),
            in_path: Path::new(""),
            out_path: PathBuf::default(),
        }
    }
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
        let mut args = Args::default();

        let mut cli_args = std::env::args().peekable();

        args.program = cli_args.next().unwrap_or_else(|| "cc2".into());

        while let Some(arg) = cli_args.peek() {
            if arg.starts_with('-') {
                let flag_name = cli_args
                    .next()
                    .expect("already peeked the next argument, should be present");

                if let Some(opt_level) = flag_name.strip_prefix("-O") {
                    // '-O' implies '-O1'.
                    if opt_level.is_empty() {
                        args.opts.enable();
                    } else {
                        match opt_level.parse::<u8>() {
                            Ok(0) => {
                                args.opts.disable();
                            }
                            Ok(1) => {
                                args.opts.enable();
                            }
                            _ => {
                                report_err!(
                                    &args.program,
                                    "invalid optimization level: expected '-O0' or '-O1'"
                                );
                                print_usage(&args.program);
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
                        ["-s", "--stage"] => match cli_args.peek().map(|s| &**s) {
                            Some("lex" | "parse" | "ir" | "mir" | "asm") => {
                                args.stage = cli_args
                                    .next()
                                    .expect("already peeked the next argument, should be present");
                            }
                            Some(s) => {
                                report_err!(&args.program, "invalid stage: '{s}'");
                                print_usage(&args.program);
                            }
                            None => {
                                report_err!(
                                    &args.program,
                                    "missing stage name after '-s'|'--stage'"
                                );
                                print_usage(&args.program);
                            }
                        },
                        ["-p", "--preprocess"] => args.preprocess = true,
                        ["", "--fold"] => args.opts.fold = true,
                        ["", "--copy-prop"] => args.opts.copy_prop = true,
                        ["", "--uce"] => args.opts.uce = true,
                        ["", "--dse"] => args.opts.dse = true,
                        ["", "--reg-alloc"] => args.opts.reg_alloc = true,
                        ["", "--coalesce"] => args.opts.coalesce = true,
                        ["-o", "--output"] => {
                            if let Some(path) = cli_args.next() {
                                args.out_path = PathBuf::from(path);
                            } else {
                                report_err!(
                                    &args.program,
                                    "missing file path after '-o'|'--output'"
                                );
                                print_usage(&args.program);
                            }
                        }
                        _ => {
                            if let Some(run) = flag.run {
                                run(&args.program);
                            }
                        }
                    }
                } else {
                    report_err!(&args.program, "invalid flag: '{flag_name}'");
                    print_usage(&args.program);
                }
            } else if args.in_path.to_str() == Some("") {
                // NOTE: String argument is leaked since it is used throughout
                // the compiler for diagnostics.
                args.in_path = Path::new(
                    cli_args
                        .next()
                        .expect("already peeked the next argument, should be present")
                        .leak(),
                );
            } else {
                report_err!(&args.program, "invalid argument: '{arg}'");
                print_usage(&args.program);
            }
        }

        if !args.in_path.exists() {
            report_err!(
                &args.program,
                "'{}': no such file or directory",
                args.in_path.display()
            );
            process::exit(1);
        }

        if args.in_path == args.out_path.as_path() {
            report_err!(
                &args.program,
                "input file '{}' is the same as output file",
                args.in_path.display()
            );
            process::exit(1);
        }

        if args.out_path.capacity() == 0 {
            args.out_path = args.in_path.with_extension("s");
        }

        args
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
        names: ["", "--reg-alloc"],
        description: "          enable register allocation optimization.",
        run: None,
    },
    Flag {
        names: ["", "--coalesce"],
        description: "           enable register coalescing optimization.",
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
