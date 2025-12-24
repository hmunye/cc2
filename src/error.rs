//! Error types and macros for the compiler.

/// Prints the provided error message to `stderr`.
#[macro_export]
macro_rules! print_err {
    // General error reporting: prints program name and error message.
    ($program:expr, $($arg:tt)+) => {{
        eprintln!("\x1b[1;1m{}\x1b[0m: \x1b[1;31merror:\x1b[0m {}", $program, format!($($arg)+));
    }};
}
