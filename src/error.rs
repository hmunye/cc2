//! Macros and result type for formatting and reporting compiler errors.

pub type Result<T> = std::result::Result<T, String>;

/// Report a generic error message, printing to `stderr`.
#[macro_export]
macro_rules! report_err {
    ($program:expr, $($arg:tt)+) => {{
        eprintln!("\x1b[1;1m{}\x1b[0m: \x1b[1;31merror:\x1b[0m {}", $program, format!($($arg)+));
    }};
}

/// Report an error related to a token (with token position and line content),
/// printing to `stderr`.
#[macro_export]
macro_rules! report_token_err {
    ($file:expr, $line:expr, $col:expr, $token:expr, $marker_len:expr, $line_content:expr, $($arg:tt)+) => {{
        eprintln!(
            "\x1b[1;1m{}:{line}:{col}:\x1b[0m \x1b[1;31merror:\x1b[0m {}\n{:>5} | {:<10}\n{:>5} | \x1b[1;31m{:>col$}{}\x1b[0m",
            $file,
            format!($($arg)+),
            $line,
            $line_content,
            "",
            "^",
            "~".repeat($marker_len),
            line = $line,
            col = $col
        );
    }};
}

/// Format a generic error message.
#[macro_export]
macro_rules! fmt_err {
    ($program:expr, $($arg:tt)+) => {{
        format!("\x1b[1;1m{}\x1b[0m: \x1b[1;31merror:\x1b[0m {}", $program, format!($($arg)+))
    }};
}

/// Format an error related to a token (with token position and line content).
#[macro_export]
macro_rules! fmt_token_err {
    ($file:expr, $line:expr, $col:expr, $token:expr, $marker_len:expr, $line_content:expr, $($arg:tt)+) => {{
        format!(
            "\x1b[1;1m{}:{line}:{col}:\x1b[0m \x1b[1;31merror:\x1b[0m {}\n{:>5} | {:<10}\n{:>5} | \x1b[1;31m{:>col$}{}\x1b[0m",
            $file,
            format!($($arg)+),
            $line,
            $line_content,
            "",
            "^",
            "~".repeat($marker_len),
            line = $line,
            col = $col
        )
    }};
}
