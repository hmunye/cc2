//! Macros for reporting compiler errors.

/// Report a generic error message, printing to `stderr`.
#[macro_export]
macro_rules! report_err {
    ($program:expr, $($arg:tt)+) => {{
        eprintln!("\x1b[1;1m{}\x1b[0m: \x1b[1;31merror:\x1b[0m {}", $program, format!($($arg)+));
    }};
}

/// Report a generic error message with context, printing to `stderr`.
#[macro_export]
macro_rules! report_ctx_err {
    ($file:expr, $line:expr, $col:expr, $($arg:tt)+) => {{
        eprintln!(
            "\x1b[1;1m{}:{}:{}:\x1b[0m \x1b[1;31merror:\x1b[0m {}",
            $file, $line, $col, format!($($arg)+)
        );
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

/// Format a generic error message into a `String`.
#[macro_export]
macro_rules! fmt_err {
    ($program:expr, $($arg:tt)+) => {{
        format!("\x1b[1;1m{}\x1b[0m: \x1b[1;31merror:\x1b[0m {}", $program, format!($($arg)+))
    }};
}

/// Format a generic error message with context into a `String`.
#[macro_export]
macro_rules! fmt_ctx_err {
    ($file:expr, $line:expr, $col:expr, $($arg:tt)+) => {{
        format!(
            "\x1b[1;1m{}:{}:{}:\x1b[0m \x1b[1;31merror:\x1b[0m {}",
            $file, $line, $col, format!($($arg)+)
        )
    }};
}

/// Format an error related to a token (with token position and line content),
/// into a String.
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
