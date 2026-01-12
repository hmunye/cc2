//! Multi-stage pipeline for compiling _C_ translation unit into textual
//! assembly.

pub mod emit;
pub mod ir;
pub mod lexer;
pub mod mir;
pub mod parser;

/// A convenience wrapper around `Result` for `Result<Token, String>`.
pub type Result<T> = std::result::Result<T, String>;
