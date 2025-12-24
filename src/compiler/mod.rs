//! Multi-stage pipeline for compiling C source code into textual assembly.

pub mod emit;
pub mod ir;
pub mod lexer;
pub mod parser;

pub use lexer::Lexer;
