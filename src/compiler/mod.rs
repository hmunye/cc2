//! Multi-stage pipeline for compiling C source code into textual assembly.

pub mod asm;
pub mod emit;
pub mod lexer;
pub mod parser;

pub use lexer::Lexer;
