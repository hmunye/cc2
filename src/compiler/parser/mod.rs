//! Syntax and Semantics
//!
//! Compiler passes responsible for building and analyzing the program’s
//! abstract representation.

pub mod ast;
pub mod sema;

pub use ast::parse_program;
