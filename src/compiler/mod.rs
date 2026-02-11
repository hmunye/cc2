//! Multi-stage pipeline for compiling a _C_ translation unit into textual
//! assembly.

pub mod driver;
pub mod emit;
pub mod ir;
pub mod lexer;
pub mod mir;
pub mod opt;
pub mod parser;

pub use driver::{Context, Result};
