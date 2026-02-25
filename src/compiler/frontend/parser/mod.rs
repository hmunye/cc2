//! Syntax and Semantics
//!
//! Compiler passes responsible for building and analyzing a _C_ program’s
//! abstract representation.

pub mod ast;
pub mod semantics;
pub mod types;

pub use semantics::symbols;
