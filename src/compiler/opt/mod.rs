//! Optimization Passes
//!
//! Compiler passes that transform a parsed _AST_ or _IR_ to improve
//! performance, reduce redundancy, or otherwise optimize code without changing
//! its observable behavior.

pub mod const_folding;

pub use const_folding::fold_constants;
