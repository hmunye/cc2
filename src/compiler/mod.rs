//! Multi-stage pipeline for compiling a _C_ translation unit into textual
//! assembly.

pub mod driver;
pub mod frontend;
pub mod ir;
pub mod opt;
pub mod targets;

pub use driver::Context;
