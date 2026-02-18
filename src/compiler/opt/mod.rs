//! Compiler Optimization
//!
//! Compiler's optimization logic, including passes over the intermediate
//! representation (_IR_) and machine-specific transformations, which aim to
//! improve performance, reduce redundancy, and generate more efficient code
//! without changing observable behavior.

pub mod analysis;
pub mod cfg;
pub mod passes;
pub mod target;

pub use cfg::{Block, CFG};
