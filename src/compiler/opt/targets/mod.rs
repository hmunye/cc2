//! Machine-dependent Optimization
//!
//! Target-specific passes that operate on the machine intermediate
//! representation (_MIR_) to improve code generation and efficiency.

pub mod x86_64;

pub use x86_64::optimize_x86_64_mir;
