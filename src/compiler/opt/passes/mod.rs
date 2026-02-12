//! Machine-independent Optimization
//!
//! Optimization passes that transform an intermediate representation (_IR_) in
//! a target-agnostic way.

pub mod cfg;
pub mod copy_prop;
pub mod dse;
pub mod fold;
pub mod optimize;
pub mod uce;

pub use copy_prop::propagate_copy;
pub use dse::dead_store;
pub use fold::{fold_ir_const, try_fold_ast};
pub use optimize::optimize_ir;
pub use uce::unreachable_code;
