//! Compiler Targets
//!
//! Responsible for lowering the target-agnostic intermediate representation
//! (_IR_) to target-specific machine intermediate representation (_MIR_) and
//! assembly emission.

pub mod x86_64;
