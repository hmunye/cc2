//! Code Emission
//!
//! Compiler pass responsible for emitting textual assembly from the
//! _x86-64_ machine intermediate representation (_MIR_) in a specific syntax.

pub mod gas_x86_64_linux;

pub use gas_x86_64_linux::emit_gas_x86_64_linux;
