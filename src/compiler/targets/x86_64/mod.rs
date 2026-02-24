//! Compiler Backend (_x86-64_)
//!
//! Responsible for the compilation of the target-agnostic intermediate
//! representation (_IR_) into _x86-64_ machine intermediate representation
//! (_MIR_) and assembly emission.

pub mod asm;
pub mod mir;

pub use mir::{
    BinaryOperator, Function, Instruction, Item, MIRX86, Operand, Reg, UnaryOperator,
    generate_x86_64_mir,
};
