//! Control-Flow Graph
//!
//! Implementation of control-flow graph for analysis of functions in an
//! intermediate representation (_IR_), representing basic blocks and control
//! flow edges, used in intraprocedural optimizations.

pub mod graph;
pub mod iter;

pub use graph::{Block, CFG, CFGInstruction};
