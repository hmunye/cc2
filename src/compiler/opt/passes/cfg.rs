//! Control Flow Graph
//!
//! Constructs and analyzes the control flow graph (CFG) for functions in an
//! intermediate representation (_IR_), representing basic blocks and control
//! flow edges, used for intraprocedural optimizations and analysis.

use crate::compiler::ir::Function;

/// Control Flow Graph (_CFG_) for a given _IR_ function.
#[derive(Debug)]
pub struct CFG {}

impl CFG {
    /// Constructs a new `CFG` for a given _IR_ function.
    #[inline]
    #[must_use]
    pub const fn new(_f: &Function<'_>) -> Self {
        Self {}
    }

    /// Applies optimizations to the _IR_ function using the optimized control
    /// flow graph. Returns `true` if changes were made, indicating further
    /// optimizations are possible.
    #[inline]
    #[must_use]
    pub const fn apply(&mut self, _f: &mut Function<'_>) -> bool {
        false
    }
}
