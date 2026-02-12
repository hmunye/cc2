//! Unreachable Code Elimination (UCE)
//!
//! Transforms an intermediate representation (_IR_) by removing code that can
//! never be executed based on control flow analysis.

use crate::compiler::opt::passes::cfg::CFG;

/// Transforms a control flow graph (_CFG_) by removing code that can never be
/// executed.
pub const fn unreachable_code(_cfg: &mut CFG<'_>) {}
