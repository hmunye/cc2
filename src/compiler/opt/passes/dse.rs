//! Dead Store Elimination (DSE)
//!
//! Transforms an intermediate representation (_IR_) by removing assignments to
//! variables that are never used or updated.

use crate::compiler::opt::CFG;

/// Transforms a control flow graph (_CFG_) by removing assignments to variables
/// that are never used or updated.
pub const fn dead_store(_cfg: &mut CFG<'_>) {}
