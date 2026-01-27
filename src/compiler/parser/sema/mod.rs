//! Semantic Analysis
//!
//! Compiler passes that checks the semantic correctness of an abstract syntax
//! tree (_AST_).

pub mod ctrl_flow;
pub mod labels;
pub mod switches;
pub mod symbols;

pub use ctrl_flow::resolve_escapable_ctrl;
pub use labels::resolve_labels;
pub use switches::resolve_switches;
pub use symbols::resolve_symbols;
