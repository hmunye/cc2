//! Semantic Analysis
//!
//! Compiler passes that checks the semantic correctness of an abstract syntax
//! tree (_AST_).

pub mod ctrl;
pub mod labels;
pub mod switches;
pub mod symbols;
pub mod typeck;

pub use ctrl::resolve_escapable_ctrl;
pub use labels::resolve_labels;
pub use switches::resolve_switches;
pub use symbols::{SymbolTable, resolve_symbols};
pub use typeck::resolve_types;
