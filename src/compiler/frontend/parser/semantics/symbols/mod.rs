//! Symbol Resolution
//!
//! Performs symbol lookup, lexical scoping, and mapping identifiers to
//! canonical forms.

pub mod resolve;
pub mod scope;
pub mod symbol_table;

pub use resolve::resolve_symbols;
pub use scope::Scope;
pub use symbol_table::{
    Linkage, StorageDuration, SymbolInfo, SymbolState, SymbolTable, convert_bindings_map,
};
