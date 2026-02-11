//! Symbol Resolution
//!
//! Performs symbol lookup, lexical scoping, and mapping identifiers to
//! canonical forms.

pub mod resolve;
pub mod scope;
pub mod symbol_map;

pub use resolve::resolve_symbols;
pub use scope::Scope;
pub use symbol_map::{
    Linkage, StorageDuration, SymbolInfo, SymbolMap, SymbolState, convert_bindings_map,
};
