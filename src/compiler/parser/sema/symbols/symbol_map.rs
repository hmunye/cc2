use std::collections::HashMap;

use crate::compiler::parser::types::{Type, c_int};

use super::resolve::{BindingInfo, BindingKey};

/// Linkage of a symbol across translation units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Symbol is visible across translation units.
    External,
    /// Symbol is local to the current translation unit.
    Internal,
}

/// Storage duration of a symbol.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum StorageDuration {
    /// Block-scope/local variables.
    Automatic,
    /// File-scope or `static` local variables.
    Static,
}

/// Declaration state of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolState {
    /// Declaration only.
    Declared,
    /// Tentative definition.
    Tentative,
    /// Fully defined without constant initializer (e.g., functions, automatic
    /// variables).
    Defined,
    /// Fully defined with constant initializer (e.g., static storage duration).
    ConstDefined(c_int),
}

/// Resolved information about a canonical symbol.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub state: SymbolState,
    pub linkage: Option<Linkage>,
    pub ty: Type,
    pub duration: Option<StorageDuration>,
}

/// Mapping of canonical identifier to symbol information.
pub type SymbolMap = HashMap<String, SymbolInfo>;

/// Converts a map of scoped symbol bindings into a map of canonical symbols
/// with their resolved information.
#[must_use]
pub fn convert_bindings_map<S: std::hash::BuildHasher>(
    binding_map: HashMap<BindingKey, BindingInfo, S>,
) -> SymbolMap {
    let mut sym_map = SymbolMap::new();

    for bind_info in binding_map.into_values() {
        sym_map.insert(
            bind_info.canonical,
            SymbolInfo {
                state: bind_info.state,
                linkage: bind_info.linkage,
                ty: bind_info.ty,
                duration: bind_info.duration,
            },
        );
    }

    sym_map
}
