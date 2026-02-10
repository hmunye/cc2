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

impl SymbolState {
    /// Returns `true` if the symbol state represents a definition.
    #[inline]
    #[must_use]
    pub const fn is_definition(&self) -> bool {
        !matches!(self, SymbolState::Declared)
    }

    /// Returns `true` if the new state promotes this existing state. Symbol
    /// states may promote upward (e.g., declared -> tentative -> defined, etc.)
    /// and are never demoted.
    #[inline]
    #[must_use]
    pub const fn promotes(&self, new: &SymbolState) -> bool {
        match (self, new) {
            (
                SymbolState::Declared,
                SymbolState::Tentative | SymbolState::Defined | SymbolState::ConstDefined(_),
            )
            | (SymbolState::Tentative, SymbolState::Defined | SymbolState::ConstDefined(_)) => true,
            // All other cases are not promotions.
            _ => false,
        }
    }
}

/// Resolved information about a canonical symbol.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub state: SymbolState,
    pub linkage: Option<Linkage>,
    pub ty: Type,
    pub duration: Option<StorageDuration>,
    /// Has this symbol been emitted into _IR_ instructions.
    pub emitted: bool,
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
        sym_map
            .entry(bind_info.canonical)
            // Need to modify existing entries to avoid order-dependent
            // overwrites, since `binding_map` may contains the same symbol
            // declared in different scopes, and `HashMap` iteration is
            // non-deterministic.
            .and_modify(|existing| {
                // Never overwrite an existing definition with a declaration.
                if !existing.state.is_definition() {
                    existing.state = bind_info.state;
                }

                existing.linkage = existing.linkage.or(bind_info.linkage);
                existing.duration = existing.duration.or(bind_info.duration);
                existing.ty = bind_info.ty;
            })
            .or_insert_with(|| SymbolInfo {
                state: bind_info.state,
                linkage: bind_info.linkage,
                ty: bind_info.ty,
                duration: bind_info.duration,
                emitted: false,
            });
    }

    sym_map
}
