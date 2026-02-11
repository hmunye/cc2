//! _C_ Types
//!
//! Primitive types that match _C_ type definitions (_C17_).

use std::fmt;

/// Equivalent to _C_ signed integer (`int`) type.
#[allow(non_camel_case_types)]
pub type c_int = i32;

/// Type specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Func { params: usize },
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Func { .. } => write!(f, "Fn"),
        }
    }
}
