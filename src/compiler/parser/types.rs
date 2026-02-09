/// Equivalent to C’s signed integer (`int`) type.
#[allow(non_camel_case_types)]
pub type c_int = i32;

/// Type specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Func { params: usize },
}
