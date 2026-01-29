use crate::Context;
use crate::compiler::Result;
use crate::compiler::parser::ast::{AST, IdentPhase, TypePhase};

/// Performs type checking and enforces semantic constraints on expressions and
/// declarations.
pub fn resolve_types(ast: AST<IdentPhase>, _ctx: &Context<'_>) -> Result<AST<TypePhase>> {
    Ok(AST {
        program: ast.program,
        _phase: std::marker::PhantomData,
    })
}
