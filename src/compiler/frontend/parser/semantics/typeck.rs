//! Type Check Analysis
//!
//! Semantic analysis pass which performs type checking, enforcing semantic
//! constraints on expressions and declarations within an _AST_.

use crate::compiler::Context;
use crate::compiler::frontend::SymbolTable;
use crate::compiler::frontend::ast::{
    AST, Block, BlockItem, Declaration, Expression, ForInit, Function, IdentPhase, Labeled,
    Statement, StorageClass, TypePhase,
};
use crate::compiler::frontend::types::Type;
use crate::{diag::Result, fmt_token_err};

/// Performs type checking using the provided `symbol_map`, enforcing semantic
/// constraints on expressions and declarations within the given _AST_.
///
/// # Errors
///
/// Returns an error if an identifier is undeclared, used with the wrong type,
/// called as a function incorrectly, etc.
pub fn resolve_types<'a>(
    ast: AST<'a, IdentPhase>,
    ctx: &Context<'_>,
    sym_table: &SymbolTable,
) -> Result<AST<'a, TypePhase>> {
    for decl in &ast.program {
        type_check_declaration(decl, true, ctx, sym_table)?;
    }

    Ok(AST {
        program: ast.program,
        _phase: std::marker::PhantomData,
    })
}

fn type_check_declaration(
    decl: &Declaration<'_>,
    is_file_scope: bool,
    ctx: &Context<'_>,
    sym_table: &SymbolTable,
) -> Result<()> {
    match decl {
        var @ Declaration::Var { .. } => type_check_variable(var, is_file_scope, ctx, sym_table),
        Declaration::Fn(f) => type_check_function(f, ctx, sym_table),
    }
}

fn type_check_function(f: &Function<'_>, ctx: &Context<'_>, sym_table: &SymbolTable) -> Result<()> {
    let Function {
        specs,
        ident,
        body,
        token,
        ..
    } = f;

    let entry = sym_table
        .get(ident)
        .expect("identifier should be in symbol table after symbol resolution");

    if entry.ty != specs.ty {
        let tok_str = format!("{token:?}");
        let line_content = ctx.src_slice(token.loc.line_span.clone());

        return Err(fmt_token_err!(
            token.loc.file_path.display(),
            token.loc.line,
            token.loc.col,
            tok_str,
            tok_str.len() - 1,
            line_content,
            "'{tok_str}' redeclared as different kind of symbol",
        ));
    }

    if let Some(body) = body {
        type_check_block(body, ctx, sym_table)?;
    }

    Ok(())
}

fn type_check_variable(
    var: &Declaration<'_>,
    is_file_scope: bool,
    ctx: &Context<'_>,
    sym_table: &SymbolTable,
) -> Result<()> {
    if let Declaration::Var {
        specs,
        ident,
        init,
        token,
    } = var
    {
        if specs.storage == Some(StorageClass::Extern) && !is_file_scope && init.is_some() {
            let tok_str = format!("{token:?}");
            let line_content = ctx.src_slice(token.loc.line_span.clone());

            return Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                line_content,
                "'{tok_str}' has both 'extern' and initializer",
            ));
        }

        let entry = sym_table
            .get(ident)
            .expect("identifier should be in symbol table after symbol resolution");

        if entry.ty != specs.ty {
            let tok_str = format!("{token:?}");
            let line_content = ctx.src_slice(token.loc.line_span.clone());

            return Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                line_content,
                "'{tok_str}' redeclared as different kind of symbol",
            ));
        }

        if let Some(init) = init {
            type_check_expression(init, ctx, sym_table)?;
        }
    }

    Ok(())
}

fn type_check_block(block: &Block<'_>, ctx: &Context<'_>, sym_table: &SymbolTable) -> Result<()> {
    for block_item in &block.0 {
        match block_item {
            BlockItem::Stmt(stmt) => type_check_statement(stmt, ctx, sym_table)?,
            BlockItem::Decl(decl) => type_check_declaration(decl, false, ctx, sym_table)?,
        }
    }

    Ok(())
}

fn type_check_statement(
    stmt: &Statement<'_>,
    ctx: &Context<'_>,
    sym_table: &SymbolTable,
) -> Result<()> {
    match stmt {
        Statement::Return(expr) | Statement::Expression(expr) => {
            type_check_expression(expr, ctx, sym_table)
        }
        Statement::If {
            cond,
            then,
            opt_else,
        } => {
            type_check_expression(cond, ctx, sym_table)?;
            type_check_statement(then, ctx, sym_table)?;

            if let Some(stmt) = opt_else {
                type_check_statement(stmt, ctx, sym_table)?;
            }

            Ok(())
        }
        Statement::LabeledStatement(labeled) => {
            let stmt = match labeled {
                Labeled::Label { stmt, .. } | Labeled::Default { stmt, .. } => stmt,
                Labeled::Case { expr, stmt, .. } => {
                    type_check_expression(expr, ctx, sym_table)?;
                    stmt
                }
            };

            type_check_statement(stmt, ctx, sym_table)
        }
        Statement::Compound(block) => type_check_block(block, ctx, sym_table),
        Statement::While { cond, stmt, .. }
        | Statement::Do { stmt, cond, .. }
        | Statement::Switch { cond, stmt, .. } => {
            type_check_expression(cond, ctx, sym_table)?;
            type_check_statement(stmt, ctx, sym_table)
        }
        Statement::For {
            init,
            opt_cond,
            opt_post,
            stmt,
            ..
        } => {
            match &**init {
                ForInit::Decl(decl) => match decl {
                    Declaration::Var { specs, token, .. } => {
                        if specs.storage == Some(StorageClass::Static) {
                            let tok_str = format!("{token:?}");
                            let line_content = ctx.src_slice(token.loc.line_span.clone());

                            return Err(fmt_token_err!(
                                token.loc.file_path.display(),
                                token.loc.line,
                                token.loc.col,
                                tok_str,
                                tok_str.len() - 1,
                                line_content,
                                "declaration of static variable '{tok_str}' in 'for' loop initial declaration"
                            ));
                        }

                        type_check_variable(decl, false, ctx, sym_table)?;
                    }
                    Declaration::Fn(_) => panic!(
                        "`for` loop initial declaration should never contain a function declaration"
                    ),
                },
                ForInit::Expr(opt_init) => {
                    if let Some(init) = opt_init {
                        type_check_expression(init, ctx, sym_table)?;
                    }
                }
            }

            if let Some(cond) = opt_cond {
                type_check_expression(cond, ctx, sym_table)?;
            }

            if let Some(post) = opt_post {
                type_check_expression(post, ctx, sym_table)?;
            }

            type_check_statement(stmt, ctx, sym_table)
        }
        Statement::Goto { .. }
        | Statement::Break { .. }
        | Statement::Continue { .. }
        | Statement::Empty => Ok(()),
    }
}

fn type_check_expression(
    expr: &Expression<'_>,
    ctx: &Context<'_>,
    sym_table: &SymbolTable,
) -> Result<()> {
    match expr {
        Expression::Assignment { lvalue, rvalue, .. } => {
            type_check_expression(lvalue, ctx, sym_table)?;
            type_check_expression(rvalue, ctx, sym_table)
        }
        Expression::Unary { expr, .. } => type_check_expression(expr, ctx, sym_table),
        Expression::Binary { lhs, rhs, .. } => {
            type_check_expression(lhs, ctx, sym_table)?;
            type_check_expression(rhs, ctx, sym_table)
        }
        Expression::Conditional {
            cond,
            second,
            third,
        } => {
            type_check_expression(cond, ctx, sym_table)?;
            type_check_expression(second, ctx, sym_table)?;
            type_check_expression(third, ctx, sym_table)
        }
        Expression::Var { ident, token } => {
            let entry = sym_table
                .get(ident)
                .expect("identifier should be in symbol table after symbol resolution");

            if entry.ty != Type::Int {
                let tok_str = format!("{token:?}");
                let line_content = ctx.src_slice(token.loc.line_span.clone());

                return Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    line_content,
                    "incompatible integer conversion initialization",
                ));
            }

            Ok(())
        }
        Expression::FnCall { ident, args, token } => {
            let entry = sym_table
                .get(ident)
                .expect("identifier should be in symbol table after symbol resolution");

            if let Type::Fn {
                param_count: params,
            } = entry.ty
            {
                let args_len = args.len();

                if params != args_len {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    return Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "too few arguments to function '{tok_str}'; expected {params}, have {args_len}",
                    ));
                }

                for expr in args {
                    type_check_expression(expr, ctx, sym_table)?;
                }

                Ok(())
            } else {
                let tok_str = format!("{token:?}");
                let line_content = ctx.src_slice(token.loc.line_span.clone());

                Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    line_content,
                    "called object '{tok_str}' is not a function",
                ))
            }
        }
        Expression::IntConstant(_) => Ok(()),
    }
}
