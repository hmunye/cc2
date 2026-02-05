use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::compiler::parser::ast::{
    AST, Block, BlockItem, Declaration, Expression, ForInit, Function, IdentPhase, Labeled,
    Statement, Type, TypePhase,
};
use crate::{Context, Result, fmt_token_err};

/// Maps a canonical identifier to its type information.
pub type TypeMap<'a> = HashMap<&'a str, Type>;

/// Performs type checking and enforces semantic constraints on expressions and
/// declarations within the given _AST_.
///
/// # Errors
///
/// Returns an error if an identifier is undeclared, used with the wrong type,
/// or called as a function incorrectly.
pub fn resolve_types<'a>(
    ast: AST<'a, IdentPhase>,
    ctx: &Context<'_>,
) -> Result<AST<'a, TypePhase>> {
    fn type_check_function<'a>(
        func: &'a Function<'_>,
        ctx: &Context<'_>,
        type_map: &mut TypeMap<'a>,
    ) -> Result<()> {
        let Function {
            ident,
            params,
            body,
            token,
        } = func;

        let ty = Type::Func {
            params: params.len(),
        };

        match type_map.entry(ident.as_str()) {
            Entry::Occupied(entry) => {
                let entry_ty = entry.get();

                if *entry_ty != ty {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    return Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "conflicting types for '{tok_str}'",
                    ));
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(ty);
            }
        }

        if let Some(body) = body {
            for param in params {
                type_map.insert(param.ident.as_str(), param.ty);
            }

            type_check_block(body, ctx, type_map)?;
        }

        Ok(())
    }

    fn type_check_block<'a>(
        block: &'a Block<'_>,
        ctx: &Context<'_>,
        type_map: &mut TypeMap<'a>,
    ) -> Result<()> {
        for block_item in &block.0 {
            match block_item {
                BlockItem::Stmt(stmt) => type_check_statement(stmt, ctx, type_map)?,
                BlockItem::Decl(decl) => type_check_declaration(decl, ctx, type_map)?,
            }
        }

        Ok(())
    }

    fn type_check_statement<'a>(
        stmt: &'a Statement<'_>,
        ctx: &Context<'_>,
        type_map: &mut TypeMap<'a>,
    ) -> Result<()> {
        match stmt {
            Statement::Return(expr) | Statement::Expression(expr) => {
                type_check_expression(expr, ctx, type_map)
            }
            Statement::If {
                cond,
                then,
                opt_else,
            } => {
                type_check_expression(cond, ctx, type_map)?;
                type_check_statement(then, ctx, type_map)?;

                if let Some(stmt) = opt_else {
                    type_check_statement(stmt, ctx, type_map)?;
                }

                Ok(())
            }
            Statement::LabeledStatement(labeled) => {
                let stmt = match labeled {
                    Labeled::Label { stmt, .. } | Labeled::Default { stmt, .. } => stmt,
                    Labeled::Case { expr, stmt, .. } => {
                        type_check_expression(expr, ctx, type_map)?;
                        stmt
                    }
                };

                type_check_statement(stmt, ctx, type_map)
            }
            Statement::Compound(block) => type_check_block(block, ctx, type_map),
            Statement::While { cond, stmt, .. }
            | Statement::Do { stmt, cond, .. }
            | Statement::Switch { cond, stmt, .. } => {
                type_check_expression(cond, ctx, type_map)?;
                type_check_statement(stmt, ctx, type_map)
            }
            Statement::For {
                init,
                opt_cond,
                opt_post,
                stmt,
                ..
            } => {
                match &**init {
                    ForInit::Decl(decl) => {
                        type_check_declaration(decl, ctx, type_map)?;
                    }
                    ForInit::Expr(opt_init) => {
                        if let Some(init) = opt_init {
                            type_check_expression(init, ctx, type_map)?;
                        }
                    }
                }

                if let Some(cond) = opt_cond {
                    type_check_expression(cond, ctx, type_map)?;
                }

                if let Some(post) = opt_post {
                    type_check_expression(post, ctx, type_map)?;
                }

                type_check_statement(stmt, ctx, type_map)
            }
            Statement::Goto { .. }
            | Statement::Break { .. }
            | Statement::Continue { .. }
            | Statement::Empty => Ok(()),
        }
    }

    fn type_check_declaration<'a>(
        decl: &'a Declaration<'_>,
        ctx: &Context<'_>,
        type_map: &mut TypeMap<'a>,
    ) -> Result<()> {
        match decl {
            Declaration::Var { ident, init, .. } => {
                type_check_variable((ident, init), ctx, type_map)
            }
            Declaration::Func(func) => type_check_function(func, ctx, type_map),
        }
    }

    fn type_check_variable<'a>(
        var: (&'a str, &'a Option<Expression<'_>>),
        ctx: &Context<'_>,
        type_map: &mut TypeMap<'a>,
    ) -> Result<()> {
        type_map.insert(var.0, Type::Int);

        if let Some(init) = var.1 {
            type_check_expression(init, ctx, type_map)?;
        }

        Ok(())
    }

    fn type_check_expression<'a>(
        expr: &'a Expression<'_>,
        ctx: &Context<'_>,
        type_map: &mut TypeMap<'a>,
    ) -> Result<()> {
        match expr {
            Expression::Assignment { lvalue, rvalue, .. } => {
                type_check_expression(lvalue, ctx, type_map)?;
                type_check_expression(rvalue, ctx, type_map)
            }
            Expression::Unary { expr, .. } => type_check_expression(expr, ctx, type_map),
            Expression::Binary { lhs, rhs, .. } => {
                type_check_expression(lhs, ctx, type_map)?;
                type_check_expression(rhs, ctx, type_map)
            }
            Expression::Conditional {
                cond,
                second,
                third,
            } => {
                type_check_expression(cond, ctx, type_map)?;
                type_check_expression(second, ctx, type_map)?;
                type_check_expression(third, ctx, type_map)
            }
            Expression::Var { ident, token } => {
                if *type_map
                    .get(ident.as_str())
                    .expect("variable type should be available after symbol resolution")
                    != Type::Int
                {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    return Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "'{tok_str}' undeclared",
                    ));
                }

                Ok(())
            }
            Expression::FuncCall { ident, args, token } => {
                let f_type = type_map
                    .get(ident.as_str())
                    .expect("function type should be available after symbol resolution");

                if let Type::Func { params } = f_type {
                    let args_len = args.len();

                    if *params != args_len {
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
                        type_check_expression(expr, ctx, type_map)?;
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

    let mut type_map: TypeMap<'_> = HashMap::new();

    for func in &ast.program {
        type_check_function(func, ctx, &mut type_map)?;
    }

    Ok(AST {
        program: ast.program,
        _phase: std::marker::PhantomData,
    })
}
