use std::collections::HashMap;

use crate::compiler::Result;
use crate::compiler::parser::ast::{
    AST, Block, BlockItem, Declaration, Expression, ForInit, Labeled, Statement,
};
use crate::{Context, fmt_token_err};

/// Helper to track the current scope for symbol resolution.
struct Scope {
    scopes: Vec<usize>,
    next_scope: usize,
}

impl Scope {
    #[inline]
    fn new() -> Self {
        Scope {
            scopes: vec![0], // Function scope.
            next_scope: 1,
        }
    }

    #[inline]
    fn current_scope(&self) -> usize {
        *self.scopes.last().unwrap_or(&0)
    }

    #[inline]
    fn enter_scope(&mut self) {
        let scope = self.next_scope;
        self.next_scope += 1;

        self.scopes.push(scope);
    }

    #[inline]
    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    #[inline]
    fn reset(&mut self) {
        self.scopes.clear();
        self.next_scope = 1;
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to perform semantic analysis on symbols within an _AST_.
#[derive(Default)]
struct SymbolResolver {
    /// `key` = (ident, scope)
    ///
    /// `value` = resolved symbol
    symbol_map: HashMap<(String, usize), String>,
    scope: Scope,
}

impl SymbolResolver {
    /// Returns a new temporary variable identifier using the provided prefix.
    #[inline]
    fn new_tmp(&self, prefix: &str) -> String {
        // The `@` in variable identifiers guarantees they won’t conflict
        // with user-defined identifiers, since the _C_ standard forbids `@` in
        // identifiers.
        format!("{prefix}@{}", self.scope.current_scope())
    }

    /// Returns `true` if the given `symbol` has already been declared in the
    /// current scope.
    #[inline]
    fn is_redeclaration(&self, symbol: &str) -> bool {
        self.symbol_map
            .contains_key(&(symbol.to_string(), self.scope.current_scope()))
    }

    /// Returns a unique identifier for the given `symbol`, recording the
    /// declaration in the current scope.
    fn declare_symbol(&mut self, symbol: &str) -> String {
        let resolved_ident = self.new_tmp(symbol);

        let res = self.symbol_map.insert(
            (symbol.to_string(), self.scope.current_scope()),
            resolved_ident.clone(),
        );

        // Ensure the key was not present before insertion.
        debug_assert!(res.is_none());

        resolved_ident
    }

    /// Returns the unique identifier for a given `symbol`, searching the
    /// current scope and all outer scopes (up to the function scope), or `None`
    /// if no existing declaration could be found.
    fn resolve_symbol(&self, symbol: &str) -> Option<String> {
        let mut resolved_ident = None;

        // Starts at the current scope then searches all outer scopes until
        // reaching the function scope (0).
        let scopes = (0..=self.scope.current_scope()).rev();

        for scope in scopes {
            if let Some(ident) = self.symbol_map.get(&(symbol.to_string(), scope)) {
                resolved_ident = Some(ident.clone());
                break;
            }
        }

        resolved_ident
    }

    /// Resets the resolver state so it may be used within another function
    /// scope.
    #[inline]
    fn reset(&mut self) {
        self.symbol_map.clear();
        self.scope.reset();
    }
}

/// Assigns a unique identifier to every symbol and performs semantic checks
/// (e.g., duplicate definitions, undeclared references).
pub fn resolve_symbols(ast: &mut AST, ctx: &Context<'_>) -> Result<()> {
    fn resolve_block(
        block: &mut Block,
        ctx: &Context<'_>,
        resolver: &mut SymbolResolver,
    ) -> Result<()> {
        for block_item in &mut block.0 {
            match block_item {
                BlockItem::Stmt(stmt) => resolve_statement(stmt, ctx, resolver)?,
                BlockItem::Decl(decl) => resolve_declaration(decl, ctx, resolver)?,
            }
        }

        resolver.scope.exit_scope();

        Ok(())
    }

    fn resolve_declaration(
        decl: &mut Declaration,
        ctx: &Context<'_>,
        resolver: &mut SymbolResolver,
    ) -> Result<()> {
        match decl {
            Declaration::Var { ident, init, token } => {
                if resolver.is_redeclaration(ident) {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    return Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "redeclaration of '{tok_str}'",
                    ));
                }

                let resolved_ident = resolver.declare_symbol(ident);
                *ident = resolved_ident;

                if let Some(init) = init {
                    resolve_expression(init, ctx, resolver)?;
                }

                Ok(())
            }
            Declaration::Func(_) => todo!(),
        }
    }

    fn resolve_statement(
        stmt: &mut Statement,
        ctx: &Context<'_>,
        resolver: &mut SymbolResolver,
    ) -> Result<()> {
        match stmt {
            Statement::Return(expr) => resolve_expression(expr, ctx, resolver),
            Statement::Expression(expr) => resolve_expression(expr, ctx, resolver),
            Statement::If {
                cond,
                then,
                opt_else,
            } => {
                resolve_expression(cond, ctx, resolver)?;
                resolve_statement(then, ctx, resolver)?;

                if let Some(stmt) = opt_else {
                    resolve_statement(stmt, ctx, resolver)?;
                }

                Ok(())
            }
            Statement::LabeledStatement(labeled) => {
                let stmt = match labeled {
                    Labeled::Label { stmt, .. } => stmt,
                    Labeled::Case { expr, stmt, .. } => {
                        resolve_expression(expr, ctx, resolver)?;
                        stmt
                    }
                    Labeled::Default { stmt, .. } => stmt,
                };

                resolve_statement(stmt, ctx, resolver)
            }
            Statement::Compound(block) => {
                resolver.scope.enter_scope();
                resolve_block(block, ctx, resolver)
            }
            Statement::While { cond, stmt, .. } => {
                resolve_expression(cond, ctx, resolver)?;
                resolve_statement(stmt, ctx, resolver)
            }
            Statement::Do { stmt, cond, .. } => {
                resolve_statement(stmt, ctx, resolver)?;
                resolve_expression(cond, ctx, resolver)
            }
            Statement::For {
                init,
                opt_cond,
                opt_post,
                stmt,
                ..
            } => {
                let enter_scope = matches!(&**init, ForInit::Decl(_));

                if enter_scope {
                    resolver.scope.enter_scope();
                }

                match &mut **init {
                    // New scope introduced enclosing for-loop header and body.
                    ForInit::Decl(decl) => {
                        resolve_declaration(decl, ctx, resolver)?;
                    }
                    // No new scope introduced - using current scope.
                    ForInit::Expr(opt_init) => {
                        if let Some(init) = opt_init {
                            resolve_expression(init, ctx, resolver)?;
                        }
                    }
                }

                if let Some(cond) = opt_cond {
                    resolve_expression(cond, ctx, resolver)?;
                }

                if let Some(post) = opt_post {
                    resolve_expression(post, ctx, resolver)?;
                }

                resolve_statement(stmt, ctx, resolver)?;

                if enter_scope {
                    resolver.scope.exit_scope();
                }

                Ok(())
            }
            Statement::Switch { cond, stmt, .. } => {
                resolve_expression(cond, ctx, resolver)?;
                resolve_statement(stmt, ctx, resolver)
            }
            Statement::Goto(_) => Ok(()),
            Statement::Break(_) => Ok(()),
            Statement::Continue(_) => Ok(()),
            Statement::Empty => Ok(()),
        }
    }

    fn resolve_expression(
        expr: &mut Expression,
        ctx: &Context<'_>,
        resolver: &mut SymbolResolver,
    ) -> Result<()> {
        match expr {
            Expression::Assignment {
                lvalue,
                rvalue,
                token,
            } => match **lvalue {
                Expression::Var(_) => {
                    resolve_expression(lvalue, ctx, resolver)?;
                    resolve_expression(rvalue, ctx, resolver)
                }
                _ => {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "lvalue required as left operand of assignment",
                    ))
                }
            },
            Expression::Var((v, token)) => {
                if let Some(ident) = resolver.resolve_symbol(v) {
                    // Use the unique symbol mapped from the original
                    // identifier.
                    *v = ident;
                } else {
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
            Expression::Unary { expr, .. } => resolve_expression(expr, ctx, resolver),
            Expression::Binary { lhs, rhs, .. } => {
                resolve_expression(lhs, ctx, resolver)?;
                resolve_expression(rhs, ctx, resolver)
            }
            Expression::Conditional(lhs, mid, rhs) => {
                resolve_expression(lhs, ctx, resolver)?;
                resolve_expression(mid, ctx, resolver)?;
                resolve_expression(rhs, ctx, resolver)
            }
            Expression::FuncCall { .. } => todo!(),
            Expression::IntConstant(_) => Ok(()),
        }
    }

    let mut sym_resolver: SymbolResolver = Default::default();

    match ast {
        AST::Program(funcs) => {
            for func in funcs {
                if let Some(body) = &mut func.body {
                    resolve_block(body, ctx, &mut sym_resolver)?;
                    sym_resolver.reset();
                }
            }
        }
    }

    Ok(())
}
