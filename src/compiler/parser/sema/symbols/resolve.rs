use std::collections::HashMap;

use crate::compiler::parser::ast::{
    AST, Block, BlockItem, Declaration, Expression, ForInit, Function, IdentPhase, Labeled, Param,
    Parsed, Statement, StorageClass,
};
use crate::compiler::parser::types::Type;
use crate::{Context, Result, fmt_token_err};

use super::{Linkage, Scope, StorageDuration, SymbolMap, SymbolState, convert_bindings_map};

/// Conflicts in the declaration/definition of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConflictStatus {
    /// Attempted redeclaration/definition of an already declared variable.
    Var,
    /// Attempted redeclaration/definition with conflicting linkage (stores
    /// previous declaration linkage).
    Linkage(Option<Linkage>),
    /// Attempted redefinition of existing function.
    Func,
}

/// Symbol binding within a scope.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BindingKey {
    /// Symbol as appears in source.
    pub ident: String,
    pub scope: usize,
}

/// Resolved information about a symbol binding within a scope.
#[derive(Debug, Clone)]
pub struct BindingInfo {
    /// Canonicalized identifier.
    pub canonical: String,
    pub state: SymbolState,
    pub linkage: Option<Linkage>,
    pub duration: Option<StorageDuration>,
    pub ty: Type,
    /// If this entry exists only for file-scope visibility. Proxy (second)
    /// entry doesn’t represent a real declaration in the source code.
    pub is_proxy: bool,
}

/// Helper to perform semantic analysis on symbols within an _AST_.
#[derive(Debug, Default)]
struct SymbolResolver {
    bindings: HashMap<BindingKey, BindingInfo>,
    scope: Scope,
}

impl SymbolResolver {
    /// Returns a new temporary identifier using the provided prefix.
    #[inline]
    fn new_tmp(&self, prefix: &str) -> String {
        // `.` guarantees it won’t conflict with user-defined identifiers, since
        // the _C_ standard forbids using `.` in identifiers.
        format!("{prefix}.{}", self.scope.current_scope())
    }

    /// Checks if the given symbol has a conflicting declaration/definition,
    /// returning the appropriate linkage given any prior declarations.
    ///
    /// # Errors
    ///
    /// Returns an error if the symbol was redeclared with conflicting type,
    /// linkage, or other variable within the same scope.
    fn check_ident_conflict(
        &self,
        symbol: &str,
        state: SymbolState,
        mut linkage: Option<Linkage>,
        storage: Option<StorageClass>,
        ty: &Type,
    ) -> core::result::Result<Option<Linkage>, ConflictStatus> {
        let mut key = BindingKey {
            ident: symbol.to_string(),
            scope: self.scope.current_scope(),
        };

        if let Some(bind_info) = self.bindings.get(&key) {
            // Linkage vs no-linkage is always a conflict.
            if linkage.is_some() != bind_info.linkage.is_some() {
                return Err(ConflictStatus::Linkage(bind_info.linkage));
            }

            if bind_info.linkage.is_some()
                && (matches!(
                    (ty, storage),
                    (Type::Func { .. }, Some(StorageClass::Extern) | None)
                ) || (*ty == Type::Int && storage == Some(StorageClass::Extern)))
            {
                // Linkage matches the prior visible declaration.
                linkage = bind_info.linkage;
            }

            if matches!(
                (linkage, bind_info.linkage),
                (Some(Linkage::External), Some(Linkage::Internal))
                    | (Some(Linkage::Internal), Some(Linkage::External))
            ) {
                return Err(ConflictStatus::Linkage(bind_info.linkage));
            }

            match (ty, bind_info.ty) {
                (Type::Func { .. }, Type::Func { .. }) => {
                    // Multiple function declarations always allowed, not
                    // definitions.
                    if matches!(
                        (state, bind_info.state),
                        (SymbolState::Defined, SymbolState::Defined)
                    ) {
                        return Err(ConflictStatus::Func);
                    }
                }
                // Multiple variables declarations only allowed if previous
                // declaration was tentative or not a definition.
                (Type::Int, Type::Int) => {
                    match (linkage, state, bind_info.state) {
                        // Internal linkage: multiple tentative or nonexistent
                        // definitions are allowed.
                        (
                            Some(Linkage::Internal),
                            SymbolState::Tentative | SymbolState::Declared,
                            _,
                        ) |
                        // Previous definition was tentative or nonexistent.
                        (_, _, SymbolState::Tentative | SymbolState::Declared) => {}
                        _ => return Err(ConflictStatus::Var),
                    }
                }
                // NOTE: Type mismatch is handled during type checking semantic
                // pass.
                _ => {}
            }
        } else if storage == Some(StorageClass::Extern) {
            key.scope = Scope::FILE_SCOPE;

            if let Some(bind_info) = self.bindings.get(&key) {
                // Linkage matches the prior visible declaration.
                linkage = bind_info.linkage;
            }
        }

        Ok(linkage)
    }

    /// Returns the canonical identifier for the given binding context,
    /// recording the declaration in the appropriate scope according to its
    /// linkage.
    ///
    /// Will update any previous declaration of the same identifier and scope.
    fn declare_ident(
        &mut self,
        ident: &str,
        state: SymbolState,
        linkage: Option<Linkage>,
        duration: Option<StorageDuration>,
        ty: Type,
    ) -> String {
        let resolved_ident = if linkage.is_some() {
            ident.to_string()
        } else {
            self.new_tmp(ident)
        };

        let mut insert_binding = |scope, is_proxy| {
            let key = BindingKey {
                ident: ident.to_string(),
                scope,
            };

            self.bindings
                .entry(key)
                .and_modify(|binding| {
                    // Promote proxy entries to real declarations and update
                    // state for non-defined symbols, ensuring declarations do
                    // not overwrite existing definitions.
                    if binding.state.promotes(&state) {
                        binding.is_proxy = false;
                        binding.state = state;
                    }

                    // Only file-scope extern declarations establish linkage,
                    // block-scope extern declarations inherit existing
                    // declaration linkage.
                    if linkage == Some(Linkage::External) && self.scope.at_file_scope() {
                        binding.linkage = linkage;
                    }
                })
                .or_insert_with(|| BindingInfo {
                    canonical: resolved_ident.clone(),
                    state,
                    linkage,
                    duration,
                    ty,
                    is_proxy,
                });
        };

        insert_binding(self.scope.current_scope(), false);

        // Insert proxy entry at file scope.
        if !self.scope.at_file_scope() && linkage == Some(Linkage::External) {
            insert_binding(Scope::FILE_SCOPE, true);
        }

        resolved_ident
    }

    /// Returns the binding information for a given identifier, searching the
    /// appropriate scopes, or `None` if not found.
    fn resolve_ident(&self, ident: &str) -> Option<BindingInfo> {
        let mut key = BindingKey {
            ident: ident.to_string(),
            scope: usize::MAX,
        };

        for scope in self.scope.active_scopes.iter().rev() {
            key.scope = *scope;

            if let Some(bind_info) = self.bindings.get(&key)
                // Ignores any entries made for file-scope resolution purposes.
                && !bind_info.is_proxy
            {
                return Some(bind_info.clone());
            }
        }

        None
    }
}

/// Converts each encountered symbol to its canonical form, while performing
/// semantic checks, returning an initialized symbol map.
///
/// # Errors
///
/// Returns an error if an identifier is redeclared with conflicting types or
/// linkage, used without prior declaration, or misused in a way that violates
/// semantic rules defined by the _C17_ standard.
pub fn resolve_symbols<'a>(
    mut ast: AST<'a, Parsed>,
    ctx: &Context<'_>,
) -> Result<(AST<'a, IdentPhase>, SymbolMap)> {
    let mut ident_resolver = SymbolResolver::default();

    for decl in &mut ast.program {
        resolve_declaration(decl, ctx, &mut ident_resolver)?;
    }

    let sym_map = convert_bindings_map(ident_resolver.bindings);

    Ok((
        AST {
            program: ast.program,
            _phase: std::marker::PhantomData,
        },
        sym_map,
    ))
}

fn resolve_declaration(
    decl: &mut Declaration<'_>,
    ctx: &Context<'_>,
    resolver: &mut SymbolResolver,
) -> Result<()> {
    match decl {
        var @ Declaration::Var { .. } => resolve_variable(var, ctx, resolver),
        Declaration::Func(func) => resolve_function(func, ctx, resolver),
    }
}

fn resolve_function(
    func: &mut Function<'_>,
    ctx: &Context<'_>,
    resolver: &mut SymbolResolver,
) -> Result<()> {
    let Function {
        specs,
        ident,
        params,
        body,
        token,
    } = func;

    if !resolver.scope.at_file_scope() && body.is_some() {
        let tok_str = format!("{token:?}");
        let line_content = ctx.src_slice(token.loc.line_span.clone());

        return Err(fmt_token_err!(
            token.loc.file_path.display(),
            token.loc.line,
            token.loc.col,
            tok_str,
            tok_str.len() - 1,
            line_content,
            "ISO C forbids nested functions"
        ));
    }

    let mut linkage = match specs.storage {
        Some(StorageClass::Extern) | None => Some(Linkage::External),
        Some(StorageClass::Static) => {
            if !resolver.scope.at_file_scope() {
                let tok_str = format!("{token:?}");
                let line_content = ctx.src_slice(token.loc.line_span.clone());

                return Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    line_content,
                    "invalid storage class for function '{tok_str}'",
                ));
            }

            Some(Linkage::Internal)
        }
    };

    let state = if body.is_some() {
        SymbolState::Defined
    } else {
        SymbolState::Declared
    };

    let ty = Type::Func {
        params: params.len(),
    };

    // Updates linkage only if `extern` adopts prior declaration linkage.
    linkage = match resolver.check_ident_conflict(ident, state, linkage, specs.storage, &ty) {
        Ok(resolved_linkage) => resolved_linkage,
        Err(status) => {
            let tok_str = format!("{token:?}");
            let line_content = ctx.src_slice(token.loc.line_span.clone());

            let msg = match status {
                ConflictStatus::Func => {
                    format!("redefinition of '{tok_str}'")
                }
                ConflictStatus::Linkage(_) => {
                    if specs.storage == Some(StorageClass::Static) {
                        format!("static declaration of '{tok_str}' follows non-static declaration")
                    } else {
                        format!("non-static declaration of '{tok_str}' follows static declaration")
                    }
                }
                ConflictStatus::Var => {
                    unreachable!("function cannot conflict based on variable redeclaration")
                }
            };

            return Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                line_content,
                "{msg}",
            ));
        }
    };

    *ident = resolver.declare_ident(ident, state, linkage, None, ty);

    resolver.scope.enter_scope();

    for param in params {
        let Param {
            ty, ident, token, ..
        } = param;
        let state = SymbolState::Defined;

        if let Err(status) = resolver.check_ident_conflict(ident, state, None, None, ty) {
            debug_assert!(
                status == ConflictStatus::Var,
                "function parameters can only conflict based on variable redeclaration"
            );

            let tok_str = format!("{token:?}");
            let line_content = ctx.src_slice(token.loc.line_span.clone());

            return Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                line_content,
                "redefinition of parameter '{tok_str}'",
            ));
        }

        *ident = resolver.declare_ident(ident, state, None, None, *ty);
    }

    if let Some(body) = body {
        resolve_block(body, ctx, resolver)?;
        resolver.scope.reset();
    } else {
        resolver.scope.exit_scope();
    }

    Ok(())
}

fn resolve_variable(
    var: &mut Declaration<'_>,
    ctx: &Context<'_>,
    resolver: &mut SymbolResolver,
) -> Result<()> {
    if let Declaration::Var {
        specs,
        ident,
        init,
        token,
    } = var
    {
        let mut linkage = match specs.storage {
            Some(StorageClass::Extern) => Some(Linkage::External),
            Some(StorageClass::Static) => {
                if resolver.scope.at_file_scope() {
                    Some(Linkage::Internal)
                } else {
                    None
                }
            }
            None if resolver.scope.at_file_scope() => Some(Linkage::External),
            _ => None,
        };

        // All file-scope variables have `static` storage duration.
        let duration = if resolver.scope.at_file_scope() || specs.storage.is_some() {
            Some(StorageDuration::Static)
        } else {
            Some(StorageDuration::Automatic)
        };

        let state = if duration == Some(StorageDuration::Static) {
            if let Some(init) = init {
                // NOTE: Update when constant-expression eval is available to
                // the compiler.
                if let Expression::IntConstant(value) = init {
                    SymbolState::ConstDefined(*value)
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
                        "initializer element is not constant (currently only support integer literals)",
                    ));
                }
            } else if specs.storage == Some(StorageClass::Extern) {
                // File/block-scope `extern` variable.
                SymbolState::Declared
            } else if resolver.scope.at_file_scope() {
                // Static/non-static file scope variable.
                SymbolState::Tentative
            } else {
                // Static block-scope variables are always considered defined.
                SymbolState::Defined
            }
        } else {
            // Automatic variables are always considered defined.
            SymbolState::Defined
        };

        // Updates linkage only if `extern` adopts prior declaration linkage.
        linkage = match resolver.check_ident_conflict(
            ident,
            state,
            linkage,
            specs.storage,
            &specs.ty,
        ) {
            Ok(resolved_linkage) => resolved_linkage,
            Err(status) => {
                let tok_str = format!("{token:?}");
                let line_content = ctx.src_slice(token.loc.line_span.clone());

                let msg = match status {
                    ConflictStatus::Var => {
                        format!("redefinition of '{tok_str}'")
                    }
                    ConflictStatus::Linkage(prev_linkage) => match specs.storage {
                        Some(StorageClass::Static) => {
                            let prev_msg = match prev_linkage {
                                Some(Linkage::External) => "extern declaration",
                                None => "declaration with no linkage",
                                Some(Linkage::Internal) => unreachable!(
                                    "internal linkage conflict should not occur since they do not differ"
                                ),
                            };

                            format!("static declaration of '{tok_str}' follows {prev_msg}")
                        }
                        Some(StorageClass::Extern) => {
                            let prev_msg = match prev_linkage {
                                Some(Linkage::Internal) => "static declaration",
                                None => "declaration with no linkage",
                                Some(Linkage::External) => unreachable!(
                                    "external linkage conflict should not occur since they do not differ"
                                ),
                            };

                            format!("extern declaration of '{tok_str}' follows {prev_msg}")
                        }
                        None => {
                            debug_assert!(
                                prev_linkage.is_some(),
                                "no linkage conflict should not occur since they do not differ"
                            );
                            debug_assert!(
                                prev_linkage != Some(Linkage::Internal),
                                "internal linkage conflict should not occur with a variable in the current scope"
                            );

                            match prev_linkage {
                                Some(Linkage::Internal) => {
                                    format!(
                                        "non-static declaration of '{tok_str}' follows static declaration"
                                    )
                                }
                                Some(Linkage::External) => {
                                    format!(
                                        "non-extern declaration of '{tok_str}' follows extern declaration"
                                    )
                                }
                                None => unreachable!(
                                    "no linkage conflict cannot occur when they do not differ"
                                ),
                            }
                        }
                    },
                    ConflictStatus::Func => {
                        unreachable!(
                            "variable conflicts cannot trigger function redefinition error"
                        )
                    }
                };

                return Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    line_content,
                    "{msg}",
                ));
            }
        };

        *ident = resolver.declare_ident(ident, state, linkage, duration, specs.ty);

        if let Some(init) = init {
            resolve_expression(init, ctx, resolver)?;
        }
    }

    Ok(())
}

fn resolve_block(
    block: &mut Block<'_>,
    ctx: &Context<'_>,
    resolver: &mut SymbolResolver,
) -> Result<()> {
    for block_item in &mut block.0 {
        match block_item {
            BlockItem::Stmt(stmt) => resolve_statement(stmt, ctx, resolver)?,
            BlockItem::Decl(decl) => resolve_declaration(decl, ctx, resolver)?,
        }
    }

    Ok(())
}

fn resolve_statement(
    stmt: &mut Statement<'_>,
    ctx: &Context<'_>,
    resolver: &mut SymbolResolver,
) -> Result<()> {
    match stmt {
        Statement::Return(expr) | Statement::Expression(expr) => {
            resolve_expression(expr, ctx, resolver)
        }
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
                Labeled::Label { stmt, .. } | Labeled::Default { stmt, .. } => stmt,
                Labeled::Case { expr, stmt, .. } => {
                    resolve_expression(expr, ctx, resolver)?;
                    stmt
                }
            };

            resolve_statement(stmt, ctx, resolver)
        }
        Statement::Compound(block) => {
            resolver.scope.enter_scope();
            resolve_block(block, ctx, resolver)?;
            resolver.scope.exit_scope();

            Ok(())
        }
        Statement::While { cond, stmt, .. }
        | Statement::Do { stmt, cond, .. }
        | Statement::Switch { cond, stmt, .. } => {
            resolve_expression(cond, ctx, resolver)?;
            resolve_statement(stmt, ctx, resolver)
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
                // New scope enclosing for-loop header and body.
                ForInit::Decl(decl) => match decl {
                    var @ Declaration::Var { .. } => {
                        resolve_variable(var, ctx, resolver)?;
                    }
                    Declaration::Func(_) => unreachable!(
                        "'for' loop initial declaration should never contain a function declaration"
                    ),
                },
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
        Statement::Goto { .. }
        | Statement::Break { .. }
        | Statement::Continue { .. }
        | Statement::Empty => Ok(()),
    }
}

fn resolve_expression(
    expr: &mut Expression<'_>,
    ctx: &Context<'_>,
    resolver: &mut SymbolResolver,
) -> Result<()> {
    match expr {
        Expression::Assignment {
            lvalue,
            rvalue,
            token,
        } => {
            if let Expression::Var { .. } = **lvalue {
                resolve_expression(lvalue, ctx, resolver)?;
                resolve_expression(rvalue, ctx, resolver)
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
                    "lvalue required as left operand of assignment"
                ))
            }
        }
        Expression::Var { ident, token } => {
            if let Some(bind_info) = resolver.resolve_ident(ident) {
                // Use the canonical identifier.
                *ident = bind_info.canonical;
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
                    "'{tok_str}' undeclared"
                ));
            }

            Ok(())
        }
        Expression::Unary { expr, .. } => resolve_expression(expr, ctx, resolver),
        Expression::Binary { lhs, rhs, .. } => {
            resolve_expression(lhs, ctx, resolver)?;
            resolve_expression(rhs, ctx, resolver)
        }
        Expression::Conditional {
            cond,
            second,
            third,
        } => {
            resolve_expression(cond, ctx, resolver)?;
            resolve_expression(second, ctx, resolver)?;
            resolve_expression(third, ctx, resolver)
        }
        Expression::FuncCall { ident, args, token } => {
            if let Some(bind_info) = resolver.resolve_ident(ident) {
                *ident = bind_info.canonical;

                for expr in args {
                    resolve_expression(expr, ctx, resolver)?;
                }
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
                    "implicit declaration of function '{tok_str}'"
                ));
            }

            Ok(())
        }
        Expression::IntConstant(_) => Ok(()),
    }
}
