use std::collections::HashMap;

use crate::compiler::lexer::Token;
use crate::compiler::parser::ast::{
    AST, Block, BlockItem, Declaration, Expression, ForInit, Function, IdentPhase, Labeled, Parsed,
    Statement,
};
use crate::{Context, Result, fmt_token_err};

/// Helper to track scopes in _AST_ traversal.
#[derive(Debug)]
struct Scope {
    /// Currently active scope IDs.
    scopes: Vec<usize>,
    /// Monotonic counter for unique scope IDs.
    next_scope: usize,
}

impl Scope {
    /// Global scope (e.g, functions, global variables).
    const FILE_SCOPE: usize = 0;

    #[inline]
    fn new() -> Self {
        Scope {
            scopes: vec![Self::FILE_SCOPE],
            next_scope: Self::FILE_SCOPE + 1,
        }
    }

    #[inline]
    fn current_scope(&self) -> usize {
        *self
            .scopes
            .last()
            .expect("file scope should always be on the stack")
    }

    #[inline]
    fn enter_scope(&mut self) {
        let scope = self.next_scope;
        self.next_scope += 1;

        self.scopes.push(scope);
    }

    #[inline]
    fn exit_scope(&mut self) {
        debug_assert!(!self.at_file_scope(), "attempting to exit file scope");
        self.scopes.pop();
    }

    #[inline]
    const fn at_file_scope(&self) -> bool {
        self.scopes.len() == 1
    }

    #[inline]
    fn reset_active(&mut self) {
        // `FILE_SCOPE` always remains active
        self.scopes.truncate(1);
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

/// Declaration status of an identifier within a scope.
#[derive(Debug, PartialEq)]
enum DeclStatus {
    /// Invalid re-declaration with different binding types.
    TypeConflict,
    /// Invalid variable re-declaration.
    VarConflict,
    /// No re-declaration (e.g., first declaration, multiple function
    /// declarations).
    None,
}

/// Types of linkage for identifiers.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum Linkage {
    /// Visible across translation units.
    External,
    #[allow(unused)]
    /// Local to a translation unit.
    Internal,
}

/// Binding context for an identifier within a scope.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct NameBinding {
    ident: String,
    scope: usize,
    /// `None` -> no linkage (lexically scoped/local).
    linkage: Option<Linkage>,
}

/// Types of bindings that can be declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingType {
    Func,
    Var,
}

/// Resolved binding information for an identifier within a scope.
#[derive(Debug, Clone)]
struct BindingInfo {
    /// Resolved binding identifier.
    canonical: String,
    is_definition: bool,
    ty: BindingType,
}

/// Helper to perform semantic analysis on identifiers within an _AST_.
#[derive(Default)]
struct IdentResolver {
    bindings: HashMap<NameBinding, BindingInfo>,
    scope: Scope,
}

impl IdentResolver {
    /// Returns a new temporary identifier using the provided prefix.
    #[inline]
    fn new_tmp(&self, prefix: &str) -> String {
        // `@` guarantees it won’t conflict with user-defined identifiers, since
        // the _C_ standard forbids using `@` in identifiers.
        format!("{prefix}@{}", self.scope.current_scope())
    }

    /// Returns `true` if the given identifier has already been defined in the
    /// current scope (or file scope for external/internal linkage).
    #[inline]
    fn is_redefinition(&self, ident: &str, linkage: Option<Linkage>) -> bool {
        let scope = match linkage {
            Some(Linkage::External | Linkage::Internal) => Scope::FILE_SCOPE,
            None => self.scope.current_scope(),
        };

        self.bindings
            .get(&NameBinding {
                ident: ident.to_string(),
                scope,
                linkage,
            })
            .is_some_and(|bind_info| bind_info.is_definition)
    }

    /// Checks if an identifier has already been declared in the current scope,
    /// returning its declaration status.
    #[inline]
    fn is_redeclaration(&self, ident: &str, ty: BindingType) -> DeclStatus {
        // Check if there is a function declaration with external linkage
        // already in scope with the same identifier as a variable.
        let mut binding = NameBinding {
            ident: ident.to_string(),
            scope: self.scope.current_scope(),
            linkage: Some(Linkage::External),
        };

        if self.bindings.contains_key(&binding)
            && let BindingType::Var = ty
        {
            return DeclStatus::TypeConflict;
        }

        // Check for local variable declarations.
        binding.linkage = None;

        if let Some(bind_info) = self.bindings.get(&binding) {
            if bind_info.ty != ty {
                return DeclStatus::TypeConflict;
            }

            return DeclStatus::VarConflict;
        }

        DeclStatus::None
    }

    /// Returns the binding information for the given identifier, recording the
    /// declaration in the appropriate scope according to its linkage.
    fn declare_ident(
        &mut self,
        ident: &str,
        linkage: Option<Linkage>,
        ty: BindingType,
        is_definition: bool,
    ) -> BindingInfo {
        let (resolved_ident, scope) = match linkage {
            Some(Linkage::External) => {
                let scope = if is_definition {
                    Scope::FILE_SCOPE
                } else {
                    // Function declarations declared within their current
                    // scope.
                    self.scope.current_scope()
                };

                (ident.to_string(), scope)
            }
            Some(Linkage::Internal) => todo!(),
            None => (self.new_tmp(ident), self.scope.current_scope()),
        };

        let bind_info = BindingInfo {
            canonical: resolved_ident.clone(),
            is_definition,
            ty,
        };

        self.bindings.insert(
            NameBinding {
                ident: ident.to_string(),
                scope,
                linkage,
            },
            bind_info.clone(),
        );

        bind_info
    }

    /// Returns the binding information for a given identifier, searching the
    /// appropriate scopes, or `None` if not found.
    fn resolve_ident(&self, ident: &str, ty: BindingType) -> Option<BindingInfo> {
        let mut key = NameBinding {
            ident: ident.to_string(),
            scope: 0,
            linkage: None,
        };

        for scope in self.scope.scopes.iter().rev() {
            key.scope = *scope;

            if let Some(bind_info) = self.bindings.get(&key) {
                return Some(bind_info.clone());
            }

            // Try to resolve with any external linkage binding for function
            // calls.
            if let BindingType::Func = ty {
                key.linkage = Some(Linkage::External);

                if let Some(bind_info) = self.bindings.get(&key) {
                    return Some(bind_info.clone());
                }
            }

            key.linkage = None;
        }

        None
    }
}

/// Assigns a canonical identifier to each identifier encountered, performing
/// semantic checks (e.g., duplicate definitions, undeclared references).
///
/// # Errors
///
/// Returns an error if a variable or function is redeclared, used without being
/// declared, assigned incorrectly, or implicitly declared without a prior
/// definition.
pub fn resolve_idents<'a>(
    mut ast: AST<'a, Parsed>,
    ctx: &Context<'_>,
) -> Result<AST<'a, IdentPhase>> {
    fn resolve_function(
        func: &mut Function<'_>,
        ctx: &Context<'_>,
        resolver: &mut IdentResolver,
    ) -> Result<()> {
        let Function {
            ident,
            params,
            body,
            token,
        } = func;

        if resolver.is_redefinition(ident, Some(Linkage::External)) {
            let tok_str = format!("{token:?}");
            let line_content = ctx.src_slice(token.loc.line_span.clone());

            return Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                line_content,
                "redefinition of '{tok_str}'",
            ));
        }

        let is_definition = body.is_some();

        let bind_info = resolver.declare_ident(
            ident,
            Some(Linkage::External),
            BindingType::Func,
            is_definition,
        );
        *ident = bind_info.canonical;

        resolver.scope.enter_scope();

        for param in params {
            resolve_variable(
                (&mut param.ident, &mut None, &mut param.token),
                ctx,
                resolver,
            )?;
        }

        if let Some(body) = body {
            resolve_block(body, ctx, resolver)?;
            resolver.scope.reset_active();
        } else {
            resolver.scope.exit_scope();
        }

        Ok(())
    }

    fn resolve_variable(
        var: (&mut String, &mut Option<Expression<'_>>, &mut Token<'_>),
        ctx: &Context<'_>,
        resolver: &mut IdentResolver,
    ) -> Result<()> {
        let (ident, init, token) = var;

        match resolver.is_redeclaration(ident, BindingType::Var) {
            DeclStatus::None => {
                // Has no linkage, meaning it is considered a definition.
                let bind_info = resolver.declare_ident(ident, None, BindingType::Var, true);
                *ident = bind_info.canonical;

                if let Some(init) = init {
                    resolve_expression(init, ctx, resolver)?;
                }
            }
            status => {
                let tok_str = format!("{token:?}");
                let line_content = ctx.src_slice(token.loc.line_span.clone());

                let msg = if let DeclStatus::TypeConflict = status {
                    format!("'{tok_str}' redeclared as different kind of symbol")
                } else {
                    format!("redeclaration of '{tok_str}'")
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
        }

        Ok(())
    }

    fn resolve_block(
        block: &mut Block<'_>,
        ctx: &Context<'_>,
        resolver: &mut IdentResolver,
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

    fn resolve_statement(
        stmt: &mut Statement<'_>,
        ctx: &Context<'_>,
        resolver: &mut IdentResolver,
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
                resolve_block(block, ctx, resolver)
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
            Statement::Goto { .. }
            | Statement::Break { .. }
            | Statement::Continue { .. }
            | Statement::Empty => Ok(()),
        }
    }

    fn resolve_declaration(
        decl: &mut Declaration<'_>,
        ctx: &Context<'_>,
        resolver: &mut IdentResolver,
    ) -> Result<()> {
        match decl {
            Declaration::Var { ident, init, token } => {
                resolve_variable((ident, init, token), ctx, resolver)?;
            }
            Declaration::Func(func) => {
                if func.body.is_some() {
                    let token = &func.token;

                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    return Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "ISO C forbids nested functions",
                    ));
                }

                // Check if the function declaration is in conflict with a
                // different identifier within the current scope (no-linkage).
                match resolver.is_redeclaration(&func.ident, BindingType::Func) {
                    DeclStatus::None => {
                        resolver.declare_ident(
                            &func.ident,
                            Some(Linkage::External),
                            BindingType::Func,
                            false,
                        );

                        resolver.scope.enter_scope();

                        for param in &mut func.params {
                            resolve_variable(
                                (&mut param.ident, &mut None, &mut param.token),
                                ctx,
                                resolver,
                            )?;
                        }

                        resolver.scope.exit_scope();
                    }
                    status => {
                        debug_assert!(status != DeclStatus::VarConflict);

                        let token = &func.token;

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
                }
            }
        }

        Ok(())
    }

    fn resolve_expression(
        expr: &mut Expression<'_>,
        ctx: &Context<'_>,
        resolver: &mut IdentResolver,
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
                        "lvalue required as left operand of assignment",
                    ))
                }
            }
            Expression::Var { ident, token } => {
                if let Some(bind_info) = resolver.resolve_ident(ident, BindingType::Var) {
                    // Use the canonical identifier mapped from the original
                    // identifier.
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
                if let Some(bind_info) = resolver.resolve_ident(ident, BindingType::Func) {
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
                        "implicit declaration of function '{tok_str}'",
                    ));
                }

                Ok(())
            }
            Expression::IntConstant(_) => Ok(()),
        }
    }

    let mut ident_resolver = IdentResolver::default();

    for func in &mut ast.program {
        resolve_function(func, ctx, &mut ident_resolver)?;
    }

    Ok(AST {
        program: ast.program,
        _phase: std::marker::PhantomData,
    })
}
