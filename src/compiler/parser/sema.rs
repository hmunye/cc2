//! Semantic Analysis
//!
//! Compiler pass that checks the semantic correctness of an abstract syntax
//! tree (_AST_).

use std::collections::{HashMap, HashSet};

use super::ast::{AST, Block, BlockItem, Declaration, Expression, ForInit, Labeled, Statement};

use crate::compiler::Result;
use crate::compiler::lexer::Token;
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
}

/// Helper to perform semantic analysis on label/`goto` statements within an
/// _AST_.
///
/// Labels live in a different namespace from ordinary identifiers (variables,
/// functions, types, etc.) within the same function scope, so they are
/// collected separately.
#[derive(Default)]
struct LabelResolver<'a> {
    /// `key` = label
    labels: HashSet<&'a str>,
    /// Collected label–token pairs for every `goto` statement within a function
    /// scope.
    pending_gotos: Vec<(&'a str, &'a Token)>,
}

impl<'a> LabelResolver<'a> {
    /// Returns `true` if the label was not encountered in the current function
    /// scope, recording it as seen.
    #[inline]
    fn mark_label(&mut self, label: &'a str) -> bool {
        self.labels.insert(label)
    }

    /// Records a `goto` statement's contents so they can be validated after
    /// processing labels.
    #[inline]
    fn mark_goto(&mut self, pair: (&'a str, &'a Token)) {
        self.pending_gotos.push(pair);
    }

    /// Validates all pending `goto` statements and ensures they point to valid
    /// targets within the current function scope.
    ///
    /// # Errors
    ///
    /// Will return `Err` with (label, token) pair of the missing target if it
    /// was not found in the current function scope.
    fn check_gotos(&self) -> core::result::Result<(), (&'a str, &'a Token)> {
        for (label, token) in &self.pending_gotos {
            if !self.labels.contains(label) {
                return Err((label, token));
            }
        }

        Ok(())
    }
}

/// Kind of a escapable control-flow statement.
enum CtrlKind {
    /// A loop statement (`for`, `while`, `do-while`).
    Loop,
    /// A switch statement.
    Switch,
}

/// Uniquely labeled escapable control-flow statement.
struct CtrlLabel<'a> {
    label: &'a str,
    kind: CtrlKind,
}

/// Helper to uniquely label all escapable control-flow statements
/// (loops/switches) and resolve `break` and `continue` targets during
/// semantic analysis within an _AST_.
#[derive(Default)]
struct CtrlResolver<'a> {
    labels: Vec<CtrlLabel<'a>>,
    loop_count: usize,
    switch_count: usize,
}

impl<'a> CtrlResolver<'a> {
    /// Returns a new unique label identifier based on the provided `CtrlKind`.
    #[inline]
    fn new_label(&mut self, kind: CtrlKind) -> String {
        // The `.` in variable identifiers guarantees they won’t conflict
        // with user-defined identifiers, since the _C_ standard forbids `.` in
        // identifiers.
        match kind {
            CtrlKind::Loop => {
                let label = format!("loop.{}", self.loop_count);
                self.loop_count += 1;
                label
            }
            CtrlKind::Switch => {
                let label = format!("switch.{}", self.switch_count);
                self.switch_count += 1;
                label
            }
        }
    }

    /// Begins a new control-flow context (loop/switch) with the specified
    /// `label` and `kind`.
    #[inline]
    fn enter_ctx(&mut self, label: &'a str, kind: CtrlKind) {
        self.labels.push(CtrlLabel { label, kind });
    }

    /// Ends the most recent control-flow context (loop/switch).
    #[inline]
    fn exit_ctx(&mut self) {
        self.labels.pop();
    }

    /// Returns a reference to the current active control-flow context, or
    /// `None` if no context is available.
    #[inline]
    fn current_ctrl(&self) -> Option<&'a CtrlLabel<'_>> {
        self.labels.last()
    }

    /// Returns a reference to the nearest enclosing loop, skipping switches, or
    /// `None` if no loop context is active.
    #[inline]
    fn current_loop(&self) -> Option<&'a CtrlLabel<'_>> {
        self.labels
            .iter()
            .rev()
            .find(|ctrl| matches!(ctrl.kind, CtrlKind::Loop))
    }
}

/// Kind of labeled statement within `switch`.
enum LabelKind {
    Case,
    Default,
}

/// `key` = switch statement label
///
/// `value` = (case values, default label, case label/expression list)
type ScopedSwitches = HashMap<String, (HashSet<i32>, Option<String>, Vec<(String, Expression)>)>;

/// Helper for _AST_ to perform semantic analysis on `switch` statement cases.
#[derive(Default)]
struct SwitchResolver<'a> {
    scope_cases: ScopedSwitches,
    /// Stack of switch statement labels.
    labels: Vec<&'a str>,
    case_count: usize,
    default_count: usize,
}

impl<'a> SwitchResolver<'a> {
    /// Returns a new unique label identifier based on the provided `LabelKind`.
    #[inline]
    fn new_label(&mut self, kind: LabelKind) -> String {
        // The `.` in variable identifiers guarantees they won’t conflict
        // with user-defined identifiers, since the _C_ standard forbids `.` in
        // identifiers.
        match kind {
            LabelKind::Case => {
                let label = format!("case.{}", self.case_count);
                self.case_count += 1;
                label
            }
            LabelKind::Default => {
                let label = format!("default.{}", self.default_count);
                self.default_count += 1;
                label
            }
        }
    }

    /// Returns a unique label for the given `case` expression, or `None` if it
    /// has been encountered within the current `switch` context.
    #[inline]
    fn mark_case(&mut self, label: String, expr: &Expression) -> Option<String> {
        let case_label = self.new_label(LabelKind::Case);

        let entry = self
            .scope_cases
            .entry(label)
            .or_insert((HashSet::new(), None, Vec::new()));

        entry.2.push((case_label.clone(), expr.clone()));

        // NOTE: Update when constant-expression eval is available to the
        // compiler.
        let val = match expr {
            Expression::IntConstant(i) => *i,
            _ => unreachable!(),
        };

        if entry.0.insert(val) {
            Some(case_label)
        } else {
            None
        }
    }

    /// Returns a unique label for the given `default` statement label, or
    /// `None` if one has been encountered within the current `switch` context.
    #[inline]
    fn mark_default(&mut self, label: String) -> Option<String> {
        let default_label = self.new_label(LabelKind::Default);

        let entry = self
            .scope_cases
            .entry(label)
            .or_insert((HashSet::new(), None, Vec::new()));

        if entry.1.is_some() {
            None
        } else {
            entry.1 = Some(default_label.clone());
            Some(default_label)
        }
    }

    /// Begins a new `switch` context with the specified `label`.
    #[inline]
    fn enter_switch(&mut self, label: &'a str) {
        self.labels.push(label);
    }

    /// Ends the most recent `switch` context, appending the collected cases to
    /// the provided container.
    #[inline]
    fn exit_switch(&mut self, cases: &mut Vec<(String, Expression)>) {
        let label = self
            .current_switch()
            .expect("exit_switch should always be called in the context of a switch")
            .to_string();

        let entry = self
            .scope_cases
            .entry(label)
            .or_insert((HashSet::new(), None, Vec::new()));

        self.labels.pop();

        cases.append(&mut entry.2);
    }

    /// Returns a reference to the label of the active `switch` context, or
    /// `None` if no context is available.
    #[inline]
    fn current_switch(&self) -> Option<&'a &str> {
        self.labels.last()
    }

    /// Returns the `default` statement label if one has been encountered in the
    /// current `switch` context.
    #[inline]
    fn current_default_lbl(&mut self) -> Option<String> {
        let label = self.current_switch()?.to_string();
        let entry = self.scope_cases.get_mut(&label)?;

        entry.1.take()
    }
}

/// Assigns a unique identifier to every variable and performs semantic checks
/// (e.g., duplicate definitions, undeclared references).
pub fn resolve_variables(ast: &mut AST, ctx: &Context<'_>) -> Result<()> {
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
        if resolver.is_redeclaration(&decl.ident) {
            let token = &decl.token;

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

        let resolved_ident = resolver.declare_symbol(&decl.ident);
        decl.ident = resolved_ident;

        if let Some(init) = &mut decl.init {
            resolve_expression(init, ctx, resolver)?;
        }

        Ok(())
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
            Expression::IntConstant(_) => Ok(()),
        }
    }

    let mut sym_resolver: SymbolResolver = Default::default();

    match ast {
        AST::Program(func) => resolve_block(&mut func.body, ctx, &mut sym_resolver),
    }
}

/// Ensures every label declared is unique within it's function scope and
/// performs semantic checks (e.g., missing `goto` targets, unreachable labels).
pub fn resolve_labels(ast: &AST, ctx: &Context<'_>) -> Result<()> {
    fn resolve_block<'a>(
        block: &'a Block,
        ctx: &Context<'_>,
        resolver: &mut LabelResolver<'a>,
    ) -> Result<()> {
        for block_item in &block.0 {
            if let BlockItem::Stmt(stmt) = block_item {
                resolve_statement_labels(stmt, ctx, resolver)?;
            }
        }

        Ok(())
    }

    fn resolve_statement_labels<'a>(
        stmt: &'a Statement,
        ctx: &Context<'_>,
        resolver: &mut LabelResolver<'a>,
    ) -> Result<()> {
        match stmt {
            Statement::If { then, opt_else, .. } => {
                resolve_statement_labels(then, ctx, resolver)?;

                if let Some(else_stmt) = opt_else {
                    resolve_statement_labels(else_stmt, ctx, resolver)?;
                }
            }
            Statement::Goto((label, token)) => resolver.mark_goto((label, token)),
            Statement::LabeledStatement(labeled) => match labeled {
                Labeled::Label { label, token, stmt } => {
                    if !resolver.mark_label(label) {
                        let tok_str = format!("{token:?}");
                        let line_content = ctx.src_slice(token.loc.line_span.clone());

                        return Err(fmt_token_err!(
                            token.loc.file_path.display(),
                            token.loc.line,
                            token.loc.col,
                            tok_str,
                            tok_str.len() - 1,
                            line_content,
                            "duplicate label '{tok_str}'",
                        ));
                    }

                    resolve_statement_labels(stmt, ctx, resolver)?;
                }
                Labeled::Case { stmt, .. } | Labeled::Default { stmt, .. } => {
                    resolve_statement_labels(stmt, ctx, resolver)?;
                }
            },
            Statement::While { stmt, .. }
            | Statement::Do { stmt, .. }
            | Statement::For { stmt, .. } => {
                resolve_statement_labels(stmt, ctx, resolver)?;
            }
            Statement::Compound(block) => resolve_block(block, ctx, resolver)?,
            Statement::Switch { stmt, .. } => {
                resolve_statement_labels(stmt, ctx, resolver)?;
            }
            Statement::Return(_)
            | Statement::Expression(_)
            | Statement::Break(_)
            | Statement::Continue(_)
            | Statement::Empty => {}
        }

        Ok(())
    }

    let mut lbl_resolver: LabelResolver<'_> = Default::default();

    match ast {
        AST::Program(func) => {
            // Collect and validate all labels within the function in the
            // first pass.
            resolve_block(&func.body, ctx, &mut lbl_resolver)?;

            // Second pass ensures all `goto` statements point to a valid
            // target within the same function scope.
            if let Err((label, token)) = lbl_resolver.check_gotos() {
                let tok_str = format!("{token:?}");
                let line_content = ctx.src_slice(token.loc.line_span.clone());

                return Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    line_content,
                    "label '{label}' used but not defined",
                ));
            }
        }
    }

    Ok(())
}

/// Assigns unique labels to all escapable control-flow statements
/// (loops/switches) and resolves `break`/`continue` targets, as well as
/// performing semantic checks (e.g., invalid `break`/`continue` location).
pub fn resolve_escapable_ctrl(ast: &mut AST, ctx: &Context<'_>) -> Result<()> {
    fn resolve_block<'a>(
        block: &'a mut Block,
        ctx: &Context<'_>,
        resolver: &mut CtrlResolver<'a>,
    ) -> Result<()> {
        for block_item in &mut block.0 {
            if let BlockItem::Stmt(stmt) = block_item {
                resolve_loop_statement(stmt, ctx, resolver)?;
            }
        }

        Ok(())
    }

    fn resolve_loop_statement<'a>(
        stmt: &'a mut Statement,
        ctx: &Context<'_>,
        resolver: &mut CtrlResolver<'a>,
    ) -> Result<()> {
        match stmt {
            Statement::Break((label, token)) => {
                if let Some(ctrl) = resolver.current_ctrl() {
                    *label = ctrl.label.to_string();
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
                        "{tok_str} statement not within a loop or switch"
                    ));
                }
            }
            Statement::Continue((label, token)) => {
                // Ensures a `continue` inside a `switch` context doesn’t
                // immediately trigger a error.
                if let Some(ctrl) = resolver.current_loop() {
                    *label = ctrl.label.to_string();
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
                        "continue statement not within a loop"
                    ));
                }
            }
            Statement::While { stmt, label, .. }
            | Statement::Do { stmt, label, .. }
            | Statement::For { stmt, label, .. } => {
                *label = resolver.new_label(CtrlKind::Loop);

                resolver.enter_ctx(label, CtrlKind::Loop);

                resolve_loop_statement(stmt, ctx, resolver)?;

                resolver.exit_ctx();
            }
            Statement::Switch { stmt, label, .. } => {
                *label = resolver.new_label(CtrlKind::Switch);

                resolver.enter_ctx(label, CtrlKind::Switch);

                resolve_loop_statement(stmt, ctx, resolver)?;

                resolver.exit_ctx();
            }
            Statement::If { then, opt_else, .. } => {
                resolve_loop_statement(then, ctx, resolver)?;
                if let Some(else_stmt) = opt_else {
                    resolve_loop_statement(else_stmt, ctx, resolver)?;
                }
            }
            Statement::LabeledStatement(labeled) => {
                let stmt = match labeled {
                    Labeled::Label { stmt, .. } => stmt,
                    Labeled::Case { stmt, .. } => stmt,
                    Labeled::Default { stmt, .. } => stmt,
                };

                resolve_loop_statement(stmt, ctx, resolver)?;
            }
            Statement::Compound(block) => {
                resolve_block(block, ctx, resolver)?;
            }
            Statement::Return(_)
            | Statement::Expression(_)
            | Statement::Goto(_)
            | Statement::Empty => {}
        }

        Ok(())
    }

    let mut ctrl_resolver: CtrlResolver<'_> = Default::default();

    match ast {
        AST::Program(func) => resolve_block(&mut func.body, ctx, &mut ctrl_resolver),
    }
}

/// Collects all `case` and `default` labels for each `switch` statement, as
/// well as performing semantic checks (e.g., duplicate `case`,
/// duplicate `default`, invalid `case`/`default` location).
pub fn resolve_switches(ast: &mut AST, ctx: &Context<'_>) -> Result<()> {
    fn resolve_block<'a>(
        block: &'a mut Block,
        ctx: &Context<'_>,
        resolver: &mut SwitchResolver<'a>,
    ) -> Result<()> {
        for block_item in &mut block.0 {
            if let BlockItem::Stmt(stmt) = block_item {
                resolve_statement(stmt, ctx, resolver)?;
            }
        }

        Ok(())
    }

    fn resolve_statement<'a>(
        stmt: &'a mut Statement,
        ctx: &Context<'_>,
        resolver: &mut SwitchResolver<'a>,
    ) -> Result<()> {
        match stmt {
            Statement::Switch {
                stmt,
                cases,
                default,
                label,
                ..
            } => {
                resolver.enter_switch(label);

                resolve_statement(stmt, ctx, resolver)?;

                *default = resolver.current_default_lbl();

                resolver.exit_switch(cases);
            }
            Statement::LabeledStatement(labeled) => match labeled {
                Labeled::Label { stmt, .. } => {
                    resolve_statement(stmt, ctx, resolver)?;
                }
                Labeled::Case {
                    expr,
                    token,
                    stmt,
                    label,
                    ..
                } => {
                    if let Some(ctx_label) = resolver.current_switch() {
                        if let Some(case_label) = resolver.mark_case(ctx_label.to_string(), expr) {
                            *label = case_label;
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
                                "duplicate case value"
                            ));
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
                            "case label not within a switch statement"
                        ));
                    }

                    resolve_statement(stmt, ctx, resolver)?;
                }
                Labeled::Default { token, stmt, label } => {
                    if let Some(ctx_label) = resolver.current_switch() {
                        if let Some(default_label) = resolver.mark_default(ctx_label.to_string()) {
                            *label = default_label;
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
                                "multiple default labels in one switch"
                            ));
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
                            "default label not within a switch statement"
                        ));
                    }

                    resolve_statement(stmt, ctx, resolver)?;
                }
            },
            Statement::While { stmt, .. }
            | Statement::Do { stmt, .. }
            | Statement::For { stmt, .. } => {
                resolve_statement(stmt, ctx, resolver)?;
            }
            Statement::If { then, opt_else, .. } => {
                resolve_statement(then, ctx, resolver)?;
                if let Some(else_stmt) = opt_else {
                    resolve_statement(else_stmt, ctx, resolver)?;
                }
            }
            Statement::Compound(block) => {
                resolve_block(block, ctx, resolver)?;
            }
            Statement::Return(_)
            | Statement::Expression(_)
            | Statement::Break(_)
            | Statement::Continue(_)
            | Statement::Goto(_)
            | Statement::Empty => {}
        }

        Ok(())
    }

    let mut switch_resolver: SwitchResolver<'_> = Default::default();

    match ast {
        AST::Program(func) => resolve_block(&mut func.body, ctx, &mut switch_resolver),
    }
}
