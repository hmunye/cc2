use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::compiler::lexer::Token;
use crate::compiler::parser::ast::{
    AST, Block, BlockItem, LabelPhase, Labeled, Statement, TypePhase,
};
use crate::{Context, Result, fmt_token_err};

/// Helper to perform semantic analysis on label/`goto` statements within an
/// _AST_.
///
/// Labels live in a different namespace from ordinary identifiers (variables,
/// functions, types, etc.) within the same function scope, so they are
/// collected separately.
#[derive(Debug, Default)]
struct LabelResolver<'a> {
    /// `key` = label
    ///
    /// `value` = canonical identifier
    labels: HashMap<String, String>,
    /// Collected label/token pairs for every `goto` statement within a function
    /// scope.
    pending_gotos: Vec<(String, Token<'a>)>,
    /// Current function identifier for canonical identifier generation.
    fn_ident: String,
}

impl<'a> LabelResolver<'a> {
    /// Returns a new label using the provided suffix.
    #[inline]
    fn new_label(&self, suffix: &str) -> String {
        // `.` guarantees it won’t conflict with user-defined identifiers, since
        // the _C_ standard forbids using `.` in identifiers.
        format!("{}.{suffix}", self.fn_ident)
    }

    /// Returns `true` if the label was not encountered in the current function
    /// scope, recording it as seen.
    #[inline]
    fn mark_label(&mut self, label: &str) -> bool {
        let canonical = self.new_label(label);

        match self.labels.entry(label.to_string()) {
            Entry::Occupied(_) => false,
            Entry::Vacant(entry) => {
                entry.insert(canonical);
                true
            }
        }
    }

    /// Records a `goto` statement's contents so they can be validated after
    /// processing labels.
    #[inline]
    fn mark_goto(&mut self, pair: (&str, &Token<'a>)) {
        self.pending_gotos
            .push((pair.0.to_string(), pair.1.clone()));
    }

    /// Validates and removes all pending `goto` statements and ensures they
    /// point to valid targets within the current function scope.
    ///
    /// # Errors
    ///
    /// Returns an error if a `goto` target was not found in
    /// the current function scope.
    fn check_gotos(&mut self) -> core::result::Result<(), (String, Token<'_>)> {
        for (label, token) in self.pending_gotos.drain(..) {
            if !self.labels.contains_key(&label) {
                return Err((label, token));
            }
        }

        Ok(())
    }

    /// Resets the resolver state so it may be used within another function
    /// scope.
    #[inline]
    fn reset(&mut self, ident: &str) {
        self.labels.clear();
        self.fn_ident = ident.to_string();
    }
}

/// Verifies labels and `goto` targets within each function scope, ensuring
/// uniqueness across different function scopes.
///
/// # Errors
///
/// Returns an error if a duplicate label is found or an
/// undefined label is used within a function scope.
pub fn resolve_labels<'a>(
    mut ast: AST<'a, TypePhase>,
    ctx: &Context<'_>,
) -> Result<AST<'a, LabelPhase>> {
    fn resolve_block<'a>(
        block: &Block<'a>,
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
        stmt: &Statement<'a>,
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
            Statement::Goto { target, token } => resolver.mark_goto((target, token)),
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
            | Statement::For { stmt, .. }
            | Statement::Switch { stmt, .. } => {
                resolve_statement_labels(stmt, ctx, resolver)?;
            }
            Statement::Compound(block) => resolve_block(block, ctx, resolver)?,
            Statement::Return(_)
            | Statement::Expression(_)
            | Statement::Break { .. }
            | Statement::Continue { .. }
            | Statement::Empty => {}
        }

        Ok(())
    }

    let mut lbl_resolver = LabelResolver::default();

    for func in &mut ast.program {
        if let Some(body) = &mut func.body {
            lbl_resolver.reset(&func.ident);

            // Collect and validate all labels within the function in
            // the first pass.
            resolve_block(body, ctx, &mut lbl_resolver)?;

            // Second pass ensures all `goto` statements point to a
            // valid target within the same function scope.
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

            // Third pass updates each label within the current function scope
            // with its canonical identifier.
            update_labels(body, &lbl_resolver);
        }
    }

    Ok(AST {
        program: ast.program,
        _phase: std::marker::PhantomData,
    })
}

/// Updates original labels/`goto` target identifiers with their canonical
/// identifier.
fn update_labels(body: &mut Block<'_>, resolver: &LabelResolver<'_>) {
    fn resolve_block(block: &mut Block<'_>, resolver: &LabelResolver<'_>) {
        for block_item in &mut block.0 {
            if let BlockItem::Stmt(stmt) = block_item {
                resolve_statement_labels(stmt, resolver);
            }
        }
    }

    fn resolve_statement_labels(stmt: &mut Statement<'_>, resolver: &LabelResolver<'_>) {
        match stmt {
            Statement::If { then, opt_else, .. } => {
                resolve_statement_labels(then, resolver);

                if let Some(else_stmt) = opt_else {
                    resolve_statement_labels(else_stmt, resolver);
                }
            }
            Statement::Goto { target, .. } => {
                target.clone_from(
                    resolver.labels.get(target.as_str()).expect(
                        "`goto` target should have been encountered during label resolution",
                    ),
                );
            }
            Statement::LabeledStatement(labeled) => match labeled {
                Labeled::Label { label, stmt, .. } => {
                    label.clone_from(
                        resolver
                            .labels
                            .get(label.as_str())
                            .expect("label should have been encountered during label resolution"),
                    );

                    resolve_statement_labels(stmt, resolver);
                }
                Labeled::Case { stmt, .. } | Labeled::Default { stmt, .. } => {
                    resolve_statement_labels(stmt, resolver);
                }
            },
            Statement::While { stmt, .. }
            | Statement::Do { stmt, .. }
            | Statement::For { stmt, .. }
            | Statement::Switch { stmt, .. } => {
                resolve_statement_labels(stmt, resolver);
            }
            Statement::Compound(block) => resolve_block(block, resolver),
            Statement::Return(_)
            | Statement::Expression(_)
            | Statement::Break { .. }
            | Statement::Continue { .. }
            | Statement::Empty => {}
        }
    }

    resolve_block(body, resolver);
}
