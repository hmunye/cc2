use std::collections::HashSet;

use crate::compiler::Result;
use crate::compiler::lexer::Token;
use crate::compiler::parser::ast::{AST, Block, BlockItem, Labeled, Statement};
use crate::{Context, fmt_token_err};

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

    /// Validates and removes all pending `goto` statements and ensures they
    /// point to valid targets within the current function scope.
    ///
    /// # Errors
    ///
    /// Will return `Err` with (label, token) pair of the missing target if it
    /// was not found in the current function scope.
    fn check_gotos(&mut self) -> core::result::Result<(), (&'a str, &'a Token)> {
        for (label, token) in self.pending_gotos.drain(..) {
            if !self.labels.contains(label) {
                return Err((label, token));
            }
        }

        Ok(())
    }

    /// Resets the resolver state so it may be used within another function
    /// scope.
    #[inline]
    fn reset(&mut self) {
        self.labels.clear();
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
        AST::Program(funcs) => {
            for func in funcs {
                if let Some(body) = &func.body {
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

                    lbl_resolver.reset();
                }
            }
        }
    }

    Ok(())
}
