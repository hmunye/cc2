use std::collections::{HashMap, HashSet};

use crate::compiler::parser::ast::{
    AST, Analyzed, Block, BlockItem, CtrlFlowPhase, Expression, Labeled, Statement, SwitchCase,
};
use crate::{Context, Result, fmt_token_err};

/// Kind of labeled statement within `switch`.
enum LabelKind {
    Case,
    Default,
}

/// `key` = switch statement label
///
/// `value` = (case values, default label, case label/expression list)
type ScopedSwitches<'a> = HashMap<String, (HashSet<i32>, Option<String>, Vec<SwitchCase<'a>>)>;

/// Helper for _AST_ to perform semantic analysis on `switch` statement cases.
#[derive(Default)]
struct SwitchResolver<'a> {
    /// Stack of switch statement labels.
    labels: Vec<String>,
    scope_cases: ScopedSwitches<'a>,
    case_count: usize,
    default_count: usize,
}

impl<'a> SwitchResolver<'a> {
    /// Returns a new unique label identifier based on the provided `LabelKind`.
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
    fn mark_case(&mut self, label: &str, expr: &Expression<'a>) -> Option<String> {
        let case_label = self.new_label(LabelKind::Case);

        let entry =
            self.scope_cases
                .entry(label.to_string())
                .or_insert((HashSet::new(), None, Vec::new()));

        entry.2.push(SwitchCase {
            jmp_label: case_label.clone(),
            expr: expr.clone(),
        });

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
    fn mark_default(&mut self, label: &str) -> Option<String> {
        let default_label = self.new_label(LabelKind::Default);

        let entry =
            self.scope_cases
                .entry(label.to_string())
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
    fn enter_switch(&mut self, label: &str) {
        self.labels.push(label.to_string());
    }

    /// Ends the most recent `switch` context, appending the collected cases to
    /// the provided container.
    fn exit_switch(&mut self, cases: &mut Vec<SwitchCase<'a>>) {
        let label = self
            .labels
            .pop()
            .expect("exit_switch should always be called in the context of a switch");

        let entry = self
            .scope_cases
            .entry(label)
            .or_insert((HashSet::new(), None, Vec::new()));

        cases.append(&mut entry.2);
    }

    /// Returns a reference to the label of the active `switch` context, or
    /// `None` if no context is available.
    #[inline]
    fn current_switch(&self) -> Option<String> {
        self.labels.last().cloned()
    }

    /// Returns the `default` statement label if one has been encountered in the
    /// current `switch` context.
    fn current_default_lbl(&mut self) -> Option<String> {
        let label = self.current_switch()?;
        let entry = self.scope_cases.get_mut(&label)?;

        entry.1.take()
    }

    /// Resets the resolver state so it may be used within another function
    /// scope.
    #[inline]
    fn reset(&mut self) {
        self.scope_cases.clear();
        self.labels.clear();
    }
}

/// Resolves `switch` statements and their `case`/`default` labels.
///
/// # Errors
///
/// Returns an error if a `switch` contains duplicate cases,
/// multiple `default` labels, or if a `case`/`default` label appears outside of
/// a `switch`.
pub fn resolve_switches<'a>(
    mut ast: AST<'a, CtrlFlowPhase>,
    ctx: &Context<'_>,
) -> Result<AST<'a, Analyzed>> {
    fn resolve_block<'a>(
        block: &mut Block<'a>,
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
        stmt: &mut Statement<'a>,
        ctx: &Context<'_>,
        resolver: &mut SwitchResolver<'a>,
    ) -> Result<()> {
        match stmt {
            Statement::Switch {
                stmt,
                cases,
                default,
                switch_label,
                ..
            } => {
                resolver.enter_switch(switch_label);

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
                    jmp_label: label,
                    ..
                } => {
                    if let Some(ctx_label) = resolver.current_switch() {
                        if let Some(case_label) = resolver.mark_case(ctx_label.as_str(), expr) {
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
                Labeled::Default {
                    token,
                    stmt,
                    jmp_label: label,
                } => {
                    if let Some(ctx_label) = resolver.current_switch() {
                        if let Some(default_label) = resolver.mark_default(ctx_label.as_str()) {
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
            | Statement::Break { .. }
            | Statement::Continue { .. }
            | Statement::Goto { .. }
            | Statement::Empty => {}
        }

        Ok(())
    }

    let mut switch_resolver = SwitchResolver::default();

    for func in &mut ast.program {
        if let Some(body) = &mut func.body {
            resolve_block(body, ctx, &mut switch_resolver)?;
            switch_resolver.reset();
        }
    }

    Ok(AST {
        program: ast.program,
        _phase: std::marker::PhantomData,
    })
}
