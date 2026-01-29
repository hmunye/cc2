use crate::compiler::Result;
use crate::compiler::parser::ast::{
    AST, Block, BlockItem, CtrlFlowPhase, LabelPhase, Labeled, Statement,
};
use crate::{Context, fmt_token_err};

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

/// Resolves labels for escapable control-flow statements (loops/switches) and
/// validates `break`/`continue` usage within each function scope.
pub fn resolve_escapable_ctrl(
    mut ast: AST<LabelPhase>,
    ctx: &Context<'_>,
) -> Result<AST<CtrlFlowPhase>> {
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

    for func in &mut ast.program {
        if let Some(body) = &mut func.body {
            resolve_block(body, ctx, &mut ctrl_resolver)?;
        }
    }

    Ok(AST {
        program: ast.program,
        _phase: std::marker::PhantomData,
    })
}
