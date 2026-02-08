use crate::compiler::parser::ast::{
    AST, Block, BlockItem, CtrlFlowPhase, Declaration, LabelPhase, Labeled, Statement,
};
use crate::{Context, Result, fmt_token_err};

/// Kind of escapable control-flow statement.
#[derive(Debug, Clone, Copy)]
enum CtrlKind {
    /// A loop statement (`for`, `while`, `do-while`).
    Loop,
    /// A switch statement.
    Switch,
}

/// Uniquely labeled escapable control-flow statement.
#[derive(Debug, Clone, Copy)]
struct CtrlLabel<'a> {
    label: &'a str,
    kind: CtrlKind,
}

/// Helper to uniquely label all escapable control-flow statements and resolve
/// `break` and `continue` targets during semantic analysis of an _AST_.
#[derive(Debug, Default)]
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
///
/// # Errors
///
/// Returns an error if a `continue` statement is not found within a loop or a
/// `break` statement is not found within a loop or `switch`.
pub fn resolve_escapable_ctrl<'a>(
    mut ast: AST<'a, LabelPhase>,
    ctx: &Context<'_>,
) -> Result<AST<'a, CtrlFlowPhase>> {
    let mut ctrl_resolver = CtrlResolver::default();

    for decl in &mut ast.program {
        if let Declaration::Func(func) = decl
            && let Some(body) = &mut func.body
        {
            resolve_block(body, ctx, &mut ctrl_resolver)?;
        }
    }

    Ok(AST {
        program: ast.program,
        _phase: std::marker::PhantomData,
    })
}

fn resolve_block<'a>(
    block: &'a mut Block<'_>,
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
    stmt: &'a mut Statement<'_>,
    ctx: &Context<'_>,
    resolver: &mut CtrlResolver<'a>,
) -> Result<()> {
    match stmt {
        Statement::Break { jmp_label, token } => {
            if let Some(ctrl) = resolver.current_ctrl() {
                *jmp_label = ctrl.label.to_string();
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
                    "break statement not within a loop or switch"
                ));
            }
        }
        Statement::Continue { jmp_label, token } => {
            // Ensures a `continue` inside a `switch` context doesn’t
            // immediately trigger an error.
            if let Some(ctrl) = resolver.current_loop() {
                *jmp_label = ctrl.label.to_string();
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
        Statement::While {
            stmt, loop_label, ..
        }
        | Statement::Do {
            stmt, loop_label, ..
        }
        | Statement::For {
            stmt, loop_label, ..
        } => {
            *loop_label = resolver.new_label(CtrlKind::Loop);

            resolver.enter_ctx(loop_label, CtrlKind::Loop);

            resolve_loop_statement(stmt, ctx, resolver)?;

            resolver.exit_ctx();
        }
        Statement::Switch {
            stmt, switch_label, ..
        } => {
            *switch_label = resolver.new_label(CtrlKind::Switch);

            resolver.enter_ctx(switch_label, CtrlKind::Switch);

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
                Labeled::Label { stmt, .. }
                | Labeled::Case { stmt, .. }
                | Labeled::Default { stmt, .. } => stmt,
            };

            resolve_loop_statement(stmt, ctx, resolver)?;
        }
        Statement::Compound(block) => {
            resolve_block(block, ctx, resolver)?;
        }
        Statement::Return(_)
        | Statement::Expression(_)
        | Statement::Goto { .. }
        | Statement::Empty => {}
    }

    Ok(())
}
