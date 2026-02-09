//! Intermediate Representation
//!
//! Compiler pass that lowers an abstract syntax tree (_AST_) into three-address
//! code (_TAC_) intermediate representation (_IR_).

use std::{borrow::Cow, fmt};

use crate::compiler::parser::{
    ast::{self, Analyzed, BinaryOperator, Signedness, StorageClass, UnaryOperator},
    sema::symbols::{Linkage, StorageDuration, SymbolMap, SymbolState},
    types::c_int,
};

/// Intermediate representation (_IR_).
#[derive(Debug)]
pub struct IR<'a> {
    /// Items that represent the structure of the program.
    pub program: Vec<Item<'a>>,
}

impl fmt::Display for IR<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IR Program")?;
        for item in &self.program {
            writeln!(f, "{:4}{item}", "")?;
        }

        Ok(())
    }
}

/// _IR_ top-level construct
#[derive(Debug)]
pub enum Item<'a> {
    Func(Function<'a>),
    /// Declaration with `static` storage duration.
    Static {
        init: c_int,
        ident: &'a str,
        is_global: bool,
    },
}

impl fmt::Display for Item<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Item::Func(func) => write!(f, "{func}"),
            Item::Static {
                init,
                ident,
                is_global,
            } => writeln!(
                f,
                "Static ({}) {ident:?} = {init}",
                if *is_global { "G" } else { "L" }
            ),
        }
    }
}

/// _IR_ function definition.
#[derive(Debug)]
pub struct Function<'a> {
    pub ident: &'a str,
    pub params: Vec<&'a str>,
    pub instructions: Vec<Instruction<'a>>,
    pub is_global: bool,
}

impl fmt::Display for Function<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params = self.params.join(", ");

        writeln!(
            f,
            "Fn ({}) {:?}({params})",
            if self.is_global { "G" } else { "L" },
            self.ident
        )?;

        for inst in &self.instructions {
            writeln!(f, "{:8}{inst}", "")?;
        }

        Ok(())
    }
}

/// _IR_ instruction.
#[derive(Debug)]
pub enum Instruction<'a> {
    /// Returns a value to the caller.
    Return(Value<'a>),
    /// Perform a unary operation on `src`, storing the result in `dst`.
    Unary {
        op: UnaryOperator,
        src: Value<'a>,
        dst: Value<'a>,
        // NOTE: Temporary hack for arithmetic right shift.
        sign: Signedness,
    },
    /// Perform a binary operation on `lhs` and `rhs`, storing the result in
    /// `dst`.
    Binary {
        op: BinaryOperator,
        lhs: Value<'a>,
        rhs: Value<'a>,
        dst: Value<'a>,
        // NOTE: Temporary hack for arithmetic right shift.
        sign: Signedness,
    },
    /// Copies the value from `src` into `dst`.
    Copy { src: Value<'a>, dst: Value<'a> },
    /// Unconditionally jumps to the point in code labeled by an identifier.
    Jump(String),
    /// Conditionally jumps to the point in code labeled by an identifier if
    /// `cond` evaluates to zero.
    JumpIfZero { cond: Value<'a>, target: String },
    /// Conditionally jumps to the point in code labeled by an identifier if
    /// `cond` does not evaluates to zero.
    JumpIfNotZero { cond: Value<'a>, target: String },
    /// Associates an identifier with instructions.
    Label(String),
    /// Transfers control to the callee labeled `ident`, passes `args`, and
    /// stores the return value in `dst`.
    Call {
        ident: &'a str,
        args: Vec<Value<'a>>,
        dst: Value<'a>,
    },
}

impl fmt::Display for Instruction<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Return(v) => write!(f, "{:<17}{}", "Return", v),
            Instruction::Unary { op, src, dst, .. } => {
                let src_str = format!("{src}");
                let len = src_str.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<17}{src_str} {:>width$}  {dst}",
                    format!("{op:?}"),
                    "->",
                    width = width
                )
            }
            Instruction::Binary {
                op, lhs, rhs, dst, ..
            } => {
                let lhs_str = format!("{lhs}");
                let rhs_str = format!("{rhs}");
                let len = lhs_str.len() + rhs_str.len();

                let max_width: usize = 30;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<17}{lhs_str}, {rhs_str} {:>width$}  {dst}",
                    format!("{op:?}"),
                    "->",
                    width = width
                )
            }
            Instruction::Copy { src, dst } => {
                let src_str = format!("{src}");
                let len = src_str.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<17}{src_str} {:>width$}  {dst}",
                    "Copy",
                    "->",
                    width = width
                )
            }
            Instruction::Jump(label) => write!(f, "{:<17}{:?}", "Jump", label),
            Instruction::JumpIfZero { cond, target } => {
                let cond_str = format!("{cond}");
                let len = cond_str.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<17}{cond_str} {:>width$}  {target:?}",
                    "JumpIfZero",
                    "->",
                    width = width
                )
            }
            Instruction::JumpIfNotZero { cond, target } => {
                let cond_str = format!("{cond}");
                let len = cond_str.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<17}{cond_str} {:>width$}  {target:?}",
                    "JumpIfNotZero",
                    "->",
                    width = width
                )
            }
            Instruction::Call { ident, args, dst } => {
                let args_str = args
                    .iter()
                    .map(|val| format!("{val}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let len = args_str.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len);

                let prefix = format!("Call({ident:?})");

                write!(
                    f,
                    "{:<17}{args_str} {:>width$}  {dst}",
                    prefix,
                    "->",
                    width = width
                )
            }
            Instruction::Label(label) => write!(f, "{:<17}{:?}", "Label", label),
        }
    }
}

/// _IR_ value.
#[derive(Debug, Clone)]
pub enum Value<'a> {
    /// Integer constant.
    IntConstant(c_int),
    /// Temporary variable.
    Var(Cow<'a, str>),
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::IntConstant(v) => write!(f, "{v}"),
            Value::Var(ident) => write!(f, "{ident:?}"),
        }
    }
}

/// Helper for lowering nested _AST_ expressions into three-address code (_TAC_)
/// instructions.
#[derive(Debug)]
struct TACBuilder<'a> {
    instructions: Vec<Instruction<'a>>,
    tmp_count: usize,
    label_count: usize,
    /// Current _AST_ function label.
    label: &'a str,
    /// Canonical names of block-scope static objects, later resolved to _IR_
    /// static items.
    block_statics: Vec<&'a str>,
}

impl<'a> TACBuilder<'a> {
    /// Returns a new temporary variable identifier.
    fn new_tmp(&mut self) -> String {
        // The `.` in temporary identifiers guarantees they won’t conflict
        // with user-defined identifiers, since the _C_ standard forbids `.` in
        // identifiers.
        let ident = format!("{}.tmp.{}", self.label, self.tmp_count);
        self.tmp_count += 1;
        ident
    }

    /// Returns a new label identifier, appending the provided suffix.
    fn new_label(&mut self, suffix: &str) -> String {
        // The `.` in labels  guarantees they won’t conflict with user-defined
        // identifiers, since the _C_ standard forbids `.` in identifiers.
        let label = format!("{}.lbl.{}.{suffix}", self.label, self.label_count);
        self.label_count += 1;
        label
    }

    /// Resets the builder state, providing the next _AST_ function label.
    const fn reset(&mut self, label: &'a str) {
        // Function label already ensures uniqueness for identifiers and labels
        // across the translation unit.
        self.tmp_count = 0;
        self.label_count = 0;

        self.label = label;
    }
}

/// Generate intermediate representation (_IR_), given an abstract syntax tree
/// (_AST_) and symbol map.
///
/// # Panics
///
/// Panics if an identifier could not be found in the symbol map.
#[must_use]
pub fn generate_ir<'a>(ast: &'a ast::AST<'_, Analyzed>, sym_map: &SymbolMap) -> IR<'a> {
    let mut ir_items = vec![];

    let mut builder = TACBuilder {
        instructions: vec![],
        tmp_count: 0,
        label_count: 0,
        label: "",
        block_statics: vec![],
    };

    for decl in &ast.program {
        match decl {
            ast::Declaration::Var {
                specs, ident, init, ..
            } => {
                if specs.storage == Some(StorageClass::Extern) && init.is_none() {
                    // Skip any `extern` _AST_ declarations at file-scope.
                    continue;
                }

                let sym_info = sym_map.get(ident.as_str()).expect(
                    "semantic analysis ensures every identifier is registered in the symbol map",
                );

                let init = match sym_info.state {
                    SymbolState::ConstDefined(i) => i,
                    SymbolState::Tentative => 0,
                    _ => continue,
                };

                ir_items.push(Item::Static {
                    init,
                    ident,
                    // File-scope declarations can either have external or
                    // internal linkage, never none.
                    is_global: sym_info.linkage == Some(Linkage::External),
                });
            }
            ast::Declaration::Func(func) => {
                if func.body.is_some() {
                    builder.reset(&func.ident);

                    ir_items.push(Item::Func(generate_ir_function(
                        func,
                        &mut builder,
                        sym_map,
                    )));
                }
            }
        }
    }

    for ident in builder.block_statics {
        let sym_info = sym_map
            .get(ident)
            .expect("semantic analysis ensures every identifier is registered in the symbol map");

        let init = match sym_info.state {
            SymbolState::ConstDefined(i) => i,
            SymbolState::Defined => 0,
            _ => unreachable!(
                "block-scope static declarations can only be zero-initialized or defined with a constant expression"
            ),
        };

        ir_items.push(Item::Static {
            init,
            ident,
            is_global: false,
        });
    }

    IR { program: ir_items }
}

/// Generate an _IR_ function definition from the provided _AST_ function.
fn generate_ir_function<'a>(
    func: &'a ast::Function<'_>,
    builder: &mut TACBuilder<'a>,
    sym_map: &SymbolMap,
) -> Function<'a> {
    fn process_ast_block<'a>(
        block: &'a ast::Block<'_>,
        builder: &mut TACBuilder<'a>,
        sym_map: &SymbolMap,
    ) {
        for block_item in &block.0 {
            match block_item {
                ast::BlockItem::Stmt(stmt) => process_ast_statement(stmt, builder, sym_map),
                ast::BlockItem::Decl(decl) => process_ast_declaration(decl, builder, sym_map),
            }
        }
    }

    fn process_ast_declaration<'a>(
        decl: &'a ast::Declaration<'_>,
        builder: &mut TACBuilder<'a>,
        sym_map: &SymbolMap,
    ) {
        if let ast::Declaration::Var {
            specs, ident, init, ..
        } = decl
        {
            if specs.storage == Some(StorageClass::Extern) {
                // Skip any `extern` _AST_ declarations at block-scope.
                return;
            }

            let sym_info = sym_map.get(ident.as_str()).expect(
                "semantic analysis ensures every identifier is registered in the symbol map",
            );

            match sym_info.duration {
                Some(StorageDuration::Static) => {
                    builder.block_statics.push(ident.as_str());
                }
                Some(StorageDuration::Automatic) => {
                    if let Some(init) = &init {
                        // Generate and append any instructions needed to encode the
                        // declaration's initializer.
                        let ir_val = generate_ir_value(init, builder);

                        // Ensure the initializer expression result is copied to the
                        // destination.
                        builder.instructions.push(Instruction::Copy {
                            src: ir_val,
                            dst: Value::Var(Cow::Borrowed(ident.as_str())),
                        });
                    }
                }
                None => unreachable!(
                    "only function declarations/definitions should have no storage duration"
                ),
            }
        }
    }

    fn process_ast_statement<'a>(
        stmt: &'a ast::Statement<'_>,
        builder: &mut TACBuilder<'a>,
        sym_map: &SymbolMap,
    ) {
        match stmt {
            ast::Statement::Return(expr) => {
                let ir_val = generate_ir_value(expr, builder);
                builder.instructions.push(Instruction::Return(ir_val));
            }
            ast::Statement::Expression(expr) => {
                // Generate and append any instructions needed to encode the
                // expression.
                let _ = generate_ir_value(expr, builder);
            }
            ast::Statement::If {
                cond,
                then,
                opt_else,
            } => {
                let cond = generate_ir_value(cond, builder);

                let e_lbl = builder.new_label("if.end");

                if let Some(else_stmt) = opt_else {
                    let else_lbl = builder.new_label("else.end");

                    builder.instructions.push(Instruction::JumpIfZero {
                        cond,
                        target: else_lbl.clone(),
                    });

                    // Handle appending instructions for statements recursively.
                    process_ast_statement(then, builder, sym_map);

                    builder.instructions.push(Instruction::Jump(e_lbl.clone()));
                    builder.instructions.push(Instruction::Label(else_lbl));

                    // Handle appending instructions for statements recursively.
                    process_ast_statement(else_stmt, builder, sym_map);
                } else {
                    builder.instructions.push(Instruction::JumpIfZero {
                        cond,
                        target: e_lbl.clone(),
                    });

                    // Handle appending instructions for statements recursively.
                    process_ast_statement(then, builder, sym_map);
                }

                builder.instructions.push(Instruction::Label(e_lbl));
            }
            ast::Statement::Goto { target, .. } => {
                builder.instructions.push(Instruction::Jump(target.clone()));
            }
            ast::Statement::LabeledStatement(labeled) => {
                match labeled {
                    ast::Labeled::Label { label, stmt, .. }
                    | ast::Labeled::Case {
                        jmp_label: label,
                        stmt,
                        ..
                    }
                    | ast::Labeled::Default {
                        jmp_label: label,
                        stmt,
                        ..
                    } => {
                        builder.instructions.push(Instruction::Label(label.clone()));

                        // Handle appending instructions for statements
                        // recursively.
                        process_ast_statement(stmt, builder, sym_map);
                    }
                }
            }
            ast::Statement::Compound(block) => process_ast_block(block, builder, sym_map),
            ast::Statement::Break { jmp_label, .. } => {
                builder
                    .instructions
                    .push(Instruction::Jump(format!("break_{jmp_label}")));
            }
            ast::Statement::Continue { jmp_label, .. } => {
                builder
                    .instructions
                    .push(Instruction::Jump(format!("cont_{jmp_label}")));
            }
            ast::Statement::Do {
                stmt,
                cond,
                loop_label: label,
            } => {
                let start_label = builder.new_label("do.start");
                let cont_label = format!("cont_{label}");
                let break_label = format!("break_{label}");

                builder
                    .instructions
                    .push(Instruction::Label(start_label.clone()));

                // Handle appending instructions for statements recursively.
                process_ast_statement(stmt, builder, sym_map);

                builder.instructions.push(Instruction::Label(cont_label));

                let cond = generate_ir_value(cond, builder);

                builder.instructions.push(Instruction::JumpIfNotZero {
                    cond,
                    target: start_label,
                });

                builder.instructions.push(Instruction::Label(break_label));
            }
            ast::Statement::While {
                cond,
                stmt,
                loop_label: label,
            } => {
                let cont_label = format!("cont_{label}");
                let break_label = format!("break_{label}");

                builder
                    .instructions
                    .push(Instruction::Label(cont_label.clone()));

                let cond = generate_ir_value(cond, builder);

                builder.instructions.push(Instruction::JumpIfZero {
                    cond,
                    target: break_label.clone(),
                });

                // Handle appending instructions for statements recursively.
                process_ast_statement(stmt, builder, sym_map);

                builder.instructions.push(Instruction::Jump(cont_label));

                builder.instructions.push(Instruction::Label(break_label));
            }
            ast::Statement::For {
                init,
                opt_cond,
                opt_post,
                stmt,
                loop_label: label,
            } => {
                match &**init {
                    ast::ForInit::Decl(decl) => process_ast_declaration(decl, builder, sym_map),
                    ast::ForInit::Expr(opt_expr) => {
                        if let Some(expr) = opt_expr {
                            // Generate and append any instructions needed to
                            // encode the expression.
                            let _ = generate_ir_value(expr, builder);
                        }
                    }
                }

                let start_label = builder.new_label("for.start");
                let cont_label = format!("cont_{label}");
                let break_label = format!("break_{label}");

                builder
                    .instructions
                    .push(Instruction::Label(start_label.clone()));

                // _C17_ 6.8.5.3 (The for statement)
                //
                // An omitted expression-2 is replaced by a nonzero constant.
                //
                // This is essentially the same as not including the jump
                // instruction at all, since a nonzero constant would never
                // equal zero.
                if let Some(cond) = opt_cond {
                    let cond = generate_ir_value(cond, builder);

                    builder.instructions.push(Instruction::JumpIfZero {
                        cond,
                        target: break_label.clone(),
                    });
                }

                // Handle appending instructions for statements recursively.
                process_ast_statement(stmt, builder, sym_map);

                builder.instructions.push(Instruction::Label(cont_label));

                if let Some(post) = opt_post {
                    // Generate and append any instructions needed to encode the
                    // expression.
                    let _ = generate_ir_value(post, builder);
                }

                builder.instructions.push(Instruction::Jump(start_label));

                // Always emitting labels for `break` statements.
                builder.instructions.push(Instruction::Label(break_label));
            }
            ast::Statement::Switch {
                cond,
                stmt,
                cases,
                default,
                switch_label: label,
            } => {
                let lhs = generate_ir_value(cond, builder);

                // Only generate code for the switch if there are case labels or
                // default label.
                if !cases.is_empty() || default.is_some() {
                    let break_label = format!("break_{label}");

                    for case in cases {
                        let dst = Value::Var(Cow::Owned(builder.new_tmp()));

                        let rhs = generate_ir_value(&case.expr, builder);

                        builder.instructions.push(Instruction::Binary {
                            op: BinaryOperator::Eq,
                            lhs: lhs.clone(),
                            rhs,
                            dst: dst.clone(),
                            // NOTE: Temporary hack for arithmetic right shift.
                            sign: Signedness::Unsigned,
                        });

                        builder.instructions.push(Instruction::JumpIfNotZero {
                            cond: dst,
                            target: case.jmp_label.clone(),
                        });
                    }

                    if let Some(default_label) = default {
                        builder
                            .instructions
                            .push(Instruction::Jump(default_label.clone()));
                    } else {
                        builder
                            .instructions
                            .push(Instruction::Jump(break_label.clone()));
                    }

                    // Handle appending instructions for statements recursively.
                    process_ast_statement(stmt, builder, sym_map);

                    builder.instructions.push(Instruction::Label(break_label));
                }
            }
            ast::Statement::Empty => {}
        }
    }

    let label = func.ident.as_str();
    let params = func
        .params
        .iter()
        .map(|param| param.ident.as_str())
        .collect::<Vec<_>>();

    let body = &func
        .body
        .as_ref()
        .expect("should not generate IR for function declarations");

    process_ast_block(body, builder, sym_map);

    // _C17_ 5.1.2.2.3 (Program termination)
    //
    // For the `main` function only, reaching the closing `}` implicitly returns
    // 0.
    //
    // _C17_ 6.9.1 (Function definitions)
    //
    // Unless otherwise specified, if the `}` that terminates a function is
    // reached, and the value of the function call is used by the caller, the
    // behavior is undefined.
    //
    // Just appending an extra `Instruction::Return` handles the edge cases of
    // the return value being used by the caller or ignored (no undefined
    // behavior if the value is never used).
    builder
        .instructions
        .push(Instruction::Return(Value::IntConstant(0)));

    let sym_info = sym_map
        .get(label)
        .expect("semantic analysis ensures every identifier is registered in the symbol map");

    Function {
        instructions: builder.instructions.drain(..).collect(),
        ident: label,
        params,
        // File-scope declarations can either have external or internal linkage,
        // never none.
        is_global: sym_info.linkage == Some(Linkage::External),
    }
}

/// Generate an _IR_ value from the provided _AST_ expression.
fn generate_ir_value<'a>(expr: &'a ast::Expression<'_>, builder: &mut TACBuilder<'a>) -> Value<'a> {
    match expr {
        ast::Expression::IntConstant(v) => Value::IntConstant(*v),
        ast::Expression::Var { ident, .. } => Value::Var(Cow::Borrowed(ident.as_str())),
        ast::Expression::Unary {
            op, expr, prefix, ..
        } => {
            // The sign of an _IR_ instruction is determined by the
            // sub-expressions (here `expr`), not by when the operator is
            // applied to them.
            //
            // NOTE: Temporary hack for arithmetic right shift.
            let sign = match **expr {
                ast::Expression::Unary { sign, .. } | ast::Expression::Binary { sign, .. } => sign,
                _ => Signedness::Unsigned,
            };

            // Recursively process the expression until the base case is reached.
            // This ensures the inner expression is processed initially before
            // unwinding.
            let src = generate_ir_value(expr, builder);
            let dst = Value::Var(Cow::Owned(builder.new_tmp()));

            match op {
                UnaryOperator::Increment | UnaryOperator::Decrement => {
                    debug_assert!(matches!(src, Value::Var(_)));

                    let binop = match op {
                        UnaryOperator::Increment => BinaryOperator::Add,
                        UnaryOperator::Decrement => BinaryOperator::Subtract,
                        _ => unreachable!(
                            "only increment/decrement unary operators should reach this match arm"
                        ),
                    };

                    let tmp = Value::Var(Cow::Owned(builder.new_tmp()));

                    builder.instructions.push(Instruction::Binary {
                        op: binop,
                        lhs: src.clone(),
                        rhs: Value::IntConstant(1),
                        dst: tmp.clone(),
                        sign,
                    });

                    if *prefix {
                        builder.instructions.push(Instruction::Copy {
                            src: tmp,
                            dst: src.clone(),
                        });

                        // Updated value of `src` is moved to `dst`.
                        builder.instructions.push(Instruction::Copy {
                            src,
                            dst: dst.clone(),
                        });
                    } else {
                        // Original value of `src` is moved to `dst`.
                        builder.instructions.push(Instruction::Copy {
                            src: src.clone(),
                            dst: dst.clone(),
                        });

                        builder
                            .instructions
                            .push(Instruction::Copy { src: tmp, dst: src });
                    }
                }
                op => {
                    builder.instructions.push(Instruction::Unary {
                        op: *op,
                        src,
                        dst: dst.clone(),
                        sign,
                    });
                }
            }

            dst
        }
        ast::Expression::Binary { op, lhs, rhs, .. } => {
            // The sign of an _IR_ instruction is determined by the
            // sub-expressions (here `expr`), not by when the operator is
            // applied to them.
            //
            // NOTE: Temporary hack for arithmetic right shift.
            let sign = match **lhs {
                ast::Expression::Unary { sign, .. } | ast::Expression::Binary { sign, .. } => sign,
                _ => Signedness::Unsigned,
            };

            match op {
                ast::BinaryOperator::LogAnd | ast::BinaryOperator::LogOr => {
                    let lhs = generate_ir_value(lhs, builder);
                    let dst = Value::Var(Cow::Owned(builder.new_tmp()));

                    // Need to short-circuit if the `lhs` is 0 for `&&`. In
                    // those cases, `rhs` instructions are never executed.
                    if matches!(op, ast::BinaryOperator::LogAnd) {
                        let f_lbl = builder.new_label("and.false");
                        let e_lbl = builder.new_label("and.end");

                        builder.instructions.push(Instruction::JumpIfZero {
                            cond: lhs,
                            target: f_lbl.clone(),
                        });

                        // Evaluate `rhs` iff `lhs` is non-zero.
                        let rhs = generate_ir_value(rhs, builder);

                        builder.instructions.extend([
                            Instruction::JumpIfZero {
                                cond: rhs,
                                target: f_lbl.clone(),
                            },
                            Instruction::Copy {
                                src: Value::IntConstant(1),
                                dst: dst.clone(),
                            },
                            Instruction::Jump(e_lbl.clone()),
                            Instruction::Label(f_lbl),
                            Instruction::Copy {
                                src: Value::IntConstant(0),
                                dst: dst.clone(),
                            },
                            Instruction::Label(e_lbl),
                        ]);
                    } else {
                        // Need to short-circuit if the `lhs` is non-zero for
                        // `||`. In those cases, `rhs` instructions are never
                        // executed.
                        let t_lbl = builder.new_label("or.true");
                        let e_lbl = builder.new_label("or.end");

                        builder.instructions.push(Instruction::JumpIfNotZero {
                            cond: lhs,
                            target: t_lbl.clone(),
                        });

                        // Evaluate `rhs` iff `lhs` is zero.
                        let rhs = generate_ir_value(rhs, builder);

                        builder.instructions.extend([
                            Instruction::JumpIfNotZero {
                                cond: rhs,
                                target: t_lbl.clone(),
                            },
                            Instruction::Copy {
                                src: Value::IntConstant(0),
                                dst: dst.clone(),
                            },
                            Instruction::Jump(e_lbl.clone()),
                            Instruction::Label(t_lbl),
                            Instruction::Copy {
                                src: Value::IntConstant(1),
                                dst: dst.clone(),
                            },
                            Instruction::Label(e_lbl),
                        ]);
                    }

                    dst
                }
                _ => {
                    // _C17_ 5.1.2.3 (Program execution)
                    //
                    // Sub-expressions of the same operation are _unsequenced_
                    // (few exceptions). This means either the `lhs` or the
                    // `rhs` can be processed first.
                    let lhs = generate_ir_value(lhs, builder);
                    let rhs = generate_ir_value(rhs, builder);
                    let dst = Value::Var(Cow::Owned(builder.new_tmp()));

                    builder.instructions.push(Instruction::Binary {
                        op: *op,
                        lhs,
                        rhs,
                        dst: dst.clone(),
                        sign,
                    });

                    dst
                }
            }
        }
        ast::Expression::Assignment { lvalue, rvalue, .. } => {
            let dst = match &**lvalue {
                ast::Expression::Var { ident, .. } => Value::Var(Cow::Borrowed(ident.as_str())),
                _ => unreachable!("lvalue of an expression should be Expression::Var"),
            };

            let result = generate_ir_value(rvalue, builder);

            builder.instructions.push(Instruction::Copy {
                src: result,
                dst: dst.clone(),
            });

            dst
        }
        ast::Expression::Conditional {
            cond,
            second,
            third,
        } => {
            let dst = Value::Var(Cow::Owned(builder.new_tmp()));
            let e_lbl = builder.new_label("cond.end");
            let rhs_lbl = builder.new_label("cond.false");

            let cond = generate_ir_value(cond, builder);

            builder.instructions.push(Instruction::JumpIfZero {
                cond,
                target: rhs_lbl.clone(),
            });

            let second = generate_ir_value(second, builder);
            builder.instructions.push(Instruction::Copy {
                src: second,
                dst: dst.clone(),
            });

            builder.instructions.push(Instruction::Jump(e_lbl.clone()));
            builder.instructions.push(Instruction::Label(rhs_lbl));

            let third = generate_ir_value(third, builder);
            builder.instructions.push(Instruction::Copy {
                src: third,
                dst: dst.clone(),
            });

            builder.instructions.push(Instruction::Label(e_lbl));

            dst
        }
        ast::Expression::FuncCall { ident, args, .. } => {
            let mut ir_args = vec![];
            let dst = Value::Var(Cow::Owned(builder.new_tmp()));

            for expr in args {
                ir_args.push(generate_ir_value(expr, builder));
            }

            builder.instructions.push(Instruction::Call {
                ident: ident.as_str(),
                args: ir_args,
                dst: dst.clone(),
            });

            dst
        }
    }
}
