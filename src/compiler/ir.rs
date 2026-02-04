//! Intermediate Representation
//!
//! Compiler pass that lowers an abstract syntax tree (_AST_) into intermediate
//! representation (_IR_) using three-address code (_TAC_).

use std::{borrow::Cow, fmt};

use crate::compiler::parser::ast::{self, Analyzed, BinaryOperator, Signedness, UnaryOperator};

/// Intermediate representation (_IR_).
#[derive(Debug)]
pub struct IR<'a> {
    /// Function that represent the structure of the program.
    pub program: Vec<Function<'a>>,
}

impl fmt::Display for IR<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IR Program")?;
        for func in &self.program {
            writeln!(f, "{:4}{func}", "")?;
        }

        Ok(())
    }
}

/// _IR_ function definition.
#[derive(Debug)]
pub struct Function<'a> {
    pub ident: &'a str,
    pub params: Vec<&'a str>,
    pub instructions: Vec<Instruction<'a>>,
}

impl fmt::Display for Function<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params = self.params.join(", ");

        writeln!(f, "Fn {:?}({})", self.ident, params)?;

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
    ///
    /// The `dst` of any unary instruction must be `Value::Var`.
    Unary {
        op: UnaryOperator,
        src: Value<'a>,
        dst: Value<'a>,
        // NOTE: Temporary hack for arithmetic right shift.
        sign: Signedness,
    },
    /// Perform a binary operation on `lhs` and `rhs`, storing the result in
    /// `dst`.
    ///
    /// The `dst` of any binary instruction must be `Value::Var`.
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
    /// Unconditionally jumps to the point in code labeled by an "identifier".
    Jump(String),
    /// Conditionally jumps to the point in code labeled by an "identifier" if
    /// the condition evaluates to zero.
    JumpIfZero { cond: Value<'a>, target: String },
    /// Conditionally jumps to the point in code labeled by an "identifier" if
    /// the condition does not evaluates to zero.
    JumpIfNotZero { cond: Value<'a>, target: String },
    /// Associates an "identifier" with a location in the program.
    Label(String),
    /// Function call.
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
            Instruction::Jump(i) => write!(f, "{:<17}{:?}", "Jump", i),
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
                let src_str = args
                    .iter()
                    .map(|val| format!("{val}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let len = src_str.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len);

                let prefix = format!("Call({ident:?})");

                write!(
                    f,
                    "{:<17}{src_str} {:>width$}  {dst}",
                    prefix,
                    "->",
                    width = width
                )
            }
            Instruction::Label(i) => write!(f, "{:<17}{:?}", "Label", i),
        }
    }
}

/// _IR_ value.
#[derive(Debug, Clone)]
pub enum Value<'a> {
    /// Constant int value (32-bit).
    IntConstant(i32),
    /// Temporary variable.
    Var(Cow<'a, str>),
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::IntConstant(v) => write!(f, "{v}"),
            Value::Var(i) => write!(f, "{i:?}"),
        }
    }
}

/// Helper for lowering nested _AST_ expressions into three-address code (_TAC_)
/// instructions.
struct TACBuilder<'a> {
    instructions: Vec<Instruction<'a>>,
    // For temporary variables.
    tmp_count: usize,
    // For `jmp` labels.
    label_count: usize,
    // Function label.
    label: &'a str,
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

    /// Resets the builder state, given the next _AST_ function label.
    fn reset(&mut self, label: &'a str) {
        // Function label already ensures uniqueness for identifiers and labels
        // across the translation unit.
        self.tmp_count = 0;
        self.label_count = 0;

        self.label = label;
    }
}

/// Generate intermediate representation (_IR_), given an abstract syntax tree
/// (_AST_).
#[must_use]
pub fn generate_ir<'a>(ast: &'a ast::AST<'_, Analyzed>) -> IR<'a> {
    let mut ir_funcs = vec![];

    let mut builder = TACBuilder {
        instructions: vec![],
        tmp_count: 0,
        label_count: 0,
        label: "",
    };

    for func in &ast.program {
        if func.body.is_some() {
            builder.reset(&func.ident);
            ir_funcs.push(generate_ir_function(func, &mut builder));
        }
    }

    IR { program: ir_funcs }
}

/// Generate an _IR_ function definition from the provided _AST_ function.
fn generate_ir_function<'a>(
    func: &'a ast::Function<'_>,
    builder: &mut TACBuilder<'a>,
) -> Function<'a> {
    fn process_ast_block<'a>(block: &'a ast::Block<'_>, builder: &mut TACBuilder<'a>) {
        for block_item in &block.0 {
            match block_item {
                ast::BlockItem::Stmt(stmt) => process_ast_statement(stmt, builder),
                ast::BlockItem::Decl(decl) => process_ast_declaration(decl, builder),
            }
        }
    }

    fn process_ast_declaration<'a>(decl: &'a ast::Declaration<'_>, builder: &mut TACBuilder<'a>) {
        if let ast::Declaration::Var { ident, init, .. } = decl
            && let Some(init) = &init
        {
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

    fn process_ast_statement<'a>(stmt: &'a ast::Statement<'_>, builder: &mut TACBuilder<'a>) {
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
                    process_ast_statement(then, builder);

                    builder.instructions.push(Instruction::Jump(e_lbl.clone()));
                    builder.instructions.push(Instruction::Label(else_lbl));

                    // Handle appending instructions for statements recursively.
                    process_ast_statement(else_stmt, builder);

                    builder.instructions.push(Instruction::Label(e_lbl));
                } else {
                    builder.instructions.push(Instruction::JumpIfZero {
                        cond,
                        target: e_lbl.clone(),
                    });

                    // Handle appending instructions for statements recursively.
                    process_ast_statement(then, builder);

                    builder.instructions.push(Instruction::Label(e_lbl));
                }
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
                        process_ast_statement(stmt, builder);
                    }
                }
            }
            ast::Statement::Compound(block) => process_ast_block(block, builder),
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
                process_ast_statement(stmt, builder);

                // Always emitting labels for `continue` statements.
                builder.instructions.push(Instruction::Label(cont_label));

                let cond = generate_ir_value(cond, builder);

                builder.instructions.push(Instruction::JumpIfNotZero {
                    cond,
                    target: start_label,
                });

                // Always emitting labels for `break` statements.
                builder.instructions.push(Instruction::Label(break_label));
            }
            ast::Statement::While {
                cond,
                stmt,
                loop_label: label,
            } => {
                let cont_label = format!("cont_{label}");
                let break_label = format!("break_{label}");

                // Always emitting labels for `continue` statements.
                builder
                    .instructions
                    .push(Instruction::Label(cont_label.clone()));

                let cond = generate_ir_value(cond, builder);

                builder.instructions.push(Instruction::JumpIfZero {
                    cond,
                    target: break_label.clone(),
                });

                // Handle appending instructions for statements recursively.
                process_ast_statement(stmt, builder);

                builder.instructions.push(Instruction::Jump(cont_label));

                // Always emitting labels for `break` statements.
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
                    ast::ForInit::Decl(decl) => process_ast_declaration(decl, builder),
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
                process_ast_statement(stmt, builder);

                // Always emitting labels for `continue` statements.
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
                // a default.
                if !cases.is_empty() || default.is_some() {
                    let break_label = format!("break_{label}");

                    // for (label, expr) in cases {

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
                    process_ast_statement(stmt, builder);

                    // Always emitting labels for `break` statements.
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

    process_ast_block(body, builder);

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

    Function {
        instructions: builder.instructions.drain(..).collect(),
        ident: label,
        params,
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

            // Recursively process the expression until the base case is
            // reached. This ensures the inner expression is processed initially
            // before unwinding.
            let src = generate_ir_value(expr, builder);
            let dst = Value::Var(Cow::Owned(builder.new_tmp()));

            match op {
                UnaryOperator::Increment | UnaryOperator::Decrement => {
                    debug_assert!(matches!(src, Value::Var(_)));

                    let binop = match op {
                        UnaryOperator::Increment => BinaryOperator::Add,
                        UnaryOperator::Decrement => BinaryOperator::Subtract,
                        _ => unreachable!(),
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

                        // Prefix - the updated value of `src` is moved to
                        // `dst`.
                        builder.instructions.push(Instruction::Copy {
                            src,
                            dst: dst.clone(),
                        });
                    } else {
                        // Postfix - the original value of `src` is moved to
                        // `dst`.
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

            // The returned `dst` is used as the new `src` in each unwind of
            // the recursion.
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
                // Need to short-circuit if the `lhs` is 0 for `&&`, or `lhs` is
                // non-zero for `||`. In those cases, `rhs` instructions are
                // never executed.
                ast::BinaryOperator::LogAnd | ast::BinaryOperator::LogOr => {
                    let lhs = generate_ir_value(lhs, builder);
                    let dst = Value::Var(Cow::Owned(builder.new_tmp()));

                    if let ast::BinaryOperator::LogAnd = op {
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
                _ => unreachable!("lvalue of an expression should be an `Expression::Var`"),
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
