//! Intermediate Representation
//!
//! Compiler pass that lowers an abstract syntax tree (_AST_) into intermediate
//! representation (_IR_) using three-address code (_TAC_).

use std::fmt;

use crate::compiler::parser::ast::{self, BinaryOperator, Signedness, UnaryOperator};

/// Intermediate representation (_IR_).
#[derive(Debug)]
pub enum IR {
    /// Function that represent the structure of the program.
    Program(Function),
}

impl fmt::Display for IR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IR::Program(func) => {
                write!(f, "IR Program\n{:4}{func}", "")
            }
        }
    }
}

/// _IR_ function definition.
#[derive(Debug)]
pub struct Function {
    pub ident: String,
    pub instructions: Vec<Instruction>,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Ident({:?})", self.ident)?;

        for inst in &self.instructions {
            writeln!(f, "{:8}{inst}", "")?;
        }

        Ok(())
    }
}

/// _IR_ instruction.
#[derive(Debug)]
pub enum Instruction {
    /// Returns a value to the caller.
    Return(Value),
    /// Perform a unary operation on `src`, storing the result in `dst`.
    ///
    /// The `dst` of any unary instruction must be `Value::Var`.
    Unary {
        op: UnaryOperator,
        src: Value,
        dst: Value,
        /// NOTE: Temporary hack for arithmetic right shift.
        sign: Signedness,
    },
    /// Perform a binary operation on `lhs` and `rhs`, storing the result in
    /// `dst`.
    ///
    /// The `dst` of any binary instruction must be `Value::Var`.
    Binary {
        op: BinaryOperator,
        lhs: Value,
        rhs: Value,
        dst: Value,
        /// NOTE: Temporary hack for arithmetic right shift.
        sign: Signedness,
    },
    /// Copies the value from `src` into `dst`.
    Copy { src: Value, dst: Value },
    /// Unconditionally jumps to the point in code labeled by an "identifier".
    Jump(String),
    /// Conditionally jumps to the point in code labeled by an "identifier" if
    /// the condition evaluates to zero.
    JumpIfZero { cond: Value, target: String },
    /// Conditionally jumps to the point in code labeled by an "identifier" if
    /// the condition does not evaluates to zero.
    JumpIfNotZero { cond: Value, target: String },
    /// Associates an "identifier" with a location in the program.
    Label(String),
}

impl fmt::Display for Instruction {
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
            Instruction::Label(i) => write!(f, "{:<17}{:?}", "Label", i),
        }
    }
}

/// _IR_ value.
#[derive(Debug, Clone)]
pub enum Value {
    /// Constant int value (32-bit).
    IntConstant(i32),
    /// Temporary variable.
    Var(String),
}

impl fmt::Display for Value {
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
    instructions: Vec<Instruction>,
    // For temporary variables.
    tmp_count: usize,
    // For `jmp` labels.
    label_count: usize,
    // Function label.
    label: &'a str,
}

impl TACBuilder<'_> {
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
}

/// Generate intermediate representation (_IR_), given an abstract syntax tree
/// (_AST_). [Exits] on error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn generate_ir(ast: &ast::AST) -> IR {
    match ast {
        ast::AST::Program(func) => IR::Program(generate_ir_function(func)),
    }
}

/// Generate an _IR_ function definition from the provided _AST_ function.
fn generate_ir_function(func: &ast::Function) -> Function {
    fn process_ast_block(block: &ast::Block, builder: &mut TACBuilder<'_>) {
        for block_item in &block.0 {
            match block_item {
                ast::BlockItem::Stmt(stmt) => process_ast_statement(stmt, builder),
                ast::BlockItem::Decl(decl) => process_ast_declaration(decl, builder),
            }
        }
    }

    fn process_ast_declaration(decl: &ast::Declaration, builder: &mut TACBuilder<'_>) {
        if let Some(init) = &decl.init {
            // Generate and append any instructions needed to encode the
            // declaration's initializer.
            let ir_val = generate_ir_value(init, builder);

            // Ensure the initializer expression result is copied to the
            // destination.
            builder.instructions.push(Instruction::Copy {
                src: ir_val,
                dst: Value::Var(decl.ident.clone()),
            });
        }
    }

    fn process_ast_statement(stmt: &ast::Statement, builder: &mut TACBuilder<'_>) {
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
            ast::Statement::Goto((target, _)) => {
                builder.instructions.push(Instruction::Jump(target.clone()));
            }
            ast::Statement::LabeledStatement(labeled) => {
                match labeled {
                    ast::Labeled::Label { label, stmt, .. }
                    | ast::Labeled::Case { label, stmt, .. }
                    | ast::Labeled::Default { label, stmt, .. } => {
                        builder.instructions.push(Instruction::Label(label.clone()));

                        // Handle appending instructions for statements
                        // recursively.
                        process_ast_statement(stmt, builder);
                    }
                }
            }
            ast::Statement::Compound(block) => process_ast_block(block, builder),
            ast::Statement::Break((label, _)) => {
                builder
                    .instructions
                    .push(Instruction::Jump(format!("break_{label}")));
            }
            ast::Statement::Continue((label, _)) => {
                builder
                    .instructions
                    .push(Instruction::Jump(format!("cont_{label}")));
            }
            ast::Statement::Do { stmt, cond, label } => {
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
            ast::Statement::While { cond, stmt, label } => {
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
                label,
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
                label,
            } => {
                let break_label = format!("break_{label}");

                let lhs = generate_ir_value(cond, builder);

                for (label, expr) in cases {
                    let dst = Value::Var(builder.new_tmp());

                    let rhs = generate_ir_value(expr, builder);

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
                        target: label.clone(),
                    });
                }

                if let Some(default_label) = default {
                    builder
                        .instructions
                        .push(Instruction::Jump(default_label.clone()));
                }

                // Handle appending instructions for statements recursively.
                process_ast_statement(stmt, builder);

                // Always emitting labels for `break` statements.
                builder.instructions.push(Instruction::Label(break_label));
            }
            ast::Statement::Empty => {}
        }
    }

    let label = func.ident.clone();

    let mut builder = TACBuilder {
        instructions: vec![],
        tmp_count: 0,
        label_count: 0,
        label: &label,
    };

    process_ast_block(&func.body, &mut builder);

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
    // As a hack, just appending an extra `Instruction::Return` handles the edge
    // cases of the return value being used by the caller or ignored (no
    // undefined behavior if the value is never used).
    builder
        .instructions
        .push(Instruction::Return(Value::IntConstant(0)));

    Function {
        instructions: builder.instructions,
        ident: label,
    }
}

/// Generate an _IR_ value from the provided _AST_ expression.
fn generate_ir_value(expr: &ast::Expression, builder: &mut TACBuilder<'_>) -> Value {
    match expr {
        ast::Expression::IntConstant(v) => Value::IntConstant(*v),
        ast::Expression::Var((v, _)) => Value::Var(v.clone()),
        ast::Expression::Unary {
            op, expr, prefix, ..
        } => {
            // The sign of an _IR_ instruction is determined by the
            // sub-expressions (here `expr`), not by when the operator is
            // applied to them.
            //
            // NOTE: Temporary hack for arithmetic right shift.
            let sign = match **expr {
                ast::Expression::Unary { sign, .. } => sign,
                ast::Expression::Binary { sign, .. } => sign,
                _ => Signedness::Unsigned,
            };

            // Recursively process the expression until the base case is
            // reached. This ensures the inner expression is processed initially
            // before unwinding.
            let src = generate_ir_value(expr, builder);
            let dst = Value::Var(builder.new_tmp());

            match op {
                UnaryOperator::Increment | UnaryOperator::Decrement => {
                    debug_assert!(matches!(src, Value::Var(_)));

                    let binop = match op {
                        UnaryOperator::Increment => BinaryOperator::Add,
                        UnaryOperator::Decrement => BinaryOperator::Subtract,
                        _ => unreachable!(),
                    };

                    let tmp = Value::Var(builder.new_tmp());

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
                ast::Expression::Unary { sign, .. } => sign,
                ast::Expression::Binary { sign, .. } => sign,
                _ => Signedness::Unsigned,
            };

            match op {
                // Need to short-circuit if the `lhs` is 0 for `&&`, or `lhs` is
                // non-zero for `||`. In those cases, `rhs` instructions are
                // never executed.
                ast::BinaryOperator::LogAnd | ast::BinaryOperator::LogOr => {
                    let lhs = generate_ir_value(lhs, builder);
                    let dst = Value::Var(builder.new_tmp());

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
                    let dst = Value::Var(builder.new_tmp());

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
                ast::Expression::Var((v, _)) => Value::Var(v.clone()),
                _ => unreachable!("lvalue of an expression should be an `Expression::Var`"),
            };

            let result = generate_ir_value(rvalue, builder);

            builder.instructions.push(Instruction::Copy {
                src: result,
                dst: dst.clone(),
            });

            dst
        }
        ast::Expression::Conditional(lhs, mid, rhs) => {
            let dst = Value::Var(builder.new_tmp());
            let e_lbl = builder.new_label("cond.end");
            let rhs_lbl = builder.new_label("cond.false");

            let cond = generate_ir_value(lhs, builder);

            builder.instructions.push(Instruction::JumpIfZero {
                cond,
                target: rhs_lbl.clone(),
            });

            let e1 = generate_ir_value(mid, builder);
            builder.instructions.push(Instruction::Copy {
                src: e1,
                dst: dst.clone(),
            });

            builder.instructions.push(Instruction::Jump(e_lbl.clone()));
            builder.instructions.push(Instruction::Label(rhs_lbl));

            let e2 = generate_ir_value(rhs, builder);
            builder.instructions.push(Instruction::Copy {
                src: e2,
                dst: dst.clone(),
            });

            builder.instructions.push(Instruction::Label(e_lbl));

            dst
        }
    }
}
