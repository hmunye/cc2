//! Intermediate Representation
//!
//! Compiler pass that lowers an abstract syntax tree (_AST_) into intermediate
//! representation (_IR_) using three-address code (_TAC_).

use std::fmt;

use crate::compiler::parser::{self, BinaryOperator, Signedness, UnaryOperator};

type Ident = String;

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
#[allow(missing_docs)]
pub struct Function {
    pub ident: Ident,
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

/// _IR_ instructions.
#[derive(Debug)]
pub enum Instruction {
    /// Returns a value to the caller.
    Return(Value),
    /// Perform a unary operation on `src`, storing the result in `dst`.
    ///
    /// The `dst` of any unary instruction must be `Value::Var`.
    #[allow(missing_docs)]
    Unary {
        op: UnaryOperator,
        src: Value,
        dst: Value,
        sign: Signedness,
    },
    /// Perform a binary operation on `lhs` and `rhs`, storing the result in
    /// `dst`.
    ///
    /// The `dst` of any binary instruction must be `Value::Var`.
    #[allow(missing_docs)]
    Binary {
        op: BinaryOperator,
        lhs: Value,
        rhs: Value,
        dst: Value,
        sign: Signedness,
    },
    /// Copies the value from `src` into `dst`.
    #[allow(missing_docs)]
    Copy { src: Value, dst: Value },
    /// Unconditionally jumps to the point in code labeled by an "identifier".
    Jump(Ident),
    /// Conditionally jumps to the point in code labeled by an "identifier" if
    /// the condition evaluates to zero.
    #[allow(missing_docs)]
    JumpIfZero { cond: Value, target: Ident },
    /// Conditionally jumps to the point in code labeled by an "identifier" if
    /// the condition does not evaluates to zero.
    #[allow(missing_docs)]
    JumpIfNotZero { cond: Value, target: Ident },
    /// Associates an "identifier" with a location in the program.
    Label(Ident),
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Return(v) => write!(f, "{:<17}{}", "Return", v),
            Instruction::Unary { op, src, dst, sign } => {
                let src_str = format!("{src}");
                let len = src_str.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<17}{src_str} {:>width$}  {dst}",
                    format!(
                        "{op:?}({})",
                        if let Signedness::Signed = sign {
                            "S"
                        } else {
                            "U"
                        }
                    ),
                    "->",
                    width = width
                )
            }
            Instruction::Binary {
                op,
                lhs,
                rhs,
                dst,
                sign,
            } => {
                let lhs_str = format!("{lhs}");
                let rhs_str = format!("{rhs}");
                let len = lhs_str.len() + rhs_str.len();

                let max_width: usize = 30;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<17}{lhs_str}, {rhs_str} {:>width$}  {dst}",
                    format!(
                        "{op:?}({})",
                        if let Signedness::Signed = sign {
                            "S"
                        } else {
                            "U"
                        }
                    ),
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

/// _IR_ values.
#[derive(Debug, Clone)]
pub enum Value {
    /// Constant int value (32-bit).
    ConstantInt(i32),
    /// Temporary variable.
    Var(Ident),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::ConstantInt(v) => write!(f, "{v}"),
            Value::Var(i) => write!(f, "{i:?}"),
        }
    }
}

/// Helper for lowering nested _AST_ expressions into three-address code (_TAC_)
/// instructions.
struct TACBuilder<'a> {
    instructions: Vec<Instruction>,
    tmp_count: usize,
    label_count: usize,
    // Function label.
    label: &'a str,
}

impl TACBuilder<'_> {
    /// Allocates and returns a fresh temporary variable identifier.
    fn new_tmp(&mut self) -> Ident {
        // The `.` in temporary identifiers guarantees they won’t conflict
        // with user-defined identifiers, since the _C_ standard forbids `.` in
        // identifiers.
        let ident = format!("{}.tmp.{}", self.label, self.tmp_count);
        self.tmp_count += 1;
        ident
    }

    /// Allocates and returns a fresh label identifier.
    fn new_label(&mut self, suffix: &str) -> Ident {
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
pub fn generate_ir(ast: &parser::AST) -> IR {
    match ast {
        parser::AST::Program(func) => {
            let ir_function = generate_ir_function(func);
            IR::Program(ir_function)
        }
    }
}

/// Generate an _IR_ function definition from the provided _AST_ function.
fn generate_ir_function(func: &parser::Function) -> Function {
    let label = func.ident.clone();

    let mut builder = TACBuilder {
        instructions: vec![],
        tmp_count: 0,
        label_count: 0,
        label: &label,
    };

    for block in &func.body {
        match block {
            parser::Block::Stmt(stmt) => match stmt {
                parser::Statement::Return(expr) => {
                    let ir_val = generate_ir_value(expr, &mut builder);
                    builder.instructions.push(Instruction::Return(ir_val));
                }
                parser::Statement::Expression(expr) => {
                    // Generate and append any instructions needed to encode the
                    // expression.
                    let _ = generate_ir_value(expr, &mut builder);
                }
                parser::Statement::Null => {}
            },
            parser::Block::Decl(decl) => {
                if let Some(init) = &decl.init {
                    // Generate and append any instructions needed to encode the
                    // declaration's initializer.
                    let ir_val = generate_ir_value(init, &mut builder);

                    builder.instructions.push(Instruction::Copy {
                        src: ir_val,
                        dst: Value::Var(decl.ident.clone()),
                    });
                }
            }
        }
    }

    // According to the _C_ standard, the "main" function without a return
    // statement implicitly returns 0. For other functions that declare a return
    // type but have no return statement, the use of that return value by the
    // caller is undefined behavior.
    //
    // As a hack, just appending an extra `Instruction::Return` handles the edge
    // cases of the return value being used by the caller or ignored (no
    // undefined behavior if the value is never used).
    builder
        .instructions
        .push(Instruction::Return(Value::ConstantInt(0)));

    Function {
        instructions: builder.instructions,
        ident: label,
    }
}

/// Generate an _IR_ value from the provided _AST_ expression.
fn generate_ir_value(expr: &parser::Expression, builder: &mut TACBuilder<'_>) -> Value {
    match expr {
        parser::Expression::Constant(v) => Value::ConstantInt(*v),
        parser::Expression::Var((v, _)) => Value::Var(v.clone()),
        parser::Expression::Unary { op, expr, .. } => {
            // The sign of an _IR_ instruction is determined by the
            // sub-expressions (here `expr`), not by when the operator is
            // applied to them. Subsequent instructions that use the result
            // will interpret its sign based on the operator's effect on the
            // operand(s) at the time of usage.
            let sign = match **expr {
                parser::Expression::Unary { sign, .. } => sign,
                parser::Expression::Binary { sign, .. } => sign,
                _ => Signedness::Unsigned,
            };

            // Recursively process the expression until the base case
            // (`ConstantInt`) is reached. This ensures the inner expression is
            // processed initially before unwinding.
            let src = generate_ir_value(expr, builder);
            let dst = Value::Var(builder.new_tmp());

            builder.instructions.push(Instruction::Unary {
                op: *op,
                src,
                dst: dst.clone(),
                sign,
            });

            // The returned `dst` is used as the new `src` in each unwind of
            // the recursion.
            dst
        }
        parser::Expression::Binary { op, lhs, rhs, .. } => {
            // The sign of an _IR_ instruction is determined by the
            // sub-expressions (here `lhs`), not by when the operator is
            // applied to them. Subsequent instructions that use the result
            // will interpret its sign based on the operator's effect on the
            // operand(s) at the time of usage.
            let sign = match **lhs {
                parser::Expression::Unary { sign, .. } => sign,
                parser::Expression::Binary { sign, .. } => sign,
                _ => Signedness::Unsigned,
            };

            match op {
                // Need to short-circuit if the `lhs` is 0 for `&&`, or `lhs` is
                // non-zero for `||`.
                parser::BinaryOperator::LogAnd | parser::BinaryOperator::LogOr => {
                    let lhs = generate_ir_value(lhs, builder);
                    let rhs = generate_ir_value(rhs, builder);
                    let dst = Value::Var(builder.new_tmp());

                    let instructions = if let parser::BinaryOperator::LogAnd = op {
                        let f_lbl = builder.new_label("and.false");
                        let e_lbl = builder.new_label("and.end");

                        [
                            Instruction::JumpIfZero {
                                cond: lhs,
                                target: f_lbl.clone(),
                            },
                            Instruction::JumpIfZero {
                                cond: rhs,
                                target: f_lbl.clone(),
                            },
                            Instruction::Copy {
                                src: Value::ConstantInt(1),
                                dst: dst.clone(),
                            },
                            Instruction::Jump(e_lbl.clone()),
                            Instruction::Label(f_lbl),
                            Instruction::Copy {
                                src: Value::ConstantInt(0),
                                dst: dst.clone(),
                            },
                            Instruction::Label(e_lbl),
                        ]
                    } else {
                        let t_lbl = builder.new_label("or.true");
                        let e_lbl = builder.new_label("or.end");

                        [
                            Instruction::JumpIfNotZero {
                                cond: lhs,
                                target: t_lbl.clone(),
                            },
                            Instruction::JumpIfNotZero {
                                cond: rhs,
                                target: t_lbl.clone(),
                            },
                            Instruction::Copy {
                                src: Value::ConstantInt(0),
                                dst: dst.clone(),
                            },
                            Instruction::Jump(e_lbl.clone()),
                            Instruction::Label(t_lbl),
                            Instruction::Copy {
                                src: Value::ConstantInt(1),
                                dst: dst.clone(),
                            },
                            Instruction::Label(e_lbl),
                        ]
                    };

                    builder.instructions.extend(instructions);
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
        parser::Expression::Assignment(lvalue, rvalue, _) => {
            let dst = match &**lvalue {
                parser::Expression::Var((v, _)) => Value::Var(v.clone()),
                _ => panic!("lvalue of an expression should be an AST var"),
            };

            let result = generate_ir_value(rvalue, builder);

            builder.instructions.push(Instruction::Copy {
                src: result,
                dst: dst.clone(),
            });

            dst
        }
    }
}
