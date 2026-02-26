//! Intermediate Representation
//!
//! Compiler pass that lowers an abstract syntax tree (_AST_) into intermediate
//! representation (_IR_).

use std::borrow::Cow;
use std::fmt;

use crate::compiler::frontend::SymbolTable;
use crate::compiler::frontend::ast::{
    self, Analyzed, BinaryOperator, Signedness, StorageClass, UnaryOperator,
};
use crate::compiler::frontend::parser::symbols::{Linkage, StorageDuration, SymbolState};
use crate::compiler::frontend::types::c_int;

/// Intermediate representation (_IR_).
#[derive(Debug)]
pub struct IR<'a> {
    /// Items that represent the target-independent structure of the program.
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

/// _IR_ top-level constructs.
#[derive(Debug)]
pub enum Item<'a> {
    Fn(Function<'a>),
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
            Item::Fn(function) => write!(f, "{function}"),
            Item::Static {
                init,
                ident,
                is_global,
            } => write!(
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

/// _IR_ instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction<'a> {
    /// Return a value to the caller.
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
    /// Copy the value from `src` into `dst`.
    Copy { src: Value<'a>, dst: Value<'a> },
    /// Unconditionally jump to the target label.
    Jump(String),
    /// Conditionally jump to the target label if `cond` is zero.
    JumpIfZero { cond: Value<'a>, target: String },
    /// Conditionally jump to the target label if `cond` is non-zero.
    JumpIfNotZero { cond: Value<'a>, target: String },
    /// Label identifier.
    Label(String),
    /// Transfers control to the callee with arguments, storing the return value
    /// in `dst`.
    FnCall {
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
            Instruction::Jump(target) => write!(f, "{:<17}{:?}", "Jump", target),
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
            Instruction::FnCall { ident, args, dst } => {
                let len = ident.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len + 2);

                let args_str = args
                    .iter()
                    .map(|a| format!("{a}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(
                    f,
                    "{:<17}{ident:?} {:>width$}  {dst}  args: ({args_str})",
                    "FnCall",
                    "->",
                    width = width
                )
            }
            Instruction::Label(label) => write!(f, "{:<17}{:?}", "Label", label),
        }
    }
}

/// _IR_ values.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Value<'a> {
    /// Integer constant (32-bit).
    IntConstant(c_int),
    /// Temporary variable.
    Var {
        ident: Cow<'a, str>,
        is_static: bool,
    },
}

impl Value<'_> {
    /// Returns the identifier of the value, or `None` if it is not a variable.
    #[inline]
    #[must_use]
    pub fn as_var(&self) -> Option<&str> {
        match self {
            Value::IntConstant(_) => None,
            Value::Var { ident, .. } => Some(ident),
        }
    }

    /// Returns `true` if the value has `static` storage duration.
    #[inline]
    #[must_use]
    pub const fn is_static(&self) -> bool {
        match self {
            Value::IntConstant(_) => false,
            Value::Var { is_static, .. } => *is_static,
        }
    }
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::IntConstant(i) => write!(f, "{i}"),
            Value::Var { ident, .. } => write!(f, "{ident:?}"),
        }
    }
}

/// Helper for lowering nested _AST_ expressions into three-address code (_TAC_)
/// representation.
#[derive(Debug, Default)]
struct TACBuilder<'a> {
    instructions: Vec<Instruction<'a>>,
    tmp_count: usize,
    label_count: usize,
    /// _AST_ function label.
    fn_ident: &'a str,
    /// Canonical names of `static` objects, later resolved to _IR_ static
    /// items.
    statics: Vec<&'a str>,
}

impl<'a> TACBuilder<'a> {
    /// Returns a new, unique, temporary variable identifier.
    #[inline]
    fn new_tmp(&mut self) -> String {
        // `.` in temporary identifiers guarantees they won’t conflict with
        // user-defined identifiers.
        let ident = format!("{}.tmp.{}", self.fn_ident, self.tmp_count);
        self.tmp_count += 1;
        ident
    }

    /// Returns a new, unique, label identifier, appending the provided suffix.
    #[inline]
    fn new_label(&mut self, suffix: &str) -> String {
        // `.` in labels guarantees they won’t conflict with user-defined
        // labels.
        let label = format!("{}.lbl.{}.{suffix}", self.fn_ident, self.label_count);
        self.label_count += 1;
        label
    }

    /// Resets the builder state for the next _AST_ function definition
    /// identifier.
    #[inline]
    const fn reset(&mut self, ident: &'a str) {
        // NOTE: Since `ident` is enough ensures the uniqueness of each
        // generated temporary and label, counts can be reset.
        self.tmp_count = 0;
        self.label_count = 0;

        self.fn_ident = ident;
    }
}

/// Generates an intermediate representation (_IR_) from an analyzed abstract
/// syntax tree (_AST_) and symbol table.
///
/// # Panics
///
/// Panics if an identifier could not be found in the symbol table or the _AST_
/// is malformed.
#[must_use]
pub fn generate_ir<'a>(ast: &'a ast::AST<'_, Analyzed>, sym_table: &mut SymbolTable) -> IR<'a> {
    let mut ir_items = vec![];

    let mut builder = TACBuilder::default();

    for decl in &ast.program {
        match decl {
            ast::Declaration::Var { ident, .. } => {
                let sym_info = sym_table
                    .get_mut(ident)
                    .expect("identifier should be in symbol table after semantic analysis");

                if sym_info.is_emitted {
                    continue;
                }

                sym_info.is_emitted = true;

                let init = match sym_info.state {
                    SymbolState::ConstDefined(i) => i,
                    SymbolState::Tentative => 0,
                    // `extern` declarations at file-scope.
                    _ => continue,
                };

                ir_items.push(Item::Static {
                    init,
                    ident,
                    is_global: sym_info.linkage == Some(Linkage::External),
                });
            }
            ast::Declaration::Fn(f) => {
                if f.body.is_some() {
                    builder.reset(&f.ident);

                    ir_items.push(Item::Fn(generate_ir_function(f, &mut builder, sym_table)));
                }
            }
        }
    }

    ir_items.reserve(builder.statics.len());

    for ident in builder.statics {
        let sym_info = sym_table
            .get(ident)
            .expect("identifier should be in symbol table after semantic analysis");

        let init = match sym_info.state {
            SymbolState::ConstDefined(i) => i,
            SymbolState::Defined => 0,
            _ => panic!("block-scope static declarations should be defined"),
        };

        ir_items.push(Item::Static {
            init,
            ident,
            is_global: false,
        });
    }

    IR { program: ir_items }
}

/// Generates an _IR_ function definition from the provided _AST_ function
/// definition.
fn generate_ir_function<'a>(
    f: &'a ast::Function<'_>,
    builder: &mut TACBuilder<'a>,
    sym_table: &SymbolTable,
) -> Function<'a> {
    fn process_ast_block<'a>(
        block: &'a ast::Block<'_>,
        builder: &mut TACBuilder<'a>,
        sym_table: &SymbolTable,
    ) {
        for block_item in &block.0 {
            match block_item {
                ast::BlockItem::Stmt(stmt) => process_ast_statement(stmt, builder, sym_table),
                ast::BlockItem::Decl(decl) => process_ast_declaration(decl, builder, sym_table),
            }
        }
    }

    fn process_ast_declaration<'a>(
        decl: &'a ast::Declaration<'_>,
        builder: &mut TACBuilder<'a>,
        sym_table: &SymbolTable,
    ) {
        if let ast::Declaration::Var {
            specs, ident, init, ..
        } = decl
        {
            if specs.storage == Some(StorageClass::Extern) {
                // `extern` declarations at block-scope.
                return;
            }

            let sym_info = sym_table
                .get(ident)
                .expect("identifier should be in symbol table after semantic analysis");

            match sym_info.duration {
                Some(StorageDuration::Static) => {
                    builder.statics.push(ident.as_str());
                }
                Some(StorageDuration::Automatic) => {
                    if let Some(init) = &init {
                        // Appends any instructions needed to represent the
                        // declaration initializer.
                        let ir_val = generate_ir_value(init, builder, sym_table);

                        // Ensure the initialized result is copied to the
                        // destination.
                        builder.instructions.push(Instruction::Copy {
                            src: ir_val,
                            dst: Value::Var {
                                ident: Cow::Borrowed(ident),
                                is_static: false,
                            },
                        });
                    }
                }
                None => {
                    panic!("block-scope variables should have a storage duration");
                }
            }
        }
    }

    fn process_ast_statement<'a>(
        stmt: &'a ast::Statement<'_>,
        builder: &mut TACBuilder<'a>,
        sym_table: &SymbolTable,
    ) {
        match stmt {
            ast::Statement::Return(expr) => {
                let ir_val = generate_ir_value(expr, builder, sym_table);
                builder.instructions.push(Instruction::Return(ir_val));
            }
            ast::Statement::Expression(expr) => {
                // Appends any instructions needed to represent the expression,
                // discarding the destination value.
                let _ = generate_ir_value(expr, builder, sym_table);
            }
            ast::Statement::If {
                cond,
                then,
                opt_else,
            } => {
                let cond = generate_ir_value(cond, builder, sym_table);

                let e_lbl = builder.new_label("if.end");

                if let Some(else_stmt) = opt_else {
                    let else_lbl = builder.new_label("else.end");

                    builder.instructions.push(Instruction::JumpIfZero {
                        cond,
                        target: else_lbl.clone(),
                    });

                    process_ast_statement(then, builder, sym_table);

                    builder.instructions.extend([
                        Instruction::Jump(e_lbl.clone()),
                        Instruction::Label(else_lbl),
                    ]);

                    process_ast_statement(else_stmt, builder, sym_table);
                } else {
                    builder.instructions.push(Instruction::JumpIfZero {
                        cond,
                        target: e_lbl.clone(),
                    });

                    process_ast_statement(then, builder, sym_table);
                }

                builder.instructions.push(Instruction::Label(e_lbl));
            }
            ast::Statement::Goto { target, .. } => {
                builder.instructions.push(Instruction::Jump(target.clone()));
            }
            ast::Statement::LabeledStatement(labeled) => match labeled {
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

                    process_ast_statement(stmt, builder, sym_table);
                }
            },
            ast::Statement::Compound(block) => process_ast_block(block, builder, sym_table),
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

                process_ast_statement(stmt, builder, sym_table);

                builder.instructions.push(Instruction::Label(cont_label));

                let cond = generate_ir_value(cond, builder, sym_table);

                builder.instructions.extend([
                    Instruction::JumpIfNotZero {
                        cond,
                        target: start_label,
                    },
                    Instruction::Label(break_label),
                ]);
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

                let cond = generate_ir_value(cond, builder, sym_table);

                builder.instructions.push(Instruction::JumpIfZero {
                    cond,
                    target: break_label.clone(),
                });

                process_ast_statement(stmt, builder, sym_table);

                builder.instructions.extend([
                    Instruction::Jump(cont_label),
                    Instruction::Label(break_label),
                ]);
            }
            ast::Statement::For {
                init,
                opt_cond,
                opt_post,
                stmt,
                loop_label: label,
            } => {
                match &**init {
                    ast::ForInit::Decl(decl) => process_ast_declaration(decl, builder, sym_table),
                    ast::ForInit::Expr(opt_expr) => {
                        if let Some(expr) = opt_expr {
                            // Appends any instructions needed to represent the
                            // expression, discarding the destination value.
                            let _ = generate_ir_value(expr, builder, sym_table);
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
                    let cond = generate_ir_value(cond, builder, sym_table);

                    builder.instructions.push(Instruction::JumpIfZero {
                        cond,
                        target: break_label.clone(),
                    });
                }

                process_ast_statement(stmt, builder, sym_table);

                builder.instructions.push(Instruction::Label(cont_label));

                if let Some(post) = opt_post {
                    // Appends any instructions needed to represent the
                    // expression, discarding the destination value.
                    let _ = generate_ir_value(post, builder, sym_table);
                }

                builder.instructions.extend([
                    Instruction::Jump(start_label),
                    Instruction::Label(break_label),
                ]);
            }
            ast::Statement::Switch {
                cond,
                stmt,
                cases,
                default,
                switch_label: label,
            } => {
                let lhs = generate_ir_value(cond, builder, sym_table);

                // Only append instructions for the `switch` if there are case
                // labels or a default label.
                if !cases.is_empty() || default.is_some() {
                    let break_label = format!("break_{label}");

                    for case in cases {
                        let dst = Value::Var {
                            ident: Cow::Owned(builder.new_tmp()),
                            is_static: false,
                        };

                        let rhs = generate_ir_value(&case.expr, builder, sym_table);

                        builder.instructions.extend([
                            Instruction::Binary {
                                op: BinaryOperator::Eq,
                                lhs: lhs.clone(),
                                rhs,
                                dst: dst.clone(),
                                // NOTE: Temporary hack for arithmetic right
                                // shift.
                                sign: Signedness::Signed,
                            },
                            Instruction::JumpIfNotZero {
                                cond: dst,
                                target: case.jmp_label.clone(),
                            },
                        ]);
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

                    process_ast_statement(stmt, builder, sym_table);

                    builder.instructions.push(Instruction::Label(break_label));
                }
            }
            ast::Statement::Empty => {}
        }
    }

    let ident = f.ident.as_str();
    let params = f
        .params
        .iter()
        .map(|param| param.ident.as_str())
        .collect::<Vec<_>>();

    let body = &f
        .body
        .as_ref()
        .expect("IR should not be generates for function declarations");

    process_ast_block(body, builder, sym_table);

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
    // Appending an extra `Instruction::Return` handles the edge cases of
    // the return value being used by the caller or ignored (no undefined
    // behavior if the value is never used). Will be optimized away during
    // unreachable code elimination, if enabled.

    builder
        .instructions
        .push(Instruction::Return(Value::IntConstant(0)));

    let sym_info = sym_table
        .get(ident)
        .expect("identifier should be in symbol table after semantic analysis");

    Function {
        instructions: builder.instructions.drain(..).collect(),
        ident,
        params,
        is_global: sym_info.linkage == Some(Linkage::External),
    }
}

/// Generates an _IR_ value from the provided _AST_ expression.
fn generate_ir_value<'a>(
    expr: &'a ast::Expression<'_>,
    builder: &mut TACBuilder<'a>,
    sym_table: &SymbolTable,
) -> Value<'a> {
    match expr {
        ast::Expression::IntConstant(i) => Value::IntConstant(*i),
        ast::Expression::Var { ident, .. } => {
            let sym_info = sym_table
                .get(ident)
                .expect("identifier should be in symbol table after semantic analysis");

            Value::Var {
                ident: Cow::Borrowed(ident),
                is_static: sym_info.duration == Some(StorageDuration::Static),
            }
        }
        ast::Expression::Unary {
            op,
            expr,
            is_prefix,
            ..
        } => {
            let src = generate_ir_value(expr, builder, sym_table);
            let dst = Value::Var {
                ident: Cow::Owned(builder.new_tmp()),
                is_static: false,
            };

            match op {
                UnaryOperator::Increment | UnaryOperator::Decrement => {
                    debug_assert!(matches!(src, Value::Var { .. }));

                    let binop = match op {
                        UnaryOperator::Increment => BinaryOperator::Add,
                        UnaryOperator::Decrement => BinaryOperator::Subtract,
                        _ => unreachable!(
                            "only increment/decrement unary operators should reach this match arm"
                        ),
                    };

                    let tmp = Value::Var {
                        ident: Cow::Owned(builder.new_tmp()),
                        is_static: false,
                    };

                    if *is_prefix {
                        builder.instructions.extend([
                            Instruction::Binary {
                                op: binop,
                                lhs: src.clone(),
                                rhs: Value::IntConstant(1),
                                dst: tmp.clone(),
                                // NOTE: Temporary hack for arithmetic right
                                // shift.
                                sign: Signedness::Signed,
                            },
                            Instruction::Copy {
                                src: tmp,
                                dst: src.clone(),
                            },
                            // Updated value of `src` is moved to `dst`.
                            Instruction::Copy {
                                src,
                                dst: dst.clone(),
                            },
                        ]);
                    } else {
                        builder.instructions.extend([
                            Instruction::Binary {
                                op: binop,
                                lhs: src.clone(),
                                rhs: Value::IntConstant(1),
                                dst: tmp.clone(),
                                // NOTE: Temporary hack for arithmetic right
                                // shift.
                                sign: Signedness::Signed,
                            },
                            // Previous value of `src` is moved to `dst`.
                            Instruction::Copy {
                                src: src.clone(),
                                dst: dst.clone(),
                            },
                            Instruction::Copy { src: tmp, dst: src },
                        ]);
                    }
                }
                op => {
                    builder.instructions.push(Instruction::Unary {
                        op: *op,
                        src,
                        dst: dst.clone(),
                        // NOTE: Temporary hack for arithmetic right shift.
                        sign: Signedness::Signed,
                    });
                }
            }

            dst
        }
        ast::Expression::Binary { op, lhs, rhs, .. } => {
            match op {
                ast::BinaryOperator::LogAnd | ast::BinaryOperator::LogOr => {
                    let lhs = generate_ir_value(lhs, builder, sym_table);
                    let dst = Value::Var {
                        ident: Cow::Owned(builder.new_tmp()),
                        is_static: false,
                    };

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
                        let rhs = generate_ir_value(rhs, builder, sym_table);

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
                        let rhs = generate_ir_value(rhs, builder, sym_table);

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
                    let lhs = generate_ir_value(lhs, builder, sym_table);
                    let rhs = generate_ir_value(rhs, builder, sym_table);
                    let dst = Value::Var {
                        ident: Cow::Owned(builder.new_tmp()),
                        is_static: false,
                    };

                    builder.instructions.push(Instruction::Binary {
                        op: *op,
                        lhs,
                        rhs,
                        dst: dst.clone(),
                        // NOTE: Temporary hack for arithmetic right shift.
                        sign: Signedness::Signed,
                    });

                    dst
                }
            }
        }
        ast::Expression::Assignment { lvalue, rvalue, .. } => {
            let ast::Expression::Var { ident, .. } = &**lvalue else {
                panic!("lvalue of an expression should be a variable");
            };

            let sym_info = sym_table
                .get(ident)
                .expect("identifier should be in symbol table after semantic analysis");

            let dst = Value::Var {
                ident: Cow::Borrowed(ident),
                is_static: sym_info.duration == Some(StorageDuration::Static),
            };

            let result = generate_ir_value(rvalue, builder, sym_table);

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
            let dst = Value::Var {
                ident: Cow::Owned(builder.new_tmp()),
                is_static: false,
            };

            let e_lbl = builder.new_label("cond.end");
            let rhs_lbl = builder.new_label("cond.false");

            let cond = generate_ir_value(cond, builder, sym_table);

            builder.instructions.push(Instruction::JumpIfZero {
                cond,
                target: rhs_lbl.clone(),
            });

            let second = generate_ir_value(second, builder, sym_table);
            builder.instructions.push(Instruction::Copy {
                src: second,
                dst: dst.clone(),
            });

            builder.instructions.extend([
                Instruction::Jump(e_lbl.clone()),
                Instruction::Label(rhs_lbl),
            ]);

            let third = generate_ir_value(third, builder, sym_table);
            builder.instructions.push(Instruction::Copy {
                src: third,
                dst: dst.clone(),
            });

            builder.instructions.push(Instruction::Label(e_lbl));

            dst
        }
        ast::Expression::FnCall { ident, args, .. } => {
            let mut ir_args = Vec::with_capacity(args.len());

            let dst = Value::Var {
                ident: Cow::Owned(builder.new_tmp()),
                is_static: false,
            };

            ir_args.extend(
                args.iter()
                    .map(|expr| generate_ir_value(expr, builder, sym_table)),
            );

            builder.instructions.push(Instruction::FnCall {
                ident: ident.as_str(),
                args: ir_args,
                dst: dst.clone(),
            });

            dst
        }
    }
}
