//! Intermediate Representation.
//!
//! Compiler pass that lowers an abstract syntax tree (_AST_) into intermediate
//! representation (_IR_) using three-address code (_TAC_).

use crate::compiler::parser::{self, UnaryOperator};

type Ident = String;

/// Intermediate representation (_IR_).
#[derive(Debug)]
pub enum IR {
    /// Function that represent the structure of the program.
    Program(Function),
}

/// Represents an IR _function_ definition.
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Function {
    pub ident: Ident,
    pub instructions: Vec<Instruction>,
}

/// Represents different IR _instructions_.
#[derive(Debug)]
pub enum Instruction {
    /// Returns a _value_ to the caller.
    Return(Value),
    /// Perform a _unary_ operation on `src`, storing the result in `dst`.
    ///
    /// NOTE: The `dst` of any unary instruction must be `Value::Var`.
    #[allow(missing_docs)]
    Unary {
        op: UnaryOperator,
        src: Value,
        dst: Value,
    },
}

/// Represents different IR _values_.
#[derive(Debug, Clone)]
pub enum Value {
    /// Constant _int_ value (32-bit).
    ConstantInt(i32),
    /// Temporary _variable_.
    Var(Ident),
}

/// Helper for lowering nested _AST_ expressions into three-address code (_TAC_)
/// instructions.
struct TACBuilder<'a> {
    instructions: Vec<Instruction>,
    count: usize,
    label: &'a str,
}

impl TACBuilder<'_> {
    /// Allocates and returns a fresh temp variable identifier.
    fn new_tmp(&mut self) -> Ident {
        // The `.` in temporary identifiers guarantees they won’t conflict
        // with user-defined identifiers, since the _C17_ standard forbids `.`
        // in identifiers.
        let ident = format!("{}.tmp.{}", self.label, self.count);
        self.count += 1;
        ident
    }
}

/// Generate intermediate representation (`IR`), given an abstract syntax tree
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

/// Generate an IR _function definition_ from the provided _AST_ function.
fn generate_ir_function(func: &parser::Function) -> Function {
    let label = func.ident.clone();

    let mut builder = TACBuilder {
        instructions: vec![],
        count: 0,
        label: &label,
    };

    match func.body {
        parser::Statement::Return(ref expr) => {
            let ir_expr = generate_ir_expression(expr, &mut builder);
            builder.instructions.push(Instruction::Return(ir_expr));
        }
    }

    Function {
        instructions: builder.instructions,
        ident: label,
    }
}

/// Generate an IR _expression_ from the provided _AST_ expression`.
fn generate_ir_expression(expr: &parser::Expression, builder: &mut TACBuilder<'_>) -> Value {
    match expr {
        parser::Expression::ConstantInt(v) => Value::ConstantInt(*v),
        parser::Expression::Unary { op, expr } => {
            // Recursively process the expression until the base case
            // (`Expression::ConstantInt`) is reached. This ensures the inner
            // expression is processed initially before unwinding.
            let src = generate_ir_expression(expr, builder);
            let dst = Value::Var(builder.new_tmp());

            builder.instructions.push(Instruction::Unary {
                op: *op,
                src,
                dst: dst.clone(),
            });

            // The returned `dst` is used as the new `src` when unwinding.
            dst
        }
    }
}
