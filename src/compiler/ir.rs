//! Intermediate Code Generation.
//!
//! Compiler pass that lowers an abstract syntax tree (_AST_) into an
//! intermediate assembly representation (_x86-64_).

use crate::compiler::parser;

/// Intermediate representation (_IR_) derived from an _AST_.
#[derive(Debug)]
pub enum IR {
    /// Function that represent the structure of the assembly program.
    Program(Function),
}

/// Represents a _function_ definition.
#[derive(Debug)]
pub struct Function {
    pub(crate) label: String,
    pub(crate) instructions: Vec<Instruction>,
}

/// Represents different assembly (_x86-64_) instructions.
#[derive(Debug)]
pub enum Instruction {
    /// Move instruction (copies _src_ to _dst_).
    Mov(Operand, Operand),
    /// Yields control back to the caller.
    Ret,
}

/// Represents different operands for assembly (_x86-64_) instructions.
#[derive(Debug)]
pub enum Operand {
    /// Immediate value (32-bit).
    Imm(i32),
    /// Register name (e.g., "eax", "ebx").
    Register(&'static str),
}

/// Generate intermediate representation (`IR`), given an abstract syntax tree
/// (_AST_).
pub fn generate_ir(ast: &parser::AST) -> Result<IR, String> {
    match ast {
        parser::AST::Program(func) => {
            let ir_function = generate_ir_function(func)?;
            Ok(IR::Program(ir_function))
        }
    }
}

/// Generate an IR _function definition_ from the provided `parser::Function`.
fn generate_ir_function(func: &parser::Function) -> Result<Function, String> {
    let mut instructions = vec![];

    match func.body {
        parser::Statement::Return(ref expr) => {
            let ir_expr = generate_ir_expression(expr)?;

            // Register `eax` is used for return values according to System-V
            // ABI.
            instructions.push(Instruction::Mov(ir_expr, Operand::Register("eax")));
            instructions.push(Instruction::Ret);
        }
    }

    Ok(Function {
        label: func.ident.clone(),
        instructions,
    })
}

/// Generate an IR _expression_ from the provided `parser::Expression`.
fn generate_ir_expression(expr: &parser::Expression) -> Result<Operand, String> {
    match expr {
        parser::Expression::ConstantInt(v) => Ok(Operand::Imm(*v)),
    }
}
