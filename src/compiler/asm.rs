//! Intermediate Code Generation
//!
//! Compiler pass that translates three-address code (_TAC_) intermediate
//! representation (_IR_) into a structured assembly representation (_x86-64_).

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::compiler::ir::{self, IR};
use crate::compiler::parser::UnaryOperator;

type Ident = String;

/// Structured _x86-64_ assembly representation.
#[derive(Debug)]
pub enum ASM {
    /// Function that represent the structure of the assembly program.
    Program(Function),
}

impl ASM {
    /// Replaces each _pseudoregister_ encountered with a stack offset,
    /// returning the stack offset of the final temporary variable.
    fn replace_pseudo(&mut self) -> i32 {
        let mut map: HashMap<Ident, i32> = Default::default();

        // NOTE: Currently "allocating" in 4-byte offsets.
        let mut stack_offset = 0;

        match self {
            ASM::Program(func) => {
                for inst in &mut func.instructions {
                    match inst {
                        Instruction::Mov(src, dst) => {
                            if let Operand::Pseudo(ident) = src {
                                // Either increment the current stack offset, or
                                // use the stored offset if the identifier has
                                // already been seen.
                                let offset = match map.entry(ident.clone()) {
                                    Entry::Occupied(entry) => *entry.get(),
                                    Entry::Vacant(entry) => {
                                        stack_offset += 4;
                                        entry.insert(stack_offset);
                                        stack_offset
                                    }
                                };
                                *src = Operand::Stack(offset);
                            }
                            if let Operand::Pseudo(ident) = dst {
                                // Either increment the current stack offset, or
                                // use the stored offset if the identifier has
                                // already been seen.
                                let offset = match map.entry(ident.clone()) {
                                    Entry::Occupied(entry) => *entry.get(),
                                    Entry::Vacant(entry) => {
                                        stack_offset += 4;
                                        entry.insert(stack_offset);
                                        stack_offset
                                    }
                                };
                                *dst = Operand::Stack(offset);
                            }
                        }
                        Instruction::Unary(_, op) => {
                            if let Operand::Pseudo(ident) = op {
                                // Either increment the current stack offset, or
                                // use the stored offset if the identifier has
                                // already been seen.
                                let offset = match map.entry(ident.clone()) {
                                    Entry::Occupied(entry) => *entry.get(),
                                    Entry::Vacant(entry) => {
                                        stack_offset += 4;
                                        entry.insert(stack_offset);
                                        stack_offset
                                    }
                                };
                                *op = Operand::Stack(offset);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        stack_offset
    }

    /// Prepends a `AllocateStack` instruction to the instruction sequence,
    /// reserving the specified number of bytes for local storage.
    fn emit_stack_allocation(&mut self, offset: i32) {
        match self {
            ASM::Program(func) => {
                // _O(n)_ time complexity.
                func.instructions
                    .insert(0, Instruction::AllocateStack(offset));
            }
        }
    }

    /// Normalizes `MOV` instructions that contain both memory operands,
    /// converting them into valid forms.
    fn rewrite_invalid_mov(&mut self) {
        match self {
            ASM::Program(func) => {
                let mut i = 0;

                while i < func.instructions.len() {
                    let inst = &mut func.instructions[i];

                    // `Mov` instruction that uses a memory address for both
                    // operands (illegal).
                    if let Instruction::Mov(src, dst) = inst
                        && let Operand::Stack(_) = src
                        && let Operand::Stack(_) = dst
                    {
                        let src = src.clone();
                        let dst = dst.clone();

                        // Use the `R10D` scratch register as temporary storage
                        // for intermediate values in the new instructions.
                        //
                        // `R10D`, unlike other hardware registers, serves no
                        // special purpose, so we are less likely to encounter
                        // a conflict.
                        func.instructions.splice(
                            i..=i,
                            [
                                Instruction::Mov(src.clone(), Operand::Register(Reg::R10D)),
                                Instruction::Mov(Operand::Register(Reg::R10D), dst.clone()),
                            ],
                        );

                        // Increment `i` by 1 to ensure the two new instructions
                        // inserted are skipped.
                        i += 1;
                    }

                    i += 1;
                }
            }
        }
    }
}

/// Represents an x86-64 _function_ definition.
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Function {
    pub label: Ident,
    pub instructions: Vec<Instruction>,
}

/// Represents x86-64 _instructions_.
#[derive(Debug)]
pub enum Instruction {
    /// Move instruction (copies _src_ to _dst_).
    Mov(Operand, Operand),
    /// Apply the given unary operator to the operand.
    Unary(UnaryOperator, Operand),
    /// Subtract the specified number of bytes from `RSP` register.
    AllocateStack(i32),
    /// Yields control back to the caller.
    Ret,
}

/// Represents x86-64 _operands_.
#[derive(Debug, Clone)]
pub enum Operand {
    /// Immediate value (32-bit).
    Imm(i32),
    /// Register name.
    Register(Reg),
    /// Pseudoregister to represent temporary variable.
    Pseudo(Ident),
    /// Stack address with the specified offset from the `RBP` register.
    Stack(i32),
}

/// Represents x86_64 _hardware registers_ (size agnostic).
#[derive(Debug, Copy, Clone)]
#[allow(missing_docs)]
pub enum Reg {
    AX,
    R10D,
}

/// Generate structured assembly representation, given an intermediate
/// representation (_IR_). [Exits] on error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn generate_asm(ir: &IR) -> ASM {
    match ir {
        IR::Program(func) => {
            let asm_func = generate_asm_function(func);

            // Pass 1 - Structured assembly representation initialized.
            let mut asm = ASM::Program(asm_func);

            // Pass 2 - Each pseudoregister replace with stack offsets.
            let stack_offset = asm.replace_pseudo();

            // Pass 3 - Rewrite invalid `Mov` instructions (both operands may
            // now be memory addresses).
            asm.rewrite_invalid_mov();

            // Pass 4 - Insert instruction for allocating `stack_offset` bytes
            asm.emit_stack_allocation(stack_offset);

            asm
        }
    }
}

/// Generate an assembly representation _function definition_ from the provided
/// _IR_ function.
fn generate_asm_function(func: &ir::Function) -> Function {
    let mut instructions = vec![];

    for inst in &func.instructions {
        match inst {
            ir::Instruction::Return(v) => {
                instructions.push(Instruction::Mov(
                    generate_asm_operand(v),
                    Operand::Register(Reg::AX),
                ));
                instructions.push(Instruction::Ret);
            }
            ir::Instruction::Unary { op, src, dst } => {
                let dst = generate_asm_operand(dst);

                instructions.push(Instruction::Mov(generate_asm_operand(src), dst.clone()));
                instructions.push(Instruction::Unary(*op, dst));
            }
        }
    }

    Function {
        label: func.ident.clone(),
        instructions,
    }
}

/// Generate an assembly representation _operand_ from the provided _IR_ value.
fn generate_asm_operand(val: &ir::Value) -> Operand {
    match val {
        ir::Value::ConstantInt(v) => Operand::Imm(*v),
        ir::Value::Var(v) => Operand::Pseudo(v.clone()),
    }
}
