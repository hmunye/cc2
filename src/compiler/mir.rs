//! Machine Intermediate Representation
//!
//! Compiler pass that lowers three-address code (_TAC_) intermediate
//! representation (_IR_) into machine intermediate representation (_x86-64_).

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;

use crate::compiler::ir::{self, IR};
use crate::compiler::parser::UnaryOperator;

type Ident = String;

/// Machine _IR_: structured _x86-64_ assembly representation.
#[derive(Debug)]
pub enum MIR {
    /// Function that represent the structure of the assembly program.
    Program(Function),
}

impl MIR {
    /// Replaces each _pseudoregister_ encountered with a stack offset,
    /// returning the stack offset of the final temporary variable.
    fn replace_pseudo_registers(&mut self) -> i32 {
        let mut map: HashMap<Ident, i32> = Default::default();

        // NOTE: Currently "allocating" in 4-byte offsets.
        let mut stack_offset = 0;

        match self {
            MIR::Program(func) => {
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

    /// Prepends a `StackAlloc` instruction to the current instruction sequence
    /// given the number of bytes to allocate.
    fn emit_stack_allocation(&mut self, bytes: i32) {
        match self {
            MIR::Program(func) => {
                // NOTE: O(n) time complexity.
                func.instructions.insert(0, Instruction::StackAlloc(bytes));
            }
        }
    }

    /// Normalizes `mov` instructions that contain both memory address operands,
    /// converting them into valid forms.
    fn rewrite_invalid_mov_instructions(&mut self) {
        match self {
            MIR::Program(func) => {
                let mut i = 0;

                while i < func.instructions.len() {
                    let inst = &mut func.instructions[i];

                    // `mov` instruction that uses a memory address for both
                    // operands (illegal).
                    if let Instruction::Mov(src, dst) = inst
                        && let Operand::Stack(_) = src
                        && let Operand::Stack(_) = dst
                    {
                        let src = src.clone();
                        let dst = dst.clone();

                        // Use the `r10d` scratch register as temporary storage
                        // for intermediate values in the new instructions.
                        //
                        // `r10d`, unlike other hardware registers, serves no
                        // special purpose, so we are less likely to encounter
                        // a conflict.
                        func.instructions.splice(
                            i..=i,
                            [
                                Instruction::Mov(src.clone(), Operand::Register(Reg::R10)),
                                Instruction::Mov(Operand::Register(Reg::R10), dst.clone()),
                            ],
                        );

                        // Increment `i` to ensure the two new instructions
                        // inserted are skipped.
                        i += 1;
                    }

                    i += 1;
                }
            }
        }
    }
}

impl fmt::Display for MIR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MIR::Program(func) => {
                write!(f, "MIR Program\n{:4}{func}", "")
            }
        }
    }
}

/// _MIR_ function definition.
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Function {
    pub label: Ident,
    pub instructions: Vec<Instruction>,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Label {:?}:", self.label)?;

        for inst in &self.instructions {
            writeln!(f, "{:8}{inst}", "")?;
        }

        Ok(())
    }
}

/// _MIR_ instructions.
#[derive(Debug)]
pub enum Instruction {
    /// Move instruction (copies `src` to `dst`).
    Mov(Operand, Operand),
    /// Apply the given unary operator to the operand.
    Unary(UnaryOperator, Operand),
    /// Subtract the specified number of bytes from `rsp` register.
    StackAlloc(i32),
    /// Yields control back to the caller.
    Ret,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Mov(src, dst) => {
                let src_str = format!("{src}");
                let len = src_str.len();

                let max_width = 15;
                let width = if len >= max_width { 0 } else { max_width - len };

                write!(
                    f,
                    "{:<15}{src_str} {:>width$}  {dst}",
                    "Mov",
                    "->",
                    width = width
                )
            }
            Instruction::Unary(op, operand) => write!(f, "{:<15}{operand}", format!("{op:?}")),
            Instruction::StackAlloc(v) => write!(f, "{:<15}{v}", "StackAlloc"),
            Instruction::Ret => write!(f, "Ret"),
        }
    }
}

/// _MIR_ operands.
#[derive(Debug, Clone)]
pub enum Operand {
    /// Immediate value (32-bit).
    Imm32(i32),
    /// Register name.
    Register(Reg),
    /// Pseudoregister to represent temporary variable.
    Pseudo(Ident),
    /// Stack address with the specified offset from the `RBP` register.
    Stack(i32),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Imm32(v) => write!(f, "{v}"),
            Operand::Register(r) => fmt::Display::fmt(r, f),
            Operand::Pseudo(i) => write!(f, "{i:?}"),
            Operand::Stack(v) => write!(f, "stack({v})"),
        }
    }
}

/// _MIR x86-64_ registers (size agnostic).
#[derive(Debug, Copy, Clone)]
#[allow(missing_docs)]
pub enum Reg {
    AX,
    R10,
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::AX => write!(f, "%eax"),
            Reg::R10 => write!(f, "%r10d"),
        }
    }
}

/// Generate machine intermediate representation (_MIR_), given intermediate
/// representation (_IR_). [Exits] on error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn generate_mir(ir: &IR) -> MIR {
    match ir {
        IR::Program(func) => {
            let mir_func = generate_mir_function(func);

            // Pass 1 - MIR initialized.
            let mut mir = MIR::Program(mir_func);

            // Pass 2 - Each pseudoregister replace with stack offsets.
            let stack_offset = mir.replace_pseudo_registers();

            // Pass 3 - Rewrite invalid `mov` instructions (both operands may
            // now be stack memory addresses).
            mir.rewrite_invalid_mov_instructions();

            // Pass 4 - Insert instruction for allocating `stack_offset` bytes.
            mir.emit_stack_allocation(stack_offset);

            mir
        }
    }
}

/// Generate a _MIR_ function definition from the provided _IR_ function.
fn generate_mir_function(func: &ir::Function) -> Function {
    let mut instructions = vec![];

    for inst in &func.instructions {
        match inst {
            ir::Instruction::Return(v) => {
                instructions.push(Instruction::Mov(
                    generate_mir_operand(v),
                    Operand::Register(Reg::AX),
                ));
                instructions.push(Instruction::Ret);
            }
            ir::Instruction::Unary { op, src, dst } => {
                let dst = generate_mir_operand(dst);

                instructions.push(Instruction::Mov(generate_mir_operand(src), dst.clone()));
                instructions.push(Instruction::Unary(*op, dst));
            }
        }
    }

    Function {
        label: func.ident.clone(),
        instructions,
    }
}

/// Generate a _MIR_ operand from the provided _IR_ value.
fn generate_mir_operand(val: &ir::Value) -> Operand {
    match val {
        ir::Value::ConstantInt(v) => Operand::Imm32(*v),
        ir::Value::Var(v) => Operand::Pseudo(v.clone()),
    }
}
