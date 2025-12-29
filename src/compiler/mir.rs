//! Machine Intermediate Representation
//!
//! Compiler pass that lowers three-address code (_TAC_) intermediate
//! representation (_IR_) into machine intermediate representation (_x86-64_).

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;

use crate::compiler::ir::{self, IR};

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
                        Instruction::Unary(_, operand) => {
                            if let Operand::Pseudo(ident) = operand {
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
                                *operand = Operand::Stack(offset);
                            }
                        }
                        Instruction::Binary(_, lhs, rhs) => {
                            if let Operand::Pseudo(ident) = lhs {
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
                                *lhs = Operand::Stack(offset);
                            }
                            if let Operand::Pseudo(ident) = rhs {
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
                                *rhs = Operand::Stack(offset);
                            }
                        }
                        Instruction::Idiv(div) => {
                            if let Operand::Pseudo(ident) = div {
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
                                *div = Operand::Stack(offset);
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

    /// Normalizes instructions with invalid operand forms into valid
    /// representations.
    fn rewrite_invalid_instructions(&mut self) {
        match self {
            MIR::Program(func) => {
                let mut i = 0;

                while i < func.instructions.len() {
                    let inst = &mut func.instructions[i];

                    match inst {
                        // `mov` instruction that uses a memory address for both
                        // operands (illegal).
                        Instruction::Mov(src, dst)
                            if matches!(src, Operand::Stack(_))
                                && matches!(dst, Operand::Stack(_)) =>
                        {
                            let src = src.clone();
                            let dst = dst.clone();

                            // Use the `r10d` register as temporary storage for
                            // intermediate values in the new instructions.
                            //
                            // `r10d`, unlike other hardware registers, serves
                            // no special purpose, so we are less likely to
                            // encounter a conflict.
                            func.instructions.splice(
                                i..=i,
                                [
                                    Instruction::Mov(src, Operand::Register(Reg::R10)),
                                    Instruction::Mov(Operand::Register(Reg::R10), dst),
                                ],
                            );

                            // Increment `i` to ensure the two new instructions
                            // inserted are skipped.
                            i += 1;
                        }
                        // `idivl` instruction that uses an immediate value as
                        // it's operand (illegal)
                        Instruction::Idiv(div) if matches!(div, Operand::Imm32(_)) => {
                            let div = div.clone();

                            func.instructions.splice(
                                i..=i,
                                [
                                    Instruction::Mov(div, Operand::Register(Reg::R10)),
                                    Instruction::Idiv(Operand::Register(Reg::R10)),
                                ],
                            );

                            // Increment `i` to ensure the two new instructions
                            // inserted are skipped.
                            i += 1;
                        }
                        // `add` or `sub` instruction that uses a memory address
                        // for both operands (illegal).
                        Instruction::Binary(op, lhs, rhs)
                            if matches!(op, BinaryOperator::Add | BinaryOperator::Sub)
                                && matches!(lhs, Operand::Stack(_))
                                && matches!(rhs, Operand::Stack(_)) =>
                        {
                            let lhs = lhs.clone();
                            let rhs = rhs.clone();

                            let binop = if let BinaryOperator::Add = op {
                                BinaryOperator::Add
                            } else {
                                BinaryOperator::Sub
                            };

                            func.instructions.splice(
                                i..=i,
                                [
                                    Instruction::Mov(lhs, Operand::Register(Reg::R10)),
                                    Instruction::Binary(binop, Operand::Register(Reg::R10), rhs),
                                ],
                            );

                            // Increment `i` to ensure the two new instructions
                            // inserted are skipped.
                            i += 1;
                        }
                        // `imul` instruction that uses a memory address as it's
                        // destination operand (illegal).
                        Instruction::Binary(BinaryOperator::Imul, lhs, rhs)
                            if matches!(rhs, Operand::Stack(_)) =>
                        {
                            let lhs = lhs.clone();
                            let rhs = rhs.clone();

                            // Use the `r11d` register as temporary storage for
                            // intermediate values in the new instructions.
                            func.instructions.splice(
                                i..=i,
                                [
                                    Instruction::Mov(rhs.clone(), Operand::Register(Reg::R11)),
                                    Instruction::Binary(
                                        BinaryOperator::Imul,
                                        lhs,
                                        Operand::Register(Reg::R11),
                                    ),
                                    Instruction::Mov(Operand::Register(Reg::R11), rhs),
                                ],
                            );

                            // Increment `i` to ensure the three new
                            // instructions inserted are skipped.
                            i += 2;
                        }
                        _ => {}
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
    /// Apply the given binary operator to both operands.
    Binary(BinaryOperator, Operand, Operand),
    /// Perform a signed division operation where the dividend is in `edx:eax`
    /// and the divisor is the operand.
    Idiv(Operand),
    /// Sign-extend the 32-bit value in `eax` to a 64-bit signed value across
    /// `edx:eax`.
    Cdq,
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

                let max_width: usize = 13;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<15}{src_str} {:>width$}  {dst}",
                    "Mov",
                    "->",
                    width = width
                )
            }
            Instruction::Unary(op, operand) => write!(f, "{:<15}{operand}", format!("{op:?}")),
            Instruction::Binary(op, lhs, rhs) => {
                let lhstr = format!("{lhs}");
                let len = lhstr.len();

                let max_width: usize = 14;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<15}{lhstr} {:<width$} {rhs}",
                    format!("{op:?}"),
                    "",
                    width = width
                )
            }
            Instruction::Idiv(div) => write!(f, "{:<15}{div}", "Idiv"),
            Instruction::Cdq => write!(f, "Cdq"),
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
    DX,
    R10,
    R11,
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::AX => write!(f, "%eax"),
            Reg::DX => write!(f, "%edx"),
            Reg::R10 => write!(f, "%r10d"),
            Reg::R11 => write!(f, "%r11d"),
        }
    }
}

/// _MIR_ unary operators.
#[derive(Debug, Copy, Clone)]
pub enum UnaryOperator {
    /// Instruction for bitwise negation.
    Not,
    /// Instruction for two's complement negation.
    Neg,
}

impl From<&crate::compiler::parser::UnaryOperator> for UnaryOperator {
    fn from(unop: &crate::compiler::parser::UnaryOperator) -> UnaryOperator {
        match unop {
            crate::compiler::parser::UnaryOperator::Complement => UnaryOperator::Not,
            crate::compiler::parser::UnaryOperator::Negate => UnaryOperator::Neg,
        }
    }
}

/// _MIR_ binary operators.
#[derive(Debug, Copy, Clone)]
pub enum BinaryOperator {
    /// Instruction for addition.
    Add,
    /// Instruction for subtraction.
    Sub,
    /// Instruction for signed multiplication.
    Imul,
}

impl TryFrom<&crate::compiler::parser::BinaryOperator> for BinaryOperator {
    type Error = ();

    fn try_from(binop: &crate::compiler::parser::BinaryOperator) -> Result<Self, Self::Error> {
        match binop {
            crate::compiler::parser::BinaryOperator::Add => Ok(BinaryOperator::Add),
            crate::compiler::parser::BinaryOperator::Subtract => Ok(BinaryOperator::Sub),
            crate::compiler::parser::BinaryOperator::Multiply => Ok(BinaryOperator::Imul),
            _ => Err(()),
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

            // Pass 3 - Rewrite invalid instructions.
            mir.rewrite_invalid_instructions();

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
                instructions.push(Instruction::Unary(op.into(), dst));
            }
            ir::Instruction::Binary { op, lhs, rhs, dst } => {
                let dst = generate_mir_operand(dst);

                match op {
                    crate::compiler::parser::BinaryOperator::Divide => {
                        instructions.push(Instruction::Mov(
                            generate_mir_operand(lhs),
                            Operand::Register(Reg::AX),
                        ));
                        instructions.push(Instruction::Cdq);
                        instructions.push(Instruction::Idiv(generate_mir_operand(rhs)));
                        instructions.push(Instruction::Mov(Operand::Register(Reg::AX), dst));
                    }
                    crate::compiler::parser::BinaryOperator::Remainder => {
                        instructions.push(Instruction::Mov(
                            generate_mir_operand(lhs),
                            Operand::Register(Reg::AX),
                        ));
                        instructions.push(Instruction::Cdq);
                        instructions.push(Instruction::Idiv(generate_mir_operand(rhs)));
                        instructions.push(Instruction::Mov(Operand::Register(Reg::DX), dst));
                    }
                    _ => {
                        instructions.push(Instruction::Mov(generate_mir_operand(lhs), dst.clone()));
                        instructions.push(Instruction::Binary(
                            op.try_into().unwrap_or_else(|_| panic!("parser::BinaryOperator '{:?}' is an invalid mir::BinaryOperator",
                                op)),
                            generate_mir_operand(rhs),
                            dst,
                        ));
                    }
                }
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
