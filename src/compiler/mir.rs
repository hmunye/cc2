//! Machine Intermediate Representation
//!
//! Compiler pass that lowers three-address code (_TAC_) intermediate
//! representation (_IR_) into machine intermediate representation (_x86-64_).

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;

use crate::compiler::ir::{self, IR};
use crate::compiler::parser::ast::{self, Signedness};

/// Machine _IR_: structured _x86-64_ assembly representation.
#[derive(Debug)]
pub enum MIRX86 {
    /// Function that represent the structure of the assembly program.
    Program(Function),
}

impl MIRX86 {
    /// Replaces each _pseudoregister_ encountered with a stack offset,
    /// returning the stack offset of the final temporary variable.
    fn replace_pseudo_registers(&mut self) -> i32 {
        let mut map: HashMap<String, i32> = Default::default();

        // Currently allocating in 4-byte offsets.
        let mut stack_offset = 0;

        // Either increment the current stack offset, or use the stored offset
        // if the identifier has already been seen.
        let mut get_offset = |ident: &mut String| match map.entry(ident.clone()) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                stack_offset += 4;
                entry.insert(stack_offset);
                stack_offset
            }
        };

        match self {
            MIRX86::Program(func) => {
                for inst in &mut func.instructions {
                    match inst {
                        Instruction::Mov(src, dst) => {
                            if let Operand::Pseudo(ident) = src {
                                *src = Operand::Stack(get_offset(ident));
                            }
                            if let Operand::Pseudo(ident) = dst {
                                *dst = Operand::Stack(get_offset(ident));
                            }
                        }
                        Instruction::Unary(_, operand) => {
                            if let Operand::Pseudo(ident) = operand {
                                *operand = Operand::Stack(get_offset(ident));
                            }
                        }
                        Instruction::Binary(_, lhs, rhs) => {
                            if let Operand::Pseudo(ident) = lhs {
                                *lhs = Operand::Stack(get_offset(ident));
                            }
                            if let Operand::Pseudo(ident) = rhs {
                                *rhs = Operand::Stack(get_offset(ident));
                            }
                        }
                        Instruction::Idiv(div) => {
                            if let Operand::Pseudo(ident) = div {
                                *div = Operand::Stack(get_offset(ident));
                            }
                        }
                        Instruction::Cmp(src, dst) => {
                            if let Operand::Pseudo(ident) = src {
                                *src = Operand::Stack(get_offset(ident));
                            }
                            if let Operand::Pseudo(ident) = dst {
                                *dst = Operand::Stack(get_offset(ident));
                            }
                        }
                        Instruction::SetC(_, dst) => {
                            if let Operand::Pseudo(ident) = dst {
                                *dst = Operand::Stack(get_offset(ident));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        stack_offset
    }

    /// Prepends a `Instruction::StackAlloc` to the current instruction
    /// sequence, given the number of bytes to allocate.
    fn emit_stack_allocation(&mut self, bytes: i32) {
        match self {
            MIRX86::Program(func) => {
                // NOTE: O(n) time complexity.
                func.instructions.insert(0, Instruction::StackAlloc(bytes));
            }
        }
    }

    /// Normalizes instructions with invalid operand forms into valid _x86-64_
    /// instruction representations.
    fn rewrite_invalid_instructions(&mut self) {
        match self {
            MIRX86::Program(func) => {
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
                            // `r10d`, unlike other hardware registers on
                            // x86-64, serves no special purpose, so we are less
                            // likely to encounter a conflict.
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
                        // `add`, `sub`, `and`, `or`, or, `xor` instruction that
                        // uses a memory address for both operands (illegal).
                        Instruction::Binary(op, lhs, rhs)
                            if matches!(
                                op,
                                BinaryOperator::Add
                                    | BinaryOperator::Sub
                                    | BinaryOperator::And
                                    | BinaryOperator::Or
                                    | BinaryOperator::Xor
                            ) && matches!(lhs, Operand::Stack(_))
                                && matches!(rhs, Operand::Stack(_)) =>
                        {
                            let lhs = lhs.clone();
                            let rhs = rhs.clone();
                            let binop = *op;

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
                        // `shl` or `shr`/`sar` instruction that uses a memory
                        // address or register not `%cl` as source operand
                        // (illegal).
                        Instruction::Binary(op, lhs, rhs)
                            if matches!(
                                op,
                                BinaryOperator::Shl | BinaryOperator::Shr | BinaryOperator::Sar
                            ) && !matches!(
                                lhs,
                                Operand::Imm32(_) | Operand::Register(Reg::CX)
                            ) =>
                        {
                            let lhs = lhs.clone();
                            let rhs = rhs.clone();
                            let binop = *op;

                            // `Reg::CX` here represents the `%cl` register.
                            func.instructions.splice(
                                i..=i,
                                [
                                    Instruction::Mov(lhs, Operand::Register(Reg::CX)),
                                    Instruction::Binary(binop, Operand::Register(Reg::CX), rhs),
                                ],
                            );

                            // Increment `i` to ensure the two new instructions
                            // inserted are skipped.
                            i += 1;
                        }
                        // `cmp` instruction that uses a memory address for both
                        // operands or immediate value as destination operand
                        // (illegal).
                        Instruction::Cmp(src, dst)
                            if (matches!(src, Operand::Stack(_))
                                && matches!(dst, Operand::Stack(_)))
                                || matches!(dst, Operand::Imm32(_)) =>
                        {
                            let src = src.clone();
                            let dst = dst.clone();

                            if let Operand::Imm32(_) = dst {
                                func.instructions.splice(
                                    i..=i,
                                    [
                                        Instruction::Mov(dst, Operand::Register(Reg::R11)),
                                        Instruction::Cmp(src, Operand::Register(Reg::R11)),
                                    ],
                                );
                            } else {
                                func.instructions.splice(
                                    i..=i,
                                    [
                                        Instruction::Mov(src, Operand::Register(Reg::R10)),
                                        Instruction::Cmp(Operand::Register(Reg::R10), dst),
                                    ],
                                );
                            }

                            // Increment `i` to ensure the two new instructions
                            // inserted are skipped.
                            i += 1;
                        }
                        _ => {}
                    }

                    i += 1;
                }
            }
        }
    }
}

impl fmt::Display for MIRX86 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MIRX86::Program(func) => {
                write!(f, "MIR Program\n{:4}{func}", "")
            }
        }
    }
}

/// _MIR_ function definition.
#[derive(Debug)]
pub struct Function {
    pub label: String,
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

/// _MIR_ instruction.
#[derive(Debug)]
pub enum Instruction {
    /// Move instruction (copies `src` to `dst`).
    Mov(Operand, Operand),
    /// Apply the given unary operator to the operand.
    Unary(UnaryOperator, Operand),
    /// Apply the given binary operator to both operands.
    Binary(BinaryOperator, Operand, Operand),
    /// Compares both operands (operand.1 - operand.0), and updates the relevant
    /// `RFLAGS`.
    Cmp(Operand, Operand),
    /// Perform a signed division operation where the dividend is in `edx:eax`
    /// and the divisor is the operand.
    Idiv(Operand),
    /// Sign-extend the 32-bit value in `eax` to a 64-bit signed value across
    /// `edx:eax`.
    Cdq,
    /// Unconditionally jump to the point instructions after the label
    /// identifier.
    Jmp(String),
    /// Conditionally jump to the point instructions after the label
    /// identifier, based on the conditional code.
    JmpC(CondCode, String),
    /// Move the value of the bit in `RFLAGS` based on the conditional code to
    /// the operand destination (1-byte).
    SetC(CondCode, Operand),
    /// Associates an "identifier" with instruction(s).
    Label(String),
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

                let max_width: usize = 32;
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

                let max_width: usize = 33;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<15}{lhstr} {:<width$} {rhs}",
                    format!("{op:?}"),
                    "",
                    width = width
                )
            }
            Instruction::Cmp(lhs, rhs) => {
                let lhstr = format!("{lhs}");
                let len = lhstr.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<15}{lhstr} {:>width$}  {rhs}",
                    "Cmp",
                    "->",
                    width = width
                )
            }
            Instruction::Idiv(div) => write!(f, "{:<15}{div}", "Idiv"),
            Instruction::Cdq => write!(f, "Cdq"),
            Instruction::Jmp(i) => write!(f, "{:<15}{i}", "Jmp"),
            Instruction::JmpC(code, i) => write!(f, "{:<15}{i}", format!("Jmp{code:?}")),
            Instruction::SetC(code, o) => write!(f, "{:<15}{o}", format!("Set{code:?}")),
            Instruction::Label(i) => write!(f, "{:<15}{i}", "Label"),
            Instruction::StackAlloc(v) => write!(f, "{:<15}{v}", "StackAlloc"),
            Instruction::Ret => write!(f, "Ret"),
        }
    }
}

/// _MIR_ operand.
#[derive(Debug, Clone)]
pub enum Operand {
    /// Immediate value (32-bit).
    Imm32(i32),
    /// Register name.
    Register(Reg),
    /// Pseudoregister to represent temporary variable.
    Pseudo(String),
    /// Stack address with the specified offset from the `RBP` register.
    Stack(i32),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Imm32(v) => write!(f, "{v}"),
            Operand::Register(r) => write!(f, "%{r:?}"),
            Operand::Pseudo(i) => write!(f, "{i:?}"),
            Operand::Stack(v) => write!(f, "stack({v})"),
        }
    }
}

/// _MIR x86-64_ registers (size agnostic).
#[derive(Debug, Copy, Clone)]
pub enum Reg {
    /// `rax` (64-bit), `eax` (32-bit), `ax` (16-bit), `al` (8-bit low),
    /// `ah` (8-bit high).
    AX,
    /// `rcx` (64-bit), `ecx` (32-bit), `cx` (16-bit), `cl` (8-bit low),
    /// `ch` (8-bit high).
    CX,
    /// `rdx` (64-bit), `edx` (32-bit), `dx` (16-bit), `dl` (8-bit low),
    /// `dh` (8-bit high).
    DX,
    /// `r10` (64-bit), `r10d` (32-bit), `r10w` (16-bit), `r10b` (8-bit low)
    R10,
    /// `r11` (64-bit), `r11d` (32-bit), `r11w` (16-bit), `r11b` (8-bit low)
    R11,
}

/// _MIR x86-64_ conditional codes.
#[derive(Debug, Copy, Clone)]
pub enum CondCode {
    /// Equal.
    E,
    /// Not-equal.
    NE,
    /// Greater.
    G,
    /// Greater-or-equal.
    GE,
    /// Less.
    L,
    /// Less-or-equal.
    LE,
}

impl fmt::Display for CondCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CondCode::E => write!(f, "e"),
            CondCode::NE => write!(f, "ne"),
            CondCode::G => write!(f, "g"),
            CondCode::GE => write!(f, "ge"),
            CondCode::L => write!(f, "l"),
            CondCode::LE => write!(f, "le"),
        }
    }
}

/// _MIR_ unary operators.
#[derive(Debug, Copy, Clone)]
pub enum UnaryOperator {
    /// Instruction for one's complement negation.
    Not,
    /// Instruction for two's complement negation.
    Neg,
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
    /// Instructions for logical AND.
    And,
    /// Instructions for logical OR.
    Or,
    /// Instructions for logical XOR.
    Xor,
    /// Instructions left shift (logical).
    Shl,
    /// Instructions right shift (logical).
    Shr,
    /// Instructions right shift (arithmetic).
    Sar,
}

impl TryFrom<&ast::BinaryOperator> for BinaryOperator {
    type Error = ();

    fn try_from(binop: &ast::BinaryOperator) -> Result<Self, Self::Error> {
        match binop {
            ast::BinaryOperator::Add => Ok(BinaryOperator::Add),
            ast::BinaryOperator::Subtract => Ok(BinaryOperator::Sub),
            ast::BinaryOperator::Multiply => Ok(BinaryOperator::Imul),
            ast::BinaryOperator::BitAnd => Ok(BinaryOperator::And),
            ast::BinaryOperator::BitOr => Ok(BinaryOperator::Or),
            ast::BinaryOperator::BitXor => Ok(BinaryOperator::Xor),
            ast::BinaryOperator::ShiftLeft => Ok(BinaryOperator::Shl),
            // Defaults to logical `shr` operator instead of arithmetic.
            ast::BinaryOperator::ShiftRight => Ok(BinaryOperator::Shr),
            _ => Err(()),
        }
    }
}

/// Generate _x86-64_ machine intermediate representation (_MIR_), given an
/// intermediate representation (_IR_). [Exits] on error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn generate_x86_64_mir(ir: &IR) -> MIRX86 {
    match ir {
        IR::Program(func) => {
            let mir_func = generate_mir_function(func);

            let mut mir = MIRX86::Program(mir_func);

            // Pass 1 - Each pseudoregister replaced with stack offsets.
            let stack_offset = mir.replace_pseudo_registers();

            // Pass 2 - Rewrite invalid instructions.
            mir.rewrite_invalid_instructions();

            // Pass 3 - Insert instruction for allocating `stack_offset` bytes.
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
            ir::Instruction::Unary { op, src, dst, .. } => {
                let dst = generate_mir_operand(dst);

                if let ast::UnaryOperator::Not = op {
                    instructions.push(Instruction::Cmp(
                        Operand::Imm32(0),
                        generate_mir_operand(src),
                    ));

                    // Zero-out the destination.
                    instructions.push(Instruction::Mov(Operand::Imm32(0), dst.clone()));
                    instructions.push(Instruction::SetC(CondCode::E, dst));
                } else {
                    let unop = match op {
                        ast::UnaryOperator::Complement => UnaryOperator::Not,
                        ast::UnaryOperator::Negate => UnaryOperator::Neg,
                        _ => unreachable!(),
                    };

                    instructions.push(Instruction::Mov(generate_mir_operand(src), dst.clone()));
                    instructions.push(Instruction::Unary(unop, dst));
                }
            }
            ir::Instruction::Binary {
                op,
                lhs,
                rhs,
                dst,
                sign,
            } => {
                let dst = generate_mir_operand(dst);

                match op {
                    ast::BinaryOperator::Divide | ast::BinaryOperator::Modulo => {
                        instructions.push(Instruction::Mov(
                            generate_mir_operand(lhs),
                            Operand::Register(Reg::AX),
                        ));
                        instructions.push(Instruction::Cdq);
                        instructions.push(Instruction::Idiv(generate_mir_operand(rhs)));

                        let src = if let ast::BinaryOperator::Divide = op {
                            // Quotient is in `eax` register.
                            Operand::Register(Reg::AX)
                        } else {
                            // Remainder is in `edx` register.
                            Operand::Register(Reg::DX)
                        };

                        instructions.push(Instruction::Mov(src, dst));
                    }
                    ast::BinaryOperator::OrdGreater
                    | ast::BinaryOperator::OrdLess
                    | ast::BinaryOperator::OrdLessEq
                    | ast::BinaryOperator::OrdGreaterEq
                    | ast::BinaryOperator::Eq
                    | ast::BinaryOperator::NotEq => {
                        let cond_code = match op {
                            ast::BinaryOperator::OrdGreater => CondCode::G,
                            ast::BinaryOperator::OrdLess => CondCode::L,
                            ast::BinaryOperator::OrdLessEq => CondCode::LE,
                            ast::BinaryOperator::OrdGreaterEq => CondCode::GE,
                            ast::BinaryOperator::Eq => CondCode::E,
                            ast::BinaryOperator::NotEq => CondCode::NE,
                            _ => unreachable!(),
                        };

                        instructions.push(Instruction::Cmp(
                            generate_mir_operand(rhs),
                            generate_mir_operand(lhs),
                        ));

                        // Zero-out the destination.
                        instructions.push(Instruction::Mov(Operand::Imm32(0), dst.clone()));
                        instructions.push(Instruction::SetC(cond_code, dst));
                    }
                    _ => {
                        let binop =
                            if let ast::BinaryOperator::ShiftRight = op
                                && let Signedness::Signed = sign
                            {
                                // Since the shift-right instruction is determined
                                // to be signed, use an arithmetic right shift
                                // instead of logical.
                                BinaryOperator::Sar
                            } else {
                                op.try_into().unwrap_or_else(|_| panic!(
                                "ast::BinaryOperator '{:?}' is an invalid mir::BinaryOperator",
                                op
                            ))
                            };

                        instructions.push(Instruction::Mov(generate_mir_operand(lhs), dst.clone()));
                        // ```
                        //     mov lhs, dst      # copy lhs into destination
                        //     op  rhs, dst      # dst = dst op rhs
                        // ```
                        instructions.push(Instruction::Binary(
                            binop,
                            generate_mir_operand(rhs),
                            dst,
                        ));
                    }
                }
            }
            ir::Instruction::Copy { src, dst } => {
                instructions.push(Instruction::Mov(
                    generate_mir_operand(src),
                    generate_mir_operand(dst),
                ));
            }
            ir::Instruction::Jump(label) => {
                instructions.push(Instruction::Jmp(label.clone()));
            }
            ir::Instruction::JumpIfZero { cond, target } => {
                instructions.push(Instruction::Cmp(
                    Operand::Imm32(0),
                    generate_mir_operand(cond),
                ));
                instructions.push(Instruction::JmpC(CondCode::E, target.clone()));
            }
            ir::Instruction::JumpIfNotZero { cond, target } => {
                instructions.push(Instruction::Cmp(
                    Operand::Imm32(0),
                    generate_mir_operand(cond),
                ));
                instructions.push(Instruction::JmpC(CondCode::NE, target.clone()));
            }
            ir::Instruction::Label(label) => {
                instructions.push(Instruction::Label(label.clone()));
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
        ir::Value::IntConstant(v) => Operand::Imm32(*v),
        ir::Value::Var(v) => Operand::Pseudo(v.clone()),
    }
}
