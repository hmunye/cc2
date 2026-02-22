//! Machine Intermediate Representation
//!
//! Compiler pass that lowers an intermediate representation (_IR_) into machine
//! intermediate representation (_x86-64_).

use std::collections::HashSet;
use std::fmt;

use crate::compiler::ir::{self, IR};
use crate::compiler::parser::ast::{self, Signedness};
use crate::compiler::parser::sema::symbols::SymbolMap;
use crate::compiler::parser::types::{Type, c_int};

/// Structured _x86-64_ assembly representation.
#[derive(Debug)]
pub struct MIRX86<'a> {
    /// Items that represent the structure of the assembly program.
    pub program: Vec<Item<'a>>,
    /// Set of functions defined within the translation unit.
    pub locales: HashSet<&'a str>,
}

impl fmt::Display for MIRX86<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MIR (x86-64) Program")?;
        for item in &self.program {
            writeln!(f, "{:4}{item}", "")?;
        }

        Ok(())
    }
}

/// _MIR x86-64_ top-level constructs.
#[derive(Debug)]
pub enum Item<'a> {
    Func(Function<'a>),
    /// Declaration with `static` storage duration.
    Static {
        init: c_int,
        label: &'a str,
        is_global: bool,
    },
}

impl fmt::Display for Item<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Item::Func(func) => write!(f, "{func}"),
            Item::Static {
                init,
                label: ident,
                is_global,
            } => writeln!(
                f,
                "Static ({}) {ident:?} = {init}",
                if *is_global { "G" } else { "L" }
            ),
        }
    }
}

/// _MIR x86-64_ function definition.
#[derive(Debug)]
pub struct Function<'a> {
    pub label: &'a str,
    pub instructions: Vec<Instruction<'a>>,
    pub is_global: bool,
}

impl fmt::Display for Function<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Label ({}) {:?}:",
            if self.is_global { "G" } else { "L" },
            self.label
        )?;

        for inst in &self.instructions {
            writeln!(f, "{:8}{inst}", "")?;
        }

        Ok(())
    }
}

/// _MIR x86-64_ instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction<'a> {
    /// Moves `src` -> `dst`.
    Mov { src: Operand<'a>, dst: Operand<'a> },
    /// Unary operator applied to `dst`.
    ///
    /// (`dst` = `<unop> dst`).
    Unary {
        unop: UnaryOperator,
        dst: Operand<'a>,
    },
    /// Binary operator applied to `rhs` and `dst`
    ///
    /// (`dst` = `dst <binop> rhs`).
    Binary {
        binop: BinaryOperator,
        rhs: Operand<'a>,
        dst: Operand<'a>,
    },
    /// Compares both operands (`lhs` - `rhs`), and updates the relevant CPU
    /// `RFLAGS`.
    Cmp { rhs: Operand<'a>, lhs: Operand<'a> },
    /// Performs signed division with a dividend of `%edx:%eax` and divisor as
    /// the operand.
    Idiv(Operand<'a>),
    /// Sign-extend the 32-bit value in `%eax` to a 64-bit signed value across
    /// `%edx:%eax`.
    Cdq,
    /// Unconditionally jump to the instruction after the target label.
    Jmp(&'a str),
    /// Conditionally jump to the instruction after the target label, based on
    /// the conditional code.
    JmpC { code: CondCode, target: &'a str },
    /// Move the value of the bit in CPU `RFLAGS` corresponding to `code` to the
    /// `dst` (1-byte operand).
    SetC { code: CondCode, dst: Operand<'a> },
    /// Defines a label identifier.
    Label(&'a str),
    /// Subtract the specified number of bytes from `%rsp`.
    StackAlloc(isize),
    /// Add the specified number of bytes to `%rsp`.
    StackDealloc(isize),
    /// Pushes the specified operand onto the call-stack.
    Push(Operand<'a>),
    /// Pops the top value from the call-stack into the specified register.
    Pop(Reg),
    /// Calls the function specified by the identifier, transferring control.
    Call(&'a str),
    /// Yields control back to the caller.
    Ret,
}

impl<'a> Instruction<'a> {
    /// Appends operands used by this instruction to `used`.
    ///
    /// # Panics
    ///
    /// Panics if a function call label is missing from the symbol map.
    #[inline]
    pub fn find_used(&self, used: &mut Vec<Operand<'a>>, sym_map: &SymbolMap) {
        match self {
            Instruction::Mov { src: val, .. }
            | Instruction::Unary { dst: val, .. }
            | Instruction::Push(val) => {
                used.push(*val);
            }
            Instruction::Binary { rhs, dst: val, .. } | Instruction::Cmp { rhs, lhs: val } => {
                used.extend([*rhs, *val]);
            }
            Instruction::Idiv(divisor) => {
                used.extend([
                    *divisor,
                    Operand::Register(Reg::AX),
                    Operand::Register(Reg::DX),
                ]);
            }
            Instruction::Cdq => {
                used.push(Operand::Register(Reg::AX));
            }
            Instruction::Call(label) => {
                let sym_info = sym_map.get(*label).expect(
                    "semantic analysis ensures every identifier is registered in the symbol map",
                );

                if let Type::Func { param_count } = sym_info.ty {
                    used.extend(
                        // According to the System-V ABI calling convention, the
                        // first six function parameters are accessed from the
                        // following registers:
                        //
                        // 64-bit: `%rdi`, `%rsi`, `%rdx`, `%rcx`, `%r8`, `%r9`
                        // 32-bit: `%edi`, `%esi`, `%edx`, `%ecx`, `%r8d`, `%r9d`
                        [Reg::DI, Reg::SI, Reg::DX, Reg::CX, Reg::R8, Reg::R9]
                            .into_iter()
                            .take(param_count.min(6))
                            .map(Operand::Register),
                    );
                }
            }
            _ => {}
        }
    }

    /// Appends operands updated by this instruction to `updated`.
    #[inline]
    pub fn find_updated(&self, updated: &mut Vec<Operand<'a>>) {
        match self {
            Instruction::Mov { dst, .. }
            | Instruction::Unary { dst, .. }
            | Instruction::Binary { dst, .. }
            | Instruction::SetC { dst, .. } => {
                updated.push(*dst);
            }
            Instruction::Idiv(_) => {
                updated.extend([Operand::Register(Reg::AX), Operand::Register(Reg::DX)]);
            }
            Instruction::Cdq => {
                updated.push(Operand::Register(Reg::DX));
            }
            Instruction::Call(_) => {
                updated.extend(
                    [
                        Reg::DI,
                        Reg::SI,
                        Reg::DX,
                        Reg::CX,
                        Reg::R8,
                        Reg::R9,
                        Reg::AX,
                    ]
                    .into_iter()
                    .map(Operand::Register),
                );
            }
            Instruction::Pop(reg) => updated.push(Operand::Register(*reg)),
            _ => {}
        }
    }

    /// Updates the two provided containers: one for the operands used and one
    /// for the operands updated by this instruction.
    #[inline]
    pub fn find_used_and_updated(
        &self,
        used: &mut Vec<Operand<'a>>,
        updated: &mut Vec<Operand<'a>>,
        sym_map: &SymbolMap,
    ) {
        self.find_used(used, sym_map);
        self.find_updated(updated);
    }
}

impl fmt::Display for Instruction<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Mov { src, dst } => {
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
            Instruction::Unary { unop, dst } => write!(f, "{:<15}{dst}", format!("{unop:?}")),
            Instruction::Binary { binop, rhs, dst } => {
                let rhstr = format!("{rhs}");
                let len = rhstr.len();

                let max_width: usize = 33;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<15}{rhstr} {:<width$} {dst}",
                    format!("{binop:?}"),
                    "",
                    width = width
                )
            }
            Instruction::Cmp { rhs, lhs } => {
                let rhstr = format!("{rhs}");
                let len = rhstr.len();

                let max_width: usize = 32;
                let width = max_width.saturating_sub(len);

                write!(
                    f,
                    "{:<15}{rhstr} {:>width$}  {lhs}",
                    "Cmp",
                    "->",
                    width = width
                )
            }
            Instruction::Idiv(div) => write!(f, "{:<15}{div}", "Idiv"),
            Instruction::Cdq => write!(f, "Cdq"),
            Instruction::Jmp(target) => write!(f, "{:<15}{target:?}", "Jmp"),
            Instruction::JmpC { code, target } => {
                write!(f, "{:<15}{target:?}", format!("Jmp{code:?}"))
            }
            Instruction::SetC { code, dst } => write!(f, "{:<15}{dst}", format!("Set{code:?}")),
            Instruction::Label(label) => write!(f, "{:<15}{label:?}", "Label"),
            Instruction::StackAlloc(i) => write!(f, "{:<15}{i}", "StackAlloc"),
            Instruction::StackDealloc(i) => write!(f, "{:<15}{i}", "StackDealloc"),
            Instruction::Push(op) => write!(f, "{:<15}{op}", "Push"),
            Instruction::Pop(reg) => write!(f, "{:<15}{reg:?}", "Pop"),
            Instruction::Call(ident) => write!(f, "{:<15}{ident:?}", "Call"),
            Instruction::Ret => write!(f, "Ret"),
        }
    }
}

/// _MIR x86-64_ operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand<'a> {
    /// Immediate value (32-bit).
    Imm32(c_int),
    /// Register name.
    Register(Reg),
    /// Symbolic operand (e.g., global, static, temporary variable).
    Symbol { ident: &'a str, is_static: bool },
    /// Stack address with specified offset from `%rbp`.
    Stack(isize),
    /// Identifier for data located in the `.bss` / `.data` _ELF_ section.
    Data(&'a str),
}

impl Operand<'_> {
    /// Returns `true` if the operand refers to a memory location.
    #[inline]
    #[must_use]
    pub const fn is_memory_operand(&self) -> bool {
        matches!(self, Operand::Stack(_) | Operand::Data(_))
    }
}

impl fmt::Display for Operand<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Imm32(i) => write!(f, "{i}"),
            Operand::Register(r) => write!(f, "%{r:?}"),
            Operand::Symbol { ident, .. } => write!(f, "{ident:?}"),
            Operand::Stack(i) => write!(f, "stack({i})"),
            Operand::Data(ident) => write!(f, "{ident:?} [static memory]"),
        }
    }
}

/// _MIR x86-64_ registers (size agnostic).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Reg {
    // =====================================================================
    // ====================== Caller-saved Registers =======================
    // =====================================================================
    /// `%rax` (64-bit), `%eax` (32-bit), `%ax` (16-bit), `%al` (8-bit low),
    /// `%ah` (8-bit high).
    AX,
    /// `%rcx` (64-bit), `%ecx` (32-bit), `%cx` (16-bit), `%cl` (8-bit low),
    /// `%ch` (8-bit high).
    CX,
    /// `%rdx` (64-bit), `%edx` (32-bit), `%dx` (16-bit), `%dl` (8-bit low),
    /// `%dh` (8-bit high).
    DX,
    /// `%rdi` (64-bit), `%edi` (32-bit), `%di` (16-bit), `%dil` (8-bit low).
    DI,
    /// `%rsi` (64-bit), `%esi` (32-bit), `%si` (16-bit), `%sil` (8-bit low).
    SI,
    /// `%r8` (64-bit), `%r8d` (32-bit), `%r8w` (16-bit), `%r8b` (8-bit low).
    R8,
    /// `%r9` (64-bit), `%r9d` (32-bit), `%r9w` (16-bit), `%r9b` (8-bit low).
    R9,
    /// `%r10` (64-bit), `%r10d` (32-bit), `%r10w` (16-bit), `%r10b`
    /// (8-bit low).
    R10,
    /// `%r11` (64-bit), `%r11d` (32-bit), `%r11w` (16-bit), `%r11b`
    /// (8-bit low).
    R11,
    // =====================================================================
    // ====================== Callee-saved Registers =======================
    // =====================================================================
    /// `%rbx` (64-bit), `%ebx` (32-bit), `%bx` (16-bit), `%bl` (8-bit low),
    /// `%bh` (8-bit high).
    BX,
    /// `%r12` (64-bit), `%r12d` (32-bit), `%r12w` (16-bit), `%r12b`
    /// (8-bit low).
    R12,
    /// `%r13` (64-bit), `%r13d` (32-bit), `%r13w` (16-bit), `%r13b`
    /// (8-bit low).
    R13,
    /// `%r14` (64-bit), `%r14d` (32-bit), `%r14w` (16-bit), `%r14b`
    /// (8-bit low).
    R14,
    /// `%r15` (64-bit), `%r15d` (32-bit), `%r15w` (16-bit), `%r15b`
    /// (8-bit low).
    R15,
}

impl Reg {
    /// Number of available hardware registers for allocation.
    pub const ALLOCATABLE_REGS: [Reg; 12] = [
        Reg::AX,
        Reg::BX,
        Reg::CX,
        Reg::DX,
        Reg::DI,
        Reg::SI,
        Reg::R8,
        Reg::R9,
        Reg::R12,
        Reg::R13,
        Reg::R14,
        Reg::R15,
        // `Reg::R10` and `Reg::R11` are used in instruction fix-up.
        //
        // `RSP` and `RBP` are used in stack-frame management.
    ];

    /// Returns `true` if the register is caller-saved according to the
    /// System-V AMD64 ABI.
    #[inline]
    #[must_use]
    pub const fn is_caller_saved(&self) -> bool {
        matches!(
            self,
            Reg::AX
                | Reg::CX
                | Reg::DX
                | Reg::DI
                | Reg::SI
                | Reg::R8
                | Reg::R9
                | Reg::R10
                | Reg::R11
        )
    }

    /// Returns `true` if the register is callee-saved according to the
    /// System-V AMD64 ABI.
    #[inline]
    #[must_use]
    pub const fn is_callee_saved(&self) -> bool {
        matches!(self, Reg::BX | Reg::R12 | Reg::R13 | Reg::R14 | Reg::R15)
    }
}

/// _MIR x86-64_ conditional codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// _MIR x86-64_ unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    /// Instruction for one's complement negation.
    Not,
    /// Instruction for two's complement negation.
    Neg,
}

/// _MIR x86-64_ binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    /// Instruction for addition.
    Add,
    /// Instruction for subtraction.
    Sub,
    /// Instruction for signed multiplication.
    Imul,
    /// Instruction for logical-AND.
    And,
    /// Instruction for logical-OR.
    Or,
    /// Instruction for logical-XOR.
    Xor,
    /// Instruction for left-shift (logical).
    Shl,
    /// Instruction for right-shift (logical).
    Shr,
    /// Instruction for right-shift (arithmetic).
    Sar,
}

/// Generate _x86-64_ machine intermediate representation (_MIR_), given an
/// intermediate representation (_IR_).
#[must_use]
pub fn generate_x86_64_mir<'a>(ir: &'a IR<'_>) -> MIRX86<'a> {
    let mut mir_items = Vec::with_capacity(ir.program.len());
    let mut locales = HashSet::new();

    for item in &ir.program {
        match item {
            ir::Item::Static {
                init,
                ident,
                is_global,
            } => {
                mir_items.push(Item::Static {
                    init: *init,
                    label: ident,
                    is_global: *is_global,
                });
            }
            ir::Item::Func(func) => {
                locales.insert(func.ident);
                mir_items.push(Item::Func(generate_mir_function(func)));
            }
        }
    }

    MIRX86 {
        program: mir_items,
        locales,
    }
}

/// Generate a _MIR x86-64_ function definition from the provided _IR_ function.
fn generate_mir_function<'a>(func: &'a ir::Function<'_>) -> Function<'a> {
    let mut instructions = Vec::with_capacity(func.params.len());

    // Lower function parameters before processing `IR` instructions.
    lower_ir_function_params(&func.params, &mut instructions);

    for inst in &func.instructions {
        match inst {
            ir::Instruction::Return(v) => {
                // According to the System-V ABI calling convention, the return
                // value is always stored in `%rax` (for 64-bit) or `%eax`
                // (for 32-bit).
                instructions.extend([
                    Instruction::Mov {
                        src: generate_mir_operand(v),
                        dst: Operand::Register(Reg::AX),
                    },
                    Instruction::Ret,
                ]);
            }
            ir::Instruction::Unary { op, src, dst, .. } => {
                let dst = generate_mir_operand(dst);

                if matches!(op, ast::UnaryOperator::Not) {
                    instructions.extend([
                        Instruction::Cmp {
                            rhs: Operand::Imm32(0),
                            lhs: generate_mir_operand(src),
                        },
                        // Zero-out the destination.
                        Instruction::Mov {
                            src: Operand::Imm32(0),
                            dst,
                        },
                        Instruction::SetC {
                            code: CondCode::E,
                            dst,
                        },
                    ]);
                } else {
                    let unop = match op {
                        ast::UnaryOperator::Complement => UnaryOperator::Not,
                        ast::UnaryOperator::Negate => UnaryOperator::Neg,
                        _ => panic!(
                            "increment/decrement should already be lowered to an add/sub instruction"
                        ),
                    };

                    instructions.extend([
                        Instruction::Mov {
                            src: generate_mir_operand(src),
                            dst,
                        },
                        Instruction::Unary { unop, dst },
                    ]);
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
                        let src = if matches!(op, ast::BinaryOperator::Divide) {
                            // Quotient is in `%eax`.
                            Operand::Register(Reg::AX)
                        } else {
                            // Remainder is in `%edx`.
                            Operand::Register(Reg::DX)
                        };

                        instructions.extend([
                            Instruction::Mov {
                                src: generate_mir_operand(lhs),
                                dst: Operand::Register(Reg::AX),
                            },
                            Instruction::Cdq,
                            Instruction::Idiv(generate_mir_operand(rhs)),
                            Instruction::Mov { src, dst },
                        ]);
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
                            _ => unreachable!(
                                "non-relational/equality binary operators should not reach this match arm"
                            ),
                        };

                        instructions.extend([
                            Instruction::Cmp {
                                rhs: generate_mir_operand(rhs),
                                lhs: generate_mir_operand(lhs),
                            },
                            // Zero-out the destination.
                            Instruction::Mov {
                                src: Operand::Imm32(0),
                                dst,
                            },
                            Instruction::SetC {
                                code: cond_code,
                                dst,
                            },
                        ]);
                    }
                    _ => {
                        // NOTE: Temporary hack for arithmetic right shift.
                        let binop = if matches!(op, ast::BinaryOperator::ShiftRight)
                            && matches!(sign, Signedness::Signed)
                        {
                            BinaryOperator::Sar
                        } else {
                            ast_to_mir_binop(*op).unwrap_or_else(|| {
                                panic!(
                                    "ast::BinaryOperator '{op:?}' could not be converted to mir::BinaryOperator"
                                )
                            })
                        };

                        instructions.extend([
                            Instruction::Mov {
                                src: generate_mir_operand(lhs),
                                dst,
                            },
                            Instruction::Binary {
                                binop,
                                rhs: generate_mir_operand(rhs),
                                dst,
                            },
                        ]);
                    }
                }
            }
            ir::Instruction::Copy { src, dst } => {
                instructions.push(Instruction::Mov {
                    src: generate_mir_operand(src),
                    dst: generate_mir_operand(dst),
                });
            }
            ir::Instruction::Jump(label) => {
                instructions.push(Instruction::Jmp(label.as_str()));
            }
            ir::Instruction::JumpIfZero { cond, target }
            | ir::Instruction::JumpIfNotZero { cond, target } => {
                let code = if let ir::Instruction::JumpIfZero { .. } = inst {
                    CondCode::E
                } else {
                    CondCode::NE
                };

                instructions.extend([
                    Instruction::Cmp {
                        rhs: Operand::Imm32(0),
                        lhs: generate_mir_operand(cond),
                    },
                    Instruction::JmpC {
                        code,
                        target: target.as_str(),
                    },
                ]);
            }
            ir::Instruction::Call { ident, args, dst } => {
                // According to the System-V ABI calling convention, the first
                // six function arguments are passed in the following registers:
                //
                // 64-bit: `%rdi`, `%rsi`, `%rdx`, `%rcx`, `%r8`, `%r9`
                // 32-bit: `%edi`, `%esi`, `%edx`, `%ecx`, `%r8d`, `%r9d`
                let regs = [Reg::DI, Reg::SI, Reg::DX, Reg::CX, Reg::R8, Reg::R9];

                let stack_args = args.len().saturating_sub(6);
                let needs_padding = (stack_args > 0) && ((stack_args % 2) == 1);

                if needs_padding {
                    // Ensure a 16-byte alignment of the stack frame, required
                    // by System-V ABI.
                    instructions.push(Instruction::StackAlloc(8));
                }

                instructions.extend(args.iter().take(6).enumerate().map(|(i, arg)| {
                    Instruction::Mov {
                        src: generate_mir_operand(arg),
                        dst: Operand::Register(regs[i]),
                    }
                }));

                if stack_args > 0 {
                    let remaining_args = &args[6..];

                    // Any remaining arguments are passed on the stack in right
                    // to left order.
                    for arg in remaining_args.iter().rev() {
                        let mir_arg = generate_mir_operand(arg);

                        match &mir_arg {
                            Operand::Register(_) | Operand::Imm32(_) => {
                                instructions.push(Instruction::Push(mir_arg));
                            }
                            // `pushq` instruction requires an 8-byte operand.
                            // Pushing a 4-byte stack value directly would
                            // incorrectly include the following 4 bytes from
                            // the stack frame. Instead, we first move it into
                            // AX (caller-saved), then push onto the stack.
                            _ => {
                                instructions.extend([
                                    Instruction::Mov {
                                        src: mir_arg,
                                        dst: Operand::Register(Reg::AX),
                                    },
                                    Instruction::Push(Operand::Register(Reg::AX)),
                                ]);
                            }
                        }
                    }
                }

                instructions.push(Instruction::Call(ident));

                // Deallocate stack-passed arguments and alignment padding so
                // the stack remains 16-byte aligned for any subsequent `call`
                // instructions before the function epilogue.
                let bytes_dealloc = 8 * stack_args + (if needs_padding { 8 } else { 0 });
                if bytes_dealloc != 0 {
                    instructions.push(Instruction::StackDealloc(bytes_dealloc.cast_signed()));
                }

                instructions.push(Instruction::Mov {
                    src: Operand::Register(Reg::AX),
                    dst: generate_mir_operand(dst),
                });
            }
            ir::Instruction::Label(label) => {
                instructions.push(Instruction::Label(label.as_str()));
            }
        }
    }

    Function {
        label: func.ident,
        instructions,
        is_global: func.is_global,
    }
}

/// Generate a _MIR x86-64_ operand from the provided _IR_ value.
const fn generate_mir_operand(val: &ir::Value) -> Operand<'_> {
    match val {
        ir::Value::IntConstant(i) => Operand::Imm32(*i),
        ir::Value::Var { ident, is_static } => Operand::Symbol {
            ident: ident.as_str(),
            is_static: *is_static,
        },
    }
}

/// Lowers _IR_ function parameters into _MIR x86-64_ instructions, appending to
/// `out`.
fn lower_ir_function_params<'a>(params: &'a [&str], out: &mut Vec<Instruction<'a>>) {
    if params.is_empty() {
        return;
    }

    // According to the System-V ABI calling convention, the first six function
    // parameters are accessed from the following registers:
    //
    // 64-bit: `%rdi`, `%rsi`, `%rdx`, `%rcx`, `%r8`, `%r9`
    // 32-bit: `%edi`, `%esi`, `%edx`, `%ecx`, `%r8d`, `%r9d`
    //
    // Copying parameters to the stack ensures no caller/callee-saved registers
    // are affected.
    let regs = [Reg::DI, Reg::SI, Reg::DX, Reg::CX, Reg::R8, Reg::R9];
    for (i, param) in params.iter().take(6).enumerate() {
        out.push(Instruction::Mov {
            src: Operand::Register(regs[i]),
            dst: Operand::Symbol {
                ident: param,
                is_static: false,
            },
        });
    }

    // Any remaining parameters are accessed from the stack in right to left
    // order.
    if params.len().saturating_sub(6) > 0 {
        let remaining_params = &params[6..];

        // Since the `call` instruction pushes the return address (8 bytes) onto
        // the stack before transferring control, the seventh parameter begins
        // at `16(%rbp)`.
        let mut stack_offset = 16;

        for param in remaining_params {
            out.push(Instruction::Mov {
                // Positive offsets are used to refer to the caller stack frame.
                src: Operand::Stack(stack_offset),
                dst: Operand::Symbol {
                    ident: param,
                    is_static: false,
                },
            });

            stack_offset += 8; // 8 bytes per parameter.
        }
    }
}

/// Returns the conversion of an _AST_ binary operator to a _MIR x86-64_ binary
/// operator, or `None` if unsupported.
#[inline]
const fn ast_to_mir_binop(binop: ast::BinaryOperator) -> Option<BinaryOperator> {
    match binop {
        ast::BinaryOperator::Add => Some(BinaryOperator::Add),
        ast::BinaryOperator::Subtract => Some(BinaryOperator::Sub),
        ast::BinaryOperator::Multiply => Some(BinaryOperator::Imul),
        ast::BinaryOperator::BitAnd => Some(BinaryOperator::And),
        ast::BinaryOperator::BitOr => Some(BinaryOperator::Or),
        ast::BinaryOperator::BitXor => Some(BinaryOperator::Xor),
        ast::BinaryOperator::ShiftLeft => Some(BinaryOperator::Shl),
        // Default to `shr` instead of `sar`. `sar` is handled separately.
        ast::BinaryOperator::ShiftRight => Some(BinaryOperator::Shr),
        _ => None,
    }
}
