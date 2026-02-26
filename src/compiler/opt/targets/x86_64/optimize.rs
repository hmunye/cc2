//! Optimization Pipeline (_MIR x86-64_)
//!
//! Executes machine-dependent optimization passes on an _x86-64_ machine
//! intermediate representation (_MIR_) based on user-specified options.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::args::Opts;
use crate::compiler::frontend::parser::symbols::StorageDuration;
use crate::compiler::targets::x86_64::{BinaryOperator, Instruction, Item, MIRX86, Operand, Reg};
use crate::compiler::{self, frontend::SymbolTable};

/// Runs machine-dependent, intraprocedural optimization passes, on the given
/// _x86-64_ machine intermediate representation (_MIR_), according to the
/// optimizations specified.
pub fn optimize_x86_64_mir<'a>(mir: &mut MIRX86<'a>, opts: &Opts, sym_table: &'a SymbolTable) {
    let mut callee_alloc = vec![];

    for item in &mut mir.program {
        if let Item::Fn(f) = item {
            optimize_mir_function(&mut f.instructions, sym_table, opts, &mut callee_alloc);
        }
    }
}

/// Optimizes the provided _MIR x86-64_ instructions, applying the specified
/// optimizations based on the given `opts`.
fn optimize_mir_function<'a>(
    instructions: &mut Vec<Instruction<'a>>,
    sym_table: &'a SymbolTable,
    opts: &Opts,
    callee_alloc: &mut Vec<Reg>,
) {
    if instructions.is_empty() {
        return;
    }

    callee_alloc.clear();

    if opts.reg_alloc {
        callee_alloc.extend(compiler::opt::targets::x86_64::allocate_registers(
            instructions,
            sym_table,
            opts.coalesce,
        ));
    } else if opts.coalesce {
        let _ = compiler::opt::targets::x86_64::coalesce_loop(instructions, sym_table);
    }

    finalize_stack_frame(instructions, sym_table, callee_alloc);
}

/// Finalizes an _MIR x86-64_ stack frame, replacing symbols, rewriting invalid
/// instructions, and ensuring stack alignment.
fn finalize_stack_frame(
    instructions: &mut Vec<Instruction<'_>>,
    sym_table: &SymbolTable,
    callee_alloc: &[Reg],
) {
    let callee_bytes = (callee_alloc.len() * 8).cast_signed();

    let stack_offset = replace_symbols(instructions, sym_table) + callee_bytes;

    let padding = if stack_offset & 0xF != 0 {
        16 - (stack_offset & 0xF)
    } else {
        0
    };

    let alloc = (stack_offset + padding) - callee_bytes;

    rewrite_invalid_instructions(instructions, callee_alloc);

    if alloc > 0 {
        // NOTE: O(n) time complexity.
        instructions.insert(0, Instruction::StackAlloc(alloc));
    }
}

/// Replaces each symbolic operand within the _MIR x86-64_ instructions with its
/// corresponding location: either a stack offset from `%rbp` or an address in
/// the `.bss` / `.data` _ELF_ section, returning the final stack offset used.
fn replace_symbols(instructions: &mut Vec<Instruction<'_>>, sym_table: &SymbolTable) -> isize {
    let mut offset_map: HashMap<&str, isize> = HashMap::default();

    let mut stack_offset = 0;

    let mut convert_symbol = |ident| {
        if let Some(sym_info) = sym_table.get(ident)
            && sym_info.duration == Some(StorageDuration::Static)
        {
            Operand::Data(ident)
        } else {
            // Either we encountered an `automatic` or `IR` temporary variable.
            let offset = match offset_map.entry(ident) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    // NOTE: Allocating stack in 4-byte offsets.
                    stack_offset += 4;
                    // Negating the offset refers to a local variable in the
                    // stack frame relative to `%rbp`.
                    entry.insert(-stack_offset);
                    -stack_offset
                }
            };

            Operand::Stack(offset)
        }
    };

    for inst in instructions {
        match inst {
            Instruction::Mov { src, dst } => {
                if let Operand::Symbol { ident, .. } = src {
                    *src = convert_symbol(ident);
                }

                if let Operand::Symbol { ident, .. } = dst {
                    *dst = convert_symbol(ident);
                }
            }
            Instruction::Unary { dst, .. } | Instruction::SetC { dst, .. } => {
                if let Operand::Symbol { ident, .. } = dst {
                    *dst = convert_symbol(ident);
                }
            }
            Instruction::Binary { rhs, dst, .. } => {
                if let Operand::Symbol { ident, .. } = rhs {
                    *rhs = convert_symbol(ident);
                }

                if let Operand::Symbol { ident, .. } = dst {
                    *dst = convert_symbol(ident);
                }
            }
            Instruction::Cmp { rhs, lhs } => {
                if let Operand::Symbol { ident, .. } = rhs {
                    *rhs = convert_symbol(ident);
                }

                if let Operand::Symbol { ident, .. } = lhs {
                    *lhs = convert_symbol(ident);
                }
            }
            Instruction::Idiv(div) => {
                if let Operand::Symbol { ident, .. } = div {
                    *div = convert_symbol(ident);
                }
            }
            Instruction::Push(op) => {
                if let Operand::Symbol { ident, .. } = op {
                    *op = convert_symbol(ident);
                }
            }
            _ => {}
        }
    }

    stack_offset
}

/// Rewrite the provided _MIR x86-64_ instructions containing invalid operands
/// to valid _x86-64_ equivalents, as well as handling callee-saved registers.
fn rewrite_invalid_instructions(instructions: &mut Vec<Instruction<'_>>, callee_alloc: &[Reg]) {
    // Callee-saved registers that are used for allocation need to be saved
    // on the stack-frame (volatile).
    instructions.splice(
        0..0,
        callee_alloc
            .iter()
            .map(|reg| Instruction::Push(Operand::Register(*reg))),
    );

    let mut i = 0;

    while i < instructions.len() {
        let inst = &mut instructions[i];

        match inst {
            Instruction::Mov { src, dst } if src.is_memory_operand() && dst.is_memory_operand() => {
                let src = *src;
                let dst = *dst;

                // `%r10d` used as temporary storage for intermediate values in
                // new instructions. `%r10d`, unlike other hardware registers,
                // serves no special purpose, so we are less likely to encounter
                // a conflict.
                instructions.splice(
                    i..=i,
                    [
                        Instruction::Mov {
                            src,
                            dst: Operand::Register(Reg::R10),
                        },
                        Instruction::Mov {
                            src: Operand::Register(Reg::R10),
                            dst,
                        },
                    ],
                );

                // Ensures the two new instructions are skipped.
                i += 1;
            }
            Instruction::Idiv(div) if matches!(div, Operand::Imm32(_)) => {
                let div = *div;

                instructions.splice(
                    i..=i,
                    [
                        Instruction::Mov {
                            src: div,
                            dst: Operand::Register(Reg::R10),
                        },
                        Instruction::Idiv(Operand::Register(Reg::R10)),
                    ],
                );

                // Ensures the two new instructions are skipped.
                i += 1;
            }
            Instruction::Binary { binop, rhs, dst }
                if matches!(
                    binop,
                    BinaryOperator::Add
                        | BinaryOperator::Sub
                        | BinaryOperator::And
                        | BinaryOperator::Or
                        | BinaryOperator::Xor
                ) && rhs.is_memory_operand()
                    && dst.is_memory_operand() =>
            {
                let rhs = *rhs;
                let dst = *dst;
                let binop = *binop;

                instructions.splice(
                    i..=i,
                    [
                        Instruction::Mov {
                            src: rhs,
                            dst: Operand::Register(Reg::R10),
                        },
                        Instruction::Binary {
                            binop,
                            rhs: Operand::Register(Reg::R10),
                            dst,
                        },
                    ],
                );

                // Ensures the two new instructions are skipped.
                i += 1;
            }
            Instruction::Binary {
                binop: BinaryOperator::Imul,
                rhs,
                dst,
            } if dst.is_memory_operand() => {
                let rhs = *rhs;
                let dst = *dst;

                // `%r11d` used as temporary storage for intermediate values in
                // the new instructions.
                instructions.splice(
                    i..=i,
                    [
                        Instruction::Mov {
                            src: dst,
                            dst: Operand::Register(Reg::R11),
                        },
                        Instruction::Binary {
                            binop: BinaryOperator::Imul,
                            rhs,
                            dst: Operand::Register(Reg::R11),
                        },
                        Instruction::Mov {
                            src: Operand::Register(Reg::R11),
                            dst,
                        },
                    ],
                );

                // Ensures the three new instructions are skipped.
                i += 2;
            }
            Instruction::Binary { binop, rhs, dst }
                if matches!(
                    binop,
                    BinaryOperator::Shl | BinaryOperator::Shr | BinaryOperator::Sar
                ) && !matches!(rhs, Operand::Imm32(_) | Operand::Register(Reg::CX)) =>
            {
                let rhs = *rhs;
                let dst = *dst;
                let binop = *binop;

                instructions.splice(
                    i..=i,
                    [
                        Instruction::Mov {
                            src: rhs,
                            dst: Operand::Register(Reg::CX),
                        },
                        Instruction::Binary {
                            binop,
                            rhs: Operand::Register(Reg::CX),
                            dst,
                        },
                    ],
                );

                // Ensures the two new instructions are skipped.
                i += 1;
            }
            Instruction::Cmp { rhs, lhs }
                if (rhs.is_memory_operand() && lhs.is_memory_operand())
                    || matches!(lhs, Operand::Imm32(_)) =>
            {
                let rhs = *rhs;
                let lhs = *lhs;

                if let Operand::Imm32(_) = lhs {
                    instructions.splice(
                        i..=i,
                        [
                            Instruction::Mov {
                                src: lhs,
                                dst: Operand::Register(Reg::R11),
                            },
                            Instruction::Cmp {
                                rhs,
                                lhs: Operand::Register(Reg::R11),
                            },
                        ],
                    );
                } else {
                    instructions.splice(
                        i..=i,
                        [
                            Instruction::Mov {
                                src: rhs,
                                dst: Operand::Register(Reg::R10),
                            },
                            Instruction::Cmp {
                                rhs: Operand::Register(Reg::R10),
                                lhs,
                            },
                        ],
                    );
                }

                // Ensures the two new instructions are skipped.
                i += 1;
            }
            Instruction::Ret => {
                // Restore the values of callee-saved registers before
                // returning to the caller.
                instructions.splice(
                    i..i,
                    callee_alloc.iter().rev().map(|reg| Instruction::Pop(*reg)),
                );

                // Ensures the current and new instructions are skipped.
                i += callee_alloc.len() + 1;
            }
            _ => {}
        }

        i += 1;
    }
}
