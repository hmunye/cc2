//! Code Emission
//!
//! Compiler pass that emits textual _gas-x86-64-linux_ assembly from the
//! compiler's _MIR_.

use std::process;
use std::{fmt::Write, io::Write as IoWrite};

use crate::compiler::mir::{self, MIR};
use crate::compiler::mir::{BinaryOperator, UnaryOperator};
use crate::{Context, report_err};

/// Emits _gas-x86-64-linux_ assembly given a _MIR_ to the provided output.
/// [Exits] on error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn emit_asm(ctx: &Context<'_>, mir: &MIR, mut f: Box<dyn IoWrite>) {
    match mir {
        MIR::Program(func) => {
            // Write the function prologue in GNU `as` (assembler) format.
            writeln!(
                &mut f,
                "\t.file\t\"{}\"\n\t.text\n\t.globl\t{label}\n\t.type\t{label}, @function\n{label}:",
                ctx.in_path.display(),
                label = func.label
            ).unwrap_or_else(|err| {
               report_err!(
                   ctx.program,
                   "failed to emit assembly: {err}",
               );
               process::exit(1);
            });

            write!(&mut f, "{}", emit_asm_function(ctx, func)).unwrap_or_else(|err| {
                report_err!(ctx.program, "failed to emit assembly: {err}");
                process::exit(1);
            });
        }
    }

    // Indicates the program does not need an executable stack (Linux).
    writeln!(&mut f, "\t.section\t.note.GNU-stack,\"\",@progbits").unwrap_or_else(|err| {
        report_err!(ctx.program, "failed to emit assembly: {err}");
        process::exit(1);
    });
}

/// Return a string assembly representation of the given _MIR_ function.
fn emit_asm_function(ctx: &Context<'_>, func: &mir::Function) -> String {
    let mut asm = String::new();

    // Generate the function prologue:
    //
    // 1. Push the current base pointer (`rbp`) onto the stack to save
    // the caller's stack frame.
    //
    // 2. Move the current stack pointer (`rsp`) into the base pointer
    // (`rbp`) to establish the start of the current function's stack
    // frame.
    writeln!(&mut asm, "\tpushq\t%rbp\n\tmovq\t%rsp, %rbp").unwrap_or_else(|err| {
        report_err!(ctx.program, "failed to emit assembly: {err}");
        process::exit(1);
    });

    // Updated if a `StackAlloc` instruction is encountered, so the allocated
    // stack memory can be deallocated if returning.
    let mut alloc = 0;

    for inst in &func.instructions {
        if let mir::Instruction::StackAlloc(b) = inst {
            // Accumulate any allocations made before any `Return` instruction.
            alloc += *b;
        }

        if let mir::Instruction::Label(label) = inst {
            writeln!(&mut asm, ".L{label}:").unwrap_or_else(|err| {
                report_err!(ctx.program, "failed to emit assembly: {err}");
                process::exit(1);
            });
            continue;
        }

        writeln!(&mut asm, "\t{}", emit_asm_instruction(inst, alloc)).unwrap_or_else(|err| {
            report_err!(ctx.program, "failed to emit assembly: {err}");
            process::exit(1);
        });
    }

    asm
}

/// Return a string assembly representation of the given `MIR` instruction.
fn emit_asm_instruction(instruction: &mir::Instruction, alloc: i32) -> String {
    match instruction {
        mir::Instruction::Mov(src, dst) => {
            format!(
                "movl\t{}, {}",
                emit_asm_operand(src, 4),
                emit_asm_operand(dst, 4)
            )
        }
        mir::Instruction::Unary(unop, operand) => match unop {
            UnaryOperator::Not => format!("notl\t{}", emit_asm_operand(operand, 4)),
            UnaryOperator::Neg => format!("negl\t{}", emit_asm_operand(operand, 4)),
        },
        mir::Instruction::Binary(binop, lhs, rhs) => {
            let (instr, size) = match binop {
                BinaryOperator::Add => ("addl", 4),
                BinaryOperator::Sub => ("subl", 4),
                BinaryOperator::Imul => ("imull", 4),
                BinaryOperator::And => ("andl", 4),
                BinaryOperator::Or => ("orl", 4),
                BinaryOperator::Xor => ("xorl", 4),
                // `l` suffix on shift mnemonics indicates the destination
                // operand is 32-bit. 1 indicates the source operand is 8-bit
                // "%cl" register or immediate value.
                BinaryOperator::Shl => ("shll", 1),
                BinaryOperator::Shr => ("shrl", 1),
                BinaryOperator::Sar => ("sarl", 1),
            };

            format!(
                "{}\t{}, {}",
                instr,
                emit_asm_operand(lhs, size),
                emit_asm_operand(rhs, size)
            )
        }
        mir::Instruction::Idiv(div) => format!("idivl\t{}", emit_asm_operand(div, 4)),
        mir::Instruction::Cdq => "cdq".into(),
        mir::Instruction::Cmp(src, dst) => format!(
            "cmpl\t{}, {}",
            emit_asm_operand(src, 4),
            emit_asm_operand(dst, 4)
        ),
        // `.L` is the local label prefix for Linux.
        mir::Instruction::Jmp(label) => format!("jmp\t.L{label}"),
        mir::Instruction::JmpC(code, label) => format!("j{code}\t.L{label}"),
        mir::Instruction::SetC(code, dst) => format!("set{code}\t{}", emit_asm_operand(dst, 1)),
        mir::Instruction::StackAlloc(v) => format!("subq\t${v}, %rsp"),
        // Include the function epilogue before returning to the caller:
        //
        // 1. Move the current base pointer (`rbp`) into the stack pointer
        // (`rsp`) to restore the stack to its state before the function was
        // called.
        //
        // 2. Pop the previously saved base pointer (`rbp`) from the stack back
        // into the `rbp` register, restoring the caller's stack frame.
        //
        // 3. Return control to the caller, jumping to the return address stored
        // on the caller's stack frame.
        mir::Instruction::Ret => {
            if alloc == 0 {
                "movq\t%rbp, %rsp\n\tpopq\t%rbp\n\tret".into()
            } else {
                format!("addq\t${alloc}, %rsp\n\tmovq\t%rbp, %rsp\n\tpopq\t%rbp\n\tret")
            }
        }
        mir::Instruction::Label(_) => {
            unreachable!("arm should be handled outside the function for proper formatting")
        }
    }
}

/// Return a string assembly representation of the given `MIR` operand.
///
/// `size` formats register operands depending on the required width (in bytes).
fn emit_asm_operand(op: &mir::Operand, size: u8) -> String {
    match op {
        mir::Operand::Imm32(v) => format!("${v}"),
        mir::Operand::Register(r) => match r {
            mir::Reg::AX => match size {
                1 => "%al",
                2 => "%ax",
                4 => "%eax",
                8 => "%rax",
                _ => panic!("invalid register size for AX: '{}'", size),
            }
            .to_string(),
            mir::Reg::CX => match size {
                1 => "%cl",
                2 => "%cx",
                4 => "%ecx",
                8 => "%rcx",
                _ => panic!("invalid register size for CX: '{}'", size),
            }
            .to_string(),
            mir::Reg::DX => match size {
                1 => "%dl",
                2 => "%dx",
                4 => "%edx",
                8 => "%rdx",
                _ => panic!("invalid register size for DX: '{}'", size),
            }
            .to_string(),
            mir::Reg::R10 => match size {
                1 => "%r10b",
                2 => "%r10w",
                4 => "%r10d",
                8 => "%r10",
                _ => panic!("invalid register size for R10: '{}'", size),
            }
            .to_string(),
            mir::Reg::R11 => match size {
                1 => "%r11b",
                2 => "%r11w",
                4 => "%r11d",
                8 => "%r11",
                _ => panic!("invalid register size for R11: '{}'", size),
            }
            .to_string(),
        },
        mir::Operand::Stack(s) => format!("-{s}(%rbp)"),
        mir::Operand::Pseudo(_) => {
            panic!("pseudoregisters should not be encountered when emitting assembly")
        }
    }
}
