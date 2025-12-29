//! Code Emission
//!
//! Compiler pass that emits textual _gas-x86-64-linux_ assembly from the
//! compiler's _MIR_.

use std::process;
use std::{fmt::Write, io::Write as IoWrite};

use crate::compiler::mir::{self, MIR};
use crate::compiler::parser::UnaryOperator;
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
            format!("movl\t{}, {}", emit_asm_operand(src), emit_asm_operand(dst))
        }
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
        //
        // NOTE: If any stack memory was allocated in the current stack frame,
        // it should deallocated before returning to the caller.
        mir::Instruction::Ret => {
            if alloc == 0 {
                "movq\t%rbp, %rsp\n\tpopq\t%rbp\n\tret".into()
            } else {
                format!("addq\t${alloc}, %rsp\n\tmovq\t%rbp, %rsp\n\tpopq\t%rbp\n\tret")
            }
        }
        mir::Instruction::Unary(op, operand) => match op {
            UnaryOperator::Complement => format!("notl\t{}", emit_asm_operand(operand)),
            UnaryOperator::Negate => format!("negl\t{}", emit_asm_operand(operand)),
        },
        mir::Instruction::StackAlloc(v) => format!("subq\t${v}, %rsp"),
    }
}

/// Return a string assembly representation of the given `MIR` operand.
fn emit_asm_operand(op: &mir::Operand) -> String {
    match op {
        mir::Operand::Imm32(v) => format!("${v}"),
        mir::Operand::Register(reg) => format!("{reg}"),
        mir::Operand::Stack(s) => format!("-{s}(%rbp)"),
        mir::Operand::Pseudo(_) => {
            panic!("pseudoregisters should not be encountered when emitting textual assembly")
        }
    }
}
