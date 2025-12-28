//! Code Emission.
//!
//! Compiler pass that emits textual assembly from the compiler's structured
//! assembly representation (_x86-64_).

use std::fs;
use std::process;
use std::{fmt::Write, io::Write as IoWrite};

use crate::compiler::mir::{self, MIR};
use crate::compiler::parser::UnaryOperator;
use crate::{Context, report_err};

/// Writes _gas-x86-64-linux_ assembly from the provided structured assembly
/// representation to the output file in the provided context. [Exits] on
/// error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn emit_assembly(ctx: &Context<'_>, asm: &MIR) {
    let Ok(mut f) = fs::File::create(ctx.out_path) else {
        report_err!(ctx.program, "failed to create output file");
        process::exit(1);
    };

    match asm {
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
                   "failed to emit assembly to '{}': {err}",
                   ctx.out_path.display()
               );
               process::exit(1);
            });

            write!(&mut f, "{}", emit_function(ctx, func)).unwrap_or_else(|err| {
                report_err!(
                    ctx.program,
                    "failed to emit assembly to '{}': {err}",
                    ctx.out_path.display()
                );
                process::exit(1);
            });
        }
    }

    // Indicates the program does not need an executable stack (Linux).
    writeln!(&mut f, "\t.section\t.note.GNU-stack,\"\",@progbits").unwrap_or_else(|err| {
        report_err!(
            ctx.program,
            "failed to emit assembly to '{}': {err}",
            ctx.out_path.display()
        );
        process::exit(1);
    });
}

/// Return a string representation of the given `ASM` function.
fn emit_function(ctx: &Context<'_>, func: &mir::Function) -> String {
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
        report_err!(
            ctx.program,
            "failed to emit assembly to '{}': {err}",
            ctx.out_path.display()
        );
        process::exit(1);
    });

    for instruction in &func.instructions {
        writeln!(&mut asm, "\t{}", emit_instruction(instruction)).unwrap_or_else(|err| {
            report_err!(
                ctx.program,
                "failed to emit assembly to '{}': {err}",
                ctx.out_path.display()
            );
            process::exit(1);
        });
    }

    asm
}

/// Return a string representation of the given `ASM` instruction.
fn emit_instruction(instruction: &mir::Instruction) -> String {
    match instruction {
        mir::Instruction::Mov(src, dst) => {
            format!("movl\t{}, {}", emit_operand(src), emit_operand(dst))
        }
        mir::Instruction::Ret => "movq\t%rbp, %rsp\n\tpopq\t%rbp\n\tret".into(),
        mir::Instruction::Unary(op, operand) => match op {
            UnaryOperator::Complement => format!("notl\t{}", emit_operand(operand)),
            UnaryOperator::Negate => format!("negl\t{}", emit_operand(operand)),
        },
        mir::Instruction::AllocateStack(v) => format!("subq\t${v}, %rsp"),
    }
}

/// Return a string representation of the given `ASM` operand.
fn emit_operand(op: &mir::Operand) -> String {
    match op {
        mir::Operand::Imm(v) => format!("${v}"),
        mir::Operand::Register(reg) => format!("{reg}"),
        mir::Operand::Stack(s) => format!("-{s}(%rbp)"),
        mir::Operand::Pseudo(_) => {
            panic!("should not be emitting pseudo registers in textual assembly")
        }
    }
}
