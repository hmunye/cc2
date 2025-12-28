//! Code Emission.
//!
//! Compiler pass that emits textual assembly from the compiler's structured
//! assembly representation (_x86-64_).

use std::fs;
use std::io::Write;
use std::process;

use crate::compiler::ir::{IR, Instruction, Operand};
use crate::{Context, report_err};

/// Generate _gas-x86-64-linux_ assembly from the provided `IR`, written to the
/// output file in the passed context. [Exits] on error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn emit_assembly(ctx: &Context<'_>, ir: &IR) {
    let Ok(mut f) = fs::File::create(ctx.out_path) else {
        report_err!(ctx.program, "failed to create output file");
        process::exit(1);
    };

    match ir {
        IR::Program(func) => {
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

            for instruction in &func.instructions {
                writeln!(&mut f, "\t{}", emit_instruction(instruction)).unwrap_or_else(|err| {
                    report_err!(
                        ctx.program,
                        "failed to emit assembly to '{}': {err}",
                        ctx.out_path.display()
                    );
                    process::exit(1);
                });
            }

            // Indicates the program does not need an executable stack.
            writeln!(&mut f, ".section\t.note.GNU-stack,\"\",@progbits").unwrap_or_else(|err| {
                report_err!(
                    ctx.program,
                    "failed to emit assembly to '{}': {err}",
                    ctx.out_path.display()
                );
                process::exit(1);
            });
        }
    }
}

/// Return a string representation of the given `IR` instruction.
fn emit_instruction(instruction: &Instruction) -> String {
    match instruction {
        Instruction::Mov(src, dst) => {
            format!("movl\t{}, {}", emit_operand(src), emit_operand(dst))
        }
        Instruction::Ret => "ret".into(),
    }
}

/// Return a string representation of the given `IR` operand.
fn emit_operand(op: &Operand) -> String {
    match op {
        Operand::Imm(v) => format!("${v}"),
        Operand::Register(reg) => format!("%{reg}"),
    }
}
