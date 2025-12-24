//! Code Emission.
//!
//! Compiler pass that emits textual assembly from the compiler's intermediate
//! assembly representation.

use std::convert::AsRef;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use crate::compiler::ir::{IR, Instruction, Operand};

/// Generate an _x86-64_ assembly file from the provided `IR`, written to the
/// specified output file.
pub fn emit_assembly(
    source_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    ir: &IR,
) -> io::Result<()> {
    let mut f = fs::File::create(output_path)?;

    match ir {
        IR::Program(func) => {
            writeln!(
                &mut f,
                "\t.file\t\"{}\"\n\t.text\n\t.globl\t{label}\n\t.type\t{label}, @function\n{label}:",
                source_path.as_ref().display(),
                label = func.label
            )?;

            for instruction in &func.instructions {
                writeln!(&mut f, "\t{}", emit_instruction(instruction))?;
            }

            // Indicates the program does not need an executable stack on
            // Linux.
            writeln!(&mut f, ".section\t.note.GNU-stack,\"\",@progbits")?;
        }
    }

    Ok(())
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
