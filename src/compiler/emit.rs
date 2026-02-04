//! Code Emission
//!
//! Compiler pass that emits textual _gas-x86-64-linux_ assembly from the
//! compiler's _MIR_.

use std::collections::HashSet;
use std::fmt::Write;
use std::io::{self, Write as IoWrite};

use crate::Context;
use crate::compiler::mir::{self, MIRX86};
use crate::compiler::mir::{BinaryOperator, UnaryOperator};

/// Emits _gas-x86-64-linux_ assembly from a _x86-64_ machine intermediate
/// representation (_MIR_) to the provided `writer`.
///
/// # Errors
///
/// Returns an error if writing textual assembly fails.
pub fn emit_gas_x86_64_linux(
    ctx: &Context<'_>,
    mir: &MIRX86<'_>,
    mut writer: Box<dyn IoWrite>,
) -> io::Result<()> {
    // Write the file prologue in GNU `as` (assembler) format.
    writeln!(
        &mut writer,
        "\t.file\t\"{}\"\n\t.text",
        ctx.in_path.display()
    )?;

    for (i, func) in mir.program.iter().enumerate() {
        // `.L` is the local label prefix for Linux.
        writeln!(
            &mut writer,
            "\t.globl\t{label}\n\t.type\t{label}, @function\n{label}:\n.LFB{i}:",
            label = &func.label
        )?;

        write!(&mut writer, "{}", emit_asm_function(func, &mir.locales)?)?;

        // Records the byte size of the function in the _ELF_ symbol table.
        //
        // `.L` is the local label prefix for Linux.
        writeln!(
            &mut writer,
            ".LFE{i}:\n\t.size\t{label}, .-{label}",
            label = &func.label
        )?;
    }

    // Indicates the program does not need an executable stack on Linux.
    writeln!(
        &mut writer,
        "\t.ident\t\"cc2: {}\"\n\t.section\t.note.GNU-stack,\"\",@progbits",
        env!("CARGO_PKG_VERSION")
    )
}

/// Return a string assembly representation of the given _MIR_ function.
fn emit_asm_function(func: &mir::Function<'_>, locales: &HashSet<&str>) -> io::Result<String> {
    let mut asm = String::new();

    // Generate the function prologue:
    //
    // 1. Push the current base pointer (`rbp`) onto the stack to save
    // the caller's stack frame.
    //
    // 2. Move the current stack pointer (`rsp`) into the base pointer
    // (`rbp`) to establish the start of the current function's stack
    // frame.
    writeln!(&mut asm, "\tpushq\t%rbp\n\tmovq\t%rsp, %rbp").map_err(io::Error::other)?;

    for inst in &func.instructions {
        if let mir::Instruction::Label(label) = inst {
            writeln!(&mut asm, ".L{label}:").map_err(io::Error::other)?;
            continue;
        }

        writeln!(&mut asm, "\t{}", emit_asm_instruction(inst, locales))
            .map_err(io::Error::other)?;
    }

    Ok(asm)
}

/// Return a string assembly representation of the given _MIR_ instruction.
fn emit_asm_instruction(instruction: &mir::Instruction<'_>, locales: &HashSet<&str>) -> String {
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
        mir::Instruction::StackDealloc(v) => format!("addq\t${v}, %rsp"),
        mir::Instruction::Push(src) => format!("pushq\t{}", emit_asm_operand(src, 8)),
        // On _macOS_, function names are prefixed with underscore
        // (e.g., `call _foo`).
        //
        // On Linux, function names not defined in the current translation unit
        // are suffixed with `@PLT` (Procedure Linkage Table), a section in
        // `ELF` binaries (e.g., `call puts@PLT`).
        mir::Instruction::Call(label) => {
            format!(
                "call\t{label}{}",
                if locales.contains(label) { "" } else { "@PLT" }
            )
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
        mir::Instruction::Ret => "movq\t%rbp, %rsp\n\tpopq\t%rbp\n\tret".into(),
        mir::Instruction::Label(_) => panic!("label emission should not be handled here"),
    }
}

/// Return a string assembly representation of the given _MIR_ operand.
///
/// `size` formats register operands depending on the required bytes.
fn emit_asm_operand(op: &mir::Operand<'_>, size: u8) -> String {
    match op {
        mir::Operand::Imm32(v) => format!("${v}"),
        mir::Operand::Register(r) => match r {
            mir::Reg::AX => match size {
                1 => "%al",
                2 => "%ax",
                4 => "%eax",
                8 => "%rax",
                _ => panic!("invalid register size for AX: '{size}'"),
            }
            .to_string(),
            mir::Reg::CX => match size {
                1 => "%cl",
                2 => "%cx",
                4 => "%ecx",
                8 => "%rcx",
                _ => panic!("invalid register size for CX: '{size}'"),
            }
            .to_string(),
            mir::Reg::DX => match size {
                1 => "%dl",
                2 => "%dx",
                4 => "%edx",
                8 => "%rdx",
                _ => panic!("invalid register size for DX: '{size}'"),
            }
            .to_string(),
            mir::Reg::DI => match size {
                1 => "%dil",
                2 => "%di",
                4 => "%edi",
                8 => "%rdi",
                _ => panic!("invalid register size for DI: '{size}'"),
            }
            .to_string(),
            mir::Reg::SI => match size {
                1 => "%sil",
                2 => "%si",
                4 => "%esi",
                8 => "%rsi",
                _ => panic!("invalid register size for SI: '{size}'"),
            }
            .to_string(),
            mir::Reg::R8 => match size {
                1 => "%r8b",
                2 => "%r8w",
                4 => "%r8d",
                8 => "%r8",
                _ => panic!("invalid register size for R8: '{size}'"),
            }
            .to_string(),
            mir::Reg::R9 => match size {
                1 => "%r9b",
                2 => "%r9w",
                4 => "%r9d",
                8 => "%r9",
                _ => panic!("invalid register size for R9: '{size}'"),
            }
            .to_string(),
            mir::Reg::R10 => match size {
                1 => "%r10b",
                2 => "%r10w",
                4 => "%r10d",
                8 => "%r10",
                _ => panic!("invalid register size for R10: '{size}'"),
            }
            .to_string(),
            mir::Reg::R11 => match size {
                1 => "%r11b",
                2 => "%r11w",
                4 => "%r11d",
                8 => "%r11",
                _ => panic!("invalid register size for R11: '{size}'"),
            }
            .to_string(),
        },
        mir::Operand::Stack(s) => format!("{s}(%rbp)"),
        mir::Operand::Pseudo(_) => panic!("pseudoregisters should not be emitted"),
    }
}
