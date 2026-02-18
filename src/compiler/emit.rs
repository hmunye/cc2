//! Code Emission
//!
//! Compiler pass that emits _gas-x86-64-linux_ textual assembly from the
//! an _MIR x86-64_ representation.

use std::borrow::Cow;
use std::collections::HashSet;
use std::io::{self, BufWriter, Write};

use crate::compiler::Context;
use crate::compiler::mir::{self, BinaryOperator, MIRX86, UnaryOperator};

/// Emits _gas-x86-64-linux_ assembly from a _x86-64_ machine intermediate
/// representation (_MIR_) to the provided `writer`.
///
/// # Errors
///
/// Returns an error if textual assembly could not be written to the `writer`.
pub fn emit_gas_x86_64_linux(
    ctx: &Context<'_>,
    mir: &MIRX86<'_>,
    writer: Box<dyn Write>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(writer);

    writeln!(
        &mut writer,
        "\t.file\t\"{}\"\n\t.text",
        ctx.in_path.display()
    )?;

    // Track last emitted _ELF_ section.
    let mut curr_section = ".text";

    // let mut i = 0;

    for item in &mir.program {
        match item {
            mir::Item::Func(func) => {
                let section_emit = if curr_section == ".text" {
                    ""
                } else {
                    curr_section = ".text";
                    "\t.text\n"
                };

                let globl_emit = if func.is_global {
                    Cow::Owned(format!("\t.globl\t{}\n", func.label))
                } else {
                    Cow::Borrowed("")
                };

                // `.L` is the local label prefix for Linux.
                writeln!(
                    &mut writer,
                    "{section_emit}{globl_emit}\t.type\t{label}, @function\n{label}:",
                    label = &func.label
                )?;

                // `FB` - Function Begin
                // writeln!(&mut writer, ".LFB{i}:")?;

                emit_asm_function(func, &mir.locales, &mut writer)?;

                // `FE` - Function End
                // writeln!(&mut writer, ".LFE{i}:")?;

                // `.size` directive records the byte size of the function in
                // the _ELF_ symbol table.
                writeln!(
                    &mut writer,
                    "\t.size\t{label}, .-{label}",
                    label = &func.label
                )?;

                // i += 1;
            }
            mir::Item::Static {
                init,
                label,
                is_global,
            } => {
                let globl_emit = if *is_global {
                    Cow::Owned(format!("\t.globl\t{label}\n"))
                } else {
                    Cow::Borrowed("")
                };

                if *init == 0 {
                    let section_emit = if curr_section == ".bss" {
                        ""
                    } else {
                        curr_section = ".bss";
                        "\t.bss\n"
                    };

                    if globl_emit.is_empty() {
                        // Declare a zero-initialized internal variable as a
                        // common symbol. `.local` makes it private to the
                        // object file. `.comm` reserves `size` bytes with
                        // `align` alignment; the linker places it in the
                        // `.bss` section at runtime.
                        writeln!(
                            &mut writer,
                            "\t.local\t{label}\n\t.comm\t{label},{size},{align}",
                            size = 4,
                            align = 4
                        )?;
                    } else {
                        // `.align` directive aligns a value to `2^n` on macOS,
                        // `.balign` directive aligns a value to `n` bytes.
                        //
                        // `.align` and `.balign` are interchangeable on Linux.
                        //
                        // `.size` directive records the byte size of the object
                        // in the _ELF_ symbol table.
                        writeln!(
                            &mut writer,
                            "{globl_emit}{section_emit}\t.align\t{align}\n\t.type\t{label}, @object\n\t.size\t{label}, {size}\n{label}:\n\t.zero\t{size}",
                            align = 4,
                            size = 4
                        )?;
                    }
                } else {
                    let section_emit = if curr_section == ".data" {
                        ""
                    } else {
                        curr_section = ".data";
                        "\t.data\n"
                    };

                    writeln!(
                        &mut writer,
                        "{globl_emit}{section_emit}\t.align\t{align}\n\t.type\t{label}, @object\n\t.size\t{label}, {size}\n{label}:\n\t.long\t{init}",
                        align = 4,
                        size = 4
                    )?;
                }
            }
        }
    }

    // Indicates the program does not need an executable stack on Linux.
    writeln!(
        &mut writer,
        "\t.ident\t\"cc2: {}\"\n\t.section\t.note.GNU-stack,\"\",@progbits",
        env!("CARGO_PKG_VERSION")
    )
}

/// Emit assembly representation of the given _MIR x86-64_ function to the
/// provided `writer`.
fn emit_asm_function(
    func: &mir::Function<'_>,
    locales: &HashSet<&str>,
    writer: &mut BufWriter<Box<dyn Write>>,
) -> io::Result<()> {
    // Generate the function prologue:
    //
    // 1. Push the current base pointer (`%rbp`) onto the stack to save the
    // caller's stack frame base.
    //
    // 2. Move the current stack pointer (`%rsp`) into the base pointer (`%rbp`)
    // to establish the start of the callee's stack frame.
    writeln!(writer, "\tpushq\t%rbp\n\tmovq\t%rsp, %rbp")?;

    for inst in &func.instructions {
        if let mir::Instruction::Label(label) = inst {
            writeln!(writer, ".L{label}:")?;
            continue;
        }

        writeln!(writer, "\t{}", emit_asm_instruction(inst, locales))?;
    }

    Ok(())
}

/// Return a string assembly representation of the given _MIR x86-64_
/// instruction.
fn emit_asm_instruction(instruction: &mir::Instruction<'_>, locales: &HashSet<&str>) -> String {
    match instruction {
        mir::Instruction::Mov { src, dst } => {
            format!(
                "movl\t{}, {}",
                emit_asm_operand(src, 4),
                emit_asm_operand(dst, 4)
            )
        }
        mir::Instruction::Unary { unop, dst } => match unop {
            UnaryOperator::Not => format!("notl\t{}", emit_asm_operand(dst, 4)),
            UnaryOperator::Neg => format!("negl\t{}", emit_asm_operand(dst, 4)),
        },
        mir::Instruction::Binary { binop, rhs, dst } => {
            let (inst, size) = match binop {
                BinaryOperator::Add => ("addl", 4),
                BinaryOperator::Sub => ("subl", 4),
                BinaryOperator::Imul => ("imull", 4),
                BinaryOperator::And => ("andl", 4),
                BinaryOperator::Or => ("orl", 4),
                BinaryOperator::Xor => ("xorl", 4),
                // `l` suffix on shift mnemonics indicates the destination
                // operand is 32-bit. 1 indicates the source operand is the
                // 8-bit "%cl" register or an immediate value.
                BinaryOperator::Shl => ("shll", 1),
                BinaryOperator::Shr => ("shrl", 1),
                BinaryOperator::Sar => ("sarl", 1),
            };

            format!(
                "{}\t{}, {}",
                inst,
                emit_asm_operand(rhs, size),
                emit_asm_operand(dst, size)
            )
        }
        mir::Instruction::Idiv(div) => format!("idivl\t{}", emit_asm_operand(div, 4)),
        mir::Instruction::Cdq => "cdq".into(),
        mir::Instruction::Cmp { rhs, lhs } => format!(
            "cmpl\t{}, {}",
            emit_asm_operand(rhs, 4),
            emit_asm_operand(lhs, 4)
        ),
        // `.L` is the local label prefix for Linux.
        mir::Instruction::Jmp(label) => format!("jmp\t.L{label}"),
        mir::Instruction::JmpC { code, label } => format!("j{code}\t.L{label}"),
        mir::Instruction::SetC { code, dst } => format!("set{code}\t{}", emit_asm_operand(dst, 1)),
        mir::Instruction::StackAlloc(v) => format!("subq\t${v}, %rsp"),
        mir::Instruction::StackDealloc(v) => format!("addq\t${v}, %rsp"),
        mir::Instruction::Push(src) => format!("pushq\t{}", emit_asm_operand(src, 8)),
        // On _macOS_, function names are prefixed with underscore
        // (`call _puts`).
        //
        // On Linux, function names not defined in the current translation unit
        // are suffixed with `@PLT` (Procedure Linkage Table), a section in
        // _ELF_ binaries (`call puts@PLT`).
        //
        // `call` instruction pushes the address of the following instruction
        // onto the stack, then loads the label’s address into `%rip`.
        mir::Instruction::Call(label) => {
            format!(
                "call\t{label}{}",
                if locales.contains(label) { "" } else { "@PLT" }
            )
        }
        // Include the function epilogue before returning to the caller:
        //
        // 1. Move the saved stack pointer in `%rbp` back into the `%rsp` to
        // restore the stack frame to its state after the function prologue.
        //
        // 2. Pop the previously pushed base pointer (`%rbp`) from the stack
        // back into `%rbp`, restoring the caller's stack frame.
        //
        // 3. Return control to the caller, moving the return address pushed by
        // the caller's `call` instruction into `%rip`.
        mir::Instruction::Ret => "movq\t%rbp, %rsp\n\tpopq\t%rbp\n\tret".into(),
        mir::Instruction::Label(_) => panic!("label emission should not be handled here"),
    }
}

/// Return a string assembly representation of the given _MIR_ operand. `size`
/// formats register operands depending on the required size in bytes.
fn emit_asm_operand(op: &mir::Operand<'_>, size: u8) -> String {
    match op {
        mir::Operand::Imm32(i) => format!("${i}"),
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
        mir::Operand::Stack(i) => format!("{i}(%rbp)"),
        // On _macOS_, symbol is prefixed with underscore (e.g., `_foo(%rip)`).
        mir::Operand::Data(symbol) => format!("{symbol}(%rip)"),
        mir::Operand::Symbol(_) => panic!("pseudoregisters should not be emitted to assembly"),
    }
}
