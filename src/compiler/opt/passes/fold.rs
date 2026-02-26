//! Constant Folding
//!
//! Transforms an intermediate representation (_IR_) by evaluating constant
//! expressions at compile-time.

use crate::compiler::frontend::ast::{self, BinaryOperator, UnaryOperator};
use crate::compiler::frontend::types::c_int;
use crate::compiler::ir::{Function, Instruction, Value};

/// Transforms an intermediate representation (_IR_) function by folding
/// all constant expressions.
pub fn fold_const(f: &mut Function<'_>) {
    let mut i = 0;
    while i < f.instructions.len() {
        let inst = &mut f.instructions[i];

        match inst {
            Instruction::Unary { op, src, dst, .. } => {
                if let Value::IntConstant(x) = src {
                    let val = eval_unary(*op, *x);

                    *inst = Instruction::Copy {
                        src: Value::IntConstant(val),
                        dst: dst.clone(),
                    };
                }
            }
            Instruction::Binary {
                op, lhs, rhs, dst, ..
            } => {
                if let Value::IntConstant(x) = lhs
                    && let Value::IntConstant(y) = rhs
                {
                    if !is_safe_fold(*op, *y) {
                        i += 1;
                        continue;
                    }

                    let val = eval_binary(*op, *x, *y);

                    *inst = Instruction::Copy {
                        src: Value::IntConstant(val),
                        dst: dst.clone(),
                    };
                }
            }
            &mut (Instruction::JumpIfZero {
                ref cond,
                ref target,
            }
            | Instruction::JumpIfNotZero {
                ref cond,
                ref target,
            }) => {
                if let Value::IntConstant(x) = cond {
                    if (matches!(inst, Instruction::JumpIfZero { .. }) && *x == 0)
                        || (matches!(inst, Instruction::JumpIfNotZero { .. }) && *x != 0)
                    {
                        // We know the jump to `target` will always be taken.
                        *inst = Instruction::Jump(target.clone());
                    } else {
                        // We know the jump to `target` will never be taken.
                        f.instructions.remove(i);
                        continue;
                    }
                }
            }
            Instruction::Return(_)
            | Instruction::Copy { .. }
            | Instruction::Jump(_)
            | Instruction::Label(_)
            | Instruction::FnCall { .. } => {}
        }

        i += 1;
    }
}

/// Attempts to fold and return the provided _AST_ expression as a single
/// constant value, or `None` if the expression cannot be evaluated to a
/// compile-time constant.
#[must_use]
pub fn try_fold_ast(expr: &ast::Expression<'_>) -> Option<c_int> {
    match expr {
        ast::Expression::Unary { op, expr, .. } => {
            let expr = try_fold_ast(expr)?;

            Some(eval_unary(*op, expr))
        }
        ast::Expression::Binary { op, lhs, rhs, .. } => {
            let rhs = try_fold_ast(rhs)?;

            if !is_safe_fold(*op, rhs) {
                return None;
            }

            let lhs = try_fold_ast(lhs)?;

            Some(eval_binary(*op, lhs, rhs))
        }
        ast::Expression::Conditional {
            cond,
            second,
            third,
        } => {
            let cond = try_fold_ast(cond)?;

            if cond != 0 {
                try_fold_ast(second)
            } else {
                try_fold_ast(third)
            }
        }
        ast::Expression::IntConstant(i) => Some(*i),
        ast::Expression::Var { .. }
        | ast::Expression::FnCall { .. }
        | ast::Expression::Assignment { .. } => None,
    }
}

/// Returns `true` if the binary operation, given it's `rhs`, is safe to fold.
#[inline]
#[must_use]
const fn is_safe_fold(op: BinaryOperator, rhs: i32) -> bool {
    #[allow(clippy::cast_possible_wrap)]
    match op {
        BinaryOperator::Divide | BinaryOperator::Modulo => rhs != 0,
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
            // NOTE: Should report diagnostic warning.
            rhs < (std::mem::size_of::<c_int>() * 8) as i32
        }
        _ => true,
    }
}

/// Evaluates a unary operator on a constant integer operand.
#[inline]
#[must_use]
fn eval_unary(unop: UnaryOperator, val: c_int) -> c_int {
    match unop {
        UnaryOperator::Complement => !val,
        UnaryOperator::Negate => -val,
        UnaryOperator::Not => i32::from(val == 0),
        UnaryOperator::Increment | UnaryOperator::Decrement => {
            panic!("{unop:?} expressions should not be folded at compile-time")
        }
    }
}

/// Evaluates a binary operator on two constant integer operands.
///
/// # Panics
///
/// Panics if `rhs` is zero in division or modulo, or if the shift factor
/// (left or right) is `size_of::<c_int> * 8` or greater.
#[inline]
#[must_use]
fn eval_binary(binop: BinaryOperator, lhs: c_int, rhs: c_int) -> c_int {
    match binop {
        BinaryOperator::Add => lhs.wrapping_add(rhs),
        BinaryOperator::Subtract => lhs.wrapping_sub(rhs),
        BinaryOperator::Multiply => lhs.wrapping_mul(rhs),
        BinaryOperator::Divide => lhs / rhs, // Panics if `rhs` is zero.
        BinaryOperator::Modulo => lhs % rhs, // Panics if `rhs` is zero.
        BinaryOperator::BitAnd => lhs & rhs,
        BinaryOperator::BitOr => lhs | rhs,
        BinaryOperator::BitXor => lhs ^ rhs,
        BinaryOperator::ShiftLeft => lhs << rhs, // Panics if `rhs` >= 32.
        BinaryOperator::ShiftRight => lhs >> rhs, // Panics if `rhs` >= 32.
        BinaryOperator::LogAnd => i32::from(lhs != 0 && rhs != 0),
        BinaryOperator::LogOr => i32::from(lhs != 0 || rhs != 0),
        BinaryOperator::Eq => i32::from(lhs == rhs),
        BinaryOperator::NotEq => i32::from(lhs != rhs),
        BinaryOperator::OrdLess => i32::from(lhs < rhs),
        BinaryOperator::OrdLessEq => i32::from(lhs <= rhs),
        BinaryOperator::OrdGreater => i32::from(lhs > rhs),
        BinaryOperator::OrdGreaterEq => i32::from(lhs >= rhs),
        BinaryOperator::Assign
        | BinaryOperator::AssignAdd
        | BinaryOperator::AssignSubtract
        | BinaryOperator::AssignMultiply
        | BinaryOperator::AssignDivide
        | BinaryOperator::AssignModulo
        | BinaryOperator::AssignBitAnd
        | BinaryOperator::AssignBitOr
        | BinaryOperator::AssignBitXor
        | BinaryOperator::AssignShiftLeft
        | BinaryOperator::AssignShiftRight
        | BinaryOperator::Conditional => {
            panic!("{binop:?} expressions should not be folded at compile-time")
        }
    }
}
