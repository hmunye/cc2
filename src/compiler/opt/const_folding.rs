//! Constant Folding
//!
//! Transforms _AST_ expressions with constant operands into single expressions,
//! simplifying computations at compile-time.

use crate::compiler::parser::ast::{
    AST, Analyzed, BinaryOperator, Block, BlockItem, Declaration, Expression, Function, Labeled,
    Statement, UnaryOperator,
};
use crate::compiler::parser::types::c_int;

/// Attempts to fold each expressions within the provided analyzed _AST_.
pub fn fold_constants(ast: &mut AST<'_, Analyzed>) {
    fn fold_declaration(decl: &mut Declaration<'_>) {
        match decl {
            Declaration::Var { init, .. } => {
                if let Some(init) = init
                    && let Some(folded) = try_fold(init)
                {
                    *init = Expression::IntConstant(folded);
                }
            }
            Declaration::Func(func) => fold_function(func),
        }
    }

    fn fold_function(func: &mut Function<'_>) {
        let Function { body, .. } = func;

        if let Some(body) = body {
            fold_block(body);
        }
    }

    fn fold_block(block: &mut Block<'_>) {
        for block_item in &mut block.0 {
            match block_item {
                BlockItem::Stmt(stmt) => fold_statement(stmt),
                BlockItem::Decl(decl) => fold_declaration(decl),
            }
        }
    }

    fn fold_statement(stmt: &mut Statement<'_>) {
        match stmt {
            Statement::Return(expr) | Statement::Expression(expr) => {
                if let Some(folded) = try_fold(expr) {
                    *expr = Expression::IntConstant(folded);
                }
            }
            Statement::LabeledStatement(labeled) => {
                if let Labeled::Case { expr, .. } = labeled
                    && let Some(folded) = try_fold(expr)
                {
                    *expr = Expression::IntConstant(folded);
                }
            }
            Statement::Compound(block) => fold_block(block),
            Statement::If { cond, .. }
            | Statement::While { cond, .. }
            | Statement::Do { cond, .. }
            | Statement::Switch { cond, .. } => {
                if let Some(folded) = try_fold(cond) {
                    *cond = Expression::IntConstant(folded);
                }
            }
            Statement::For {
                opt_cond, opt_post, ..
            } => {
                if let Some(cond) = opt_cond
                    && let Some(folded) = try_fold(cond)
                {
                    *opt_cond = Some(Expression::IntConstant(folded));
                }

                if let Some(post) = opt_post
                    && let Some(folded) = try_fold(post)
                {
                    *opt_post = Some(Expression::IntConstant(folded));
                }
            }
            Statement::Goto { .. }
            | Statement::Break { .. }
            | Statement::Continue { .. }
            | Statement::Empty => {}
        }
    }

    for decl in &mut ast.program {
        fold_declaration(decl);
    }
}

/// Attempts to fold and return the provided expression into a single constant
/// value, or `None` if the expression cannot be evaluated to a constant.
#[must_use]
pub fn try_fold(expr: &Expression<'_>) -> Option<c_int> {
    match expr {
        Expression::Var { .. } | Expression::FuncCall { .. } | Expression::Assignment { .. } => {
            None
        }
        Expression::IntConstant(i) => Some(*i),
        Expression::Unary { op, expr, .. } => {
            if matches!(op, UnaryOperator::Increment | UnaryOperator::Decrement) {
                return None;
            }

            let expr = try_fold(expr)?;

            Some(eval_unary(*op, expr))
        }
        Expression::Binary { op, lhs, rhs, .. } => {
            if matches!(
                op,
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
                    | BinaryOperator::Conditional
            ) {
                return None;
            }

            let rhs = try_fold(rhs)?;

            match op {
                BinaryOperator::Divide | BinaryOperator::Modulo => {
                    if rhs == 0 {
                        // NOTE: Should report warning diagnostic.
                        return None;
                    }
                }
                BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
                    if rhs >= 32 {
                        // NOTE: Should report warning diagnostic.
                        return None;
                    }
                }
                _ => {}
            }

            let lhs = try_fold(lhs)?;

            Some(eval_binary(*op, lhs, rhs))
        }
        Expression::Conditional {
            cond,
            second,
            third,
        } => {
            let cond = try_fold(cond)?;

            if cond != 0 {
                try_fold(second)
            } else {
                try_fold(third)
            }
        }
    }
}

/// Evaluates a unary operator on a constant operand.
#[inline]
#[must_use]
fn eval_unary(unop: UnaryOperator, val: c_int) -> c_int {
    match unop {
        UnaryOperator::Complement => !val,
        UnaryOperator::Negate => -val,
        UnaryOperator::Not => i32::from(val == 0),
        UnaryOperator::Increment | UnaryOperator::Decrement => {
            unreachable!(
                "{:?} expressions should never be evaluated at compile-time",
                unop
            )
        }
    }
}

/// Evaluates a binary operator on two constants operands.
///
/// # Panics
///
/// Panics if `rhs` is zero in division or modulo, or if the shift factor
/// (left or right) is 32 or greater.
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
        | BinaryOperator::Conditional => unreachable!(
            "{:?} expressions should never be evaluated at compile-time",
            binop
        ),
    }
}
