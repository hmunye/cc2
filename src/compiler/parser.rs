//! Syntax Analysis
//!
//! Compiler pass that parses a stream of tokens into an abstract syntax tree
//! (_AST_).

use std::{fmt, process};

use crate::compiler::lexer::{Lexer, OperatorKind, TokenType};
use crate::{Context, fmt_err, fmt_token_err, report_err};

type Ident = String;

/// Abstract Syntax Tree (_AST_).
pub enum AST {
    /// Function that represent the structure of the program.
    Program(Function),
}

impl fmt::Debug for AST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(func) => f.debug_tuple("AST Program").field(func).finish(),
        }
    }
}

/// _AST_ function definition.
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Function {
    // NOTE: Currently unused.
    pub ty: Type,
    pub ident: Ident,
    pub body: Statement,
}

/// _AST_ function/object types.
#[derive(Debug)]
pub enum Type {
    /// Integer type.
    Int,
    /// Absence of a type.
    Void,
}

/// _AST_ statements.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum Statement {
    Return(Expression),
}

/// _AST_ expressions.
#[derive(Debug)]
pub enum Expression {
    /// Constant int value (32-bit).
    ConstantInt(i32),
    /// Unary operator applied to an expression.
    #[allow(missing_docs)]
    Unary {
        op: UnaryOperator,
        expr: Box<Expression>,
        sign: Signedness,
    },
    /// Binary operator applied to two expressions.
    #[allow(missing_docs)]
    Binary {
        op: BinaryOperator,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
        sign: Signedness,
    },
}

/// _AST_ unary operators.
#[derive(Debug, Copy, Clone)]
pub enum UnaryOperator {
    /// `~` unary operator.
    Complement,
    /// `-` unary operator.
    Negate,
}

/// _AST_ binary operators.
#[derive(Debug, Copy, Clone)]
pub enum BinaryOperator {
    /// `+` binary operator.
    Add,
    /// `-` binary operator.
    Subtract,
    /// `*` binary operator.
    Multiply,
    /// `/` binary operator.
    Divide,
    /// `%` binary operator.
    Remainder,
    /// `&` binary operator.
    BitAnd,
    /// `|` binary operator.
    BitOr,
    /// `^` binary operator.
    BitXor,
    /// `<<` binary operator.
    ShiftLeft,
    /// `>>` binary operator.
    ShiftRight,
}

/// Currently used to determine the signedness of an expression, particularly
/// for logical or arithmetic right shifts.
///
/// NOTE: Any unary negation or binary subtraction will treat an expression as
/// signed.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)]
pub enum Signedness {
    Signed,
    Unsigned,
}

impl BinaryOperator {
    /// Returns the _precedence_ level of the binary operator (higher numbers
    /// indicate tighter binding).
    ///
    /// Derived from the structure of the _C17_ expression grammar.
    pub fn precedence(&self) -> u8 {
        match self {
            // _C17_ 6.5.12 (inclusive-OR-expression)
            BinaryOperator::BitOr => 7,
            // _C17_ 6.5.11 (exclusive-OR-expression)
            BinaryOperator::BitXor => 8,
            // _C17_ 6.5.10 (AND-expression)
            BinaryOperator::BitAnd => 9,
            // _C17_ 6.5.9 (equality-expression)
            // => 10
            // _C17_ 6.5.8 (relational-expression)
            // => 11
            // _C17_ 6.5.7 (shift-expression)
            BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => 12,
            // _C17_ 6.5.6 (additive-expression)
            BinaryOperator::Add | BinaryOperator::Subtract => 13,
            // _C17_ 6.5.5 (multiplicative-expression)
            BinaryOperator::Multiply | BinaryOperator::Divide | BinaryOperator::Remainder => 14,
        }
    }
}

/// Parses an abstract syntax tree (_AST_) from the provided `Lexer`. [Exits] on
/// error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn parse_program(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> AST {
    let func = parse_function(ctx, lexer).unwrap_or_else(|err| {
        if cfg!(test) {
            panic!("{err}");
        }

        eprintln!("{err}");
        process::exit(1);
    });

    if !lexer.is_empty() {
        if cfg!(test) {
            panic!("tokens remaining after parsing: {}", lexer.len());
        }

        report_err!(
            ctx.in_path.display(),
            "tokens remaining after parsing: {}",
            lexer.len()
        );
        process::exit(1);
    }

    AST::Program(func)
}

/// Parses an _AST_ function definition from the provided `Lexer`.
fn parse_function(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Function, String> {
    let ty = parse_type(ctx, lexer)?;
    let ident = parse_ident(ctx, lexer)?;

    expect_token(ctx, lexer, TokenType::ParenOpen)?;

    // NOTE: Currently do not process function parameters.
    expect_token(ctx, lexer, TokenType::Keyword("void".into()))?;

    expect_token(ctx, lexer, TokenType::ParenClose)?;
    expect_token(ctx, lexer, TokenType::BraceOpen)?;

    let body = parse_statement(ctx, lexer)?;

    expect_token(ctx, lexer, TokenType::BraceClose)?;

    Ok(Function { ty, ident, body })
}

/// Parse an _AST_ statement from the provided `Lexer`.
fn parse_statement(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Statement, String> {
    expect_token(ctx, lexer, TokenType::Keyword("return".into()))?;
    let ret_val = parse_expression(ctx, lexer, 0)?;
    expect_token(ctx, lexer, TokenType::Semicolon)?;

    Ok(Statement::Return(ret_val))
}

/// Parse an _AST_ function/object type from the provided `Lexer`.
fn parse_type(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Type, String> {
    if let Some(token) = lexer.next_token() {
        match token.ty {
            TokenType::Keyword(ref s) if s == "int" => Ok(Type::Int),
            TokenType::Keyword(ref s) if s == "void" => Ok(Type::Void),
            _ => {
                let tok_str = format!("{token:?}");

                Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    token.loc.line_content,
                    "unknown type name '{tok_str}'",
                ))
            }
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '<type>' at end of input",
        ))
    }
}

/// Parse an _AST_ identifier from the provided `Lexer`.
fn parse_ident(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Ident, String> {
    if let Some(token) = lexer.next_token() {
        match token.ty {
            TokenType::Ident(ref s) => Ok(s.clone()),
            tok => {
                let tok_str = format!("{tok:?}");

                Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    token.loc.line_content,
                    "expected identifier",
                ))
            }
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '<ident>' at end of input",
        ))
    }
}

/// Parse an _AST_ expression from the provided `Lexer` using precedence
/// climbing.
fn parse_expression(
    ctx: &Context<'_>,
    lexer: &mut Lexer<'_>,
    min_precedence: u8,
) -> Result<Expression, String> {
    let mut lhs = parse_factor(ctx, lexer)?;
    let mut next = lexer.peek();

    while let Some(token) = next {
        let Some(binop) = is_binop(&token.ty) else {
            break;
        };

        let sign = if let BinaryOperator::Subtract = binop {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        };

        // Higher-precedence operators will be evaluated first, making
        // the resulting AST left-associative.
        if binop.precedence() < min_precedence {
            break;
        }

        // Consume the matched binary operator.
        let _ = lexer.next_token();

        let rhs = parse_expression(ctx, lexer, binop.precedence() + 1)?;

        lhs = Expression::Binary {
            op: binop,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            sign,
        };

        next = lexer.peek();
    }

    Ok(lhs)
}

/// Parse an _AST_ expression or sub-expression (factor) from the provided
/// `Lexer`.
fn parse_factor(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Expression, String> {
    if let Some(token) = lexer.next_token() {
        match token.ty {
            TokenType::ConstantInt(v) => Ok(Expression::ConstantInt(v)),
            TokenType::Operator(OperatorKind::BitNot | OperatorKind::Minus) => {
                let unop = match token.ty {
                    TokenType::Operator(OperatorKind::BitNot) => UnaryOperator::Complement,
                    TokenType::Operator(OperatorKind::Minus) => UnaryOperator::Negate,
                    _ => unreachable!(),
                };

                let sign = if let UnaryOperator::Negate = unop {
                    Signedness::Signed
                } else {
                    Signedness::Unsigned
                };

                // Recursively parse the factor on which the unary operator is
                // being applied to.
                let inner_fct = parse_factor(ctx, lexer)?;
                Ok(Expression::Unary {
                    op: unop,
                    expr: Box::new(inner_fct),
                    sign,
                })
            }
            TokenType::ParenOpen => {
                // Recursively parse the expression within parenthesis.
                let inner_expr = parse_expression(ctx, lexer, 0)?;
                expect_token(ctx, lexer, TokenType::ParenClose)?;
                Ok(inner_expr)
            }
            tok => {
                let tok_str = format!("{tok:?}");

                let err_msg = if let TokenType::ParenClose = tok {
                    format!("expected expression before '{tok_str}' token")
                } else {
                    format!("'{tok_str}' undeclared")
                };

                Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    token.loc.line_content,
                    "{err_msg}",
                ))
            }
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '<factor>' at end of input",
        ))
    }
}

/// Consumes the current token from the `Lexer` if it matches the expected token
/// type.
fn expect_token(
    ctx: &Context<'_>,
    lexer: &mut Lexer<'_>,
    expected: TokenType,
) -> Result<(), String> {
    if let Some(token) = lexer.peek() {
        if token.ty == expected {
            // Consume the token.
            let _ = lexer.next_token();
            Ok(())
        } else {
            let tok_str = format!("{token:?}");

            Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                token.loc.line_content,
                "expected '{expected:?}', but found '{tok_str}'",
            ))
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '{:?}' at end of input",
            expected
        ))
    }
}

/// Returns the conversion of `TokenType` to BinaryOperator, or `None` if the
/// token type is not a binary operator.
fn is_binop(ty: &TokenType) -> Option<BinaryOperator> {
    match ty {
        TokenType::Operator(OperatorKind::Plus) => Some(BinaryOperator::Add),
        TokenType::Operator(OperatorKind::Minus) => Some(BinaryOperator::Subtract),
        TokenType::Operator(OperatorKind::Asterisk) => Some(BinaryOperator::Multiply),
        TokenType::Operator(OperatorKind::Division) => Some(BinaryOperator::Divide),
        TokenType::Operator(OperatorKind::Modulo) => Some(BinaryOperator::Remainder),
        TokenType::Operator(OperatorKind::Ampersand) => Some(BinaryOperator::BitAnd),
        TokenType::Operator(OperatorKind::BitOr) => Some(BinaryOperator::BitOr),
        TokenType::Operator(OperatorKind::BitXor) => Some(BinaryOperator::BitXor),
        TokenType::Operator(OperatorKind::ShiftLeft) => Some(BinaryOperator::ShiftLeft),
        TokenType::Operator(OperatorKind::ShiftRight) => Some(BinaryOperator::ShiftRight),
        _ => None,
    }
}

// Reference: https://github.com/nlsandler/writing-a-c-compiler-tests
#[cfg(test)]
mod tests {
    use std::panic::{self, AssertUnwindSafe};
    use std::path::Path;

    use super::*;
    use crate::Context;

    fn test_ctx() -> Context<'static> {
        Context {
            program: "cc2",
            in_path: Path::new("test.c"),
            out_path: Path::new("test.s"),
        }
    }

    #[test]
    fn parser_valid_bitwise_complement() {
        let source = b"int main(void) {\n    return ~12;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bitwise_complement_i32_min() {
        // Take the bitwise complement of the largest negative integer
        // that can be safely negated in a 32-bit signed integer (-2147483647).
        let source = b"int main(void) {\n    return ~-2147483647;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bitwise_complement_zero() {
        let source = b"int main(void) {\n    return ~0;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_negation() {
        let source = b"int main(void) {\n    return -5;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_negation_zero() {
        let source = b"int main(void) {\n    return -0;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_negation_i32_max() {
        let source = b"int main(void) {\n    return -2147483647;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_nested_unary_ops() {
        let source = b"int main(void) {\n    return ~-3;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_parenthesize_constant() {
        let source = b"int main(void) {\n    return (2);\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_redundant_parens() {
        let source = b"int main(void) {\n    return -((((10))));\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_add() {
        let source = b"int main(void) {\n    return 1 + 2;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_div() {
        let source = b"int main(void) {\n    return 1 / 2;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_div_neg() {
        let source = b"int main(void) {\n    return (-12) / 2;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_mod() {
        let source = b"int main(void) {\n    return 12 % 2;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_mul() {
        let source = b"int main(void) {\n    return 12 * 2;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_parens() {
        let source = b"int main(void) {\n    return 2 * (3 + 4);\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_sub() {
        let source = b"int main(void) {\n    return 2 - 1;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_unary_add() {
        let source = b"int main(void) {\n    return ~2 + 1;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_unary_parens() {
        let source = b"int main(void) {\n    return ~(1 + 1);\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_associativity() {
        let source = b"int main(void) {\n    return (3 / 2 * 4) + (5 - 4 + 3);\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_associativity_and_precedence() {
        let source = b"int main(void) {\n    return 5 * 4 / 2 - 3 % (2 + 1);\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_bit_and() {
        let source = b"int main(void) {\n    return 3 & 5;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_bit_or() {
        let source = b"int main(void) {\n    return 1 | 2;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_bit_precedence() {
        let source = b"int main(void) {\n    return 80 >> 2 | 1 ^ 5 & 7 << 1;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    fn parser_valid_bin_bit_shift() {
        let source = b"int main(void) {\n    return 33 << 4 >> 2;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_no_expr() {
        let source = b"int main(void) {\n    return";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_extra_ident() {
        // Single identifier outside of a declaration is not a valid top-level
        // construct.
        let source = b"int main(void) {\n    return 2;\n}\nfoo";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_function_ident() {
        // Functions must have an identifier as a name.
        let source = b"int 3 (void) {\n    return 0;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_case_sensitive_keyword() {
        let source = b"int main(void) {\n    RETURN 0;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_missing_function_type() {
        // Because of backwards compatibility, `GCC` and `Clang` will compile
        // this program with a warning.
        let source = b"main(void) {\n    return 0;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_keyword() {
        let source = b"int main(void) {\n    returns 0;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_missing_semicolon() {
        let source = b"int main(void) {\n    return 0\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_expression() {
        let source = b"int main(void) {\n    return int;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_keyword_space() {
        let source = b"int main(void) {\n    retur n 0;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_parens_fn() {
        let source = b"int main)void( {\n    return 0;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_unclosed_brace_fn() {
        let source = b"int main(void) {\n    return 0;";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_unclosed_paren_fn() {
        let source = b"int main(void {\n    return 0;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_extra_paren() {
        let source = b"int main(void)\n{\n    return (3));\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_missing_constant() {
        let source = b"int main(void) {\n    return ~;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_nested_missing_constant() {
        let source = b"int main(void)\n{\n    return -~;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_parenthesize_operand() {
        let source = b"int main(void) {\n   return (-)3;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_unclose_paren_expr() {
        let source = b"int main(void) {\n   return (1;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_negation_postfix() {
        let source = b"int main(void) {\n   return 4-;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_semicolon() {
        let source = b"int main(void) {\n   return 2*2\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_rhs() {
        let source = b"int main(void) {\n   return 1 / ;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_open_paren() {
        let source = b"int main(void) {\n   return 1+2);\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_closing_paren() {
        let source = b"int main(void) {\n   return (1+2;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_lhs() {
        let source = b"int main(void) {\n   return / 3;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_misplaced_semicolon() {
        let source = b"int main(void) {\n   return 1 + (2;)\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_op() {
        let source = b"int main(void) {\n   return 2 (- 3);\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_double_op() {
        let source = b"int main(void) {\n   return 2 * / 2;\n}";
        let mut lexer = crate::compiler::lexer::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in parser test");

        parse_program(&ctx, &mut lexer);
    }
}
