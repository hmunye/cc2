//! Syntax Analysis
//!
//! Compiler pass that parses a stream of tokens into an abstract syntax tree
//! (_AST_).

use std::{fmt, process};

use crate::compiler::lexer::{Lexer, OperatorKind, TokenType};
use crate::{Context, fmt_err, fmt_token_err, report_err};

type Ident = String;

/// Abstract Syntax Tree (_AST_).
#[derive(Debug)]
pub enum AST {
    /// Function that represent the structure of the program.
    Program(Function),
}

impl fmt::Display for AST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AST::Program(func) => {
                writeln!(f, "AST Program\n{:4}{func}", "")
            }
        }
    }
}

/// _AST_ function definition.
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Function {
    pub ty: Type,
    pub ident: Ident,
    pub body: Statement,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Function({:?}, {:?})", self.ty, self.ident)?;
        write!(f, "{:8}{}", "", self.body)
    }
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

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Return(expr) => write!(f, "Return({expr})"),
        }
    }
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
    },
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::ConstantInt(v) => write!(f, "Int({v})"),
            Expression::Unary { op, expr } => write!(f, "Unary({op:?}, {expr})"),
        }
    }
}

/// _AST_ unary operators.
#[derive(Debug, Copy, Clone)]
pub enum UnaryOperator {
    /// `~` unary operator.
    Complement,
    /// `-` unary operator.
    Negate,
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
    let ret_val = parse_expression(ctx, lexer)?;
    expect_token(ctx, lexer, TokenType::Semicolon)?;

    Ok(Statement::Return(ret_val))
}

/// Parse an _AST_ function/object type from the provided `Lexer`.
fn parse_type(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Type, String> {
    if let Some(token) = lexer.next_token() {
        let tok_str = format!("{token:?}");

        match tok_str.as_str() {
            "int" => Ok(Type::Int),
            "void" => Ok(Type::Void),
            t => Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                t,
                t.len() - 1,
                token.loc.line_content,
                "unknown type name '{t}'",
            )),
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
            t => {
                let tok_str = format!("{t:?}");

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

/// Parse an _AST_ expression from the provided `Lexer`.
fn parse_expression(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Expression, String> {
    if let Some(token) = lexer.next_token() {
        match token.ty {
            TokenType::ConstantInt(v) => Ok(Expression::ConstantInt(v)),
            TokenType::Operator(OperatorKind::Complement) => {
                // Recursively parse the expression on which the operator is
                // being applied to.
                let expr = parse_expression(ctx, lexer)?;
                Ok(Expression::Unary {
                    op: UnaryOperator::Complement,
                    expr: Box::new(expr),
                })
            }
            TokenType::Operator(OperatorKind::Negate) => {
                // Recursively parse the expression on which the operator is
                // being applied to.
                let expr = parse_expression(ctx, lexer)?;
                Ok(Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: Box::new(expr),
                })
            }
            TokenType::ParenOpen => {
                // Recursively parse the expression within parenthesis.
                let expr = parse_expression(ctx, lexer)?;
                expect_token(ctx, lexer, TokenType::ParenClose)?;
                Ok(expr)
            }
            t => {
                let tok_str = format!("{t:?}");

                Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    token.loc.line_content,
                    "'{tok_str}' undeclared",
                ))
            }
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '<expr>' at end of input",
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
            let _ = lexer.next_token();
            Ok(())
        } else {
            let tok_str = format!("{token:?}");

            Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len(),
                token.loc.line_content,
                "expected '{expected}', but found '{tok_str}'",
            ))
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '{}' at end of input",
            expected
        ))
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
}
