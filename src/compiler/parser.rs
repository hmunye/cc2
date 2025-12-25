//! Syntax Analysis.
//!
//! Converts a stream of tokens (produced by lexical analysis) into an
//! abstract syntax tree (AST).

use std::process;

use crate::compiler::lexer::{Lexer, TokenType};
use crate::{Context, fmt_err, fmt_err_ctx, report_err};

type Ident = String;

/// Abstract Syntax Tree (_AST_) node types.
#[derive(Debug)]
pub enum AST {
    /// Function that represent the structure of the program.
    Program(Function),
}

/// Represents a _function_ definition.
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Function {
    pub ty: Type,
    pub ident: Ident,
    pub body: Statement,
}

/// Represents the _type_ of a lexical token.
#[derive(Debug)]
pub enum Type {
    /// Integer type.
    Int,
    /// Absence of a type.
    Void,
}

/// Represents different variants of _statements_.
#[derive(Debug)]
pub enum Statement {
    /// Return statement that yields control back to the caller.
    Return(Expression),
}

/// Represents different variants of _expressions_.
#[derive(Debug)]
pub enum Expression {
    /// Constant _int_ value (32-bit).
    ConstantInt(i32),
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
            panic!("tokens remaining after parsing");
        }

        report_err!(ctx.in_path.display(), "tokens remaining after parsing");
        process::exit(1);
    }

    AST::Program(func)
}

/// Parses a _function definition_ from the provided `Lexer`.
fn parse_function(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Function, String> {
    let ty = parse_type(ctx, lexer)?;
    let ident = parse_ident(ctx, lexer)?;

    expect_token(ctx, lexer, TokenType::ParenOpen)?;
    expect_token(ctx, lexer, TokenType::Keyword("void".into()))?;
    expect_token(ctx, lexer, TokenType::ParenClose)?;
    expect_token(ctx, lexer, TokenType::BraceOpen)?;

    let body = parse_statement(ctx, lexer)?;

    expect_token(ctx, lexer, TokenType::BraceClose)?;

    Ok(Function { ty, ident, body })
}

/// Parse a _statement_ from the provided `Lexer`.
fn parse_statement(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Statement, String> {
    expect_token(ctx, lexer, TokenType::Keyword("return".into()))?;
    let ret_val = parse_expression(ctx, lexer)?;
    expect_token(ctx, lexer, TokenType::Semicolon)?;

    Ok(Statement::Return(ret_val))
}

/// Parse a function or object _type_ from the provided `Lexer`.
fn parse_type(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Type, String> {
    if let Some(token) = lexer.next_token() {
        match token.ty {
            TokenType::Keyword(ref s) if s == "int" => Ok(Type::Int),
            TokenType::Keyword(ref s) if s == "void" => Ok(Type::Void),
            _ => Err(fmt_err_ctx!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                "expected '<type>', but found '{:?}'",
                token.ty
            )),
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "unexpected end of input, expected '<type>'"
        ))
    }
}

/// Parse an _identifier_ from the provided `Lexer`.
fn parse_ident(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Ident, String> {
    if let Some(token) = lexer.next_token() {
        match token.ty {
            TokenType::Ident(ref s) => Ok(s.clone()),
            _ => Err(fmt_err_ctx!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                "expected '<ident>', but found '{:?}'",
                token.ty
            )),
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "unexpected end of input, expected '<ident>'"
        ))
    }
}

/// Parse an _expression_ from the provided `Lexer`.
fn parse_expression(ctx: &Context<'_>, lexer: &mut Lexer<'_>) -> Result<Expression, String> {
    if let Some(token) = lexer.next_token() {
        match token.ty {
            TokenType::ConstantInt(v) => Ok(Expression::ConstantInt(v)),
            _ => Err(fmt_err_ctx!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                "expected '<expr>', but found '{:?}'",
                token.ty
            )),
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "unexpected end of input, expected '<expr>'"
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
            Err(fmt_err_ctx!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                "expected '{:?}', but found '{:?}'",
                expected,
                token.ty
            ))
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "unexpected end of input, expected '{:?}'",
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
    #[should_panic]
    fn parser_invalid_no_expr() {
        let source = b"int main(void) {\n    return";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_extra_ident() {
        // Single identifier outside of a declaration is not a valid top-level
        // construct.
        let source = b"int main(void) {\n    return 2;\n}\nfoo";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_function_ident() {
        // Functions must have an identifier as a name.
        let source = b"int 3 (void) {\n    return 0;\n}";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_case_sensitive_keyword() {
        let source = b"int main(void) {\n    RETURN 0;\n}";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_missing_function_type() {
        // Because of backwards compatibility, `GCC` and `Clang` will compile
        // this program with a warning.
        let source = b"main(void) {\n    return 0;\n}";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_keyword() {
        let source = b"int main(void) {\n    returns 0;\n}";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_missing_semicolon() {
        let source = b"int main(void) {\n    return 0\n}";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_expression() {
        let source = b"int main(void) {\n    return int;\n}";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_keyword_space() {
        let source = b"int main(void) {\n    retur n 0;\n}";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_parens() {
        let source = b"int main)void( {\n    return 0;\n}";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_unclosed_brace() {
        let source = b"int main(void) {\n    return 0;";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }

    #[test]
    #[should_panic]
    fn parser_invalid_unclosed_paren() {
        let source = b"int main(void {\n    return 0;\n}";
        let mut lexer = crate::compiler::Lexer::new(source);

        let ctx = test_ctx();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            lexer.lex(&ctx);
        }));

        assert!(result.is_ok(), "lexer should not panic in test");

        parse_program(&ctx, &mut lexer);
    }
}
