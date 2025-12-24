//! Syntax Analysis.
//!
//! Converts a stream of tokens (produced by lexical analysis) into an
//! abstract syntax tree (AST).

use std::convert::AsRef;
use std::path::Path;

use crate::compiler::lexer::{Lexer, TokenType};

/// Abstract Syntax Tree (_AST_) node types.
#[derive(Debug)]
pub enum AST {
    /// Function that represent the structure of the program.
    Program(Function),
}

/// Represents a _function_ definition.
#[derive(Debug)]
pub struct Function {
    ty: Type,
    ident: String,
    body: Statement,
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

/// Parses an abstract syntax tree (_AST_) from the provided `Lexer`.
pub fn parse_program(file_name: impl AsRef<Path>, lexer: &mut Lexer) -> Result<AST, String> {
    let function = parse_function(&file_name, lexer)?;

    if !lexer.is_empty() {
        return Err(format!(
            "\x1b[1;1m{}:\x1b[0m \x1b[1;31merror:\x1b[0m tokens remaining after parsing",
            file_name.as_ref().display()
        ));
    }

    Ok(AST::Program(function))
}

/// Parses a _function definition_ from the provided `Lexer`.
fn parse_function(file_name: impl AsRef<Path>, lexer: &mut Lexer) -> Result<Function, String> {
    let ty = parse_type(&file_name, lexer)?;
    let ident = parse_ident(&file_name, lexer)?;

    expect_token(&file_name, lexer, TokenType::ParenOpen)?;
    expect_token(&file_name, lexer, TokenType::Keyword("void".into()))?;
    expect_token(&file_name, lexer, TokenType::ParenClose)?;
    expect_token(&file_name, lexer, TokenType::BraceOpen)?;

    let body = parse_statement(&file_name, lexer)?;

    expect_token(&file_name, lexer, TokenType::BraceClose)?;

    Ok(Function { ty, ident, body })
}

/// Parse a function or object _type_ from the provided `Lexer`.
fn parse_type(file_name: impl AsRef<Path>, lexer: &mut Lexer) -> Result<Type, String> {
    if let Some(token) = lexer.consume_next() {
        match token.ty {
            TokenType::Keyword(ref s) if s == "int" => Ok(Type::Int),
            TokenType::Keyword(ref s) if s == "void" => Ok(Type::Void),
            _ => Err(format!(
                "\x1b[1;1m{}:{}:{}\x1b[0m \x1b[1;31merror:\x1b[0m expected '<type>', but found '{:?}'",
                file_name.as_ref().display(),
                token.line,
                token.col,
                token.ty
            )),
        }
    } else {
        Err(format!(
            "\x1b[1;1m{}:\x1b[0m \x1b[1;31merror:\x1b[0m unexpected end of input, expected '<type>'",
            file_name.as_ref().display()
        ))
    }
}

/// Parse an _identifier_ from the provided `Lexer`.
fn parse_ident(file_name: impl AsRef<Path>, lexer: &mut Lexer) -> Result<String, String> {
    if let Some(token) = lexer.consume_next() {
        match token.ty {
            TokenType::Ident(ref s) => Ok(s.clone()),
            _ => Err(format!(
                "\x1b[1;1m{}:{}:{}\x1b[0m \x1b[1;31merror:\x1b[0m expected '<identifier>', but found '{:?}'",
                file_name.as_ref().display(),
                token.line,
                token.col,
                token.ty
            )),
        }
    } else {
        Err(format!(
            "\x1b[1;1m{}:\x1b[0m \x1b[1;31merror:\x1b[0m unexpected end of input, expected '<identifier>'",
            file_name.as_ref().display()
        ))
    }
}

/// Parse a _statement_ from the provided `Lexer`.
fn parse_statement(file_name: impl AsRef<Path>, lexer: &mut Lexer) -> Result<Statement, String> {
    expect_token(&file_name, lexer, TokenType::Keyword("return".into()))?;

    let ret_val = parse_expression(&file_name, lexer)?;

    expect_token(&file_name, lexer, TokenType::Semi)?;

    Ok(Statement::Return(ret_val))
}

/// Parse an _expression_ from the provided `Lexer`.
fn parse_expression(file_name: impl AsRef<Path>, lexer: &mut Lexer) -> Result<Expression, String> {
    if let Some(token) = lexer.consume_next() {
        match token.ty {
            TokenType::ConstantInt(v) => Ok(Expression::ConstantInt(v)),
            _ => Err(format!(
                "\x1b[1;1m{}:{}:{}\x1b[0m \x1b[1;31merror:\x1b[0m expected '<int>', but found '{:?}'",
                file_name.as_ref().display(),
                token.line,
                token.col,
                token.ty
            )),
        }
    } else {
        Err(format!(
            "\x1b[1;1m{}:\x1b[0m \x1b[1;31merror:\x1b[0m unexpected end of input, expected '<expr>'",
            file_name.as_ref().display()
        ))
    }
}

/// Consumes the current token from the `Lexer` if it matches the expected token
/// type.
fn expect_token(
    file_name: impl AsRef<Path>,
    lexer: &mut Lexer,
    expected: TokenType,
) -> Result<(), String> {
    if let Some(token) = lexer.peek_next() {
        if token.ty == expected {
            // Consume the token when it is expected.
            let _ = lexer.consume_next();
            Ok(())
        } else {
            Err(format!(
                "\x1b[1;1m{}:{}:{}\x1b[0m \x1b[1;31merror:\x1b[0m expected '{:?}', but found '{:?}'",
                file_name.as_ref().display(),
                token.line,
                token.col,
                expected,
                token.ty
            ))
        }
    } else {
        Err(format!(
            "\x1b[1;1m{}:\x1b[0m \x1b[1;31merror:\x1b[0m unexpected end of input, expected '{:?}'",
            file_name.as_ref().display(),
            expected
        ))
    }
}

// Reference: https://github.com/nlsandler/writing-a-c-compiler-tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_invalid_no_expr() {
        let source = b"int main(void) {\n    return";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_extra_ident() {
        // Single identifier outside of a declaration is not a valid top-level
        // construct.
        let source = b"int main(void) {\n    return 2;\n}\nfoo";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_function_ident() {
        // Functions must have an identifier as a name.
        let source = b"int 3 (void) {\n    return 0;\n}";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_case_sensitive_keyword() {
        let source = b"int main(void) {\n    RETURN 0;\n}";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_missing_function_type() {
        // Because of backwards compatibility, `GCC` and `Clang` will compile
        // this program with a warning.
        let source = b"main(void) {\n    return 0;\n}";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_keyword() {
        let source = b"int main(void) {\n    returns 0;\n}";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_missing_semicolon() {
        let source = b"int main(void) {\n    return 0\n}";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_expression() {
        let source = b"int main(void) {\n    return int;\n}";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_keyword_space() {
        let source = b"int main(void) {\n    retur n 0;\n}";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_parens() {
        let source = b"int main)void( {\n    return 0;\n}";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_unclosed_brace() {
        let source = b"int main(void) {\n    return 0;";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }

    #[test]
    fn parser_invalid_unclosed_paren() {
        let source = b"int main(void {\n    return 0;\n}";
        let mut lexer = crate::compiler::Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok());
        assert!(parse_program("test.c", &mut lexer).is_err());
    }
}
