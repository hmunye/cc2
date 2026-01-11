//! Syntax Analysis
//!
//! Compiler pass that parses a stream of tokens into an abstract syntax tree
//! (_AST_).

use std::{fmt, process};

use crate::compiler::lexer::{OperatorKind, Token, TokenType};
use crate::{Context, fmt_err, fmt_token_err, report_err};

type Ident = String;
type TokenResult = Result<Token, String>;

/// Helper for resolving variable identifiers.
struct Resolver {
    map: std::collections::HashMap<Ident, Ident>,
    var_count: usize,
}

impl Resolver {
    /// Allocates and returns a fresh temporary variable identifier, prepending
    /// the provided prefix.
    fn new_tmp(&mut self, prefix: &str) -> Ident {
        // The `.` in temporary identifiers guarantees they won’t conflict
        // with user-defined identifiers, since the _C_ standard forbids `.` in
        // identifiers.
        let ident = format!("{prefix}.{}", self.var_count);
        self.var_count += 1;
        ident
    }
}

/// Abstract Syntax Tree (_AST_).
pub enum AST {
    /// Function that represent the structure of the program.
    Program(Function),
}

impl AST {
    /// Ensures all variables are uniquely identified and checks for semantic
    /// errors.
    fn resolve_vars(&mut self, ctx: &Context<'_>, resolver: &mut Resolver) -> Result<(), String> {
        match self {
            AST::Program(func) => {
                for block in &mut func.body {
                    match block {
                        Block::Stmt(stmt) => AST::resolve_statement(stmt, ctx, resolver)?,
                        Block::Decl(decl) => AST::resolve_declaration(decl, ctx, resolver)?,
                    }
                }
            }
        }

        Ok(())
    }

    fn resolve_declaration(
        decl: &mut Declaration,
        ctx: &Context<'_>,
        resolver: &mut Resolver,
    ) -> Result<(), String> {
        if resolver.map.contains_key(&decl.ident) {
            let token = &decl.token;

            let tok_str = format!("{token:?}");
            let line_content = ctx.src_slice(token.loc.line_span.clone());

            return Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                line_content,
                "redeclaration of '{tok_str}'",
            ));
        }

        let resolved_ident = resolver.new_tmp(&decl.ident);
        resolver
            .map
            .insert(decl.ident.clone(), resolved_ident.clone());

        decl.ident = resolved_ident;

        if let Some(init) = &decl.init {
            decl.init = Some(AST::resolve_expression(init, ctx, resolver)?);
        }

        Ok(())
    }

    fn resolve_statement(
        stmt: &mut Statement,
        ctx: &Context<'_>,
        resolver: &mut Resolver,
    ) -> Result<(), String> {
        *stmt = match stmt {
            Statement::Return(expr) => {
                Statement::Return(AST::resolve_expression(expr, ctx, resolver)?)
            }
            Statement::Expression(expr) => {
                Statement::Expression(AST::resolve_expression(expr, ctx, resolver)?)
            }
            Statement::Null => Statement::Null,
        };

        Ok(())
    }

    fn resolve_expression(
        expr: &Expression,
        ctx: &Context<'_>,
        resolver: &mut Resolver,
    ) -> Result<Expression, String> {
        match expr {
            Expression::Assignment(lvalue, rvalue, token) => match **lvalue {
                Expression::Var(_) => {
                    let lvalue = AST::resolve_expression(lvalue, ctx, resolver)?;
                    let rvalue = AST::resolve_expression(rvalue, ctx, resolver)?;

                    Ok(Expression::Assignment(
                        Box::new(lvalue),
                        Box::new(rvalue),
                        token.clone(),
                    ))
                }
                _ => {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "lvalue required as left operand of assignment",
                    ))
                }
            },
            Expression::Var(v) => {
                if resolver.map.contains_key(&v.0) {
                    Ok(Expression::Var(v.clone()))
                } else {
                    let token = &v.1;
                    let tok_str = format!("{token:?}");

                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "'{tok_str}' undeclared",
                    ))
                }
            }
            Expression::Constant(i) => Ok(Expression::Constant(*i)),
            Expression::Unary { op, expr, sign } => {
                let expr = AST::resolve_expression(expr, ctx, resolver)?;

                Ok(Expression::Unary {
                    op: *op,
                    expr: Box::new(expr),
                    sign: *sign,
                })
            }
            Expression::Binary { op, lhs, rhs, sign } => {
                let lhs = AST::resolve_expression(lhs, ctx, resolver)?;
                let rhs = AST::resolve_expression(rhs, ctx, resolver)?;

                Ok(Expression::Binary {
                    op: *op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    sign: *sign,
                })
            }
        }
    }
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
pub struct Function {
    /// Function identifier.
    pub ident: Ident,
    /// Compound statement.
    pub body: Vec<Block>,
}

/// _AST_ block.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum Block {
    Stmt(Statement),
    Decl(Declaration),
}

/// _AST_ statements.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum Statement {
    Return(Expression),
    Expression(Expression),
    // Expression statement without an expression.
    Null,
}

/// _AST_ declarations.
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Declaration {
    pub ident: Ident,
    pub token: Token,
    // Optional initializer.
    pub init: Option<Expression>,
}

/// _AST_ expressions.
#[derive(Debug, Clone)]
pub enum Expression {
    /// Constant int value (32-bit).
    Constant(i32),
    /// Variable expression.
    Var((Ident, Token)),
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
    /// Assigns an `rvalue` to an `lvalue`.
    Assignment(Box<Expression>, Box<Expression>, Token),
}

/// _AST_ unary operators.
#[derive(Debug, Copy, Clone)]
pub enum UnaryOperator {
    /// `~` unary operator.
    Complement,
    /// `-` unary operator.
    Negate,
    /// `!` unary operator.
    Not,
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
    Modulo,
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
    /// `&&` binary operator.
    LogAnd,
    /// `||` binary operator.
    LogOr,
    /// `==` binary operator.
    Eq,
    /// `!=` binary operator.
    NotEq,
    /// `<` binary operator.
    OrdLess,
    /// `<=` binary operator.
    OrdLessEq,
    /// `>` binary operator.
    OrdGreater,
    /// `>=` binary operator.
    OrdGreaterEq,
    /// `=` binary operator.
    Assign,
}

impl BinaryOperator {
    /// Returns the _precedence_ level of the binary operator (higher numbers
    /// indicate tighter binding).
    ///
    /// Derived from the structure of the _C17_ expression grammar.
    pub fn precedence(&self) -> u8 {
        match self {
            // _C17_ 6.5.16 (assignment-expression)
            BinaryOperator::Assign => 3,
            // _C17_ 6.5.15 (conditional-expression)
            // => 4
            // _C17_ 6.5.14 (logical-OR-expression)
            BinaryOperator::LogOr => 5,
            // _C17_ 6.5.13 (logical-AND-expression)
            BinaryOperator::LogAnd => 6,
            // _C17_ 6.5.12 (inclusive-OR-expression)
            BinaryOperator::BitOr => 7,
            // _C17_ 6.5.11 (exclusive-OR-expression)
            BinaryOperator::BitXor => 8,
            // _C17_ 6.5.10 (AND-expression)
            BinaryOperator::BitAnd => 9,
            // _C17_ 6.5.9 (equality-expression)
            BinaryOperator::Eq | BinaryOperator::NotEq => 10,
            // _C17_ 6.5.8 (relational-expression)
            BinaryOperator::OrdLess
            | BinaryOperator::OrdLessEq
            | BinaryOperator::OrdGreater
            | BinaryOperator::OrdGreaterEq => 11,
            // _C17_ 6.5.7 (shift-expression)
            BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => 12,
            // _C17_ 6.5.6 (additive-expression)
            BinaryOperator::Add | BinaryOperator::Subtract => 13,
            // _C17_ 6.5.5 (multiplicative-expression)
            BinaryOperator::Multiply | BinaryOperator::Divide | BinaryOperator::Modulo => 14,
        }
    }
}

// NOTE: Temporary hack for arithmetic right shift.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)]
pub enum Signedness {
    Signed,
    Unsigned,
}

/// Parses an abstract syntax tree (_AST_) from the provided `Token` iterator.
/// [Exits] on error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn parse_program<I: Iterator<Item = TokenResult>>(
    ctx: &Context<'_>,
    mut iter: std::iter::Peekable<I>,
) -> AST {
    let func = parse_function(ctx, &mut iter).unwrap_or_else(|err| {
        if cfg!(test) {
            panic!("{err}");
        }

        eprintln!("{err}");
        process::exit(1);
    });

    if iter.peek().is_some() {
        if cfg!(test) {
            panic!("tokens remaining after parsing");
        }

        report_err!(ctx.in_path.display(), "tokens remaining after parsing");
        process::exit(1);
    }

    let mut ast = AST::Program(func);

    let mut resolver = Resolver {
        map: Default::default(),
        var_count: 0,
    };

    // Pass 1 - Variable resolution.
    ast.resolve_vars(ctx, &mut resolver).unwrap_or_else(|err| {
        if cfg!(test) {
            panic!("{err}");
        }

        eprintln!("{err}");
        process::exit(1);
    });

    ast
}

/// Parses an _AST_ function definition from the provided `Token` iterator.
fn parse_function<I: Iterator<Item = TokenResult>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Function, String> {
    // NOTE: Currently only support the `int` type.
    expect_token(ctx, iter, TokenType::Keyword("int".into()))?;
    let (ident, _) = parse_ident(ctx, iter)?;

    expect_token(ctx, iter, TokenType::ParenOpen)?;
    // NOTE: Currently not processing function parameters.
    expect_token(ctx, iter, TokenType::Keyword("void".into()))?;
    expect_token(ctx, iter, TokenType::ParenClose)?;

    expect_token(ctx, iter, TokenType::BraceOpen)?;

    let mut body = vec![];

    // Process all block items as the function's body.
    while let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        if let TokenType::BraceClose = token.ty {
            break;
        }

        let block = parse_block(ctx, iter)?;
        body.push(block);
    }

    expect_token(ctx, iter, TokenType::BraceClose)?;

    Ok(Function { ident, body })
}

/// Parse an _AST_ block from the provided `Token` iterator.
fn parse_block<I: Iterator<Item = TokenResult>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Block, String> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        match token.ty {
            // Parse this as a declaration (starts with a type).
            TokenType::Keyword(ref s) if s == "int" => {
                Ok(Block::Decl(parse_declaration(ctx, iter)?))
            }
            // Parse this as a statement.
            _ => Ok(Block::Stmt(parse_statement(ctx, iter)?)),
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '<block>' at end of input",
        ))
    }
}

/// Parse an _AST_ declaration from the provided `Token` iterator.
fn parse_declaration<I: Iterator<Item = TokenResult>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Declaration, String> {
    // NOTE: Currently only support the `int` type.
    expect_token(ctx, iter, TokenType::Keyword("int".into()))?;

    let (ident, token) = parse_ident(ctx, iter)?;

    let mut init = None;

    if let Some(token) = iter.peek().map(Result::as_ref).transpose()?
        && let TokenType::Operator(OperatorKind::Assign) = token.ty
    {
        // Consume the "=" token.
        let _ = iter.next();
        init = Some(parse_expression(ctx, iter, 0)?);
    }

    expect_token(ctx, iter, TokenType::Semicolon)?;

    Ok(Declaration { ident, token, init })
}

/// Parse an _AST_ statement from the provided `Token` iterator.
fn parse_statement<I: Iterator<Item = TokenResult>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Statement, String> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        match token.ty {
            TokenType::Keyword(ref s) if s == "return" => {
                // Consume the "return" token.
                let _ = iter.next();

                let expr = parse_expression(ctx, iter, 0)?;
                expect_token(ctx, iter, TokenType::Semicolon)?;

                Ok(Statement::Return(expr))
            }
            TokenType::Semicolon => {
                // Consume the "semicolon" token.
                let _ = iter.next();
                Ok(Statement::Null)
            }
            _ => {
                let expr = parse_expression(ctx, iter, 0)?;
                expect_token(ctx, iter, TokenType::Semicolon)?;

                Ok(Statement::Expression(expr))
            }
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '<statement>' at end of input",
        ))
    }
}

/// Parse an _AST_ identifier from the provided `Token` iterator.
fn parse_ident<I: Iterator<Item = TokenResult>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<(Ident, Token), String> {
    if let Some(token) = iter.next() {
        let token = token?;

        match token.ty {
            TokenType::Ident(ref s) => Ok((s.clone(), token)),
            tok => {
                let tok_str = format!("{tok:?}");
                let line_content = ctx.src_slice(token.loc.line_span);

                Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    line_content,
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

/// Parse an _AST_ expression from the provided `Token` iterator using
/// `precedence climbing`.
fn parse_expression<I: Iterator<Item = TokenResult>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    min_precedence: u8,
) -> Result<Expression, String> {
    let mut lhs = parse_factor(ctx, iter)?;
    let mut next = iter.peek().map(Result::as_ref).transpose()?;

    while let Some(token) = next {
        let Some(binop) = ty_to_binop(&token.ty) else {
            break;
        };

        if binop.precedence() < min_precedence {
            break;
        }

        // Consume the peeked token.
        let token = iter.next();

        if let BinaryOperator::Assign = binop {
            // This ensures we can handle right-associative operators like `=`,
            // since operators of the same precedence still are evaluated
            // together.
            let rhs = parse_expression(ctx, iter, binop.precedence())?;
            lhs = Expression::Assignment(
                Box::new(lhs),
                Box::new(rhs),
                token
                    .expect("next token should be present")
                    .expect("next token should be valid"),
            );
        } else {
            // Handle other binary operators as left-associative by allowing
            // higher-precedence operators to be evaluated first.
            let rhs = parse_expression(ctx, iter, binop.precedence() + 1)?;

            lhs = Expression::Binary {
                op: binop,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                // NOTE: Temporary hack for arithmetic right shift.
                sign: Signedness::Unsigned,
            };
        }

        next = iter.peek().map(Result::as_ref).transpose()?;
    }

    Ok(lhs)
}

/// Parse an _AST_ expression or sub-expression (factor) from the provided
/// `Token` iterator.
fn parse_factor<I: Iterator<Item = TokenResult>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Expression, String> {
    if let Some(token) = iter.next() {
        let token = token?;

        match token.ty {
            TokenType::Constant(v) => Ok(Expression::Constant(v)),
            TokenType::Ident(ref s) => Ok(Expression::Var((s.clone(), token))),
            TokenType::Operator(
                OperatorKind::BitNot | OperatorKind::Minus | OperatorKind::LogNot,
            ) => {
                let unop = ty_to_unop(&token.ty)
                    .expect("expected unary operator when parsing factor (OperatorKind::BitNot | OperatorKind::Minus | OperatorKind::LogNot)");

                // NOTE: Temporary hack for arithmetic right shift.
                let sign = if let UnaryOperator::Negate = unop {
                    Signedness::Signed
                } else {
                    Signedness::Unsigned
                };

                // Recursively parse the factor on which the unary operator is
                // being applied to.
                let inner_fct = parse_factor(ctx, iter)?;

                Ok(Expression::Unary {
                    op: unop,
                    expr: Box::new(inner_fct),
                    sign,
                })
            }
            TokenType::ParenOpen => {
                // Recursively parse the expression within parenthesis.
                let inner_expr = parse_expression(ctx, iter, 0)?;

                expect_token(ctx, iter, TokenType::ParenClose)?;

                Ok(inner_expr)
            }
            tok => {
                let tok_str = format!("{tok:?}");
                let line_content = ctx.src_slice(token.loc.line_span);

                let err_msg = if let TokenType::ParenClose = tok {
                    format!("expected expression before '{tok_str}' token")
                } else {
                    format!("unexpected token '{tok_str}' in expression")
                };

                Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    line_content,
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

/// Advance the `Token` iterator if it matches the expected token type.
fn expect_token<I: Iterator<Item = TokenResult>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    expected: TokenType,
) -> Result<(), String> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        if token.ty == expected {
            // Consume the peeked token.
            let _ = iter.next();
            Ok(())
        } else {
            let tok_str = format!("{token:?}");
            let line_content = ctx.src_slice(token.loc.line_span.clone());

            Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                line_content,
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

/// Returns the conversion of `TokenType` to `UnaryOperator`, or `None` if the
/// token type is not a unary operator.
fn ty_to_unop(ty: &TokenType) -> Option<UnaryOperator> {
    match ty {
        TokenType::Operator(OperatorKind::BitNot) => Some(UnaryOperator::Complement),
        TokenType::Operator(OperatorKind::Minus) => Some(UnaryOperator::Negate),
        TokenType::Operator(OperatorKind::LogNot) => Some(UnaryOperator::Not),
        _ => None,
    }
}

/// Returns the conversion of `TokenType` to `BinaryOperator`, or `None` if the
/// token type is not a binary operator.
fn ty_to_binop(ty: &TokenType) -> Option<BinaryOperator> {
    match ty {
        TokenType::Operator(OperatorKind::Plus) => Some(BinaryOperator::Add),
        TokenType::Operator(OperatorKind::Minus) => Some(BinaryOperator::Subtract),
        TokenType::Operator(OperatorKind::Asterisk) => Some(BinaryOperator::Multiply),
        TokenType::Operator(OperatorKind::Division) => Some(BinaryOperator::Divide),
        TokenType::Operator(OperatorKind::Remainder) => Some(BinaryOperator::Modulo),
        TokenType::Operator(OperatorKind::Ampersand) => Some(BinaryOperator::BitAnd),
        TokenType::Operator(OperatorKind::BitOr) => Some(BinaryOperator::BitOr),
        TokenType::Operator(OperatorKind::BitXor) => Some(BinaryOperator::BitXor),
        TokenType::Operator(OperatorKind::ShiftLeft) => Some(BinaryOperator::ShiftLeft),
        TokenType::Operator(OperatorKind::ShiftRight) => Some(BinaryOperator::ShiftRight),
        TokenType::Operator(OperatorKind::LogAnd) => Some(BinaryOperator::LogAnd),
        TokenType::Operator(OperatorKind::LogOr) => Some(BinaryOperator::LogOr),
        TokenType::Operator(OperatorKind::Eq) => Some(BinaryOperator::Eq),
        TokenType::Operator(OperatorKind::NotEq) => Some(BinaryOperator::NotEq),
        TokenType::Operator(OperatorKind::LessThan) => Some(BinaryOperator::OrdLess),
        TokenType::Operator(OperatorKind::LessThanEq) => Some(BinaryOperator::OrdLessEq),
        TokenType::Operator(OperatorKind::GreaterThan) => Some(BinaryOperator::OrdGreater),
        TokenType::Operator(OperatorKind::GreaterThanEq) => Some(BinaryOperator::OrdGreaterEq),
        TokenType::Operator(OperatorKind::Assign) => Some(BinaryOperator::Assign),
        _ => None,
    }
}

// Reference: https://github.com/nlsandler/writing-a-c-compiler-tests
#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::Context;

    fn test_ctx(src: &'static [u8]) -> Context<'static> {
        Context {
            program: "cc2",
            in_path: Path::new("test.c"),
            src,
        }
    }

    #[test]
    fn parser_valid_bitwise_complement() {
        let source = b"int main(void) {\n    return ~12;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bitwise_complement_i32_min() {
        // Take the bitwise complement of the largest negative integer
        // that can be safely negated in a 32-bit signed integer (-2147483647).
        let source = b"int main(void) {\n    return ~-2147483647;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bitwise_complement_zero() {
        let source = b"int main(void) {\n    return ~0;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_negation() {
        let source = b"int main(void) {\n    return -5;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_negation_zero() {
        let source = b"int main(void) {\n    return -0;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_negation_i32_max() {
        let source = b"int main(void) {\n    return -2147483647;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_nested_unary_ops() {
        let source = b"int main(void) {\n    return ~-3;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_parenthesize_constant() {
        let source = b"int main(void) {\n    return (2);\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_redundant_parens() {
        let source = b"int main(void) {\n    return -((((10))));\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_add() {
        let source = b"int main(void) {\n    return 1 + 2;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_div() {
        let source = b"int main(void) {\n    return 1 / 2;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_div_neg() {
        let source = b"int main(void) {\n    return (-12) / 2;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_mod() {
        let source = b"int main(void) {\n    return 12 % 2;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_mul() {
        let source = b"int main(void) {\n    return 12 * 2;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_parens() {
        let source = b"int main(void) {\n    return 2 * (3 + 4);\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_sub() {
        let source = b"int main(void) {\n    return 2 - 1;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_unary_add() {
        let source = b"int main(void) {\n    return ~2 + 1;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_unary_parens() {
        let source = b"int main(void) {\n    return ~(1 + 1);\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_associativity() {
        let source = b"int main(void) {\n    return (3 / 2 * 4) + (5 - 4 + 3);\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_associativity_and_precedence() {
        let source = b"int main(void) {\n    return 5 * 4 / 2 - 3 % (2 + 1);\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_bit_and() {
        let source = b"int main(void) {\n    return 3 & 5;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_bit_or() {
        let source = b"int main(void) {\n    return 1 | 2;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_bit_precedence() {
        let source = b"int main(void) {\n    return 80 >> 2 | 1 ^ 5 & 7 << 1;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    fn parser_valid_bin_bit_shift() {
        let source = b"int main(void) {\n    return 33 << 4 >> 2;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_no_expr() {
        let source = b"int main(void) {\n    return";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_extra_ident() {
        // Single identifier outside of a declaration is not a valid top-level
        // construct.
        let source = b"int main(void) {\n    return 2;\n}\nfoo";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_function_ident() {
        // Functions must have an identifier as a name.
        let source = b"int 3 (void) {\n    return 0;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_case_sensitive_keyword() {
        let source = b"int main(void) {\n    RETURN 0;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_missing_function_type() {
        // Because of backwards compatibility, `GCC` and `Clang` will compile
        // this program with a warning.
        let source = b"main(void) {\n    return 0;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_keyword() {
        let source = b"int main(void) {\n    returns 0;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_missing_semicolon() {
        let source = b"int main(void) {\n    return 0\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_expression() {
        let source = b"int main(void) {\n    return int;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_keyword_space() {
        let source = b"int main(void) {\n    retur n 0;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_parens_fn() {
        let source = b"int main)void( {\n    return 0;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_unclosed_brace_fn() {
        let source = b"int main(void) {\n    return 0;";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_unclosed_paren_fn() {
        let source = b"int main(void {\n    return 0;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_extra_paren() {
        let source = b"int main(void)\n{\n    return (3));\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_missing_constant() {
        let source = b"int main(void) {\n    return ~;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_nested_missing_constant() {
        let source = b"int main(void)\n{\n    return -~;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_parenthesize_operand() {
        let source = b"int main(void) {\n   return (-)3;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_unclose_paren_expr() {
        let source = b"int main(void) {\n   return (1;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_negation_postfix() {
        let source = b"int main(void) {\n   return 4-;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_semicolon() {
        let source = b"int main(void) {\n   return 2*2\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_rhs() {
        let source = b"int main(void) {\n   return 1 / ;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_open_paren() {
        let source = b"int main(void) {\n   return 1+2);\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_closing_paren() {
        let source = b"int main(void) {\n   return (1+2;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_lhs() {
        let source = b"int main(void) {\n   return / 3;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_misplaced_semicolon() {
        let source = b"int main(void) {\n   return 1 + (2;)\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_missing_op() {
        let source = b"int main(void) {\n   return 2 (- 3);\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }

    #[test]
    #[should_panic]
    fn parser_invalid_bin_expr_double_op() {
        let source = b"int main(void) {\n   return 2 * / 2;\n}";
        let ctx = test_ctx(source);

        let lexer = crate::compiler::lexer::Lexer::new(&ctx);
        parse_program(&ctx, lexer.peekable());
    }
}
