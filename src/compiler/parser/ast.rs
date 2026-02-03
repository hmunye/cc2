//! Abstract Syntax Tree
//!
//! Compiler pass that parses a stream of tokens into an abstract syntax tree
//! (_AST_).

use std::{fmt, process};

use super::sema;

use crate::compiler::Result;
use crate::compiler::lexer::{OperatorKind, Reserved, Token, TokenType};
use crate::{Context, fmt_err, fmt_token_err};

/// Zero-sized marker indicating a parsed _AST_ (no semantic analysis).
#[derive(Debug)]
pub struct Parsed;

/// Zero-sized marker indicating _AST_ after identifier resolution.
#[derive(Debug)]
pub struct IdentPhase;

/// Zero-sized marker indicating _AST_ after type checking.
#[derive(Debug)]
pub struct TypePhase;

/// Zero-sized marker indicating _AST_ after label resolution.
#[derive(Debug)]
pub struct LabelPhase;

/// Zero-sized marker indicating _AST_ after control-flow resolution.
#[derive(Debug)]
pub struct CtrlFlowPhase;

/// Zero-sized marker indicating _AST_ after `switch` resolution.
#[derive(Debug)]
pub struct SwitchPhase;

/// Zero-sized marker indicating all semantic analysis completed on _AST_.
#[derive(Debug)]
pub struct Analyzed;

/// Abstract Syntax Tree (_AST_).
#[derive(Debug)]
pub struct AST<P> {
    pub program: Vec<Function>,
    pub _phase: std::marker::PhantomData<P>,
}

impl<P> fmt::Display for AST<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AST Program")?;
        for func in &self.program {
            func.fmt_with_indent(f, 2)?;
        }

        Ok(())
    }
}

/// _AST_ type specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Func { params: usize },
}

/// _AST_ declaration.
#[derive(Debug)]
pub enum Declaration {
    Var {
        ident: String,
        init: Option<Expression>,
        /// Identifier token.
        token: Token,
    },
    Func(Function),
}

impl fmt::Display for Declaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Declaration::Var { ident, init, .. } => match init {
                Some(expr) => write!(f, "{ident:?} = {expr}"),
                None => write!(f, "{ident:?} = uninit"),
            },
            Declaration::Func(func) => func.fmt_with_indent(f, 0),
        }
    }
}

/// _AST_ function parameter.
#[derive(Debug)]
pub struct Param {
    pub ty: Type,
    pub ident: String,
    /// Identifier token.
    pub token: Token,
}

/// _AST_ function declaration/definition.
#[derive(Debug)]
pub struct Function {
    pub ident: String,
    pub params: Vec<Param>,
    pub body: Option<Block>,
    /// Identifier token.
    pub token: Token,
}

impl Function {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);
        let params = self
            .params
            .iter()
            .map(|param| param.ident.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        writeln!(f, "{}Fn {:?}({})", pad, self.ident, params)?;

        if let Some(body) = &self.body {
            body.fmt_with_indent(f, indent + 2)?;
        }

        Ok(())
    }
}

/// _AST_ block.
#[derive(Debug)]
pub struct Block(pub Vec<BlockItem>);

impl Block {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);

        writeln!(f, "{pad}Block: {{")?;

        for item in &self.0 {
            item.fmt_with_indent(f, indent + 2)?;
        }

        writeln!(f, "{pad}}}")
    }
}

/// _AST_ block item.
#[derive(Debug)]
pub enum BlockItem {
    Stmt(Statement),
    Decl(Declaration),
}

impl BlockItem {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);

        match self {
            BlockItem::Stmt(stmt) => stmt.fmt_with_indent(f, indent),
            BlockItem::Decl(decl) => {
                writeln!(f, "{pad}Decl: {decl}")
            }
        }
    }
}

/// _AST_ `for` statement initial clause.
#[derive(Debug)]
pub enum ForInit {
    Decl(Declaration),
    Expr(Option<Expression>),
}

/// _AST_ labeled statement.
#[derive(Debug)]
pub enum Labeled {
    Label {
        label: String,
        stmt: Box<Statement>,
        /// `label` identifier token.
        token: Token,
    },
    Case {
        expr: Expression,
        stmt: Box<Statement>,
        /// `case` keyword token.
        token: Token,
        jmp_label: String,
    },
    Default {
        stmt: Box<Statement>,
        /// `default` keyword token.
        token: Token,
        jmp_label: String,
    },
}

/// _AST_ case label/expression.
#[derive(Debug)]
pub struct SwitchCase {
    pub jmp_label: String,
    pub expr: Expression,
}

/// _AST_ statement.
#[derive(Debug)]
pub enum Statement {
    Return(Expression),
    Expression(Expression),
    If {
        /// Controlling expression.
        cond: Expression,
        /// Executes when the result of `cond` is non-zero.
        then: Box<Statement>,
        /// Optional statement to execute when result of `cond` is zero.
        opt_else: Option<Box<Statement>>,
    },
    Goto {
        target: String,
        /// `goto` keyword token.
        token: Token,
    },
    LabeledStatement(Labeled),
    Compound(Block),
    Break {
        jmp_label: String,
        /// `break` keyword token.
        token: Token,
    },
    Continue {
        jmp_label: String,
        /// `continue` keyword token.
        token: Token,
    },
    While {
        cond: Expression,
        stmt: Box<Statement>,
        loop_label: String,
    },
    Do {
        stmt: Box<Statement>,
        cond: Expression,
        loop_label: String,
    },
    For {
        init: Box<ForInit>,
        opt_cond: Option<Expression>,
        opt_post: Option<Expression>,
        stmt: Box<Statement>,
        loop_label: String,
    },
    Switch {
        /// Controlling expression.
        cond: Expression,
        stmt: Box<Statement>,
        /// Result of `cond` used to determine which switch case to execute at.
        cases: Vec<SwitchCase>,
        /// `default` jmp label.
        default: Option<String>,
        switch_label: String,
    },
    /// Expression statement without an expression (`;`).
    Empty,
}

impl Statement {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);

        match self {
            Statement::Return(expr) => {
                writeln!(f, "{pad}Return {expr}")
            }
            Statement::Expression(expr) => {
                writeln!(f, "{pad}Expr: {expr}")
            }
            Statement::If {
                cond,
                then,
                opt_else,
            } => {
                writeln!(f, "{pad}If ({cond})")?;
                writeln!(f, "{pad}Then:")?;
                then.fmt_with_indent(f, indent + 2)?;

                if let Some(else_stmt) = opt_else {
                    writeln!(f, "{pad}Else:")?;
                    else_stmt.fmt_with_indent(f, indent + 2)?;
                }

                Ok(())
            }
            Statement::Goto { target, .. } => {
                writeln!(f, "{pad}Goto {target:?}")
            }
            Statement::LabeledStatement(labeled) => match labeled {
                Labeled::Label { label, stmt, .. } => {
                    writeln!(f, "{pad}Label {label:?}:")?;
                    stmt.fmt_with_indent(f, indent + 2)
                }
                Labeled::Case {
                    expr,
                    stmt,
                    jmp_label: label,
                    ..
                } => {
                    writeln!(f, "{pad}Case <label {label:?}> {expr}:")?;
                    stmt.fmt_with_indent(f, indent + 2)
                }
                Labeled::Default {
                    stmt,
                    jmp_label: label,
                    ..
                } => {
                    writeln!(f, "{pad}Default <label {label:?}>:")?;
                    stmt.fmt_with_indent(f, indent + 2)
                }
            },
            Statement::Compound(block) => block.fmt_with_indent(f, indent),
            Statement::Break { jmp_label, .. } => {
                writeln!(f, "{pad}Break <label {jmp_label:?}>")
            }
            Statement::Continue { jmp_label, .. } => {
                writeln!(f, "{pad}Continue <label {jmp_label:?}>")
            }
            Statement::While {
                cond,
                stmt,
                loop_label: label,
            } => {
                writeln!(f, "{pad}While <label {label:?}> ({cond})")?;
                stmt.fmt_with_indent(f, indent + 2)
            }
            Statement::Do {
                stmt,
                cond,
                loop_label: label,
            } => {
                writeln!(f, "{pad}Do <label {label:?}>")?;
                stmt.fmt_with_indent(f, indent + 2)?;
                writeln!(f, "{}/* while */ ({cond})", "  ".repeat(indent + 2))
            }
            Statement::For {
                init,
                opt_cond,
                opt_post,
                stmt,
                loop_label: label,
            } => {
                let init_fmt = match &**init {
                    ForInit::Decl(decl) => format!("Decl: {decl}"),
                    ForInit::Expr(opt_expr) => opt_expr
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_default(),
                };

                let cond_fmt = opt_cond
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_default();

                let post_fmt = opt_post
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_default();

                writeln!(
                    f,
                    "{pad}For <label {label:?}> ({init_fmt}; {cond_fmt}; {post_fmt})",
                )?;

                stmt.fmt_with_indent(f, indent + 2)
            }
            Statement::Switch {
                cond,
                stmt,
                switch_label: label,
                ..
            } => {
                writeln!(f, "{pad}Switch <label {label:?}> ({cond})")?;
                stmt.fmt_with_indent(f, indent + 2)
            }
            Statement::Empty => {
                writeln!(f, "{pad}Empty \";\"")
            }
        }
    }
}

/// _AST_ expression.
#[derive(Debug, Clone)]
pub enum Expression {
    /// Integer constant (32-bit signed).
    IntConstant(i32),
    Var {
        ident: String,
        /// Identifier token.
        token: Token,
    },
    /// Unary operator applied to an expression.
    Unary {
        op: UnaryOperator,
        expr: Box<Expression>,
        // NOTE: Temporary hack for arithmetic right shift.
        sign: Signedness,
        prefix: bool,
    },
    /// Binary operator applied to two expressions.
    Binary {
        op: BinaryOperator,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
        // NOTE: Temporary hack for arithmetic right shift.
        sign: Signedness,
    },
    /// Assigns an `rvalue` to an `lvalue`.
    Assignment {
        lvalue: Box<Expression>,
        rvalue: Box<Expression>,
        /// Assignment operator token.
        token: Token,
    },
    /// Ternary expression which evaluates the condition and returns the result
    /// of the `second` if true, otherwise `third`.
    Conditional {
        cond: Box<Expression>,
        second: Box<Expression>,
        third: Box<Expression>,
    },
    FuncCall {
        ident: String,
        args: Vec<Expression>,
        /// Identifier token.
        token: Token,
    },
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::IntConstant(i) => write!(f, "Int({i})"),
            Expression::Var { ident, .. } => write!(f, "Var({ident:?})"),
            Expression::Unary {
                op, expr, prefix, ..
            } => {
                if *prefix {
                    write!(f, "{op}{expr}")
                } else {
                    write!(f, "{expr}{op}")
                }
            }
            Expression::Binary { op, lhs, rhs, .. } => {
                write!(f, "{lhs} {op} {rhs}")
            }
            Expression::Assignment { lvalue, rvalue, .. } => {
                write!(f, "{lvalue} = {rvalue}")
            }
            Expression::Conditional {
                cond,
                second,
                third,
            } => {
                write!(f, "{cond} ? {second} : {third}")
            }
            Expression::FuncCall { ident, args, .. } => {
                let args_str = args
                    .iter()
                    .map(|a| format!("{a}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "{ident:?}({args_str})")
            }
        }
    }
}

/// _AST_ unary operator.
#[derive(Debug, Clone, Copy)]
pub enum UnaryOperator {
    /// `~` - unary operator.
    Complement,
    /// `-` - unary operator.
    Negate,
    /// `!` - unary operator.
    Not,
    /// `++` - unary operator (postfix or prefix).
    Increment,
    /// `--` - unary operator (postfix or prefix).
    Decrement,
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self {
            UnaryOperator::Complement => "~",
            UnaryOperator::Negate => "-",
            UnaryOperator::Not => "!",
            UnaryOperator::Increment => "++",
            UnaryOperator::Decrement => "--",
        };

        write!(f, "{op}")
    }
}

/// _AST_ binary operator.
#[derive(Debug, Clone, Copy)]
pub enum BinaryOperator {
    /// `+` - binary operator.
    Add,
    /// `-` - binary operator.
    Subtract,
    /// `*` - binary operator.
    Multiply,
    /// `/` - binary operator.
    Divide,
    /// `%` - binary operator.
    Modulo,
    /// `&` - binary operator.
    BitAnd,
    /// `|` - binary operator.
    BitOr,
    /// `^` - binary operator.
    BitXor,
    /// `<<` - binary operator.
    ShiftLeft,
    /// `>>` - binary operator.
    ShiftRight,
    /// `&&` - binary operator.
    LogAnd,
    /// `||` - binary operator.
    LogOr,
    /// `==` - binary operator.
    Eq,
    /// `!=` - binary operator.
    NotEq,
    /// `<` - binary operator.
    OrdLess,
    /// `<=` - binary operator.
    OrdLessEq,
    /// `>` - binary operator.
    OrdGreater,
    /// `>=` - binary operator.
    OrdGreaterEq,
    /// `=` - binary operator.
    Assign,
    /// `+=` - binary operator.
    AssignAdd,
    /// `-=` - binary operator.
    AssignSubtract,
    /// `*=` - binary operator.
    AssignMultiply,
    /// `/=` - binary operator.
    AssignDivide,
    /// `%=` - binary operator.
    AssignModulo,
    /// `&=` - binary operator.
    AssignBitAnd,
    /// `|=` - binary operator.
    AssignBitOr,
    /// `^=` - binary operator.
    AssignBitXor,
    /// `<<=` - binary operator.
    AssignShiftLeft,
    /// `>>=` - binary operator.
    AssignShiftRight,
    /// `?` - ternary operator (parsed as a binary operator but **not**
    /// evaluated as one).
    Conditional,
}

impl BinaryOperator {
    /// Returns the precedence level of the binary operator (higher number
    /// indicates tighter binding).
    #[must_use]
    pub fn precedence(&self) -> u8 {
        match self {
            // _C17_ 6.5.16 (assignment-expression)
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
            | BinaryOperator::AssignShiftRight => 3,
            // _C17_ 6.5.15 (conditional-expression)
            BinaryOperator::Conditional => 4,
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

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self {
            BinaryOperator::Add => "+",
            BinaryOperator::Subtract => "-",
            BinaryOperator::Multiply => "*",
            BinaryOperator::Divide => "/",
            BinaryOperator::Modulo => "%",
            BinaryOperator::BitAnd => "&",
            BinaryOperator::BitOr => "|",
            BinaryOperator::BitXor => "^",
            BinaryOperator::ShiftLeft => "<<",
            BinaryOperator::ShiftRight => ">>",
            BinaryOperator::LogAnd => "&&",
            BinaryOperator::LogOr => "||",
            BinaryOperator::Eq => "==",
            BinaryOperator::NotEq => "!=",
            BinaryOperator::OrdLess => "<",
            BinaryOperator::OrdLessEq => "<=",
            BinaryOperator::OrdGreater => ">",
            BinaryOperator::OrdGreaterEq => ">=",
            BinaryOperator::Assign => "=",
            BinaryOperator::AssignAdd => "+=",
            BinaryOperator::AssignSubtract => "-=",
            BinaryOperator::AssignMultiply => "*=",
            BinaryOperator::AssignDivide => "/=",
            BinaryOperator::AssignModulo => "%=",
            BinaryOperator::AssignBitAnd => "&=",
            BinaryOperator::AssignBitOr => "|=",
            BinaryOperator::AssignBitXor => "^=",
            BinaryOperator::AssignShiftLeft => "<<=",
            BinaryOperator::AssignShiftRight => ">>=",
            BinaryOperator::Conditional => "?",
        };

        write!(f, "{op}")
    }
}

/// NOTE: Temporary hack for arithmetic right shift.
#[derive(Debug, Clone, Copy)]
pub enum Signedness {
    Signed,
    Unsigned,
}

/// Parses an abstract syntax tree (_AST_) from the provided `Token` iterator.
/// [Exits] on error with non-zero status.
///
/// [Exits]: std::process::exit
pub fn parse_ast<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    mut iter: std::iter::Peekable<I>,
) -> AST<Analyzed> {
    // Run all semantic analysis passes in order after parsing _AST_.
    (|| {
        let ast = parse_program(ctx, &mut iter)?;

        // Pass 1 - Identifier resolution.
        let ast = sema::resolve_idents(ast, ctx)?;

        // Pass 2 - Type checking.
        let ast = sema::resolve_types(ast, ctx)?;

        // Pass 3 - Label/`goto` resolution.
        let ast = sema::resolve_labels(ast, ctx)?;

        // Pass 4 - Control-flow labeling.
        let ast = sema::resolve_escapable_ctrl(ast, ctx)?;

        // Pass 5 - Switch statement resolution.
        sema::resolve_switches(ast, ctx)
    })()
    .unwrap_or_else(|err| {
        eprintln!("{err}");
        process::exit(1);
    })
}

/// Parses an _AST_ program from the provided `Token` iterator.
///
/// # Errors
///
/// This function will return an error if a program could not be parsed.
pub fn parse_program<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<AST<Parsed>> {
    let mut funcs = vec![];

    while iter.peek().is_some() {
        funcs.push(parse_function(ctx, iter, None)?);
    }

    Ok(AST {
        program: funcs,
        _phase: std::marker::PhantomData,
    })
}

/// Parses an _AST_ function declaration/definition from the provided `Token`
/// iterator. Optionally accepts a partially parsed function header.
///
/// # Errors
///
/// This function will return an error if a function could not be parsed.
fn parse_function<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    parsed_header: Option<(&'static str, String, Token)>,
) -> Result<Function> {
    let (ident, token) = if let Some(parsed_header) = parsed_header {
        // NOTE: Only allow `int` return type for now.
        debug_assert!(parsed_header.0 == "int");
        (parsed_header.1, parsed_header.2)
    } else {
        expect_token(ctx, iter, TokenType::Keyword(Reserved::Int))?;
        parse_ident(ctx, iter)?
    };

    expect_token(ctx, iter, TokenType::LParen)?;
    let params = parse_params(ctx, iter)?;
    expect_token(ctx, iter, TokenType::RParen)?;

    if let Some(tok) = iter.peek().map(Result::as_ref).transpose()?
        && let TokenType::Semicolon = tok.ty
    {
        // Consume the ";" token.
        let _ = iter.next();

        return Ok(Function {
            ident,
            params,
            body: None,
            token,
        });
    }

    let body = parse_block(ctx, iter)?;

    Ok(Function {
        ident,
        params,
        body: Some(body),
        token,
    })
}

/// Parse an _AST_ parameter list from the provided `Token` iterator.
///
/// # Errors
///
/// This function will return an error if a parameter list could not be parsed.
fn parse_params<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Vec<Param>> {
    let mut params = vec![];

    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        match token.ty {
            TokenType::Keyword(Reserved::Void) => {
                // Consume the "void" token.
                let _ = iter.next();
                Ok(params)
            }
            TokenType::Keyword(Reserved::Int) => {
                // Consume the "int" token.
                let _ = iter.next();

                let (ident, token) = parse_ident(ctx, iter)?;
                params.push(Param {
                    ty: Type::Int,
                    ident,
                    token,
                });

                while let Some(token) = iter.peek().map(Result::as_ref).transpose()?
                    && let TokenType::Comma = token.ty
                {
                    // Consume the "," token.
                    let _ = iter.next();

                    expect_token(ctx, iter, TokenType::Keyword(Reserved::Int))?;

                    let (ident, token) = parse_ident(ctx, iter)?;
                    params.push(Param {
                        ty: Type::Int,
                        ident,
                        token,
                    });
                }

                Ok(params)
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
                    "unknown type name '{tok_str}'",
                ))
            }
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '<params>' at end of input",
        ))
    }
}

/// Parse an _AST_ declaration from the provided `Token` iterator.
///
/// # Errors
///
/// This function will return an error if a declaration could not be parsed.
fn parse_declaration<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Declaration> {
    expect_token(ctx, iter, TokenType::Keyword(Reserved::Int))?;

    let (ident, token) = parse_ident(ctx, iter)?;

    let mut init = None;

    if let Some(tok) = iter.peek().map(Result::as_ref).transpose()? {
        match &tok.ty {
            // Function declaration/definition.
            TokenType::LParen => {
                let func = parse_function(ctx, iter, Some(("int", ident, token)))?;
                return Ok(Declaration::Func(func));
            }
            // Variable declaration with initializer.
            TokenType::Operator(OperatorKind::Assign) => {
                // Consume the "=" token.
                let _ = iter.next();
                init = Some(parse_expression(ctx, iter, 0)?);
            }
            // Variable declaration with no initializer.
            _ => {}
        }
    }

    expect_token(ctx, iter, TokenType::Semicolon)?;

    Ok(Declaration::Var { ident, token, init })
}

/// Parse an _AST_ block from the provided `Token` iterator.
///
/// # Errors
///
/// This function will return an error if a block could not be parsed.
fn parse_block<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Block> {
    expect_token(ctx, iter, TokenType::LBrace)?;

    let mut block = vec![];

    while let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        if let TokenType::RBrace = token.ty {
            break;
        }

        let block_item = parse_block_item(ctx, iter)?;
        block.push(block_item);
    }

    expect_token(ctx, iter, TokenType::RBrace)?;

    Ok(Block(block))
}

/// Parse an _AST_ block item from the provided `Token` iterator.
///
/// # Errors
///
/// This function will return an error if a block item could not be parsed.
fn parse_block_item<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<BlockItem> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        match token.ty {
            // Parse this as a declaration (starts with a type).
            TokenType::Keyword(Reserved::Int) => Ok(BlockItem::Decl(parse_declaration(ctx, iter)?)),
            // Parse this as a statement.
            _ => Ok(BlockItem::Stmt(parse_statement(ctx, iter)?)),
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '<block>' at end of input",
        ))
    }
}

/// Parse an _AST_ `for` initial clause from the provided `Token` iterator.
///
/// # Errors
///
/// This function will return an error if a `for` statement initial clause could
/// not be parsed.
fn parse_for_init<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<ForInit> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        // Parse this as a declaration (starts with a type).
        if let TokenType::Keyword(Reserved::Int) = token.ty {
            match parse_declaration(ctx, iter)? {
                Declaration::Func(func) => {
                    let token = &func.token;

                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "declaration of non-variable '{tok_str}' in 'for' loop initial declaration",
                    ))
                }
                decl @ Declaration::Var { .. } => Ok(ForInit::Decl(decl)),
            }
        } else {
            // Parse this as an optional expression.
            let opt_expr = parse_opt_expression(ctx, iter, TokenType::Semicolon)?;
            expect_token(ctx, iter, TokenType::Semicolon)?;

            Ok(ForInit::Expr(opt_expr))
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '<for_init>' at end of input",
        ))
    }
}

/// Parse an _AST_ statement from the provided `Token` iterator.
///
/// # Errors
///
/// This function will return an error if a statement could not be parsed.
fn parse_statement<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Statement> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        match token.ty {
            TokenType::Keyword(Reserved::Return) => {
                // Consume the "return" token.
                let _ = iter.next();

                let expr = parse_expression(ctx, iter, 0)?;
                expect_token(ctx, iter, TokenType::Semicolon)?;

                Ok(Statement::Return(expr))
            }
            TokenType::Keyword(Reserved::If) => {
                // Consume the "if" token.
                let _ = iter.next();

                expect_token(ctx, iter, TokenType::LParen)?;
                let expr = parse_expression(ctx, iter, 0)?;
                expect_token(ctx, iter, TokenType::RParen)?;

                let stmt = parse_statement(ctx, iter)?;

                let mut opt_else = None;

                if let Some(token) = iter.peek().map(Result::as_ref).transpose()?
                    && let TokenType::Keyword(kw) = token.ty
                    && matches!(kw, Reserved::Else)
                {
                    // Consume the "else" token.
                    let _ = iter.next();
                    opt_else = Some(Box::new(parse_statement(ctx, iter)?));
                }

                Ok(Statement::If {
                    cond: expr,
                    then: Box::new(stmt),
                    opt_else,
                })
            }
            TokenType::Keyword(Reserved::Goto) => {
                // Consume the "goto" token.
                let token = iter
                    .next()
                    .expect("next token should be present")
                    .expect("next token should be ok");

                let (target, _) = parse_ident(ctx, iter)?;

                expect_token(ctx, iter, TokenType::Semicolon)?;

                Ok(Statement::Goto { target, token })
            }
            TokenType::Keyword(Reserved::Break) => {
                // Consume the "break" token.
                let token = iter
                    .next()
                    .expect("next token should be present")
                    .expect("next token should be ok");

                expect_token(ctx, iter, TokenType::Semicolon)?;

                Ok(Statement::Break {
                    // Placeholder label allocated during parsing, backpatched
                    // in control-flow labeling pass.
                    jmp_label: String::new(),
                    token,
                })
            }
            TokenType::Keyword(Reserved::Continue) => {
                // Consume the "continue" token.
                let token = iter
                    .next()
                    .expect("next token should be present")
                    .expect("next token should be ok");

                expect_token(ctx, iter, TokenType::Semicolon)?;

                Ok(Statement::Continue {
                    // Placeholder label allocated during parsing, backpatched
                    // in control-flow labeling pass.
                    jmp_label: String::new(),
                    token,
                })
            }
            TokenType::Keyword(Reserved::While) => {
                // Consume the "while" token.
                let _ = iter.next();

                expect_token(ctx, iter, TokenType::LParen)?;

                let cond = parse_expression(ctx, iter, 0)?;

                expect_token(ctx, iter, TokenType::RParen)?;

                let stmt = parse_statement(ctx, iter)?;

                Ok(Statement::While {
                    cond,
                    stmt: Box::new(stmt),
                    // Placeholder label allocated during parsing, backpatched
                    // in control-flow labeling pass.
                    loop_label: String::new(),
                })
            }
            TokenType::Keyword(Reserved::Do) => {
                // Consume the "do" token.
                let _ = iter.next();

                let stmt = parse_statement(ctx, iter)?;

                expect_token(ctx, iter, TokenType::Keyword(Reserved::While))?;

                expect_token(ctx, iter, TokenType::LParen)?;

                let cond = parse_expression(ctx, iter, 0)?;

                expect_token(ctx, iter, TokenType::RParen)?;
                expect_token(ctx, iter, TokenType::Semicolon)?;

                Ok(Statement::Do {
                    stmt: Box::new(stmt),
                    cond,
                    // Placeholder label allocated during parsing, backpatched
                    // in control-flow labeling pass.
                    loop_label: String::new(),
                })
            }
            TokenType::Keyword(Reserved::For) => {
                // Consume the "for" token.
                let _ = iter.next();

                expect_token(ctx, iter, TokenType::LParen)?;

                let opt_init = parse_for_init(ctx, iter)?;

                let opt_cond = parse_opt_expression(ctx, iter, TokenType::Semicolon)?;
                expect_token(ctx, iter, TokenType::Semicolon)?;

                let opt_post = parse_opt_expression(ctx, iter, TokenType::RParen)?;
                expect_token(ctx, iter, TokenType::RParen)?;

                let stmt = parse_statement(ctx, iter)?;

                Ok(Statement::For {
                    init: Box::new(opt_init),
                    opt_cond,
                    opt_post,
                    stmt: Box::new(stmt),
                    // Placeholder label allocated during parsing, backpatched
                    // in control-flow labeling pass.
                    loop_label: String::new(),
                })
            }
            TokenType::Keyword(Reserved::Switch) => {
                // Consume the "switch" token.
                let _ = iter.next();

                expect_token(ctx, iter, TokenType::LParen)?;
                let expr = parse_expression(ctx, iter, 0)?;
                expect_token(ctx, iter, TokenType::RParen)?;

                let stmt = parse_statement(ctx, iter)?;

                Ok(Statement::Switch {
                    cond: expr,
                    stmt: Box::new(stmt),
                    // Will be filled in a later semantic analysis pass.
                    cases: vec![],
                    // Will be filled in a later semantic analysis pass.
                    default: None,
                    // Placeholder label allocated during parsing, backpatched
                    // in control-flow labeling pass.
                    switch_label: String::new(),
                })
            }
            TokenType::Keyword(Reserved::Case) => {
                // Consume the "case" token.
                let token = iter
                    .next()
                    .expect("next token should be present")
                    .expect("next token should be ok");

                let expr = parse_expression(ctx, iter, 0)?;

                // NOTE: Update when constant-expression eval is available to
                // the compiler.
                if let Expression::IntConstant(_) = expr {
                    expect_token(ctx, iter, TokenType::Colon)?;

                    let stmt = parse_statement(ctx, iter)?;

                    Ok(Statement::LabeledStatement(Labeled::Case {
                        expr,
                        token,
                        stmt: Box::new(stmt),
                        // Placeholder label allocated during parsing,
                        // backpatched in semantic analysis.
                        jmp_label: String::new(),
                    }))
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
                        "case label does not reduce to an integer constant (currently only support integer literals)",
                    ))
                }
            }
            TokenType::Keyword(Reserved::Default) => {
                // Consume the "default" token.
                let token = iter
                    .next()
                    .expect("next token should be present")
                    .expect("next token should be ok");

                expect_token(ctx, iter, TokenType::Colon)?;

                let stmt = parse_statement(ctx, iter)?;

                Ok(Statement::LabeledStatement(Labeled::Default {
                    token,
                    stmt: Box::new(stmt),
                    // Placeholder label allocated during parsing, backpatched
                    // in semantic analysis.
                    jmp_label: String::new(),
                }))
            }
            TokenType::Semicolon => {
                // Consume the ";" token.
                let _ = iter.next();
                Ok(Statement::Empty)
            }
            TokenType::LBrace => {
                let block = parse_block(ctx, iter)?;
                Ok(Statement::Compound(block))
            }
            _ => {
                let expr = parse_expression(ctx, iter, 0)?;

                // Labeled statement encountered if next token is colon (`:`).
                if let Some(token) = iter.peek().map(Result::as_ref).transpose()?
                    && let TokenType::Colon = token.ty
                {
                    if let Expression::Var { ident, token } = expr {
                        // Consume the ":" token.
                        let _ = iter.next();

                        let stmt = parse_statement(ctx, iter)?;

                        Ok(Statement::LabeledStatement(Labeled::Label {
                            label: ident,
                            token,
                            stmt: Box::new(stmt),
                        }))
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
                            "expected ';' before '{tok_str}' token",
                        ))
                    }
                } else {
                    expect_token(ctx, iter, TokenType::Semicolon)?;
                    Ok(Statement::Expression(expr))
                }
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
///
/// # Errors
///
/// This function will return an error if an identifier could not be parsed.
fn parse_ident<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<(String, Token)> {
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
///
/// # Errors
///
/// This function will return an error if an expression could not be parsed.
fn parse_expression<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    min_precedence: u8,
) -> Result<Expression> {
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
        let token = iter
            .next()
            .expect("next token should be present")
            .expect("next token should be ok");

        match binop {
            BinaryOperator::Assign => {
                // This ensures we can handle right-associative operators like
                // `=`, since operators of the same precedence still are
                // evaluated together.
                let rhs = parse_expression(ctx, iter, binop.precedence())?;
                lhs = Expression::Assignment {
                    lvalue: Box::new(lhs),
                    rvalue: Box::new(rhs),
                    token,
                };
            }
            BinaryOperator::AssignAdd
            | BinaryOperator::AssignSubtract
            | BinaryOperator::AssignMultiply
            | BinaryOperator::AssignDivide
            | BinaryOperator::AssignModulo
            | BinaryOperator::AssignBitAnd
            | BinaryOperator::AssignBitOr
            | BinaryOperator::AssignBitXor
            | BinaryOperator::AssignShiftLeft
            | BinaryOperator::AssignShiftRight => {
                let op = match binop {
                    BinaryOperator::AssignAdd => BinaryOperator::Add,
                    BinaryOperator::AssignSubtract => BinaryOperator::Subtract,
                    BinaryOperator::AssignMultiply => BinaryOperator::Multiply,
                    BinaryOperator::AssignDivide => BinaryOperator::Divide,
                    BinaryOperator::AssignModulo => BinaryOperator::Modulo,
                    BinaryOperator::AssignBitAnd => BinaryOperator::BitAnd,
                    BinaryOperator::AssignBitOr => BinaryOperator::BitOr,
                    BinaryOperator::AssignBitXor => BinaryOperator::BitXor,
                    BinaryOperator::AssignShiftLeft => BinaryOperator::ShiftLeft,
                    BinaryOperator::AssignShiftRight => BinaryOperator::ShiftRight,
                    _ => unreachable!(),
                };

                // This ensures we can handle right-associative operators since
                // operators of the same precedence still are evaluated
                // together.
                let rhs = parse_expression(ctx, iter, binop.precedence())?;

                let rhs = Expression::Binary {
                    op,
                    lhs: Box::new(lhs.clone()),
                    rhs: Box::new(rhs),
                    // NOTE: Temporary hack for arithmetic right shift.
                    sign: Signedness::Unsigned,
                };

                lhs = Expression::Assignment {
                    lvalue: Box::new(lhs),
                    rvalue: Box::new(rhs),
                    token,
                };
            }
            BinaryOperator::Conditional => {
                let second = parse_expression(ctx, iter, 0)?;

                expect_token(ctx, iter, TokenType::Colon)?;

                // This ensures we can handle right-associative operators since
                // operators of the same precedence still are evaluated
                // together.
                let rhs = parse_expression(ctx, iter, binop.precedence())?;

                lhs = Expression::Conditional {
                    cond: Box::new(lhs),
                    second: Box::new(second),
                    third: Box::new(rhs),
                };
            }
            binop => {
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
        }

        next = iter.peek().map(Result::as_ref).transpose()?;
    }

    Ok(lhs)
}

/// Parse an _AST_ expression from the provided `Token` iterator, or `None` if
/// the `end_token` is encountered first.
///
/// # Errors
///
/// This function will return an error if an optional expression could not be
/// parsed.
fn parse_opt_expression<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    end_token: TokenType,
) -> Result<Option<Expression>> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        if token.ty == end_token {
            // Not consuming the `end_token`.
            Ok(None)
        } else {
            Ok(Some(parse_expression(ctx, iter, 0)?))
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected '{end_token}' or '<expr>' at end of input",
        ))
    }
}

/// Parse an _AST_ expression or sub-expression (factor) from the provided
/// `Token` iterator.
///
/// # Errors
///
/// This function will return an error if a factor could not be parsed.
fn parse_factor<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Expression> {
    if let Some(token) = iter.next() {
        let token = token?;

        match token.ty {
            TokenType::IntConstant(v) => {
                if let Some(tok) = iter.peek().map(Result::as_ref).transpose()?
                    && let TokenType::Operator(OperatorKind::Increment | OperatorKind::Decrement) =
                        tok.ty
                {
                    let tok_str = format!("{tok:?}");
                    let line_content = ctx.src_slice(tok.loc.line_span.clone());

                    let err_msg = if let TokenType::Operator(OperatorKind::Increment) = tok.ty {
                        "lvalue required as increment operand"
                    } else {
                        "lvalue required as decrement operand"
                    };

                    Err(fmt_token_err!(
                        tok.loc.file_path.display(),
                        tok.loc.line,
                        tok.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "{err_msg}",
                    ))
                } else {
                    Ok(Expression::IntConstant(v))
                }
            }
            TokenType::Ident(ref s) => {
                if let Some(tok) = iter.peek().map(Result::as_ref).transpose()? {
                    match &tok.ty {
                        TokenType::Operator(OperatorKind::Increment | OperatorKind::Decrement) => {
                            let unop = ty_to_unop(&tok.ty).expect(
                                "expected postfix increment or decrement when parsing factor",
                            );

                            // Consume the postfix increment/decrement operator.
                            let _ = iter.next();

                            return Ok(Expression::Unary {
                                op: unop,
                                expr: Box::new(Expression::Var {
                                    ident: s.clone(),
                                    token,
                                }),
                                // NOTE: Temporary hack for arithmetic right shift.
                                sign: Signedness::Unsigned,
                                // Postfix unary operator.
                                prefix: false,
                            });
                        }
                        TokenType::LParen => {
                            // Consume the "(" token.
                            let _ = iter.next();

                            let args = parse_args(ctx, iter)?;
                            expect_token(ctx, iter, TokenType::RParen)?;

                            return Ok(Expression::FuncCall {
                                ident: s.clone(),
                                args,
                                token,
                            });
                        }
                        _ => {}
                    }
                }

                Ok(Expression::Var {
                    ident: s.clone(),
                    token,
                })
            }
            TokenType::Operator(
                OperatorKind::BitNot
                | OperatorKind::Minus
                | OperatorKind::LogNot
                | OperatorKind::Increment
                | OperatorKind::Decrement,
            ) => {
                let unop = ty_to_unop(&token.ty)
                    .expect("expected prefix unary operator when parsing factor");

                // NOTE: Temporary hack for arithmetic right shift.
                let sign = if let UnaryOperator::Negate = unop {
                    Signedness::Signed
                } else {
                    Signedness::Unsigned
                };

                // Recursively parse the factor on which the unary operator is
                // being applied to.
                let inner_fct = parse_factor(ctx, iter)?;

                if let UnaryOperator::Increment | UnaryOperator::Decrement = unop
                    && !matches!(inner_fct, Expression::Var { .. })
                {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span);

                    let err_msg = if let UnaryOperator::Increment = unop {
                        "lvalue required as increment operand"
                    } else {
                        "lvalue required as decrement operand"
                    };

                    return Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "{err_msg}",
                    ));
                }

                Ok(Expression::Unary {
                    op: unop,
                    expr: Box::new(inner_fct),
                    sign,
                    prefix: true,
                })
            }
            TokenType::LParen => {
                // Recursively parse the expression within parenthesis.
                let inner_expr = parse_expression(ctx, iter, 0)?;

                expect_token(ctx, iter, TokenType::RParen)?;

                if let Some(tok) = iter.peek().map(Result::as_ref).transpose()?
                    && let TokenType::Operator(OperatorKind::Increment | OperatorKind::Decrement) =
                        tok.ty
                {
                    let unop = ty_to_unop(&tok.ty).expect(
                        "expected postfix increment or decrement when parsing parenthesized expression"
                    );

                    // Consume the postfix increment/decrement operator.
                    let token = iter
                        .next()
                        .expect("next token should be present")
                        .expect("next token should be ok");

                    if let Expression::Var { .. } = inner_expr {
                        Ok(Expression::Unary {
                            op: unop,
                            expr: Box::new(inner_expr),
                            // NOTE: Temporary hack for arithmetic right shift.
                            sign: Signedness::Unsigned,
                            // Postfix unary operator.
                            prefix: false,
                        })
                    } else {
                        let tok_str = format!("{token:?}");
                        let line_content = ctx.src_slice(token.loc.line_span);

                        let err_msg = if let UnaryOperator::Increment = unop {
                            "lvalue required as increment operand"
                        } else {
                            "lvalue required as decrement operand"
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
                } else {
                    Ok(inner_expr)
                }
            }
            tok => {
                let tok_str = format!("{tok:?}");
                let line_content = ctx.src_slice(token.loc.line_span);

                let err_msg = if let TokenType::RParen = tok {
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

/// Parse an _AST_ argument list from the provided `Token` iterator.
///
/// # Errors
///
/// This function will return an error if an argument list could not be parsed.
fn parse_args<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Vec<Expression>> {
    let mut args = vec![];

    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        if token.ty == TokenType::RParen {
            // Not consuming the ")".
            Ok(args)
        } else {
            let expr = parse_expression(ctx, iter, 0)?;
            args.push(expr);

            while let Some(token) = iter.peek().map(Result::as_ref).transpose()?
                && let TokenType::Comma = token.ty
            {
                // Consume the "," token.
                let _ = iter.next();

                let expr = parse_expression(ctx, iter, 0)?;
                args.push(expr);
            }

            Ok(args)
        }
    } else {
        Err(fmt_err!(
            ctx.in_path.display(),
            "expected ')' or '<expr>' at end of input",
        ))
    }
}

/// Advance the `Token` iterator if it matches the expected token type.
///
/// # Errors
///
/// This function will return an error if the next token does not match the
/// expected token type provided.
fn expect_token<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    expected: TokenType,
) -> Result<()> {
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
        TokenType::Operator(OperatorKind::Increment) => Some(UnaryOperator::Increment),
        TokenType::Operator(OperatorKind::Decrement) => Some(UnaryOperator::Decrement),
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
        TokenType::Operator(OperatorKind::AssignPlus) => Some(BinaryOperator::AssignAdd),
        TokenType::Operator(OperatorKind::AssignMinus) => Some(BinaryOperator::AssignSubtract),
        TokenType::Operator(OperatorKind::AssignAsterisk) => Some(BinaryOperator::AssignMultiply),
        TokenType::Operator(OperatorKind::AssignDivision) => Some(BinaryOperator::AssignDivide),
        TokenType::Operator(OperatorKind::AssignRemainder) => Some(BinaryOperator::AssignModulo),
        TokenType::Operator(OperatorKind::AssignAmpersand) => Some(BinaryOperator::AssignBitAnd),
        TokenType::Operator(OperatorKind::AssignBitOr) => Some(BinaryOperator::AssignBitOr),
        TokenType::Operator(OperatorKind::AssignBitXor) => Some(BinaryOperator::AssignBitXor),
        TokenType::Operator(OperatorKind::AssignShiftLeft) => Some(BinaryOperator::AssignShiftLeft),
        TokenType::Operator(OperatorKind::AssignShiftRight) => {
            Some(BinaryOperator::AssignShiftRight)
        }
        TokenType::Question => Some(BinaryOperator::Conditional),
        _ => None,
    }
}
