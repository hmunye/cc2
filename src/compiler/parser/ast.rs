//! Abstract Syntax Tree
//!
//! Compiler pass that parses a stream of tokens into an abstract syntax tree
//! (_AST_).

use std::fmt;

use super::sema;

use crate::compiler::lexer::{OperatorKind, Reserved, Token, TokenType};
use crate::{Context, Result, fmt_err, fmt_token_err};

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

/// Zero-sized marker indicating _AST_ after all semantic analysis.
#[derive(Debug)]
pub struct Analyzed;

/// Abstract Syntax Tree (_AST_).
#[derive(Debug)]
pub struct AST<'a, P> {
    pub program: Vec<Declaration<'a>>,
    pub _phase: std::marker::PhantomData<P>,
}

impl<P> fmt::Display for AST<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AST Program")?;
        for decl in &self.program {
            decl.fmt_with_indent(f, 2)?;
        }

        Ok(())
    }
}

/// _AST_ declaration specifiers.
#[derive(Debug, PartialEq, Eq)]
pub struct DeclSpecs {
    pub ty: Type,
    pub storage: Option<StorageClass>,
}

impl fmt::Display for DeclSpecs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(storage) = self.storage {
            write!(f, "{:?} {:?}", storage, self.ty)
        } else {
            write!(f, "{:?}", self.ty)
        }
    }
}

/// _AST_ storage-class specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Static,
    Extern,
}

/// _AST_ type specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Func { params: usize },
}

/// _AST_ declaration.
#[derive(Debug)]
pub enum Declaration<'a> {
    Var {
        specs: DeclSpecs,
        ident: String,
        init: Option<Expression<'a>>,
        /// Identifier token.
        token: Token<'a>,
    },
    Func(Function<'a>),
}

impl Declaration<'_> {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);

        match self {
            var @ Declaration::Var { .. } => writeln!(f, "{pad}{var}"),
            Declaration::Func(func) => func.fmt_with_indent(f, indent),
        }
    }
}

impl fmt::Display for Declaration<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Declaration::Var {
                specs, ident, init, ..
            } => match init {
                Some(expr) => write!(f, "{specs} {ident:?} = {expr}"),
                None => write!(
                    f,
                    "{specs} {ident:?}{}",
                    if specs.storage == Some(StorageClass::Extern) {
                        ""
                    } else {
                        " = <uninit>"
                    }
                ),
            },
            Declaration::Func(func) => func.fmt_with_indent(f, 0),
        }
    }
}

/// _AST_ function parameter.
#[derive(Debug)]
pub struct Param<'a> {
    pub ty: Type,
    pub ident: String,
    /// Identifier token.
    pub token: Token<'a>,
}

/// _AST_ function declaration/definition.
#[derive(Debug)]
pub struct Function<'a> {
    pub specs: DeclSpecs,
    pub ident: String,
    pub params: Vec<Param<'a>>,
    pub body: Option<Block<'a>>,
    /// Identifier token.
    pub token: Token<'a>,
}

impl Function<'_> {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);
        let params = self
            .params
            .iter()
            .map(|param| param.ident.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        if let Some(body) = &self.body {
            writeln!(f, "{}{} Fn {:?}({})", pad, self.specs, self.ident, params)?;
            body.fmt_with_indent(f, indent + 2)?;
        } else {
            write!(f, "{}{} Fn {:?}({})", pad, self.specs, self.ident, params)?;
        }

        Ok(())
    }
}

/// _AST_ block.
#[derive(Debug)]
pub struct Block<'a>(pub Vec<BlockItem<'a>>);

impl Block<'_> {
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
pub enum BlockItem<'a> {
    Stmt(Statement<'a>),
    Decl(Declaration<'a>),
}

impl BlockItem<'_> {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);

        match self {
            BlockItem::Stmt(stmt) => stmt.fmt_with_indent(f, indent),
            BlockItem::Decl(decl) => {
                writeln!(f, "{pad}{decl}")
            }
        }
    }
}

/// _AST_ `for` statement initial clause.
#[derive(Debug)]
pub enum ForInit<'a> {
    Decl(Declaration<'a>),
    Expr(Option<Expression<'a>>),
}

/// _AST_ labeled statement.
#[derive(Debug)]
pub enum Labeled<'a> {
    Label {
        label: String,
        stmt: Box<Statement<'a>>,
        /// `label` identifier token.
        token: Token<'a>,
    },
    Case {
        expr: Expression<'a>,
        stmt: Box<Statement<'a>>,
        /// `case` keyword token.
        token: Token<'a>,
        jmp_label: String,
    },
    Default {
        stmt: Box<Statement<'a>>,
        /// `default` keyword token.
        token: Token<'a>,
        jmp_label: String,
    },
}

/// _AST_ `case` jmp label/expression.
#[derive(Debug)]
pub struct SwitchCase<'a> {
    pub jmp_label: String,
    pub expr: Expression<'a>,
}

/// _AST_ statement.
#[derive(Debug)]
pub enum Statement<'a> {
    Return(Expression<'a>),
    Expression(Expression<'a>),
    If {
        /// Controlling expression.
        cond: Expression<'a>,
        /// Executes when the result of `cond` is non-zero.
        then: Box<Statement<'a>>,
        /// Optional statement executes when result of `cond` is zero.
        opt_else: Option<Box<Statement<'a>>>,
    },
    Goto {
        target: String,
        /// `goto` keyword token.
        token: Token<'a>,
    },
    LabeledStatement(Labeled<'a>),
    Compound(Block<'a>),
    Break {
        jmp_label: String,
        /// `break` keyword token.
        token: Token<'a>,
    },
    Continue {
        jmp_label: String,
        /// `continue` keyword token.
        token: Token<'a>,
    },
    While {
        cond: Expression<'a>,
        stmt: Box<Statement<'a>>,
        loop_label: String,
    },
    Do {
        stmt: Box<Statement<'a>>,
        cond: Expression<'a>,
        loop_label: String,
    },
    For {
        init: Box<ForInit<'a>>,
        opt_cond: Option<Expression<'a>>,
        opt_post: Option<Expression<'a>>,
        stmt: Box<Statement<'a>>,
        loop_label: String,
    },
    Switch {
        /// Controlling expression.
        cond: Expression<'a>,
        stmt: Box<Statement<'a>>,
        /// Result of `cond` used to determine which switch case to execute.
        cases: Vec<SwitchCase<'a>>,
        /// `default` jmp label.
        default: Option<String>,
        switch_label: String,
    },
    /// Expression statement without an expression (`;`).
    Empty,
}

impl Statement<'_> {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);

        match self {
            Statement::Return(expr) => {
                writeln!(f, "{pad}Return {expr}")
            }
            Statement::Expression(expr) => {
                writeln!(f, "{pad}{expr}")
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
pub enum Expression<'a> {
    /// Integer constant (32-bit signed).
    IntConstant(i32),
    Var {
        ident: String,
        /// Identifier token.
        token: Token<'a>,
    },
    /// Unary operator applied to an expression.
    Unary {
        op: UnaryOperator,
        expr: Box<Expression<'a>>,
        // NOTE: Temporary hack for arithmetic right shift.
        sign: Signedness,
        prefix: bool,
    },
    /// Binary operator applied to two expressions.
    Binary {
        op: BinaryOperator,
        lhs: Box<Expression<'a>>,
        rhs: Box<Expression<'a>>,
        // NOTE: Temporary hack for arithmetic right shift.
        sign: Signedness,
    },
    /// Assigns an `rvalue` to an `lvalue`.
    Assignment {
        lvalue: Box<Expression<'a>>,
        rvalue: Box<Expression<'a>>,
        /// Assignment operator token.
        token: Token<'a>,
    },
    /// Ternary expression which evaluates the condition and returns the result
    /// of the `second` if true, otherwise `third`.
    Conditional {
        cond: Box<Expression<'a>>,
        second: Box<Expression<'a>>,
        third: Box<Expression<'a>>,
    },
    FuncCall {
        ident: String,
        args: Vec<Expression<'a>>,
        /// Identifier token.
        token: Token<'a>,
    },
}

impl fmt::Display for Expression<'_> {
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
    /// `!` - unary logical operator.
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
    pub const fn precedence(&self) -> u8 {
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
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid _AST_.
pub fn parse_ast<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    mut iter: std::iter::Peekable<I>,
) -> Result<AST<'a, Analyzed>> {
    let ast = parse_program(ctx, &mut iter)?;

    let ast = sema::resolve_symbols(ast, ctx)?;

    let ast = sema::resolve_types(ast, ctx)?;

    let ast = sema::resolve_labels(ast, ctx)?;

    let ast = sema::resolve_escapable_ctrl(ast, ctx)?;

    sema::resolve_switches(ast, ctx)
}

/// Parses an _AST_ program from the provided `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid program.
pub fn parse_program<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<AST<'a, Parsed>> {
    let mut decls = vec![];

    while iter.peek().is_some() {
        decls.push(parse_declaration(ctx, iter)?);
    }

    Ok(AST {
        program: decls,
        _phase: std::marker::PhantomData,
    })
}

/// Parse an _AST_ declaration from the provided `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid declaration.
fn parse_declaration<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Declaration<'a>> {
    let specs = parse_declaration_specs(ctx, iter)?;

    let (ident, token) = parse_ident(ctx, iter)?;

    let mut init = None;

    if let Some(tok) = iter.peek().map(Result::as_ref).transpose()? {
        match &tok.ty {
            // Function declaration/definition.
            TokenType::LParen => {
                let func = parse_function(ctx, iter, Some((specs, ident, token)))?;
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

    Ok(Declaration::Var {
        specs,
        ident,
        token,
        init,
    })
}

/// Parse _AST_ declaration specifiers from the provided `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form valid declaration specifiers.
fn parse_declaration_specs<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<DeclSpecs> {
    let mut decl_ty = None;
    let mut storage = None;

    while let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        match token.ty {
            TokenType::Keyword(Reserved::Int) => {
                if decl_ty.is_some() {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    return Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "two or more data types in declaration specifiers"
                    ));
                }

                decl_ty = Some(Type::Int);

                // Consume the "int" token.
                let _ = iter.next();
            }
            TokenType::Keyword(Reserved::Static | Reserved::Extern) => {
                if storage.is_some() {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    return Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "multiple storage classes in declaration specifiers"
                    ));
                }

                if token.ty == TokenType::Keyword(Reserved::Static) {
                    storage = Some(StorageClass::Static);
                } else {
                    storage = Some(StorageClass::Extern);
                }

                // Consume the storage-class token.
                let _ = iter.next();
            }
            _ => break,
        }
    }

    let Some(decl_ty) = decl_ty else {
        if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
            let tok_str = format!("{token:?}");
            let line_content = ctx.src_slice(token.loc.line_span.clone());

            return Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                line_content,
                "type specifier missing in declaration"
            ));
        } else {
            return Err(fmt_err!(
                ctx.program,
                "expected '<decl_specs>' at end of input"
            ));
        }
    };

    Ok(DeclSpecs {
        ty: decl_ty,
        storage,
    })
}

/// Parses an _AST_ function declaration/definition from the provided `Token`
/// iterator. Optionally accepts a partially parsed function signature.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid function.
fn parse_function<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    opt_sig: Option<(DeclSpecs, String, Token<'a>)>,
) -> Result<Function<'a>> {
    let (specs, ident, token) = if let Some(sig) = opt_sig {
        sig
    } else {
        let specs = parse_declaration_specs(ctx, iter)?;
        let (ident, token) = parse_ident(ctx, iter)?;

        (specs, ident, token)
    };

    expect_token(ctx, iter, TokenType::LParen)?;
    let params = parse_params(ctx, iter)?;
    expect_token(ctx, iter, TokenType::RParen)?;

    if let Some(tok) = iter.peek().map(Result::as_ref).transpose()?
        && matches!(tok.ty, TokenType::Semicolon)
    {
        // Consume the ";" token.
        let _ = iter.next();

        // Function declaration.
        return Ok(Function {
            specs,
            ident,
            params,
            body: None,
            token,
        });
    }

    let body = parse_block(ctx, iter)?;

    // Function definition.
    Ok(Function {
        specs,
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
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid parameter list.
fn parse_params<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Vec<Param<'a>>> {
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
                    && matches!(token.ty, TokenType::Comma)
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
            TokenType::Keyword(Reserved::Static | Reserved::Extern) => {
                let tok_str = format!("{token:?}");
                let line_content = ctx.src_slice(token.loc.line_span.clone());

                Err(fmt_token_err!(
                    token.loc.file_path.display(),
                    token.loc.line,
                    token.loc.col,
                    tok_str,
                    tok_str.len() - 1,
                    line_content,
                    "storage class specified for parameter"
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
                    "unknown type name '{tok_str}'"
                ))
            }
        }
    } else {
        Err(fmt_err!(ctx.program, "expected '<params>' at end of input"))
    }
}

/// Parse an _AST_ block from the provided `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid block.
fn parse_block<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Block<'a>> {
    expect_token(ctx, iter, TokenType::LBrace)?;

    let mut block = vec![];

    while let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        if token.ty == TokenType::RBrace {
            break;
        }

        block.push(parse_block_item(ctx, iter)?);
    }

    expect_token(ctx, iter, TokenType::RBrace)?;

    Ok(Block(block))
}

/// Parse an _AST_ block item from the provided `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid block item.
fn parse_block_item<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<BlockItem<'a>> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        match token.ty {
            // Parse this as a declaration (starts with a type/storage-class
            // specifier).
            TokenType::Keyword(Reserved::Int | Reserved::Static | Reserved::Extern) => {
                Ok(BlockItem::Decl(parse_declaration(ctx, iter)?))
            }
            // Parse this as a statement.
            _ => Ok(BlockItem::Stmt(parse_statement(ctx, iter)?)),
        }
    } else {
        Err(fmt_err!(ctx.program, "expected '<block>' at end of input"))
    }
}

/// Parse an _AST_ `for` initial clause from the provided `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid `for` initial clause.
fn parse_for_init<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<ForInit<'a>> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        // Parse this as a declaration (starts with a type/storage-class
        // specifier).
        if matches!(
            token.ty,
            TokenType::Keyword(Reserved::Int | Reserved::Static | Reserved::Extern)
        ) {
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
                        "declaration of non-variable '{tok_str}' in 'for' loop initial declaration"
                    ))
                }
                var @ Declaration::Var { .. } => Ok(ForInit::Decl(var)),
            }
        } else {
            // Parse this as an optional expression.
            let opt_expr = parse_opt_expression(ctx, iter, TokenType::Semicolon)?;
            expect_token(ctx, iter, TokenType::Semicolon)?;

            Ok(ForInit::Expr(opt_expr))
        }
    } else {
        Err(fmt_err!(
            ctx.program,
            "expected '<for_init>' at end of input"
        ))
    }
}

/// Parse an _AST_ statement from the provided `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid statement.
fn parse_statement<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Statement<'a>> {
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

                let opt_else = if let Some(token) = iter.peek().map(Result::as_ref).transpose()?
                    && let TokenType::Keyword(kw) = token.ty
                    && matches!(kw, Reserved::Else)
                {
                    // Consume the "else" token.
                    let _ = iter.next();
                    Some(Box::new(parse_statement(ctx, iter)?))
                } else {
                    None
                };

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
                        "case label does not reduce to an integer constant (currently only support integer literals)"
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
                    && token.ty == TokenType::Colon
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
                            "expected ';' before '{tok_str}' token"
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
            ctx.program,
            "expected '<statement>' at end of input"
        ))
    }
}

/// Parse an _AST_ identifier from the provided `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid identifier.
fn parse_ident<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<(String, Token<'a>)> {
    if let Some(token) = iter.next() {
        let token = token?;

        match token.ty {
            TokenType::Ident(s) => Ok((s.to_string(), token)),
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
                    "expected identifier"
                ))
            }
        }
    } else {
        Err(fmt_err!(ctx.program, "expected '<ident>' at end of input"))
    }
}

/// Parse an _AST_ expression from the provided `Token` iterator using
/// `precedence climbing`.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid expression.
fn parse_expression<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    min_precedence: u8,
) -> Result<Expression<'a>> {
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
                    _ => unreachable!(
                        "non-assignment expression operators should not reach this match arm"
                    ),
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
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form an optional expression.
fn parse_opt_expression<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    end_token: TokenType<'_>,
) -> Result<Option<Expression<'a>>> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        if token.ty == end_token {
            // Not consuming the `end_token`.
            Ok(None)
        } else {
            Ok(Some(parse_expression(ctx, iter, 0)?))
        }
    } else {
        Err(fmt_err!(
            ctx.program,
            "expected '{end_token}' or '<expr>' at end of input"
        ))
    }
}

/// Parse an _AST_ expression or sub-expression (factor) from the provided
/// `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid factor.
fn parse_factor<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Expression<'a>> {
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

                    let err_msg = if tok.ty == TokenType::Operator(OperatorKind::Increment) {
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
                        "{err_msg}"
                    ))
                } else {
                    Ok(Expression::IntConstant(v))
                }
            }
            TokenType::Ident(s) => {
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
                                    ident: s.to_string(),
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
                                ident: s.to_string(),
                                args,
                                token,
                            });
                        }
                        _ => {}
                    }
                }

                Ok(Expression::Var {
                    ident: s.to_string(),
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
                let sign = if matches!(unop, UnaryOperator::Negate) {
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

                    let err_msg = if matches!(unop, UnaryOperator::Increment) {
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
                        "{err_msg}"
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

                        let err_msg = if matches!(unop, UnaryOperator::Increment) {
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
                            "{err_msg}"
                        ))
                    }
                } else {
                    Ok(inner_expr)
                }
            }
            tok => {
                let tok_str = format!("{tok:?}");
                let line_content = ctx.src_slice(token.loc.line_span);

                let err_msg = if tok == TokenType::RParen {
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
                    "{err_msg}"
                ))
            }
        }
    } else {
        Err(fmt_err!(ctx.program, "expected '<factor>' at end of input"))
    }
}

/// Parse an _AST_ argument list from the provided `Token` iterator.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the tokens cannot
/// form a valid argument list.
fn parse_args<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Vec<Expression<'a>>> {
    let mut args = vec![];

    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        if token.ty != TokenType::RParen {
            let expr = parse_expression(ctx, iter, 0)?;
            args.push(expr);

            while let Some(token) = iter.peek().map(Result::as_ref).transpose()?
                && token.ty == TokenType::Comma
            {
                // Consume the "," token.
                let _ = iter.next();

                let expr = parse_expression(ctx, iter, 0)?;
                args.push(expr);
            }
        }

        Ok(args)
    } else {
        Err(fmt_err!(
            ctx.program,
            "expected ')' or '<expr>' at end of input"
        ))
    }
}

/// Advance the `Token` iterator if it matches the expected token type.
///
/// # Errors
///
/// Returns an error if an invalid token is encountered or if the next token
/// does not match the expected token type provided.
fn expect_token<'a, I: Iterator<Item = Result<Token<'a>>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
    expected: TokenType<'_>,
) -> Result<()> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        if token.ty == expected {
            // Consume the peeked token.
            let _ = iter.next();
            return Ok(());
        } else {
            let tok_str = format!("{token:?}");
            let line_content = ctx.src_slice(token.loc.line_span.clone());

            return Err(fmt_token_err!(
                token.loc.file_path.display(),
                token.loc.line,
                token.loc.col,
                tok_str,
                tok_str.len() - 1,
                line_content,
                "expected '{expected:?}', but found '{tok_str}'"
            ));
        }
    }

    Err(fmt_err!(
        ctx.program,
        "expected '{:?}' at end of input",
        expected
    ))
}

/// Returns the conversion of `TokenType` to `UnaryOperator`, or `None` if the
/// token type is not a unary operator.
const fn ty_to_unop(ty: &TokenType<'_>) -> Option<UnaryOperator> {
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
const fn ty_to_binop(ty: &TokenType<'_>) -> Option<BinaryOperator> {
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
