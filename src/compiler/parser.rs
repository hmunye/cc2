//! Syntax Analysis
//!
//! Compiler pass that parses a stream of tokens into an abstract syntax tree
//! (_AST_).

use std::collections::{HashMap, HashSet};
use std::{fmt, process};

use crate::compiler::Result;
use crate::compiler::lexer::{OperatorKind, Token, TokenType};
use crate::{Context, fmt_err, fmt_token_err, report_err};

type Ident = String;

/// Helper to track the current scope for symbol resolution.
#[repr(transparent)]
struct Scope(usize);

impl Scope {
    #[inline]
    const fn new(scope: usize) -> Self {
        Scope(scope)
    }

    #[inline]
    const fn current_scope(&self) -> usize {
        self.0
    }

    #[inline]
    const fn enter_scope(&mut self) {
        self.0 += 1;
    }

    #[inline]
    const fn exit_scope(&mut self) {
        self.0 = self.0.saturating_sub(1);
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Helper for _AST_ to perform semantic analysis on symbols.
#[derive(Default)]
struct SymbolResolver {
    // key = (symbol, scope), value = resolved identifier
    symbol_map: HashMap<(Ident, usize), Ident>,
    // Used to track current scope.
    scope: Scope,
}

impl SymbolResolver {
    /// Returns a new temporary variable identifier, appending the current
    /// scope to the provided prefix.
    #[inline]
    fn new_tmp(&mut self, prefix: &str) -> Ident {
        // The `@` in variable identifiers guarantees they won’t conflict
        // with user-defined identifiers, since the _C_ standard forbids `@` in
        // identifiers.
        format!("{prefix}@{}", self.scope.current_scope())
    }

    /// Returns `true` if the given `symbol` has already been declared in the
    /// current scope.
    #[inline]
    fn is_redeclaration(&self, symbol: &str) -> bool {
        self.symbol_map
            .contains_key(&(symbol.to_string(), self.scope.current_scope()))
    }

    /// Returns a unique identifier for the given `symbol`, recording the
    /// declaration for the current scope.
    fn declare_symbol(&mut self, symbol: &str) -> Ident {
        let resolved_ident = self.new_tmp(symbol);

        let res = self.symbol_map.insert(
            (symbol.to_string(), self.scope.current_scope()),
            resolved_ident.clone(),
        );

        debug_assert!(res.is_none());

        resolved_ident
    }

    /// Returns the unique identifier for a given `symbol`, searching the
    /// current scope and all outer scopes (up to the function scope), or `None`
    /// if no existing declaration could be found.
    fn resolve_symbol(&self, symbol: &str) -> Option<Ident> {
        let mut resolved_ident = None;

        // Starts at the current scope then searches all outer scopes until
        // reaching the function scope (0).
        let scopes = (0..=self.scope.current_scope()).rev();

        for scope in scopes {
            if let Some(ident) = self.symbol_map.get(&(symbol.to_string(), scope)) {
                resolved_ident = Some(ident.clone());
                break;
            }
        }

        resolved_ident
    }
}

/// Helper for _AST_ to perform semantic analysis on label/`goto` statements.
#[derive(Default)]
struct LabelResolver<'a> {
    // key = label
    labels: HashSet<&'a str>,
    // Collected label–token pairs for every `goto` statement within a function
    // scope. After recording all local labels, a later pass verifies that each
    // target label exists within the same scope.
    pending_gotos: Vec<(&'a str, &'a Token)>,
}

impl<'a> LabelResolver<'a> {
    /// Returns `true` if the label was not encountered in the current function
    /// scope and records it as seen.
    #[inline]
    fn mark_label(&mut self, label: &'a str) -> bool {
        // Labels live in a different namespace from ordinary
        // identifiers (variables, functions, types, etc.)
        // within the same function scope, so they are collected
        // separately.
        !self.labels.insert(label)
    }

    /// Records a `goto` statement's contents so they can be validated after
    /// processing labels.
    #[inline]
    fn mark_goto(&mut self, pair: (&'a str, &'a Token)) {
        self.pending_gotos.push(pair);
    }

    /// Validates all pending `goto` statements and ensures they point to valid
    /// targets within the current function scope. On `Err`, returns the
    /// (label, token) pair of the missing target.
    fn check_gotos(&self) -> core::result::Result<(), (&'a str, &'a Token)> {
        for (label, token) in &self.pending_gotos {
            if !self.labels.contains(label) {
                return Err((label, token));
            }
        }

        Ok(())
    }
}

/// Abstract Syntax Tree (_AST_).
pub enum AST {
    /// Function that represent the structure of the program.
    Program(Function),
}

impl<'a> AST {
    /// Assigns a unique identifier to every variable and performs semantic
    /// checks (e.g., duplicate definitions, undeclared references).
    fn resolve_variables(
        &mut self,
        ctx: &Context<'_>,
        resolver: &mut SymbolResolver,
    ) -> Result<()> {
        fn resolve_declaration(
            decl: &Declaration,
            ctx: &Context<'_>,
            resolver: &mut SymbolResolver,
        ) -> Result<Declaration> {
            if resolver.is_redeclaration(&decl.ident) {
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

            let resolved_ident = resolver.declare_symbol(&decl.ident);

            let mut opt_init = None;
            if let Some(init) = &decl.init {
                opt_init = Some(resolve_expression(init, ctx, resolver)?);
            }

            Ok(Declaration {
                ident: resolved_ident,
                token: decl.token.clone(),
                init: opt_init,
            })
        }

        fn resolve_statement(
            stmt: &mut Statement,
            ctx: &Context<'_>,
            resolver: &mut SymbolResolver,
        ) -> Result<Statement> {
            match stmt {
                Statement::Return(expr) => {
                    Ok(Statement::Return(resolve_expression(expr, ctx, resolver)?))
                }
                Statement::Expression(expr) => Ok(Statement::Expression(resolve_expression(
                    expr, ctx, resolver,
                )?)),
                Statement::If {
                    cond,
                    then,
                    opt_else,
                } => {
                    let resolved_cond = resolve_expression(cond, ctx, resolver)?;
                    let resolved_then = Box::new(resolve_statement(then, ctx, resolver)?);

                    let mut resolved_opt_else = None;
                    if let Some(stmt) = opt_else {
                        resolved_opt_else = Some(Box::new(resolve_statement(stmt, ctx, resolver)?));
                    }

                    Ok(Statement::If {
                        cond: resolved_cond,
                        then: resolved_then,
                        opt_else: resolved_opt_else,
                    })
                }
                Statement::Goto((target, token)) => {
                    Ok(Statement::Goto((target.clone(), token.clone())))
                }
                Statement::Labeled { label, token, stmt } => Ok(Statement::Labeled {
                    label: label.clone(),
                    token: token.clone(),
                    stmt: Box::new(resolve_statement(stmt, ctx, resolver)?),
                }),
                Statement::Compound(block) => {
                    resolver.scope.enter_scope();

                    resolve_block(block, ctx, resolver)?;

                    Ok(Statement::Compound(std::mem::replace(block, Block(vec![]))))
                }
                Statement::Empty => Ok(Statement::Empty),
            }
        }

        fn resolve_expression(
            expr: &Expression,
            ctx: &Context<'_>,
            resolver: &mut SymbolResolver,
        ) -> Result<Expression> {
            match expr {
                Expression::Assignment {
                    lvalue,
                    rvalue,
                    token,
                } => match **lvalue {
                    Expression::Var(_) => {
                        let lvalue = resolve_expression(lvalue, ctx, resolver)?;
                        let rvalue = resolve_expression(rvalue, ctx, resolver)?;

                        Ok(Expression::Assignment {
                            lvalue: Box::new(lvalue),
                            rvalue: Box::new(rvalue),
                            token: token.clone(),
                        })
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
                Expression::Var((v, token)) => {
                    if let Some(ident) = resolver.resolve_symbol(v) {
                        // Use the unique variable identifier mapped from the
                        // original symbol.
                        Ok(Expression::Var((ident, token.clone())))
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
                            "'{tok_str}' undeclared",
                        ))
                    }
                }
                Expression::IntConstant(i) => Ok(Expression::IntConstant(*i)),
                Expression::Unary {
                    op,
                    expr,
                    sign,
                    prefix,
                } => {
                    let expr = resolve_expression(expr, ctx, resolver)?;

                    Ok(Expression::Unary {
                        op: *op,
                        expr: Box::new(expr),
                        sign: *sign,
                        prefix: *prefix,
                    })
                }
                Expression::Binary { op, lhs, rhs, sign } => {
                    let lhs = resolve_expression(lhs, ctx, resolver)?;
                    let rhs = resolve_expression(rhs, ctx, resolver)?;

                    Ok(Expression::Binary {
                        op: *op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                        sign: *sign,
                    })
                }
                Expression::Conditional(lhs, mid, rhs) => {
                    let lhs = resolve_expression(lhs, ctx, resolver)?;
                    let mid = resolve_expression(mid, ctx, resolver)?;
                    let rhs = resolve_expression(rhs, ctx, resolver)?;

                    Ok(Expression::Conditional(
                        Box::new(lhs),
                        Box::new(mid),
                        Box::new(rhs),
                    ))
                }
            }
        }

        fn resolve_block(
            block: &mut Block,
            ctx: &Context<'_>,
            resolver: &mut SymbolResolver,
        ) -> Result<()> {
            for block_item in &mut block.0 {
                match block_item {
                    BlockItem::Stmt(stmt) => *stmt = resolve_statement(stmt, ctx, resolver)?,
                    BlockItem::Decl(decl) => *decl = resolve_declaration(decl, ctx, resolver)?,
                }
            }

            resolver.scope.exit_scope();

            Ok(())
        }

        match self {
            AST::Program(func) => resolve_block(&mut func.body, ctx, resolver)?,
        }

        Ok(())
    }

    /// Ensures every label declared is unique within it's function scope
    /// and performs semantic checks (e.g., missing `goto` targets, unreachable
    /// labels).
    fn resolve_labels(&'a self, ctx: &Context<'_>, resolver: &mut LabelResolver<'a>) -> Result<()> {
        fn resolve_block<'a>(
            block: &'a Block,
            ctx: &Context<'_>,
            resolver: &mut LabelResolver<'a>,
        ) -> Result<()> {
            for block_item in &block.0 {
                if let BlockItem::Stmt(stmt) = block_item {
                    resolve_statement_labels(stmt, ctx, resolver)?;
                }
            }

            Ok(())
        }

        fn resolve_statement_labels<'a>(
            stmt: &'a Statement,
            ctx: &Context<'_>,
            resolver: &mut LabelResolver<'a>,
        ) -> Result<()> {
            match stmt {
                Statement::If { then, opt_else, .. } => {
                    resolve_statement_labels(then, ctx, resolver)?;

                    if let Some(else_stmt) = opt_else {
                        resolve_statement_labels(else_stmt, ctx, resolver)?;
                    }
                }
                Statement::Labeled { label, token, stmt } => {
                    if !resolver.mark_label(label) {
                        let tok_str = format!("{token:?}");
                        let line_content = ctx.src_slice(token.loc.line_span.clone());

                        return Err(fmt_token_err!(
                            token.loc.file_path.display(),
                            token.loc.line,
                            token.loc.col,
                            tok_str,
                            tok_str.len() - 1,
                            line_content,
                            "duplicate label '{tok_str}'",
                        ));
                    }

                    resolve_statement_labels(stmt, ctx, resolver)?;
                }
                Statement::Compound(block) => resolve_block(block, ctx, resolver)?,
                Statement::Goto((label, token)) => resolver.mark_goto((label, token)),
                _ => {}
            }

            Ok(())
        }

        match self {
            AST::Program(func) => {
                // Collect and validate all labels within the function in the
                // first pass.
                resolve_block(&func.body, ctx, resolver)?;

                // Second pass ensures all `goto` statements point to a valid
                // target within the same function scope.
                if let Err((label, token)) = resolver.check_gotos() {
                    let tok_str = format!("{token:?}");
                    let line_content = ctx.src_slice(token.loc.line_span.clone());

                    return Err(fmt_token_err!(
                        token.loc.file_path.display(),
                        token.loc.line,
                        token.loc.col,
                        tok_str,
                        tok_str.len() - 1,
                        line_content,
                        "label '{label}' used but not defined",
                    ));
                }
            }
        }

        Ok(())
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
    pub body: Block,
}

/// _AST_ block.
#[derive(Debug)]
pub struct Block(pub Vec<BlockItem>);

/// _AST_ block item.
#[derive(Debug)]
pub enum BlockItem {
    Stmt(Statement),
    Decl(Declaration),
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
    Goto((Ident, Token)),
    // Labeled statement.
    Labeled {
        label: Ident,
        token: Token,
        stmt: Box<Statement>,
    },
    // Compound statement.
    Compound(Block),
    // Expression statement without an expression (';').
    Empty,
}

/// _AST_ declaration.
#[derive(Debug)]
pub struct Declaration {
    /// Identifier of the variable.
    pub ident: Ident,
    /// Token of the identifier.
    pub token: Token,
    /// Optional initializer.
    pub init: Option<Expression>,
}

/// _AST_ expression.
#[derive(Debug, Clone)]
pub enum Expression {
    /// Constant `int` value (32-bit).
    IntConstant(i32),
    /// Variable expression with identifier token.
    Var((Ident, Token)),
    /// Unary operator applied to an expression.
    Unary {
        op: UnaryOperator,
        expr: Box<Expression>,
        sign: Signedness,
        prefix: bool,
    },
    /// Binary operator applied to two expressions.
    Binary {
        op: BinaryOperator,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
        sign: Signedness,
    },
    /// Assigns an `rvalue` to an `lvalue`, with assignment operator token.
    Assignment {
        lvalue: Box<Expression>,
        rvalue: Box<Expression>,
        token: Token,
    },
    /// Ternary expression which evaluates the first expression and returns the
    /// result of the second if true, otherwise the third.
    Conditional(Box<Expression>, Box<Expression>, Box<Expression>),
}

/// _AST_ unary operators.
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

/// _AST_ binary operators.
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
pub fn parse_program<I: Iterator<Item = Result<Token>>>(
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

    // Pass 1 - Variable resolution.
    let mut sym_resolver: SymbolResolver = Default::default();
    ast.resolve_variables(ctx, &mut sym_resolver)
        .unwrap_or_else(|err| {
            if cfg!(test) {
                panic!("{err}");
            }

            eprintln!("{err}");
            process::exit(1);
        });

    // Pass 2 - Label/`goto` resolution.
    let mut lbl_resolver: LabelResolver<'_> = Default::default();
    ast.resolve_labels(ctx, &mut lbl_resolver)
        .unwrap_or_else(|err| {
            if cfg!(test) {
                panic!("{err}");
            }

            eprintln!("{err}");
            process::exit(1);
        });

    ast
}

/// Parses an _AST_ function definition from the provided `Token` iterator.
fn parse_function<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Function> {
    // NOTE: Currently only support the `int` type.
    expect_token(ctx, iter, TokenType::Keyword("int".into()))?;
    let (ident, _) = parse_ident(ctx, iter)?;

    expect_token(ctx, iter, TokenType::LParen)?;

    // NOTE: Currently not processing function parameters.
    expect_token(ctx, iter, TokenType::Keyword("void".into()))?;

    expect_token(ctx, iter, TokenType::RParen)?;

    let block = parse_block(ctx, iter)?;

    Ok(Function { ident, body: block })
}

/// Parse an _AST_ block from the provided `Token` iterator.
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
fn parse_block_item<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<BlockItem> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        match token.ty {
            // Parse this as a declaration (starts with a type).
            TokenType::Keyword(ref s) if s == "int" => {
                Ok(BlockItem::Decl(parse_declaration(ctx, iter)?))
            }
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

/// Parse an _AST_ declaration from the provided `Token` iterator.
fn parse_declaration<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Declaration> {
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
fn parse_statement<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<Statement> {
    if let Some(token) = iter.peek().map(Result::as_ref).transpose()? {
        match token.ty {
            TokenType::Keyword(ref s) if s == "return" => {
                // Consume the "return" token.
                let _ = iter.next();

                let expr = parse_expression(ctx, iter, 0)?;
                expect_token(ctx, iter, TokenType::Semicolon)?;

                Ok(Statement::Return(expr))
            }
            TokenType::Keyword(ref s) if s == "if" => {
                // Consume the "if" token.
                let _ = iter.next();

                expect_token(ctx, iter, TokenType::LParen)?;
                let expr = parse_expression(ctx, iter, 0)?;
                expect_token(ctx, iter, TokenType::RParen)?;

                let stmt = parse_statement(ctx, iter)?;

                let mut opt_else = None;

                if let Some(token) = iter.peek().map(Result::as_ref).transpose()?
                    && let TokenType::Keyword(ref s) = token.ty
                    && s == "else"
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
            TokenType::Keyword(ref s) if s == "goto" => {
                // Consume the "goto" token.
                let token = iter
                    .next()
                    .expect("next token should be present")
                    .expect("next token should be ok");

                let (target, _) = parse_ident(ctx, iter)?;

                expect_token(ctx, iter, TokenType::Semicolon)?;

                Ok(Statement::Goto((target, token)))
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

                // A labeled statement encountered - next token is colon (`:`).
                if let Some(token) = iter.peek().map(Result::as_ref).transpose()?
                    && let TokenType::Colon = token.ty
                {
                    if let Expression::Var((ident, token)) = expr {
                        // Consume the ":" token.
                        let _ = iter.next();

                        let stmt = parse_statement(ctx, iter)?;

                        Ok(Statement::Labeled {
                            label: ident,
                            token,
                            stmt: Box::new(stmt),
                        })
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
fn parse_ident<I: Iterator<Item = Result<Token>>>(
    ctx: &Context<'_>,
    iter: &mut std::iter::Peekable<I>,
) -> Result<(Ident, Token)> {
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
                let mid = parse_expression(ctx, iter, 0)?;

                expect_token(ctx, iter, TokenType::Colon)?;

                // This ensures we can handle right-associative operators since
                // operators of the same precedence still are evaluated
                // together.
                let rhs = parse_expression(ctx, iter, binop.precedence())?;

                lhs = Expression::Conditional(Box::new(lhs), Box::new(mid), Box::new(rhs));
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

/// Parse an _AST_ expression or sub-expression (factor) from the provided
/// `Token` iterator.
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
                if let Some(tok) = iter.peek().map(Result::as_ref).transpose()?
                    && let TokenType::Operator(OperatorKind::Increment | OperatorKind::Decrement) =
                        tok.ty
                {
                    let unop = ty_to_unop(&tok.ty)
                        .expect("expected postfix increment or decrement when parsing factor");

                    // Consume the postfix increment/decrement operator.
                    let _ = iter.next();

                    Ok(Expression::Unary {
                        op: unop,
                        expr: Box::new(Expression::Var((s.clone(), token))),
                        sign: Signedness::Unsigned,
                        // Postfix unary operator.
                        prefix: false,
                    })
                } else {
                    Ok(Expression::Var((s.clone(), token)))
                }
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
                    && !matches!(inner_fct, Expression::Var(_))
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

                Ok(inner_expr)
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

/// Advance the `Token` iterator if it matches the expected token type.
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
