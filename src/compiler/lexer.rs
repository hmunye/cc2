//! Lexical Analysis
//!
//! Compiler pass that tokenizes _C_ source code, producing a sequence of
//! tokens.

use std::path::Path;
use std::{fmt, process};

use crate::{Context, report_token_err};

/// Reserved tokens defined by the _C_ language standard (_C17_).
const KEYWORDS: [&str; 3] = ["int", "void", "return"];

/// Types of operators.
#[derive(PartialEq, Clone, Copy)]
pub enum OperatorKind {
    /// `~` bitwise NOT operator.
    BitNot,
    /// `-` subtraction or negation operator.
    Minus,
    /// `+` addition or unary positive operator.
    Plus,
    /// `*` multiplication or dereference operator.
    Asterisk,
    /// `/` division operator.
    Division,
    /// `%` remainder operator.
    Modulo,
    /// `++` increment operator.
    Increment,
    /// `--` decrement operator.
    Decrement,
    /// `&` bitwise AND or address-of operator.
    Ampersand,
    /// `|` bitwise OR operator.
    BitOr,
    /// `^` bitwise XOR operator.
    BitXor,
    /// `<` less-than relational operator.
    LessThan,
    /// `>` greater-than relational operator.
    GreaterThan,
    /// `<=` less-than-or-equal relational operator.
    LessThanEq,
    /// `>=` greater-than-or-equal relational operator.
    GreaterThanEq,
    /// `<<` bitwise left-shift operator.
    ShiftLeft,
    /// `>>` bitwise right-shift operator.
    ShiftRight,
    /// `!` logical NOT operator.
    LogNot,
    /// `&&` logical AND operator.
    LogAnd,
    /// `||` logical OR operator.
    LogOr,
    /// `==` equal-to relational operator.
    Eq,
    /// `!=` not-equal relational operator.
    NotEq,
}

impl fmt::Display for OperatorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperatorKind::BitNot => write!(f, "op('~')"),
            OperatorKind::Minus => write!(f, "op('-')"),
            OperatorKind::Plus => write!(f, "op('+')"),
            OperatorKind::Asterisk => write!(f, "op('*')"),
            OperatorKind::Division => write!(f, "op('/')"),
            OperatorKind::Modulo => write!(f, "op('-')"),
            OperatorKind::Increment => write!(f, "op('++')"),
            OperatorKind::Decrement => write!(f, "op('--')"),
            OperatorKind::Ampersand => write!(f, "op('&')"),
            OperatorKind::BitOr => write!(f, "op('|')"),
            OperatorKind::BitXor => write!(f, "op('^')"),
            OperatorKind::LessThan => write!(f, "op('<')"),
            OperatorKind::GreaterThan => write!(f, "op('>')"),
            OperatorKind::LessThanEq => write!(f, "op('<=')"),
            OperatorKind::GreaterThanEq => write!(f, "op('>=')"),
            OperatorKind::ShiftLeft => write!(f, "op('<<')"),
            OperatorKind::ShiftRight => write!(f, "op('>>')"),
            OperatorKind::LogNot => write!(f, "op('!')"),
            OperatorKind::LogAnd => write!(f, "op('&&')"),
            OperatorKind::LogOr => write!(f, "op('||')"),
            OperatorKind::Eq => write!(f, "op('==')"),
            OperatorKind::NotEq => write!(f, "op('!=')"),
        }
    }
}

impl fmt::Debug for OperatorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperatorKind::BitNot => write!(f, "~"),
            OperatorKind::Minus => write!(f, "-"),
            OperatorKind::Plus => write!(f, "+"),
            OperatorKind::Asterisk => write!(f, "*"),
            OperatorKind::Division => write!(f, "/"),
            OperatorKind::Modulo => write!(f, "-"),
            OperatorKind::Increment => write!(f, "++"),
            OperatorKind::Decrement => write!(f, "--"),
            OperatorKind::Ampersand => write!(f, "&"),
            OperatorKind::BitOr => write!(f, "|"),
            OperatorKind::BitXor => write!(f, "^"),
            OperatorKind::LessThan => write!(f, "<"),
            OperatorKind::GreaterThan => write!(f, ">"),
            OperatorKind::LessThanEq => write!(f, "<="),
            OperatorKind::GreaterThanEq => write!(f, ">="),
            OperatorKind::ShiftLeft => write!(f, "<<"),
            OperatorKind::ShiftRight => write!(f, ">>"),
            OperatorKind::LogNot => write!(f, "!"),
            OperatorKind::LogAnd => write!(f, "&&"),
            OperatorKind::LogOr => write!(f, "||"),
            OperatorKind::Eq => write!(f, "=="),
            OperatorKind::NotEq => write!(f, "!="),
        }
    }
}

/// Types of lexical elements.
#[derive(PartialEq, Clone)]
#[allow(missing_docs)]
pub enum TokenType {
    Keyword(String),
    Ident(String),
    ConstantInt(i32),
    Operator(OperatorKind),
    ParenOpen,
    ParenClose,
    BraceOpen,
    BraceClose,
    Semicolon,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenType::Keyword(s) => write!(f, "keyword({s:?})"),
            TokenType::Ident(i) => write!(f, "ident({i:?})"),
            TokenType::ConstantInt(v) => write!(f, "int(\"{v}\")"),
            TokenType::Operator(op) => fmt::Display::fmt(op, f),
            TokenType::ParenOpen => write!(f, "'('"),
            TokenType::ParenClose => write!(f, "')'"),
            TokenType::BraceOpen => write!(f, "'{{'"),
            TokenType::BraceClose => write!(f, "'}}'"),
            TokenType::Semicolon => write!(f, "';'"),
        }
    }
}

impl fmt::Debug for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenType::Keyword(s) => write!(f, "{s}"),
            TokenType::Ident(i) => write!(f, "{i}"),
            TokenType::ConstantInt(v) => write!(f, "{v}"),
            TokenType::Operator(op) => fmt::Debug::fmt(op, f),
            TokenType::ParenOpen => write!(f, "("),
            TokenType::ParenClose => write!(f, ")"),
            TokenType::BraceOpen => write!(f, "{{"),
            TokenType::BraceClose => write!(f, "}}"),
            TokenType::Semicolon => write!(f, ";"),
        }
    }
}

/// Location of a processed `Token`.
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Location<'a> {
    pub line_content: &'a str,
    pub file_path: &'static Path,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for Location<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file_path.display(), self.line, self.col)
    }
}

/// Minimal lexical element.
#[allow(missing_docs)]
pub struct Token<'a> {
    pub ty: TokenType,
    pub loc: Location<'a>,
}

impl fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}\t    {}", self.loc, self.ty)
    }
}

impl fmt::Debug for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.ty)
    }
}

/// Stores the ordered sequence of tokens extracted from a _C_ translation unit.
#[derive(Debug, Default)]
pub struct Lexer<'a> {
    tokens: std::collections::VecDeque<Token<'a>>,
    src: &'a [u8],
    cur: usize,
    // Index of the beginning of a newline (used to calculate the current
    // column).
    bol: usize,
    line: usize,
}

impl<'a> Lexer<'a> {
    /// Returns a new, empty `Lexer`.   
    pub fn new(src: &'a [u8]) -> Self {
        Self {
            tokens: Default::default(),
            src,
            cur: 0,
            bol: 0,
            line: 1,
        }
    }

    /// Produces a sequence of tokens internally given _C_ source code. [Exits]
    /// on error with non-zero status.
    ///
    /// Does **not** support universal character names (only _ASCII_).
    ///
    /// [Exits]: std::process::exit
    pub fn lex(&mut self, ctx: &Context<'_>) {
        let mut fl_captured = false;
        let mut line_content = "";

        while self.has_next() {
            // Ensures the first line has the correct column count.
            let col = if self.line == 1 {
                self.cur - self.bol + 1
            } else {
                self.cur - self.bol
            };

            // Capture the first line's contents (must be done separately since
            // no '\n' has been encountered yet).
            if !fl_captured {
                let mut i = 0;
                while i < self.src.len() && self.src[i] != b'\n' {
                    i += 1;
                }

                line_content = std::str::from_utf8(&self.src[self.bol..i])
                    .expect("ASCII bytes should be valid UTF-8");

                fl_captured = true;
            }

            match self.first() {
                b'\n' => {
                    self.bol = self.cur;
                    self.cur += 1;
                    self.line += 1;

                    // Capture the new line's contents.
                    let mut i = self.cur;
                    while i < self.src.len() && self.src[i] != b'\n' {
                        i += 1;
                    }

                    line_content = std::str::from_utf8(&self.src[self.bol + 1..i])
                        .expect("ASCII bytes should be valid UTF-8");
                }
                b if b.is_ascii_whitespace() => {
                    self.cur += 1;
                }
                // NOTE: Currently only handling integer constants without
                // suffix.
                b'0'..=b'9' => {
                    let token_start = self.cur;

                    while self.has_next() && self.first().is_ascii_digit() {
                        self.cur += 1;
                    }

                    // Identifier are not allowed to begin with digits.
                    //
                    // NOTE: Currently don't allow '_' as a separator.
                    if self.first().is_ascii_alphabetic() || self.first() == b'_' {
                        let const_len = self.cur - token_start;
                        let const_end = self.cur;

                        // Continue consuming the invalid suffix.
                        while self.has_next()
                            && (self.first().is_ascii_alphabetic() || self.first() == b'_')
                        {
                            self.cur += 1;
                        }

                        let suffix = std::str::from_utf8(&self.src[const_end..self.cur])
                            .expect("ASCII bytes should be valid UTF-8");

                        if cfg!(test) {
                            panic!("invalid suffix '{suffix}' on integer constant");
                        }

                        report_token_err!(
                            ctx.in_path.display(),
                            self.line,
                            col + const_len,
                            suffix,
                            suffix.len() - 1,
                            line_content,
                            "invalid suffix '{suffix}' on integer constant",
                        );
                        process::exit(1);
                    }

                    let token = std::str::from_utf8(&self.src[token_start..self.cur])
                        .expect("ASCII bytes should be valid UTF-8");

                    let Ok(integer) = token.parse::<i32>() else {
                        if cfg!(test) {
                            panic!("integer constant is too large for its type");
                        }

                        report_token_err!(
                            ctx.in_path.display(),
                            self.line,
                            col,
                            token,
                            token.len() - 1,
                            line_content,
                            "integer constant is too large for its type (32-bit signed)",
                        );
                        process::exit(1);
                    };

                    self.add_token(
                        TokenType::ConstantInt(integer),
                        line_content,
                        ctx.in_path,
                        col,
                    );
                }
                // Handle identifier or keyword (allowed to begin with '_').
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                    let token_start = self.cur;

                    while self.has_next()
                        && (self.first().is_ascii_alphanumeric() || self.first() == b'_')
                    {
                        self.cur += 1;
                    }

                    let token = std::str::from_utf8(&self.src[token_start..self.cur])
                        .expect("ASCII bytes should be valid UTF-8");

                    if KEYWORDS.contains(&token) {
                        self.add_token(
                            TokenType::Keyword(token.into()),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                    } else {
                        self.add_token(
                            TokenType::Ident(token.into()),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                    }
                }
                b'~' => {
                    self.add_token(
                        TokenType::Operator(OperatorKind::BitNot),
                        line_content,
                        ctx.in_path,
                        col,
                    );
                    self.cur += 1;
                }
                b'-' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'-' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::Decrement),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else {
                        self.add_token(
                            TokenType::Operator(OperatorKind::Minus),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                    }
                }
                b'+' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'+' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::Increment),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else {
                        self.add_token(
                            TokenType::Operator(OperatorKind::Plus),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                    }
                }
                b'*' => {
                    self.add_token(
                        TokenType::Operator(OperatorKind::Asterisk),
                        line_content,
                        ctx.in_path,
                        col,
                    );
                    self.cur += 1;
                }
                b'/' => {
                    self.add_token(
                        TokenType::Operator(OperatorKind::Division),
                        line_content,
                        ctx.in_path,
                        col,
                    );
                    self.cur += 1;
                }
                b'%' => {
                    self.add_token(
                        TokenType::Operator(OperatorKind::Modulo),
                        line_content,
                        ctx.in_path,
                        col,
                    );
                    self.cur += 1;
                }
                b'&' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'&' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::LogAnd),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else {
                        self.add_token(
                            TokenType::Operator(OperatorKind::Ampersand),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                    }
                }
                b'|' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'|' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::LogOr),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else {
                        self.add_token(
                            TokenType::Operator(OperatorKind::BitOr),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                    }
                }
                b'^' => {
                    self.add_token(
                        TokenType::Operator(OperatorKind::BitXor),
                        line_content,
                        ctx.in_path,
                        col,
                    );
                    self.cur += 1;
                }
                b'<' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'<' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::ShiftLeft),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else if self.has_next() && self.first() == b'=' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::LessThanEq),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else {
                        self.add_token(
                            TokenType::Operator(OperatorKind::LessThan),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                    }
                }
                b'>' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'>' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::ShiftRight),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else if self.has_next() && self.first() == b'=' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::GreaterThanEq),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else {
                        self.add_token(
                            TokenType::Operator(OperatorKind::GreaterThan),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                    }
                }
                b'!' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'=' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::NotEq),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else {
                        self.add_token(
                            TokenType::Operator(OperatorKind::LogNot),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                    }
                }
                b'=' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'=' {
                        self.add_token(
                            TokenType::Operator(OperatorKind::Eq),
                            line_content,
                            ctx.in_path,
                            col,
                        );
                        self.cur += 1;
                    } else {
                        todo!("assignment unsupported")
                    }
                }
                b'(' => {
                    self.add_token(TokenType::ParenOpen, line_content, ctx.in_path, col);
                    self.cur += 1;
                }
                b')' => {
                    self.add_token(TokenType::ParenClose, line_content, ctx.in_path, col);
                    self.cur += 1;
                }
                b'{' => {
                    self.add_token(TokenType::BraceOpen, line_content, ctx.in_path, col);
                    self.cur += 1;
                }
                b'}' => {
                    self.add_token(TokenType::BraceClose, line_content, ctx.in_path, col);
                    self.cur += 1;
                }
                b';' => {
                    self.add_token(TokenType::Semicolon, line_content, ctx.in_path, col);
                    self.cur += 1;
                }
                b => {
                    let token = format!("{}", b as char);

                    if cfg!(test) {
                        panic!("stray '{token}' in program");
                    }

                    report_token_err!(
                        ctx.in_path.display(),
                        self.line,
                        col,
                        token,
                        0,
                        line_content,
                        "stray '{token}' in program"
                    );
                    process::exit(1);
                }
            }
        }
    }

    /// Returns a reference to the next token in sequence without consuming it.
    pub fn peek(&self) -> Option<&Token<'_>> {
        self.tokens.front()
    }

    /// Returns the next token in sequence.
    pub fn next_token(&mut self) -> Option<Token<'_>> {
        self.tokens.pop_front()
    }

    /// Returns the number of tokens left to consume.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` if there are no more tokens available to consume.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Generates and appends a new `Token` to the token sequence.
    #[inline]
    fn add_token(&mut self, ty: TokenType, lc: &'a str, file_path: &'static Path, col: usize) {
        self.tokens.push_back(Token {
            ty,
            loc: Location {
                line_content: lc,
                file_path,
                line: self.line,
                col,
            },
        });
    }

    /// Returns the byte from `src` at the current cursor position. Does **not**
    /// update the cursor position.
    ///
    /// # Panic
    ///
    /// Will _panic_ if the cursor position is out of bounds.
    #[inline]
    const fn first(&mut self) -> u8 {
        self.src[self.cur]
    }

    /// Returns `true` if the cursor position is within bounds of `src`.
    #[inline]
    const fn has_next(&self) -> bool {
        self.cur < self.src.len()
    }
}

impl fmt::Display for Lexer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for tok in &self.tokens {
            writeln!(f, "{}", tok)?;
        }
        Ok(())
    }
}

// Reference: https://github.com/nlsandler/writing-a-c-compiler-tests
#[cfg(test)]
mod tests {
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
    fn lexer_valid_return_zero() {
        let source = b"int main(void) { return 0; }";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    fn lexer_valid_return_non_zero() {
        let source = b"int main(void) { return 2; }";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    fn lexer_valid_multi_digit() {
        let source = b"int main(void) { return 100; }";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    fn lexer_valid_newlines() {
        let source = b"int\nmain\n(\nvoid\n)\n{\nreturn\n0\n;}";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    fn lexer_valid_no_newlines() {
        let source = b"int main(void){return 0;}";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    fn lexer_valid_whitespace() {
        let source = b"   int   main    (  void)  {   return  0 ; }";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    fn lexer_valid_tabs() {
        let source = b"int	main	(	void)	{	return	0	;	}";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    #[should_panic(expected = "stray")]
    fn lexer_invalid_unexpected_symbol() {
        // The '@' symbol doesn't appear in any token, except inside string or
        // character literals.
        let source = b"int main(void) { return 0@1; }";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    #[should_panic(expected = "stray")]
    fn lexer_invalid_backslash() {
        // Single backslash ('\') is not a valid token.
        let source = b"\\";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    #[should_panic(expected = "stray")]
    fn lexer_invalid_backtick() {
        // Backtick ('`') is not a valid token.
        let source = b"`";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    #[should_panic(expected = "invalid suffix")]
    fn lexer_invalid_identifier_digit() {
        // Identifiers are not allowed to start with digits, must begin with
        // a non-digit including ('_').
        let source = b"int main(void) { return 1foo; }";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    #[should_panic(expected = "stray")]
    fn lexer_invalid_identifier_symbol() {
        // Identifiers are not allowed to start with digits, must begin with
        // a non-digit including ('_').
        let source = b"int main(void) { return @b; }";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }
}
