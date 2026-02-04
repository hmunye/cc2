//! Lexical Analysis
//!
//! Compiler pass that tokenizes a _C_ translation unit, producing a sequence of
//! tokens.

use std::fmt;
use std::ops::Range;
use std::path::Path;

use crate::{Context, Result, fmt_token_err};

/// Reserved tokens defined by the _C_ language standard (_C17_).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reserved {
    Int,
    Void,
    Return,
    If,
    Else,
    Goto,
    Do,
    While,
    For,
    Break,
    Continue,
    Switch,
    Case,
    Default,
}

impl Reserved {
    #[inline]
    const fn as_str(self) -> &'static str {
        match self {
            Reserved::Int => "int",
            Reserved::Void => "void",
            Reserved::Return => "return",
            Reserved::If => "if",
            Reserved::Else => "else",
            Reserved::Goto => "goto",
            Reserved::Do => "do",
            Reserved::While => "while",
            Reserved::For => "for",
            Reserved::Break => "break",
            Reserved::Continue => "continue",
            Reserved::Switch => "switch",
            Reserved::Case => "case",
            Reserved::Default => "default",
        }
    }

    #[inline]
    fn contains(s: &str) -> Option<Self> {
        match s {
            "int" => Some(Reserved::Int),
            "void" => Some(Reserved::Void),
            "return" => Some(Reserved::Return),
            "if" => Some(Reserved::If),
            "else" => Some(Reserved::Else),
            "goto" => Some(Reserved::Goto),
            "do" => Some(Reserved::Do),
            "while" => Some(Reserved::While),
            "for" => Some(Reserved::For),
            "break" => Some(Reserved::Break),
            "continue" => Some(Reserved::Continue),
            "switch" => Some(Reserved::Switch),
            "case" => Some(Reserved::Case),
            "default" => Some(Reserved::Default),
            _ => None,
        }
    }
}

impl fmt::Display for Reserved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "keyword({:?})", self.as_str())
    }
}

impl fmt::Debug for Reserved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Operator symbols that can be emitted.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OperatorKind {
    /// `~` - bitwise NOT
    BitNot,
    /// `-` - subtraction or negation
    Minus,
    /// `+` - addition or unary identity
    Plus,
    /// `*` – multiplication or dereference
    Asterisk,
    /// `/` – division
    Division,
    /// `%` – remainder
    Remainder,
    /// `++` – increment
    Increment,
    /// `--` – decrement
    Decrement,
    /// `&` – bitwise AND or address‑of
    Ampersand,
    /// `|` – bitwise OR
    BitOr,
    /// `^` – bitwise XOR
    BitXor,
    /// `<` – less‑than
    LessThan,
    /// `>` – greater‑than
    GreaterThan,
    /// `<=` – less‑than or equal
    LessThanEq,
    /// `>=` – greater‑than or equal
    GreaterThanEq,
    /// `<<` – bitwise left shift
    ShiftLeft,
    /// `>>` – bitwise right shift
    ShiftRight,
    /// `!` – logical NOT
    LogNot,
    /// `&&` – logical AND
    LogAnd,
    /// `||` – logical OR
    LogOr,
    /// `==` – equality
    Eq,
    /// `!=` – inequality
    NotEq,
    /// `=` – simple assignment
    Assign,
    /// `+=` – add‑and‑assign
    AssignPlus,
    /// `-=` – sub‑and‑assign
    AssignMinus,
    /// `*=` – mul‑and‑assign
    AssignAsterisk,
    /// `/=` – div‑and‑assign
    AssignDivision,
    /// `%=` – rem‑and‑assign
    AssignRemainder,
    /// `&=` – and‑and‑assign
    AssignAmpersand,
    /// `|=` – or‑and‑assign
    AssignBitOr,
    /// `^=` – xor‑and‑assign
    AssignBitXor,
    /// `<<=` – shift‑left‑and‑assign
    AssignShiftLeft,
    /// `>>=` – shift‑right‑and‑assign
    AssignShiftRight,
}

impl fmt::Display for OperatorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperatorKind::BitNot => write!(f, "op('~')"),
            OperatorKind::Minus => write!(f, "op('-')"),
            OperatorKind::Plus => write!(f, "op('+')"),
            OperatorKind::Asterisk => write!(f, "op('*')"),
            OperatorKind::Division => write!(f, "op('/')"),
            OperatorKind::Remainder => write!(f, "op('%')"),
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
            OperatorKind::Assign => write!(f, "op('=')"),
            OperatorKind::AssignPlus => write!(f, "op('+=')"),
            OperatorKind::AssignMinus => write!(f, "op('-=')"),
            OperatorKind::AssignAsterisk => write!(f, "op('*=')"),
            OperatorKind::AssignDivision => write!(f, "op('/=')"),
            OperatorKind::AssignRemainder => write!(f, "op('%=')"),
            OperatorKind::AssignAmpersand => write!(f, "op('&=')"),
            OperatorKind::AssignBitOr => write!(f, "op('|=')"),
            OperatorKind::AssignBitXor => write!(f, "op('^=')"),
            OperatorKind::AssignShiftLeft => write!(f, "op('<<=')"),
            OperatorKind::AssignShiftRight => write!(f, "op('>>=')"),
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
            OperatorKind::Remainder => write!(f, "%"),
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
            OperatorKind::Assign => write!(f, "="),
            OperatorKind::AssignPlus => write!(f, "+="),
            OperatorKind::AssignMinus => write!(f, "-="),
            OperatorKind::AssignAsterisk => write!(f, "*="),
            OperatorKind::AssignDivision => write!(f, "/="),
            OperatorKind::AssignRemainder => write!(f, "%="),
            OperatorKind::AssignAmpersand => write!(f, "&="),
            OperatorKind::AssignBitOr => write!(f, "|="),
            OperatorKind::AssignBitXor => write!(f, "^="),
            OperatorKind::AssignShiftLeft => write!(f, "<<="),
            OperatorKind::AssignShiftRight => write!(f, ">>="),
        }
    }
}

/// Lexical elements that can be emitted.
#[derive(Clone, PartialEq, Eq)]
pub enum TokenType<'a> {
    /// Reserved word (e.g., `if`, `return`).
    Keyword(Reserved),
    /// User-defined identifier.
    Ident(&'a str),
    /// Integer constant (32-bit signed).
    IntConstant(i32),
    /// Operator token.
    Operator(OperatorKind),
    /// `(`.
    LParen,
    /// `)`.
    RParen,
    /// `{`.
    LBrace,
    /// `}`.
    RBrace,
    /// `?`.
    Question,
    /// `:`.
    Colon,
    /// `,`.
    Comma,
    /// `;`.
    Semicolon,
}

impl fmt::Display for TokenType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenType::Keyword(kw) => fmt::Display::fmt(kw, f),
            TokenType::Ident(i) => write!(f, "ident({i:?})"),
            TokenType::IntConstant(v) => write!(f, "constant(\"{v}\")"),
            TokenType::Operator(op) => fmt::Display::fmt(op, f),
            TokenType::LParen => write!(f, "'('"),
            TokenType::RParen => write!(f, "')'"),
            TokenType::LBrace => write!(f, "'{{'"),
            TokenType::RBrace => write!(f, "'}}'"),
            TokenType::Question => write!(f, "'?'"),
            TokenType::Colon => write!(f, "':'"),
            TokenType::Comma => write!(f, "','"),
            TokenType::Semicolon => write!(f, "';'"),
        }
    }
}

impl fmt::Debug for TokenType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenType::Keyword(kw) => fmt::Debug::fmt(kw, f),
            TokenType::Ident(i) => write!(f, "{i}"),
            TokenType::IntConstant(v) => write!(f, "{v}"),
            TokenType::Operator(op) => fmt::Debug::fmt(op, f),
            TokenType::LParen => write!(f, "("),
            TokenType::RParen => write!(f, ")"),
            TokenType::LBrace => write!(f, "{{"),
            TokenType::RBrace => write!(f, "}}"),
            TokenType::Question => write!(f, "?"),
            TokenType::Colon => write!(f, ":"),
            TokenType::Comma => write!(f, ","),
            TokenType::Semicolon => write!(f, ";"),
        }
    }
}

/// Location of an emitted token.
#[derive(Debug, Clone)]
pub struct Location {
    pub file_path: &'static Path,
    // Range of line from source code bytes this token appears in.
    pub line_span: Range<usize>,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file_path.display(), self.line, self.col)
    }
}

/// Minimal lexical token that can be emitted.
#[derive(Clone)]
pub struct Token<'a> {
    pub loc: Location,
    pub ty: TokenType<'a>,
}

impl fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}\t\t{}", self.loc, self.ty)
    }
}

impl fmt::Debug for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.ty)
    }
}

/// Lexer that emits tokens lazily from a _C_ translation unit.
#[derive(Debug)]
pub struct Lexer<'a> {
    ctx: &'a Context<'a>,
    cur: usize,
    /// Index of the next `\n` for current line.
    line_end: usize,
    /// Index of byte after last encountered `\n`.
    bol: usize,
    line: usize,
}

impl<'a> Lexer<'a> {
    /// Returns a new `Lexer`.
    ///
    /// Does **not** support universal character names (only _ASCII_).
    #[inline]
    #[must_use]
    pub const fn new(ctx: &'a Context<'_>) -> Self {
        Self {
            ctx,
            cur: 0,
            line_end: 0,
            bol: 0,
            line: 1,
        }
    }

    /// Skips over an identifier or keyword (_ASCII_ uppercase/lowercase letter
    /// or `_`), returning a token.
    fn consume_ident(&mut self) -> Token<'a> {
        let col = self.col();
        let token_start = self.cur;

        while self.has_next() && (self.first().is_ascii_alphanumeric() || self.first() == b'_') {
            self.cur += 1;
        }

        let token = self.ctx.src_slice(token_start..self.cur);

        if let Some(kw) = Reserved::contains(token) {
            Token {
                ty: TokenType::Keyword(kw),
                loc: self.token_loc(col),
            }
        } else {
            Token {
                ty: TokenType::Ident(token),
                loc: self.token_loc(col),
            }
        }
    }

    /// Skips over an integer constant (32-bit signed), returning a token.
    ///
    /// # Errors
    ///
    /// Returns an error if the constant contains an illegal suffix or cannot be
    /// parsed.
    fn consume_constant(&mut self) -> Result<Token<'a>> {
        let col = self.col();
        let token_start = self.cur;

        while self.has_next() && self.first().is_ascii_digit() {
            self.cur += 1;
        }

        // Identifiers are not allowed to begin with digits.
        //
        // NOTE: Currently don't allow `'` as a digit separator or any suffixes.
        if self.first().is_ascii_alphabetic() || self.first() == b'\'' {
            let const_len = self.cur - token_start;
            let const_end = self.cur;

            // Continue consuming the invalid suffix.
            while self.has_next() && (self.first().is_ascii_alphabetic() || self.first() == b'_') {
                self.cur += 1;
            }

            let suffix = self.ctx.src_slice(const_end..self.cur);
            let line_content = self.ctx.src_slice(self.bol..self.line_end);

            return Err(fmt_token_err!(
                self.ctx.in_path.display(),
                self.line,
                col + const_len,
                suffix,
                suffix.len() - 1,
                line_content,
                "invalid suffix '{suffix}' on integer constant"
            ));
        }

        let token = self.ctx.src_slice(token_start..self.cur);

        let Ok(int) = token.parse::<i32>() else {
            return Err(fmt_token_err!(
                self.ctx.in_path.display(),
                self.line,
                col,
                token,
                token.len() - 1,
                self.ctx.src_slice(self.bol..self.line_end),
                "integer constant is too large for its type (32-bit signed)"
            ));
        };

        Ok(Token {
            ty: TokenType::IntConstant(int),
            loc: self.token_loc(col),
        })
    }

    /// Skips over a line directive from the preprocessor of the form:
    ///
    /// ```text
    ///     # <decimal-line> SP* [ "<filename>" ] SP* [ <decimal-flag>... ]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the `decimal-line` cannot be parsed.
    fn consume_line_directive(&mut self) -> Result<()> {
        self.cur += 1;
        self.consume_whitespace();

        let line_token = self.consume_constant()?;
        let TokenType::IntConstant(line) = line_token.ty else {
            unreachable!("line should be  parsed as an integer constant");
        };

        debug_assert!(line >= 0);

        // Part of the preprocessor prologue - skip it.
        if line == 0 {
            while self.has_next() && self.first() != b'\n' {
                self.cur += 1;
            }

            self.consume_newline();

            return Ok(());
        }

        // Accounts for the `\n` that is consumed later.
        self.line = (line - 1) as usize;

        // NOTE: Currently don't parse [ "<filename>" ].
        //
        // NOTE: Currently don't parse [ <decimal-flag>... ].

        while self.has_next() && self.first() != b'\n' {
            self.cur += 1;
        }

        self.consume_newline();

        Ok(())
    }

    /// Skips over all consecutive _ASCII_ whitespace characters (not including
    /// `\n`).
    const fn consume_whitespace(&mut self) {
        while self.has_next() && matches!(self.first(), b'\t' | b'\x0C' | b'\r' | b' ') {
            self.cur += 1;
        }
    }

    /// Skips over `\n`, advancing to the start of the next line.
    const fn consume_newline(&mut self) {
        self.cur += 1;
        self.bol = self.cur;
        self.line += 1;

        let mut i = self.cur;
        while i < self.ctx.src.len() && self.ctx.src[i] != b'\n' {
            i += 1;
        }

        self.line_end = i;
    }

    /// Returns the location of a token beginning at `col` within the source.
    #[inline]
    const fn token_loc(&self, col: usize) -> Location {
        Location {
            file_path: self.ctx.in_path,
            line: self.line,
            col,
            line_span: self.bol..self.line_end,
        }
    }

    /// Returns the current column number (1‑based).
    #[inline]
    const fn col(&self) -> usize {
        self.cur - self.bol + 1
    }

    /// Returns the byte from source at the current cursor position. Does
    /// **not** update the cursor position.
    ///
    /// # Panics
    ///
    /// Panics if the cursor position is out of bounds.
    #[inline]
    const fn first(&self) -> u8 {
        self.ctx.src[self.cur]
    }

    /// Returns `true` if the cursor position is within bounds of source.
    #[inline]
    const fn has_next(&self) -> bool {
        self.cur < self.ctx.src.len()
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.has_next() {
            // Determine the span of the first line (special case since no `\n`
            // has been encountered yet).
            if self.line_end == 0 {
                let mut i = 0;
                while i < self.ctx.src.len() && self.ctx.src[i] != b'\n' {
                    i += 1;
                }

                self.line_end = i;
            }

            let col = self.col();

            match self.first() {
                b'\n' => self.consume_newline(),
                b'\t' | b'\x0C' | b'\r' | b' ' => self.consume_whitespace(),
                b'0'..=b'9' => return Some(self.consume_constant()),
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => return Some(Ok(self.consume_ident())),
                b'#' => {
                    if let Err(err) = self.consume_line_directive() {
                        return Some(Err(err));
                    }
                }
                b'~' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::Operator(OperatorKind::BitNot),
                        loc: self.token_loc(col),
                    }));
                }
                b'-' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'-' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Decrement),
                            loc: self.token_loc(col),
                        }));
                    } else if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::AssignMinus),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Minus),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'+' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'+' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Increment),
                            loc: self.token_loc(col),
                        }));
                    } else if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::AssignPlus),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Plus),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'*' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::AssignAsterisk),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Asterisk),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'/' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::AssignDivision),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Division),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'%' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::AssignRemainder),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Remainder),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'&' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'&' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::LogAnd),
                            loc: self.token_loc(col),
                        }));
                    } else if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::AssignAmpersand),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Ampersand),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'|' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'|' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::LogOr),
                            loc: self.token_loc(col),
                        }));
                    } else if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::AssignBitOr),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::BitOr),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'^' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::AssignBitXor),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::BitXor),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'<' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'<' {
                        self.cur += 1;

                        if self.has_next() && self.first() == b'=' {
                            self.cur += 1;

                            return Some(Ok(Token {
                                ty: TokenType::Operator(OperatorKind::AssignShiftLeft),
                                loc: self.token_loc(col),
                            }));
                        } else {
                            return Some(Ok(Token {
                                ty: TokenType::Operator(OperatorKind::ShiftLeft),
                                loc: self.token_loc(col),
                            }));
                        }
                    } else if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::LessThanEq),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::LessThan),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'>' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'>' {
                        self.cur += 1;

                        if self.has_next() && self.first() == b'=' {
                            self.cur += 1;

                            return Some(Ok(Token {
                                ty: TokenType::Operator(OperatorKind::AssignShiftRight),
                                loc: self.token_loc(col),
                            }));
                        } else {
                            return Some(Ok(Token {
                                ty: TokenType::Operator(OperatorKind::ShiftRight),
                                loc: self.token_loc(col),
                            }));
                        }
                    } else if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::GreaterThanEq),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::GreaterThan),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'!' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::NotEq),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::LogNot),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'=' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'=' {
                        self.cur += 1;

                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Eq),
                            loc: self.token_loc(col),
                        }));
                    } else {
                        return Some(Ok(Token {
                            ty: TokenType::Operator(OperatorKind::Assign),
                            loc: self.token_loc(col),
                        }));
                    }
                }
                b'(' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::LParen,
                        loc: self.token_loc(col),
                    }));
                }
                b')' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::RParen,
                        loc: self.token_loc(col),
                    }));
                }
                b'{' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::LBrace,
                        loc: self.token_loc(col),
                    }));
                }
                b'}' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::RBrace,
                        loc: self.token_loc(col),
                    }));
                }
                b'?' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::Question,
                        loc: self.token_loc(col),
                    }));
                }
                b':' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::Colon,
                        loc: self.token_loc(col),
                    }));
                }
                b',' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::Comma,
                        loc: self.token_loc(col),
                    }));
                }
                b';' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::Semicolon,
                        loc: self.token_loc(col),
                    }));
                }
                b => {
                    self.cur += 1;

                    let byte = format!("{}", b as char);
                    let line_content = self.ctx.src_slice(self.bol..self.line_end);

                    return Some(Err(fmt_token_err!(
                        self.ctx.in_path.display(),
                        self.line,
                        col,
                        byte,
                        0,
                        line_content,
                        "stray '{byte}' in program"
                    )));
                }
            }
        }

        None
    }
}

impl fmt::Display for Lexer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lexer = Lexer::new(self.ctx);

        for token in lexer {
            match token {
                Ok(token) => writeln!(f, "{token}")?,
                Err(err) => writeln!(f, "{err}")?,
            }
        }

        Ok(())
    }
}
