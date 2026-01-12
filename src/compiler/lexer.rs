//! Lexical Analysis
//!
//! Compiler pass that tokenizes _C_ translation unit, producing a sequence of
//! tokens.

use std::fmt;
use std::ops::Range;
use std::path::Path;

use crate::{Context, compiler::Result, fmt_token_err};

/// Reserved tokens defined by the _C_ language standard (_C17_).
const KEYWORDS: [&str; 3] = ["int", "void", "return"];

/// Types of operators.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OperatorKind {
    /// `~` bitwise NOT operator.
    BitNot,
    /// `-` subtraction or negation operator.
    Minus,
    /// `+` addition or unary identity operator.
    Plus,
    /// `*` multiplication or dereference operator.
    Asterisk,
    /// `/` division operator.
    Division,
    /// `%` remainder operator.
    Remainder,
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
    /// `=` assignment operator.
    Assign,
    /// `+=` assignment operator.
    AssignPlus,
    /// `-=` assignment operator.
    AssignMinus,
    /// `*=` assignment operator.
    AssignAsterisk,
    /// `/=` assignment operator.
    AssignDivision,
    /// `%=` assignment operator.
    AssignRemainder,
    /// `&=` assignment operator.
    AssignAmpersand,
    /// `|=` assignment operator.
    AssignBitOr,
    /// `^=` assignment operator.
    AssignBitXor,
    /// `<<=` assignment operator.
    AssignShiftLeft,
    /// `>>=` assignment operator.
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

/// Types of lexical elements.
#[derive(Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum TokenType {
    Keyword(String),
    Ident(String),
    Constant(i32),
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
            TokenType::Constant(v) => write!(f, "constant(\"{v}\")"),
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
            TokenType::Constant(v) => write!(f, "{v}"),
            TokenType::Operator(op) => fmt::Debug::fmt(op, f),
            TokenType::ParenOpen => write!(f, "("),
            TokenType::ParenClose => write!(f, ")"),
            TokenType::BraceOpen => write!(f, "{{"),
            TokenType::BraceClose => write!(f, "}}"),
            TokenType::Semicolon => write!(f, ";"),
        }
    }
}

/// Location of processed `Token`.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct Location {
    pub file_path: &'static Path,
    pub line: usize,
    pub col: usize,
    /// Range of line in source code this token appears.
    pub line_span: Range<usize>,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file_path.display(), self.line, self.col)
    }
}

/// Minimal lexical element.
#[derive(Clone)]
#[allow(missing_docs)]
pub struct Token {
    pub ty: TokenType,
    pub loc: Location,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}\t    {}", self.loc, self.ty)
    }
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.ty)
    }
}

/// Produces tokens lazily from a _C_ translation unit.
#[derive(Debug)]
pub struct Lexer<'a> {
    ctx: &'a Context<'a>,
    cur: usize,
    // Track source code line spans for each token.
    line_end: usize,
    // Index of the beginning of a newline (to calculate the current column and
    // line span).
    bol: usize,
    line: usize,
}

impl<'a> Lexer<'a> {
    /// Returns a new `Lexer`.
    ///
    /// Does **not** support universal character names (only _ASCII_).
    pub const fn new(ctx: &'a Context<'_>) -> Self {
        Self {
            ctx,
            cur: 0,
            line_end: 0,
            bol: 0,
            line: 1,
        }
    }

    /// Skips over all consecutive whitespace characters.
    const fn consume_whitespace(&mut self) {
        while self.has_next() && self.first().is_ascii_whitespace() {
            self.cur += 1;
        }
    }

    /// Skips over a newline, advancing to the start of the next line.
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

    /// Skips over an identifier or keyword (starting with an ASCII uppercase/
    /// lowercase letter or '_'), producing a `Token`.
    fn consume_ident(&mut self) -> Result<Token> {
        let col = self.col();
        let token_start = self.cur;

        while self.has_next() && (self.first().is_ascii_alphanumeric() || self.first() == b'_') {
            self.cur += 1;
        }

        let token = self.ctx.src_slice(token_start..self.cur);

        if KEYWORDS.contains(&token) {
            Ok(Token {
                ty: TokenType::Keyword(token.into()),
                loc: self.token_loc(col),
            })
        } else {
            Ok(Token {
                ty: TokenType::Ident(token.into()),
                loc: self.token_loc(col),
            })
        }
    }

    /// Skips over an constant, producing a `Token`.
    fn consume_constant(&mut self) -> Result<Token> {
        let col = self.col();
        let token_start = self.cur;

        while self.has_next() && self.first().is_ascii_digit() {
            self.cur += 1;
        }

        // Identifier are not allowed to begin with digits.
        //
        // NOTE: Currently don't allow '_' as a separator or suffixes.
        if self.first().is_ascii_alphabetic() || self.first() == b'_' {
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

        let Ok(integer) = token.parse::<i32>() else {
            let line_content = self.ctx.src_slice(self.bol..self.line_end);

            return Err(fmt_token_err!(
                self.ctx.in_path.display(),
                self.line,
                col,
                token,
                token.len() - 1,
                line_content,
                "integer constant is too large for its type (32-bit signed)"
            ));
        };

        Ok(Token {
            ty: TokenType::Constant(integer),
            loc: self.token_loc(col),
        })
    }

    /// Returns the source location for a token at `col`.
    #[inline]
    const fn token_loc(&self, col: usize) -> Location {
        Location {
            file_path: self.ctx.in_path,
            line: self.line,
            col,
            line_span: self.bol..self.line_end,
        }
    }

    /// Returns the current column in the line.
    #[inline]
    const fn col(&self) -> usize {
        self.cur - self.bol + 1
    }

    /// Returns the byte from `src` at the current cursor position. Does **not**
    /// update the cursor position.
    ///
    /// # Panic
    ///
    /// Will _panic_ if the cursor position is out of bounds.
    #[inline]
    const fn first(&self) -> u8 {
        self.ctx.src[self.cur]
    }

    /// Returns `true` if the cursor position is within bounds of `src`.
    #[inline]
    const fn has_next(&self) -> bool {
        self.cur < self.ctx.src.len()
    }
}

impl Iterator for Lexer<'_> {
    type Item = Result<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.has_next() {
            // Determine the span of the first line (special case since no '\n'
            // has been seen yet).
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
                b if b.is_ascii_whitespace() => self.consume_whitespace(),
                b'0'..=b'9' => return Some(self.consume_constant()),
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => return Some(self.consume_ident()),
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
                        ty: TokenType::ParenOpen,
                        loc: self.token_loc(col),
                    }));
                }
                b')' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::ParenClose,
                        loc: self.token_loc(col),
                    }));
                }
                b'{' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::BraceOpen,
                        loc: self.token_loc(col),
                    }));
                }
                b'}' => {
                    self.cur += 1;

                    return Some(Ok(Token {
                        ty: TokenType::BraceClose,
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
                Err(err) => writeln!(f, "\n{}\n", err)?,
            }
        }

        Ok(())
    }
}

// Reference: https://github.com/nlsandler/writing-a-c-compiler-tests
#[cfg(test)]
mod tests {
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
    fn lexer_valid_return_zero() {
        let source = b"int main(void) { return 0; }";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        for token in lexer {
            assert!(token.is_ok());
        }
    }

    #[test]
    fn lexer_valid_return_non_zero() {
        let source = b"int main(void) { return 2; }";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        for token in lexer {
            assert!(token.is_ok());
        }
    }

    #[test]
    fn lexer_valid_multi_digit() {
        let source = b"int main(void) { return 100; }";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        for token in lexer {
            assert!(token.is_ok());
        }
    }

    #[test]
    fn lexer_valid_newlines() {
        let source = b"int\nmain\n(\nvoid\n)\n{\nreturn\n0\n;}";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        for token in lexer {
            assert!(token.is_ok());
        }
    }

    #[test]
    fn lexer_valid_no_newlines() {
        let source = b"int main(void){return 0;}";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        for token in lexer {
            assert!(token.is_ok());
        }
    }

    #[test]
    fn lexer_valid_whitespace() {
        let source = b"   int   main    (  void)  {   return  0 ; }";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        for token in lexer {
            assert!(token.is_ok());
        }
    }

    #[test]
    fn lexer_valid_tabs() {
        let source = b"int	main	(	void)	{	return	0	;	}";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        for token in lexer {
            assert!(token.is_ok());
        }
    }

    #[test]
    fn lexer_invalid_unexpected_symbol() {
        // The '@' symbol doesn't appear in any token, except inside string or
        // character literals.
        let source = b"int main(void) { return 0@1; }";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        let mut seen_error = false;

        for token in lexer {
            if let Err(err) = token {
                assert!(err.contains("stray"));
                seen_error = true;
            }
        }

        assert!(seen_error);
    }

    #[test]
    fn lexer_invalid_backslash() {
        // Single backslash ('\') is not a valid token.
        let source = b"\\";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        let mut seen_error = false;

        for token in lexer {
            if let Err(err) = token {
                assert!(err.contains("stray"));
                seen_error = true;
            }
        }

        assert!(seen_error);
    }

    #[test]
    fn lexer_invalid_backtick() {
        // Backtick ('`') is not a valid token.
        let source = b"`";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        let mut seen_error = false;

        for token in lexer {
            if let Err(err) = token {
                assert!(err.contains("stray"));
                seen_error = true;
            }
        }

        assert!(seen_error);
    }

    #[test]
    fn lexer_invalid_identifier_digit() {
        // Identifiers are not allowed to start with digits, must begin with
        // a non-digit including ('_').
        let source = b"int main(void) { return 1foo; }";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        let mut seen_error = false;

        for token in lexer {
            if let Err(err) = token {
                assert!(err.contains("invalid suffix"));
                seen_error = true;
            }
        }

        assert!(seen_error);
    }

    #[test]
    fn lexer_invalid_identifier_symbol() {
        // Identifiers are not allowed to start with digits, must begin with
        // a non-digit including ('_').
        let source = b"int main(void) { return @b; }";
        let ctx = test_ctx(source);
        let lexer = Lexer::new(&ctx);

        let mut seen_error = false;

        for token in lexer {
            if let Err(err) = token {
                assert!(err.contains("stray"));
                seen_error = true;
            }
        }

        assert!(seen_error);
    }
}
