//! Lexical Analysis
//!
//! Compiler pass that tokenizes _C_ source code, producing a sequence of
//! tokens.

use std::path::Path;
use std::{fmt, process};

use crate::{Context, report_err_ctx, report_token_err};

/// Reserved tokens by the _C_ language standard (_C17_).
const KEYWORDS: [&str; 3] = ["int", "void", "return"];

/// Different operators defined in the _C_ language standard (_C17_).
#[derive(Debug, PartialEq)]
pub enum OperatorKind {
    /// `~` unary operator.
    Complement,
    /// `-` unary operator.
    Negate,
    /// `--` unary operator.
    Decrement,
}

/// TODO: Implement `Display`.
///
/// Types of lexical elements.
#[derive(Debug, PartialEq)]
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

/// TODO: Add line content so that the parser can report token errors.
///
/// Location of a processed `Token`.
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Location {
    pub file_path: &'static Path,
    pub line: usize,
    pub col: usize,
}

/// Minimal lexical element of the _C_ language standard (_C17_).
#[derive(Debug)]
#[allow(missing_docs)]
pub struct Token {
    pub ty: TokenType,
    pub loc: Location,
}

/// Stores the ordered sequence of tokens extracted from a _C_ translation unit.
#[derive(Default)]
pub struct Lexer<'a> {
    tokens: std::collections::VecDeque<Token>,
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

    /// Initialize the `Lexer` given _C_ source code, internally producing a
    /// sequence of tokens. [Exits] on error with non-zero status.
    ///
    /// Does **not** support universal character names (only _ASCII_).
    ///
    /// [Exits]: std::process::exit
    pub fn lex(&mut self, ctx: &Context<'_>) {
        // TODO: Track each current line, so it's contents can used in error
        // reporting.
        while self.has_next() {
            let col = self.cur - self.bol;

            match self.first() {
                b'\n' => {
                    self.bol = self.cur;
                    self.cur += 1;
                    self.line += 1;
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
                        // Continue consuming the invalid suffix.
                        while self.has_next()
                            && (self.first().is_ascii_alphabetic() || self.first() == b'_')
                        {
                            self.cur += 1;
                        }

                        let suffix = std::str::from_utf8(&self.src[token_start + 1..self.cur])
                            .expect("ASCII bytes should be valid UTF-8");

                        // Continue to consume the rest of the line.
                        while self.has_next() && self.first() != b'\n' {
                            self.cur += 1;
                        }

                        let line_contents = std::str::from_utf8(&self.src[self.bol + 1..self.cur])
                            .expect("ASCII bytes should be valid UTF-8");

                        if cfg!(test) {
                            panic!("invalid suffix '{suffix}' on integer constant");
                        }

                        report_token_err!(
                            ctx.in_path.display(),
                            self.line,
                            col,
                            suffix,
                            line_contents,
                            "invalid suffix '{suffix}' on integer constant",
                        );
                        process::exit(1);
                    }

                    let token = std::str::from_utf8(&self.src[token_start..self.cur])
                        .expect("ASCII bytes should be valid UTF-8");

                    let Ok(integer) = token.parse::<i32>() else {
                        if cfg!(test) {
                            panic!("invalid integer constant: '{token}'");
                        }

                        report_err_ctx!(
                            ctx.in_path.display(),
                            self.line,
                            col,
                            "invalid integer constant: '{token}'"
                        );
                        process::exit(1);
                    };

                    self.tokens.push_back(Token {
                        ty: TokenType::ConstantInt(integer),
                        loc: Location {
                            file_path: ctx.in_path,
                            line: self.line,
                            col,
                        },
                    });
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
                        self.tokens.push_back(Token {
                            ty: TokenType::Keyword(token.into()),
                            loc: Location {
                                file_path: ctx.in_path,
                                line: self.line,
                                col,
                            },
                        });
                    } else {
                        self.tokens.push_back(Token {
                            ty: TokenType::Ident(token.into()),
                            loc: Location {
                                file_path: ctx.in_path,
                                line: self.line,
                                col,
                            },
                        });
                    }
                }
                b'~' => {
                    self.tokens.push_back(Token {
                        ty: TokenType::Operator(OperatorKind::Complement),
                        loc: Location {
                            file_path: ctx.in_path,
                            line: self.line,
                            col,
                        },
                    });

                    self.cur += 1;
                }
                b'-' => {
                    self.cur += 1;

                    if self.has_next() && self.first() == b'-' {
                        self.tokens.push_back(Token {
                            ty: TokenType::Operator(OperatorKind::Decrement),
                            loc: Location {
                                file_path: ctx.in_path,
                                line: self.line,
                                col,
                            },
                        });

                        self.cur += 1;
                    } else {
                        self.tokens.push_back(Token {
                            ty: TokenType::Operator(OperatorKind::Negate),
                            loc: Location {
                                file_path: ctx.in_path,
                                line: self.line,
                                col,
                            },
                        });
                    }
                }
                b'(' => {
                    self.tokens.push_back(Token {
                        ty: TokenType::ParenOpen,
                        loc: Location {
                            file_path: ctx.in_path,
                            line: self.line,
                            col,
                        },
                    });

                    self.cur += 1;
                }
                b')' => {
                    self.tokens.push_back(Token {
                        ty: TokenType::ParenClose,
                        loc: Location {
                            file_path: ctx.in_path,
                            line: self.line,
                            col,
                        },
                    });

                    self.cur += 1;
                }
                b'{' => {
                    self.tokens.push_back(Token {
                        ty: TokenType::BraceOpen,
                        loc: Location {
                            file_path: ctx.in_path,
                            line: self.line,
                            col,
                        },
                    });

                    self.cur += 1;
                }
                b'}' => {
                    self.tokens.push_back(Token {
                        ty: TokenType::BraceClose,
                        loc: Location {
                            file_path: ctx.in_path,
                            line: self.line,
                            col,
                        },
                    });

                    self.cur += 1;
                }
                b';' => {
                    self.tokens.push_back(Token {
                        ty: TokenType::Semicolon,
                        loc: Location {
                            file_path: ctx.in_path,
                            line: self.line,
                            col,
                        },
                    });

                    self.cur += 1;
                }
                b => {
                    if cfg!(test) {
                        panic!("invalid character: '{}'", b as char);
                    }

                    report_err_ctx!(
                        ctx.in_path.display(),
                        self.line,
                        col,
                        "invalid character: '{}'",
                        b as char
                    );
                    process::exit(1);
                }
            }
        }
    }

    /// Returns a reference to the next token in sequence without consuming it.
    pub fn peek(&self) -> Option<&Token> {
        self.tokens.front()
    }

    /// Returns the next token in sequence.
    pub fn next_token(&mut self) -> Option<Token> {
        self.tokens.pop_front()
    }

    /// Returns `true` if there are no more tokens available to consume.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the byte from `src` at the current cursor position. Does not
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

impl fmt::Debug for Lexer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lexer")
            .field("tokens", &self.tokens)
            .finish()
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
    #[should_panic(expected = "invalid character")]
    fn lexer_invalid_unexpected_symbol() {
        // The '@' symbol doesn't appear in any token, except inside string or
        // character literals.
        let source = b"int main(void) { return 0@1; }";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    #[should_panic(expected = "invalid character")]
    fn lexer_invalid_backslash() {
        // Single backslash ('\') is not a valid token.
        let source = b"\\";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }

    #[test]
    #[should_panic(expected = "invalid character")]
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
    #[should_panic(expected = "invalid character")]
    fn lexer_invalid_identifier_symbol() {
        // Identifiers are not allowed to start with digits, must begin with
        // a non-digit including ('_').
        let source = b"int main(void) { return @b; }";
        let mut lexer = Lexer::new(source);
        lexer.lex(&test_ctx());
    }
}
