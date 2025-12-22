//! Lexical Analysis.
//!
//! Performs lexical analysis on _C_ source code, producing a sequence of
//! tokens.

#![allow(dead_code)]

use std::convert::AsRef;
use std::path::Path;

/// Reserved tokens by the _C_ language standard (_C17_).
const KEYWORDS: [&str; 3] = ["int", "void", "return"];

/// Types of lexical elements.
#[derive(Debug)]
pub(crate) enum TokenType {
    Keyword(String),
    Ident(String),
    ConstantInt(i32),
    ParenOpen,
    ParenClose,
    BraceOpen,
    BraceClose,
    Semi,
}

/// Minimal lexical element of the _C_ language standard (_C17_).
#[derive(Debug)]
pub(crate) struct Token {
    ty: TokenType,
    line: usize,
    col: usize,
}

/// Stores the ordered sequence of tokens extracted from _C_ source code.
#[derive(Debug, Default)]
pub struct Lexer {
    tokens: Vec<Token>,
}

impl Lexer {
    /// Returns a new, empty `Lexer`.   
    pub fn new() -> Self {
        Self {
            tokens: Default::default(),
        }
    }

    /// Performs lexical analysis on the provided _C_ source code `input`,
    /// internally producing a sequence of tokens.
    ///
    /// Does **not** support universal character names (only _ASCII_).
    pub fn lex(&mut self, file_name: impl AsRef<Path>, input: &[u8]) -> Result<(), String> {
        let mut i = 0;
        // Track index of last encountered newline.
        let mut nl = i;

        let mut line = 1;
        let mut col = 1;

        while i < input.len() {
            match input[i] {
                // Next character is on a new line of source code.
                b'\n' => {
                    line += 1;
                    col = 1;
                    nl = i;
                    i += 1;
                }
                // Skip ASCII values for whitespace.
                b' ' | b'\t' => {
                    col += 1;
                    i += 1;
                }
                // Handle numeric constant.
                //
                // NOTE: Currently only handling integer constants with no
                // suffixes.
                b'0'..=b'9' => {
                    let start = i;
                    let tok_col = col;

                    while i < input.len() && input[i].is_ascii_digit() {
                        col += 1;
                        i += 1;
                    }

                    // Identifier beginning with a digit encountered, which is
                    // invalid.
                    //
                    // NOTE: Currently don't allow usage of '_' as a separator
                    // between digits.
                    //
                    // NOTE: Temporary error handling.
                    if input[i].is_ascii_lowercase()
                        || input[i].is_ascii_uppercase()
                        || input[i] == b'_'
                    {
                        while i < input.len()
                            && (input[i].is_ascii_alphanumeric() || input[i] == b'_')
                        {
                            i += 1;
                        }

                        let suffix = std::str::from_utf8(&input[start + 1..i])
                            .map_err(|err| format!("{err}"))?;

                        while i < input.len() && input[i] != b'\n' {
                            i += 1;
                        }

                        let line_contents =
                            std::str::from_utf8(&input[nl..i]).map_err(|err| format!("{err}"))?;

                        return Err(format!(
                            "\x1b[1;1m{}:{line}:{tok_col}:\x1b[0m \x1b[1;31merror:\x1b[0m invalid suffix '{suffix}' on integer constant\n{line} | {:10}\n{:^line$} | \x1b[1;31m{:>tok_col$}{}\x1b[0m",
                            file_name.as_ref().display(),
                            line_contents,
                            "",
                            "^",
                            "~".repeat(suffix.len())
                        ));
                    }

                    let token =
                        std::str::from_utf8(&input[start..i]).map_err(|err| format!("{err}"))?;

                    let integer = token.parse::<i32>().map_err(|err| format!("{err}"))?;

                    self.tokens.push(Token {
                        ty: TokenType::ConstantInt(integer),
                        line,
                        col: tok_col,
                    });
                }
                // Handle identifier or keyword (allowed to begin with '_').
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                    let start = i;
                    let tok_col = col;

                    while i < input.len() && (input[i].is_ascii_alphanumeric() || input[i] == b'_')
                    {
                        col += 1;
                        i += 1;
                    }

                    let token =
                        std::str::from_utf8(&input[start..i]).map_err(|err| format!("{err}"))?;

                    if KEYWORDS.contains(&token) {
                        self.tokens.push(Token {
                            ty: TokenType::Keyword(token.into()),
                            line,
                            col: tok_col,
                        });
                    } else {
                        self.tokens.push(Token {
                            ty: TokenType::Ident(token.into()),
                            line,
                            col: tok_col,
                        });
                    }
                }
                b'(' => {
                    self.tokens.push(Token {
                        ty: TokenType::ParenOpen,
                        line,
                        col,
                    });
                    col += 1;
                    i += 1;
                }
                b')' => {
                    self.tokens.push(Token {
                        ty: TokenType::ParenClose,
                        line,
                        col,
                    });
                    col += 1;
                    i += 1;
                }
                b'{' => {
                    self.tokens.push(Token {
                        ty: TokenType::BraceOpen,
                        line,
                        col,
                    });
                    col += 1;
                    i += 1;
                }
                b'}' => {
                    self.tokens.push(Token {
                        ty: TokenType::BraceClose,
                        line,
                        col,
                    });
                    col += 1;
                    i += 1;
                }
                b';' => {
                    self.tokens.push(Token {
                        ty: TokenType::Semi,
                        line,
                        col,
                    });
                    col += 1;
                    i += 1;
                }
                _ => {
                    return Err(format!(
                        "\x1b[1;1m{}:{line}:{col}:\x1b[0m \x1b[1;31merror:\x1b[0m invalid character '{}'",
                        file_name.as_ref().display(),
                        input[i] as char
                    ));
                }
            }
        }

        Ok(())
    }
}

// Reference: https://github.com/nlsandler/writing-a-c-compiler-tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexer_valid_return_zero() {
        let source = b"int main(void) { return 0; }";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok())
    }

    #[test]
    fn lexer_valid_return_non_zero() {
        let source = b"int main(void) { return 2; }";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok())
    }

    #[test]
    fn lexer_valid_multi_digit() {
        let source = b"int main(void) { return 100; }";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok())
    }

    #[test]
    fn lexer_valid_newlines() {
        let source = b"int\nmain\n(\nvoid\n)\n{\nreturn\n0\n;}";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok())
    }

    #[test]
    fn lexer_valid_no_newlines() {
        let source = b"int main(void){return 0;}";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok())
    }

    #[test]
    fn lexer_valid_whitespace() {
        let source = b"   int   main    (  void)  {   return  0 ; }";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok())
    }

    #[test]
    fn lexer_valid_tabs() {
        let source = b"int	main	(	void)	{	return	0	;	}";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_ok())
    }

    #[test]
    fn lexer_invalid_unexpected_symbol() {
        // The '@' symbol doesn't appear in any token, except inside string or
        // character literals.
        let source = b"int main(void) { return 0@1; }";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_err())
    }

    #[test]
    fn lexer_invalid_backslash() {
        // Single backslash ('\') is not a valid token.
        let source = b"\\";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_err())
    }

    #[test]
    fn lexer_invalid_backtick() {
        // Backtick ('`') is not a valid token.
        let source = b"`";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_err())
    }

    #[test]
    fn lexer_invalid_identifier_digit() {
        // Identifiers are not allowed to start with digits, must begin with
        // a non-digit including ('_').
        let source = b"int main(void) { return 1foo; }";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_err())
    }

    #[test]
    fn lexer_invalid_identifier_symbol() {
        // Identifiers are not allowed to start with digits, must begin with
        // a non-digit including ('_').
        let source = b"int main(void) { return @b; }";
        let mut lexer = Lexer::new();

        assert!(lexer.lex("test.c", source).is_err())
    }
}
