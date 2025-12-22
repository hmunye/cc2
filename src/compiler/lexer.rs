//! Lexical Analysis.
//!
//! Performs lexical analysis on _C_ source code, producing a sequence of
//! tokens.

// FIXME:
#![allow(dead_code)]

/// Reserved tokens by the _C_ language standard (_C17_).
const KEYWORDS: [&str; 3] = ["int", "void", "return"];

/// Minimal lexical element of the _C_ language standard (_C17_).
#[derive(Debug)]
pub(crate) enum Token {
    Keyword(String),
    Identifier(String),
    ConstantInt(i32),
    ParenOpen,
    ParenClose,
    BraceOpen,
    BraceClose,
    Semicolon,
}

/// Stores the ordered sequence of tokens extracted from _C_ source code.
#[derive(Debug)]
pub struct Lexer {
    pub(crate) tokens: Vec<Token>,
}

impl Lexer {
    /// Performs lexical analysis on the provided _C_ source code, returning a
    /// new `Lexer`.
    ///
    /// Does **not** support universal character names (only _ASCII_).
    pub fn new(input: &[u8]) -> Result<Self, String> {
        // TODO: Update error messages, use other compilers as reference

        let mut tokens = vec![];
        let mut i = 0;

        while i < input.len() {
            match input[i] {
                // Skip ASCII values for whitespace.
                b' ' | b'\n' | b'\t' => {
                    i += 1;
                }
                // Handle numeric constant.
                b'0'..=b'9' => {
                    let lo = i;

                    while i < input.len() && input[i].is_ascii_digit() {
                        i += 1;
                    }

                    // This indicates an identifier beginning with a digit,
                    // which is invalid.
                    //
                    // NOTE: Currently don't allow usage of '_' as a separator
                    // between digits.
                    if input[i].is_ascii_lowercase()
                        || input[i].is_ascii_uppercase()
                        || input[i] == b'_'
                    {
                        return Err("invalid identifier encountered".into());
                    }

                    let token =
                        std::str::from_utf8(&input[lo..i]).map_err(|err| format!("{err}"))?;

                    // NOTE: Currently only handling integer constants with no
                    // suffixes.
                    let integer = token.parse::<i32>().map_err(|err| format!("{err}"))?;

                    tokens.push(Token::ConstantInt(integer));
                }
                // Handle identifier or keyword (allowed to begin with '_').
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                    let lo = i;

                    while i < input.len() && (input[i].is_ascii_alphanumeric() || input[i] == b'_')
                    {
                        i += 1;
                    }

                    let token =
                        std::str::from_utf8(&input[lo..i]).map_err(|err| format!("{err}"))?;

                    if KEYWORDS.contains(&token) {
                        tokens.push(Token::Keyword(token.into()))
                    } else {
                        tokens.push(Token::Identifier(token.into()))
                    }
                }
                b'(' => {
                    tokens.push(Token::ParenOpen);
                    i += 1;
                }
                b')' => {
                    tokens.push(Token::ParenClose);
                    i += 1;
                }
                b'{' => {
                    tokens.push(Token::BraceOpen);
                    i += 1;
                }
                b'}' => {
                    tokens.push(Token::BraceClose);
                    i += 1;
                }
                b';' => {
                    tokens.push(Token::Semicolon);
                    i += 1;
                }
                _ => {
                    return Err(format!(
                        "unexpected character encountered: {}",
                        input[i] as char
                    ));
                }
            }
        }

        Ok(Self { tokens })
    }
}

// TODO: Test the output tokens
//
// Reference: https://github.com/nlsandler/writing-a-c-compiler-tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexer_valid_return_zero() {
        let source = b"int main(void) { return 0; }";

        assert!(Lexer::new(source).is_ok())
    }

    #[test]
    fn lexer_valid_return_non_zero() {
        let source = b"int main(void) { return 2; }";

        assert!(Lexer::new(source).is_ok())
    }

    #[test]
    fn lexer_valid_multi_digit() {
        let source = b"int main(void) { return 100; }";

        assert!(Lexer::new(source).is_ok())
    }

    #[test]
    fn lexer_valid_newlines() {
        let source = b"int\nmain\n(\nvoid\n)\n{\nreturn\n0\n;}";

        assert!(Lexer::new(source).is_ok())
    }

    #[test]
    fn lexer_valid_no_newlines() {
        let source = b"int main(void){return 0;}";

        assert!(Lexer::new(source).is_ok())
    }

    #[test]
    fn lexer_valid_whitespace() {
        let source = b"   int   main    (  void)  {   return  0 ; }";

        assert!(Lexer::new(source).is_ok())
    }

    #[test]
    fn lexer_valid_tabs() {
        let source = b"int	main	(	void)	{	return	0	;	}";

        assert!(Lexer::new(source).is_ok())
    }

    #[test]
    fn lexer_invalid_unexpected_symbol() {
        // The '@' symbol doesn't appear in any token, except inside string or
        // character literals.
        let source = b"int main(void) { return 0@1; }";

        assert!(Lexer::new(source).is_err())
    }

    #[test]
    fn lexer_invalid_backslash() {
        // Single backslash ('\') is not a valid token.
        let source = b"\\";

        assert!(Lexer::new(source).is_err())
    }

    #[test]
    fn lexer_invalid_backtick() {
        // Backtick ('`') is not a valid token.
        let source = b"`";

        assert!(Lexer::new(source).is_err())
    }

    #[test]
    fn lexer_invalid_identifier_digit() {
        // Identifiers are not allowed to start with digits, must begin with
        // a non-digit including ('_').
        let source = b"int main(void) { return 1foo; }";

        assert!(Lexer::new(source).is_err())
    }

    #[test]
    fn lexer_invalid_identifier_symbol() {
        // Identifiers are not allowed to start with digits, must begin with
        // a non-digit including ('_').
        let source = b"int main(void) { return @b; }";

        assert!(Lexer::new(source).is_err())
    }
}
