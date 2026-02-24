//! Compiler Frontend
//!
//! Responsible for processing a _C_ translation unit into a structured,
//! semantically valid abstract syntax tree (_AST_).

pub mod lexer;
pub mod parser;

pub use lexer::Lexer;
pub use parser::ast::{self, parse_ast};
pub use parser::semantics::SymbolTable;
pub use parser::types;
