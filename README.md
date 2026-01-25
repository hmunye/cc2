# cc2

Tiny C Compiler (subset of _C17_).

> [!NOTE]  
> Currently only emits `gas_x86-64_linux` (`GNU` assembler) assembly. Other backends may be added in the future.

## Language Features Supported

### Types

- [x] `int` (32-bit integer on `x86-64` Linux)

### Declarations

- [x] Local variables
- [x] Lexical block scoping
- [x] Shadowing

### Statements

- [x] `return`
- [x] Expression statements
- [x] `if` / `else`
- [x] `goto`/labeled statements
- [x] Compound statements (`{ ... }`)
- [x] `for` / `do` / `while`
- [x] `break` / `continue`
- [x] Empty statements (`;`)

### Expressions

- [x] Integer constants
- [x] Variable references
- [x] Unary operators: `-`, `!`, `~`, `++`, `--`
- [x] Binary operators: `+`, `-`, `*`, `/`, `%`
- [x] Binary bitwise operators: `&`, `|`, `^`
- [x] Binary shift operators: `<<`, `>>`
- [x] Binary logical operators: `&&`, `||`
- [x] Comparison operators: `<`, `<=`, `>`, `>=`, `==`, `!=`
- [x] Assignment operators: `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`
- [x] Conditional (ternary) operator: `? :`

### Functions

- [x] `int main(void)`

## Quick Start

Clone the repository and build the project:

```bash
git clone https://github.com/hmunye/cc2.git
cd cc2
cargo build --release
```

Display help and available flags:

```bash
./target/release/cc2 --help
```

Optionally, install globally:

```bash
cargo install --path .
cc2 --help
```

## License

This project is licensed under the [MIT License].

[MIT License]: https://github.com/hmunye/cc2/blob/main/LICENSE

## References

- [Writing a C Compiler](https://norasandler.com/book/)
- [C17 Standard Draft](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2310.pdf)
- [System V x86-64 ABI](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Intel x86-64 ISA](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
