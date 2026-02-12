# cc2

Tiny C Compiler (subset of _C17_).

> [!WARNING]  
> This project is experimental and not intended for production use.

> [!NOTE]  
> Currently only emits `gas_x86-64_linux` assembly. Other backends may be added in the future.

## Language Features Supported

### Types

- [x] `int`

### Functions

- [x] Function definitions and declarations (supports types listed above)
- [x] `int main(void) { ... }`

### Declarations

- [x] Local variables
- [x] Global variables
- [x] Lexical block scoping
- [x] Shadowing
- [x] Internal and external linkage (`extern` and `static`)
- [x] Static/automatic storage duration (local/file-scope variables and `static` locals)

### Statements

- [x] `return`
- [x] Expression statements
- [x] `if` / `else`
- [x] `goto`/labeled statements
- [x] Compound statements (`{ ... }`)
- [x] `for` / `do` / `while`
- [x] `break` / `continue`
- [x] `switch`
- [x] `default` / `case` (currently only support integer literal expressions)
- [x] Empty statements (`;`)

### Expressions

- [x] Integer constants
- [x] Variable references
- [x] Unary operators: `-`, `!`, `~`, `++` (postfix and prefix), `--` (postfix and prefix)
- [x] Binary operators: `+`, `-`, `*`, `/`, `%`
- [x] Binary bitwise operators: `&`, `|`, `^`
- [x] Binary shift operators: `<<`, `>>`
- [x] Binary logical operators: `&&`, `||`
- [x] Comparison operators: `<`, `<=`, `>`, `>=`, `==`, `!=`
- [x] Assignment operators: `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`
- [x] Conditional (ternary) operator: `? :`
- [x] Function calls

### Optimizations

- [x] Constant folding

## Testing

Compiler is tested using an external test suite: [`nlsandler/writing-a-c-compiler-tests/`](https://github.com/nlsandler/writing-a-c-compiler-tests/).

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

### Example: `hello_world.c`

```bash
# 1. Preprocess separately using `GCC` or `cpp`:
gcc -E hello_world.c -o hello_world.i
# or
cpp hello_world.c -o hello_world.i

# 2. Compile the preprocessed file to assembly:
./target/release/cc2 hello_world.i

# 3a. Use `GCC` to assemble and link to an executable:
gcc hello_world.s -o hello_world

# 3b. Or manually invoke assembler and linker:
as hello_world.s -o hello_world.o   # assemble to object file
ld hello_world.s -o hello_world     # link to produce executable

# -----------------------------
# Shortcut: Use the -p flag
# -----------------------------
#
# The `-p` flag preprocesses and compiles in a single step.
# This is just a convenience; no separate preprocessed file is needed.
./target/release/cc2 -p hello_world.c

# Link as usual:
gcc hello_world.s -o hello_world
```

## License

This project is licensed under the [MIT License].

[MIT License]: https://github.com/hmunye/cc2/blob/main/LICENSE

## References

- [Writing a C Compiler](https://norasandler.com/book/)
- [C17 Standard Draft](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2310.pdf)
- [System V x86-64 ABI](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Intel x86-64 ISA](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
