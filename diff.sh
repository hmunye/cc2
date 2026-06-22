#!/usr/bin/env bash

set -uo pipefail

SRC="${1:-example.c}"
ASM="${SRC%.c}.s"
BIN=a.out

printf "\x1b[1;45m=== cc2 ===\x1b[0m\n"
cargo r -q --release -- "$SRC" -p -O -o "$ASM" &&
    (gcc "$ASM" -o "$BIN"; "./$BIN"; echo "exit code: $?")

printf "\n\x1b[1;45m=== gcc ===\x1b[0m\n"
gcc -std=c17 -pedantic "$SRC" -o "$BIN" &&
    ("./$BIN"; echo "exit code: $?")

printf "\n\x1b[1;45m=== clang ===\x1b[0m\n"
clang -std=c17 -pedantic "$SRC" -o "$BIN" &&
    ("./$BIN"; echo "exit code: $?")

rm -f "$ASM" "$BIN"
