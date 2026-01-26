#!/usr/bin/env bash

set -uo pipefail

SRC=example.c
ASM=example.s
BIN=a.out

echo "=== cc2 ==="
cargo r -q -- "$SRC" -p &&
    (gcc "$ASM" -o "$BIN"; "./$BIN"; echo "exit code: $?")

echo
echo "=== gcc ==="
gcc "$SRC" -o "$BIN" &&
    ("./$BIN"; echo "exit code: $?")

echo
echo "=== clang ==="
clang "$SRC" -o "$BIN" &&
    ("./$BIN"; echo "exit code: $?")

rm -f "$BIN" "$ASM"
