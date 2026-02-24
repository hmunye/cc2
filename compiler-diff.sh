#!/usr/bin/env bash

# set -x
set -uo pipefail

src="${1:-example.c}"
asm="${src%.c}.s"
bin=a.out

printf "\x1b[1;45m=== cc2 ===\x1b[0m\n"
cargo r -q --release -- "$src" -p -O1 -o "$asm" &&
    (gcc "$asm" -o "$bin"; "./$bin"; echo "exit code: $?")

echo
printf "\x1b[1;45m=== gcc ===\x1b[0m\n"
gcc -std=c17 -pedantic "$src" -o "$bin" &&
    ("./$bin"; echo "exit code: $?")

echo
printf "\x1b[1;45m=== clang ===\x1b[0m\n"
clang -std=c17 -pedantic "$src" -o "$bin" &&
    ("./$bin"; echo "exit code: $?")

rm -f "$bin" "$asm"
