#!/usr/bin/env bash

set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <file>" >&2
  exit 1
fi

src="${1:-example.c}"
asm=example.s
bin=a.out

printf "\x1b[1;45m=== cc2 ===\x1b[0m\n"
cargo r -q --release -- "$src" -p -o "$asm" &&
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
