#!/bin/sh
set -eu

PATTERN='(^|[^A-Za-z])(/Users/|/home/|[A-Za-z]:\\|[A-Za-z]:/)'

echo "Checking tracked file contents for absolute local paths..."
MATCHES="$(git grep -nE "$PATTERN" -- . || true)"
if [ -n "$MATCHES" ]; then
    echo "FAIL: absolute local paths found in tracked content:"
    printf '%s\n' "$MATCHES"
    exit 1
fi
echo "OK: no absolute local paths in tracked content."

echo "Checking tracked path names for absolute prefixes..."
PATH_MATCHES="$(git ls-files | grep -nE '^/|^[A-Za-z]:\\' || true)"
if [ -n "$PATH_MATCHES" ]; then
    echo "FAIL: absolute tracked path names found:"
    printf '%s\n' "$PATH_MATCHES"
    exit 1
fi
echo "OK: no absolute tracked path names."

echo "Portability hygiene check passed."
