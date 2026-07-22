#!/usr/bin/env bash
set -euo pipefail

TYPE="${1:-}"
ARCHIVE="${2:-}"
[ -f "$ARCHIVE" ] || { echo "[fatal] archive not found: $ARCHIVE" >&2; exit 1; }
case "$TYPE" in
  ordinary)
    rm -rf accounting_result
    tar -xzf "$ARCHIVE" -C .
    [ -d accounting_result ] || { echo "[fatal] archive did not create accounting_result/" >&2; exit 1; }
    python scripts/validation/validate_accounting_input.py --accounting-root accounting_result
    ;;
  advanced)
    rm -rf advanced_accounting_result
    mkdir -p advanced_accounting_result
    tar -xzf "$ARCHIVE" -C advanced_accounting_result
    python scripts/validation/validate_advanced_accounting.py --root advanced_accounting_result
    ;;
  *)
    echo "usage: $0 ordinary|advanced archive.tar.gz" >&2
    exit 2
    ;;
esac
