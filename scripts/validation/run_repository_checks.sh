#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"

python -m compileall -q src scripts tests
while IFS= read -r -d '' script; do
  bash -n "$script"
done < <(find scripts patches -type f -name '*.sh' -print0)

python scripts/validation/validate_config_paths.py --root .
python scripts/validation/check_markdown_links.py --root .
python scripts/validation/audit_public_tree.py --root .
PYTHONPATH=src:. pytest -q

if command -v ruff >/dev/null 2>&1; then
  ruff check src tests scripts/pruning/generate_compact_accounting_plans.py scripts/steering
else
  echo "[warn] ruff is not installed; syntax and tests passed, lint skipped"
fi

find . -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf .pytest_cache
printf '%s\n' 'REPOSITORY CHECKS: PASSED'
