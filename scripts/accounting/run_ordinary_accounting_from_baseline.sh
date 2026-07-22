#!/usr/bin/env bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
if [ -z "${RUNS_ROOT:-}" ]; then
  if [ ! -f ../outputs/latest_routecat_baseline_runs_dir.txt ]; then
    echo "[fatal] set RUNS_ROOT or run the RouteCat baseline first" >&2
    exit 1
  fi
  RUNS_ROOT="$(cat ../outputs/latest_routecat_baseline_runs_dir.txt)"
fi
RESULT_ROOT="${RESULT_ROOT:-../outputs/routecat_ordinary_accounting/${RUN_STAMP}}"

RUNS_ROOT="$RUNS_ROOT" \
RESULT_ROOT="$RESULT_ROOT" \
CONFIG_PATH=configs/run_config.json \
CALL_INDICES=all \
RESUME=1 \
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}" \
bash scripts/accounting/run_frequency_accounting.sh

echo "$RESULT_ROOT" > ../outputs/latest_routecat_ordinary_accounting_dir.txt
python scripts/validation/validate_accounting_input.py \
  --accounting-root "$RESULT_ROOT" \
  --output "$RESULT_ROOT/accounting_validation.json"

echo "[done] ordinary accounting: $RESULT_ROOT"
echo "[next] package it for manual upload with scripts/accounting/package_accounting_results.sh"
