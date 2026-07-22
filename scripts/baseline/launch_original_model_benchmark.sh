#!/usr/bin/env bash
set -euo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-../outputs/routecat_baseline_original_model/${RUN_STAMP}}"
RUNS_DIR="${RUNS_DIR:-artifacts/baseline_runs_${RUN_STAMP}}"

python scripts/validation/validate_evaluation_contract.py

CONFIG_PATH=configs/run_config.json \
MAX_TURNS_SCHEDULE="80 50 120" \
TASK_MAX_ATTEMPTS=3 \
RUN_ACCOUNTING=0 \
STOP_VLLM_BEFORE_ACCOUNTING=0 \
SKIP_TASK_EXPORT=1 \
OUT_DIR="$OUT_DIR" \
RUNS_DIR="$RUNS_DIR" \
bash scripts/baseline/run_baseline_220.sh

echo "$OUT_DIR" > ../outputs/latest_routecat_baseline_output_dir.txt
echo "$OUT_DIR/runs" > ../outputs/latest_routecat_baseline_runs_dir.txt

echo "[done] baseline output: $OUT_DIR"
echo "[next] validate with: python scripts/validation/validate_baseline_results.py --result-root '$OUT_DIR'"
