#!/usr/bin/env bash
set -euo pipefail
mkdir -p \
  artifacts/tasks artifacts/runs artifacts/expert_counts \
  accounting_result advanced_accounting_result pruning_info \
  models results ../outputs
for path in artifacts/tasks artifacts/runs accounting_result advanced_accounting_result pruning_info results; do
  touch "$path/.gitkeep"
done
echo "[done] runtime layout initialized"
