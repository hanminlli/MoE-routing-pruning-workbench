#!/usr/bin/env bash
set -euo pipefail

ACCOUNTING="${ACCOUNTING:?set ACCOUNTING to ordinary_response_routing_by_task.csv.gz}"
TASK_METADATA="${TASK_METADATA:-}"
KEEP_SIZES="${KEEP_SIZES:-192,128,64}"

python scripts/pruning/generate_compact_accounting_plans.py \
  --accounting "$ACCOUNTING" \
  --experiment weighted_mass \
  --output-root pruning_info/experiment_1_weighted_mass \
  --keep-sizes "$KEEP_SIZES"

python scripts/pruning/generate_compact_accounting_plans.py \
  --accounting "$ACCOUNTING" \
  --experiment task_normalized_weighted_mass \
  --output-root pruning_info/experiment_2_task_normalized_weighted_mass \
  --keep-sizes "$KEEP_SIZES"

python scripts/pruning/generate_compact_accounting_plans.py \
  --accounting "$ACCOUNTING" \
  --experiment unweighted_count \
  --output-root pruning_info/experiment_3_unweighted_count \
  --keep-sizes "$KEEP_SIZES"

if [ -n "$TASK_METADATA" ]; then
  python scripts/pruning/prepare_sector_task_lists.py \
    --task-metadata "$TASK_METADATA" \
    --output-root pruning_info/experiment_4_sector_weighted_mass/task_selection
  python scripts/pruning/generate_compact_accounting_plans.py \
    --accounting "$ACCOUNTING" \
    --task-metadata "$TASK_METADATA" \
    --experiment sector_weighted_mass \
    --output-root pruning_info/experiment_4_sector_weighted_mass \
    --keep-sizes "$KEEP_SIZES"
else
  echo "[warn] TASK_METADATA is unset; Experiment 4 plans were not generated"
fi
