#!/usr/bin/env bash
set -euo pipefail
SECTOR_SLUG="${SECTOR_SLUG:-finance_and_insurance}"
TASK_LIST_FILE="${TASK_LIST_FILE:-pruning_info/experiment_4_sector_weighted_mass/task_selection/${SECTOR_SLUG}.json}"
EXPERIMENT_NAME="experiment_4_sector_weighted_mass_${SECTOR_SLUG}" \
CONFIG_ROOT="configs/models/experiment_4_sector_weighted_mass/${SECTOR_SLUG}" \
TASK_LIST_FILE="$TASK_LIST_FILE" \
bash scripts/experiments/run_experiment_family.sh
