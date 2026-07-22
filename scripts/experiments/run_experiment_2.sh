#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_NAME="experiment_2_task_normalized_weighted_mass" \
CONFIG_ROOT="configs/models/experiment_2_task_normalized_weighted_mass" \
bash scripts/experiments/run_experiment_family.sh
