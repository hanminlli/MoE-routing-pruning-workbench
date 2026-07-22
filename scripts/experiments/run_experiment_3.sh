#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_NAME="experiment_3_unweighted_count" \
CONFIG_ROOT="configs/models/experiment_3_unweighted_count" \
bash scripts/experiments/run_experiment_family.sh
