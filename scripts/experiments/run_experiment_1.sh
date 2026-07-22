#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_NAME="experiment_1_weighted_mass" \
CONFIG_ROOT="configs/models/experiment_1_weighted_mass" \
bash scripts/experiments/run_experiment_family.sh
