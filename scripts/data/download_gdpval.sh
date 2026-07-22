#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stirrup-py312
mkdir -p GDPval_data
hf download openai/gdpval --repo-type dataset --local-dir GDPval_data
