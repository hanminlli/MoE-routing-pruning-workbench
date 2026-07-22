#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate routing-hf-py312
ACCOUNTING_ROOT="${ACCOUNTING_ROOT:-accounting_result}"
PRUNING_ROOT="${PRUNING_ROOT:-pruning_info/weighted_frequency_all_response_tokens}"
CRITERION="${CRITERION:-weighted_frequency}"
EXCLUDE_TASKS="${EXCLUDE_TASKS:-188}"
mkdir -p "$PRUNING_ROOT" ../outputs/models
python scripts/pruning/prepare_pruning_plans.py \
  --accounting-root "$ACCOUNTING_ROOT" --output-root "$PRUNING_ROOT" \
  --keep-sizes "192,128,64" --bucket generated_output_prediction \
  --exclude-tasks "$EXCLUDE_TASKS"
for keep in 192 128 64; do
  plan="$PRUNING_ROOT/plans/${CRITERION}_keep_$(printf '%03d' "$keep").json"
  model_out="../outputs/models/Qwen3.6-35B-A3B-${CRITERION}-keep${keep}"
  python scripts/pruning/prune_checkpoint.py \
    --source-model Qwen/Qwen3.6-35B-A3B --plan "$plan" \
    --output-dir "$model_out" --expected-layers 40 \
    --expected-original-experts 256 --local-files-only
  readlink -f "$model_out" > "../outputs/latest_pruned_model_keep${keep}_dir.txt"
done
python scripts/pruning/configure_pruned_configs.py \
  --keep192 "$(cat ../outputs/latest_pruned_model_keep192_dir.txt)" \
  --keep128 "$(cat ../outputs/latest_pruned_model_keep128_dir.txt)" \
  --keep64 "$(cat ../outputs/latest_pruned_model_keep64_dir.txt)"
