#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate routing-hf-py312

EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"
PLAN_ROOT="${PLAN_ROOT:?set PLAN_ROOT to the directory containing plans/}"
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
MODEL_ROOT="${MODEL_ROOT:-models/$EXPERIMENT_NAME}"
CONFIG_ROOT="${CONFIG_ROOT:-configs/models/$EXPERIMENT_NAME}"
KEEP_SIZES="${KEEP_SIZES:-192 128 64}"
PLAN_PREFIX="${PLAN_PREFIX:?set PLAN_PREFIX, e.g. weighted_mass_global}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$MODEL_ROOT" "$CONFIG_ROOT"
config_args=()
for keep in $KEEP_SIZES; do
  plan="$PLAN_ROOT/plans/${PLAN_PREFIX}_keep_$(printf '%03d' "$keep").json"
  output="$MODEL_ROOT/keep${keep}"
  [ -f "$plan" ] || { echo "[fatal] missing plan: $plan" >&2; exit 1; }

  args=(
    --source-model "$SOURCE_MODEL"
    --plan "$plan"
    --output-dir "$output"
    --expected-layers 40
    --expected-original-experts 256
  )
  [ "$LOCAL_FILES_ONLY" = "1" ] && args+=(--local-files-only)
  [ "$OVERWRITE" = "1" ] && args+=(--overwrite)
  python scripts/pruning/prune_checkpoint.py "${args[@]}"
  config_args+=(--model "${keep}=$(readlink -f "$output")")
done

python scripts/pruning/configure_pruned_configs.py \
  --base-config configs/run_config.json \
  --output-dir "$CONFIG_ROOT" \
  "${config_args[@]}"

echo "[done] model family: $MODEL_ROOT"
echo "[done] configs: $CONFIG_ROOT"
