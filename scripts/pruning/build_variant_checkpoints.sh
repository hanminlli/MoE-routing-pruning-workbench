#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate routing-hf-py312

: "${EXPERIMENT_ID:?set EXPERIMENT_ID, e.g. experiment_2}"
: "${CRITERION:?set CRITERION}"
: "${PRUNING_ROOT:?set PRUNING_ROOT}"
: "${CONFIG_OUTPUT_DIR:?set CONFIG_OUTPUT_DIR}"
: "${MODEL_LABEL:?set MODEL_LABEL}"

ACCOUNTING_ROOT="${ACCOUNTING_ROOT:-accounting_result}"
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
PLAN_KEEP_SIZES="${PLAN_KEEP_SIZES:-192,128,64}"
BUILD_KEEP_SIZES="${BUILD_KEEP_SIZES:-$PLAN_KEEP_SIZES}"
REUSE_EXISTING_CHECKPOINTS="${REUSE_EXISTING_CHECKPOINTS:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
MODEL_ROOT="${MODEL_ROOT:-../outputs/models/${EXPERIMENT_ID}}"
CALIBRATION_POLICY="${CALIBRATION_POLICY:-configs/calibration_policy.json}"

case "$CRITERION" in
  weighted_frequency|task_normalized_weighted_frequency|unweighted_frequency) ;;
  *)
    echo "[fatal] unsupported variant criterion: $CRITERION" >&2
    exit 1
    ;;
esac

if [ ! -f "$CALIBRATION_POLICY" ]; then
  echo "[fatal] missing calibration policy: $CALIBRATION_POLICY" >&2
  exit 1
fi

read_policy_field() {
  local field="$1"
  python - "$CALIBRATION_POLICY" "$field" <<'PY'
import json
import sys
from pathlib import Path
p, field = Path(sys.argv[1]), sys.argv[2]
obj = json.loads(p.read_text(encoding="utf-8"))
value = obj[field]
if isinstance(value, list):
    print(",".join(str(x) for x in value))
else:
    print(value)
PY
}

EXCLUDE_TASKS="${EXCLUDE_TASKS-$(read_policy_field exclude_tasks)}"
BUCKET="${BUCKET:-$(read_policy_field bucket)}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-$(read_policy_field expected_layers)}"
EXPECTED_EXPERTS="${EXPECTED_EXPERTS:-$(read_policy_field expected_experts_per_layer)}"

normalize_keep_sizes() {
  python - "$1" <<'PY'
import sys
seen = []
for x in sys.argv[1].replace(" ", ",").split(","):
    x = x.strip()
    if not x:
        continue
    k = int(x)
    if k not in seen:
        seen.append(k)
print(" ".join(str(x) for x in seen))
PY
}

PLAN_KEEP_LIST="$(normalize_keep_sizes "$PLAN_KEEP_SIZES")"
BUILD_KEEP_LIST="$(normalize_keep_sizes "$BUILD_KEEP_SIZES")"

mkdir -p "$PRUNING_ROOT" "$CONFIG_OUTPUT_DIR" "$MODEL_ROOT"

echo "[info] experiment=$EXPERIMENT_ID criterion=$CRITERION"
echo "[info] accounting_root=$ACCOUNTING_ROOT"
echo "[info] plan_keep_sizes=$PLAN_KEEP_LIST"
echo "[info] build_keep_sizes=$BUILD_KEEP_LIST"
echo "[info] calibration_policy=$CALIBRATION_POLICY"
echo "[info] exclude_tasks=${EXCLUDE_TASKS:-<none>} bucket=$BUCKET"
echo "[info] source_model=$SOURCE_MODEL local_files_only=$LOCAL_FILES_ONLY"

python scripts/pruning/prepare_experiment_variant_plans.py \
  --accounting-root "$ACCOUNTING_ROOT" \
  --output-root "$PRUNING_ROOT" \
  --criterion "$CRITERION" \
  --keep-sizes "$(tr ' ' ',' <<< "$PLAN_KEEP_LIST")" \
  --bucket "$BUCKET" \
  --exclude-tasks "$EXCLUDE_TASKS" \
  --expected-layers "$EXPECTED_LAYERS" \
  --expected-experts "$EXPECTED_EXPERTS"

is_valid_checkpoint() {
  local d="$1"
  [ -d "$d" ] && [ -f "$d/config.json" ] && [ -f "$d/pruning_manifest.json" ] && [ -f "$d/model.safetensors.index.json" ]
}

config_args=()
for keep in $BUILD_KEEP_LIST; do
  plan="$PRUNING_ROOT/plans/${CRITERION}_keep_$(printf '%03d' "$keep").json"
  model_out="$MODEL_ROOT/Qwen3.6-35B-A3B-${MODEL_LABEL}-keep${keep}"
  pointer="../outputs/latest_${EXPERIMENT_ID}_pruned_model_keep${keep}_dir.txt"

  if [ ! -f "$plan" ]; then
    echo "[fatal] missing pruning plan for keep${keep}: $plan" >&2
    exit 1
  fi

  if is_valid_checkpoint "$model_out" && [ "$REUSE_EXISTING_CHECKPOINTS" = "1" ] && [ "$FORCE_REBUILD" != "1" ]; then
    echo "[reuse] valid checkpoint already exists: $model_out"
  else
    if [ -e "$model_out" ]; then
      if [ "$FORCE_REBUILD" = "1" ]; then
        echo "[rebuild] removing existing path: $model_out"
        rm -rf "$model_out"
      else
        echo "[fatal] checkpoint path exists but is incomplete or reuse is disabled: $model_out" >&2
        echo "        Set FORCE_REBUILD=1 to remove and rebuild it." >&2
        exit 1
      fi
    fi

    prune_args=(
      --source-model "$SOURCE_MODEL"
      --plan "$plan"
      --output-dir "$model_out"
      --expected-layers "$EXPECTED_LAYERS"
      --expected-original-experts "$EXPECTED_EXPERTS"
    )
    if [ "$LOCAL_FILES_ONLY" = "1" ]; then
      prune_args+=(--local-files-only)
    fi

    python scripts/pruning/prune_checkpoint.py "${prune_args[@]}"
  fi

  if ! is_valid_checkpoint "$model_out"; then
    echo "[fatal] checkpoint validation failed after build: $model_out" >&2
    exit 1
  fi

  readlink -f "$model_out" > "$pointer"
  config_args+=(--model "${keep}=$(cat "$pointer")")
done

if [ "${#config_args[@]}" -eq 0 ]; then
  echo "[fatal] no checkpoint sizes selected" >&2
  exit 1
fi

python scripts/pruning/configure_pruned_configs.py \
  --base-config configs/run_config.json \
  --output-dir "$CONFIG_OUTPUT_DIR" \
  "${config_args[@]}"

printf '[done] %s selected checkpoint family created\n' "$EXPERIMENT_ID"
printf '[done] configs: %s\n' "$CONFIG_OUTPUT_DIR"
