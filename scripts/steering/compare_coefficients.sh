#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:?set MODEL to a pruned checkpoint path}"
ARTIFACT="${ARTIFACT:?set ARTIFACT to a steering artifact .pt file}"
PROMPTS_JSONL="${PROMPTS_JSONL:?set PROMPTS_JSONL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/phase2_steering}"
COEFFICIENTS="${COEFFICIENTS:--4 -2 -1 0 1 2 4}"
POSITION_MODE="${POSITION_MODE:-last}"

mkdir -p "$OUTPUT_ROOT"
for coefficient in $COEFFICIENTS; do
  tag="$(printf '%s' "$coefficient" | sed 's/-/neg_/; s/\./p/g')"
  python scripts/steering/apply_steering.py \
    --model "$MODEL" \
    --artifact "$ARTIFACT" \
    --coefficient "$coefficient" \
    --position-mode "$POSITION_MODE" \
    --prompts-jsonl "$PROMPTS_JSONL" \
    --output "$OUTPUT_ROOT/coefficient_${tag}.jsonl" \
    --trust-remote-code
done
