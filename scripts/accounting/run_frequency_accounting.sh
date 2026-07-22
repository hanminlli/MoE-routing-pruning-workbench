#!/usr/bin/env bash
set -uo pipefail

RUNS_ROOT="${RUNS_ROOT:-accounting/runs}"
RESULT_ROOT="${RESULT_ROOT:-accounting_result}"
CONFIG_PATH="${CONFIG_PATH:-configs/run_config.json}"
CALL_INDICES="${CALL_INDICES:-all}"
RESUME="${RESUME:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

# Optional task filters, inclusive lower bound and exclusive upper bound.
# Example: START_TASK=50 END_TASK=80 bash scripts/accounting/run_frequency_accounting.sh
START_TASK="${START_TASK:-}"
END_TASK="${END_TASK:-}"

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/manifests"

MASTER_LOG="$RESULT_ROOT/master_accounting.log"
SUCCESS_TSV="$RESULT_ROOT/manifests/successful_accounting.tsv"
FAILED_TSV="$RESULT_ROOT/manifests/failed_accounting.tsv"
ALL_TSV="$RESULT_ROOT/manifests/accounting_manifest.tsv"

exec > >(tee -a "$MASTER_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

get_task_num() {
  local run_dir="$1"
  local base
  base="$(basename "$run_dir")"

  local n
  n="$(echo "$base" | sed -n 's/.*gdpval_task_\([0-9][0-9][0-9][0-9]\).*/\1/p' | head -1)"

  if [ -n "$n" ]; then
    echo "$n"
    return 0
  fi

  if [ -f "$run_dir/status.json" ]; then
    python - "$run_dir/status.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    x = json.loads(p.read_text())
    print(f"{int(x['row_index']):04d}")
except Exception:
    print("unknown")
PY
    return 0
  fi

  echo "unknown"
}

task_in_range() {
  local task_num="$1"

  if [ "$task_num" = "unknown" ]; then
    return 0
  fi

  local i=$((10#$task_num))

  if [ -n "$START_TASK" ] && [ "$i" -lt "$START_TASK" ]; then
    return 1
  fi

  if [ -n "$END_TASK" ] && [ "$i" -ge "$END_TASK" ]; then
    return 1
  fi

  return 0
}

metadata_is_good() {
  local meta="$1"

  python - "$meta" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    m = json.loads(p.read_text())
except Exception:
    raise SystemExit(1)

ok = (
    m.get("prompt_matches_expected_total") is True
    and m.get("generated_matches_expected_total") is True
)
raise SystemExit(0 if ok else 1)
PY
}

verify_model_calls() {
  local model_calls="$1"

  python - "$model_calls" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
calls = []
bad = []

for lineno, line in enumerate(p.read_text(encoding="utf-8").split("\n"), start=1):
    if not line.strip():
        continue

    try:
        c = json.loads(line)
    except Exception as e:
        bad.append(f"line {lineno}: JSON parse error: {type(e).__name__}: {e}")
        continue

    calls.append(c)

    ci = c.get("call_index", len(calls))
    usage = c.get("response", {}).get("token_usage", {}) if isinstance(c.get("response"), dict) else {}

    logged_input = int(usage.get("input", 0) or 0)
    logged_answer = int(usage.get("answer", 0) or 0)
    logged_reasoning = int(usage.get("reasoning", 0) or 0)
    logged_generated = logged_answer + logged_reasoning

    replay = c.get("replay", {}) if isinstance(c.get("replay"), dict) else {}
    pids = replay.get("prompt_token_ids_from_response")
    gids = replay.get("generated_token_ids_from_response")

    if not isinstance(pids, list):
        bad.append(f"call {ci}: missing prompt_token_ids_from_response")
        continue

    if not isinstance(gids, list):
        bad.append(f"call {ci}: missing generated_token_ids_from_response")
        continue

    if len(pids) != logged_input:
        bad.append(f"call {ci}: prompt ids len {len(pids)} != logged input {logged_input}")

    if len(gids) != logged_generated:
        bad.append(f"call {ci}: generated ids len {len(gids)} != logged generated {logged_generated}")

if not calls:
    bad.append("no JSONL calls found")

print(f"num_calls={len(calls)}")

if bad:
    print("[bad model_calls]")
    for x in bad[:30]:
        print(x)
    if len(bad) > 30:
        print(f"... plus {len(bad) - 30} more")
    raise SystemExit(1)

print("[ok] model_calls token IDs are usable")
PY
}

run_one() {
  local run_dir="$1"
  local task_num="$2"

  local model_calls="$run_dir/model_calls.jsonl"
  local run_base
  run_base="$(basename "$run_dir")"

  local count_dir="$RESULT_ROOT/task_${task_num}__${run_base}"
  local count_log="$RESULT_ROOT/logs/task_${task_num}__${run_base}.log"

  mkdir -p "$count_dir"

  if [ "$RESUME" = "1" ] && [ -f "$count_dir/metadata.json" ] && metadata_is_good "$count_dir/metadata.json"; then
    log "[skip task $task_num] already accounted successfully: $count_dir"
    echo -e "$task_num\t$run_dir\t$count_dir\tskipped_existing_success\t0" >> "$ALL_TSV"
    return 0
  fi

  log "================================================================"
  log "[accounting task $task_num] run_dir=$run_dir"
  log "[accounting task $task_num] out_dir=$count_dir"
  log "================================================================"

  if [ ! -f "$model_calls" ]; then
    log "[fail task $task_num] missing model_calls.jsonl: $model_calls"
    echo -e "$task_num\t$run_dir\t$count_dir\tmissing_model_calls\t1" >> "$FAILED_TSV"
    echo -e "$task_num\t$run_dir\t$count_dir\tmissing_model_calls\t1" >> "$ALL_TSV"
    return 0
  fi

  log "[task $task_num] verifying model_calls token IDs"
  if ! verify_model_calls "$model_calls" 2>&1 | tee "$count_dir/token_id_check.log"; then
    log "[fail task $task_num] model_calls token check failed"
    echo -e "$task_num\t$run_dir\t$count_dir\tbad_model_calls\t1" >> "$FAILED_TSV"
    echo -e "$task_num\t$run_dir\t$count_dir\tbad_model_calls\t1" >> "$ALL_TSV"
    return 0
  fi

  local args=(
    --config "$CONFIG_PATH"
    --model-calls "$model_calls"
    --call-indices "$CALL_INDICES"
    --out-dir "$count_dir"
  )

  if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    args+=(--local-files-only)
  fi

  local start_epoch
  local end_epoch
  local duration_sec
  start_epoch="$(date +%s)"

  set +e
  PYTHONPATH=. python scripts/accounting/replay_frequency.py "${args[@]}" \
    2>&1 | tee "$count_log"
  local exit_code=${PIPESTATUS[0]}
  set -u

  end_epoch="$(date +%s)"
  duration_sec=$((end_epoch - start_epoch))

  if [ "$exit_code" -ne 0 ]; then
    log "[fail task $task_num] accounting script failed exit_code=$exit_code duration_sec=$duration_sec"
    echo -e "$task_num\t$run_dir\t$count_dir\taccounting_failed\t$exit_code" >> "$FAILED_TSV"
    echo -e "$task_num\t$run_dir\t$count_dir\taccounting_failed\t$exit_code" >> "$ALL_TSV"
    return 0
  fi

  if [ ! -f "$count_dir/metadata.json" ]; then
    log "[fail task $task_num] metadata.json missing after accounting"
    echo -e "$task_num\t$run_dir\t$count_dir\tmissing_metadata\t1" >> "$FAILED_TSV"
    echo -e "$task_num\t$run_dir\t$count_dir\tmissing_metadata\t1" >> "$ALL_TSV"
    return 0
  fi

  if ! metadata_is_good "$count_dir/metadata.json"; then
    log "[fail task $task_num] metadata conservation check failed"
    echo -e "$task_num\t$run_dir\t$count_dir\tmetadata_check_failed\t1" >> "$FAILED_TSV"
    echo -e "$task_num\t$run_dir\t$count_dir\tmetadata_check_failed\t1" >> "$ALL_TSV"
    return 0
  fi

  log "[ok task $task_num] accounting passed duration_sec=$duration_sec"
  echo -e "$task_num\t$run_dir\t$count_dir\tfinished\t0" >> "$SUCCESS_TSV"
  echo -e "$task_num\t$run_dir\t$count_dir\tfinished\t0" >> "$ALL_TSV"
}

main() {
  log "[info] current dir: $PWD"
  log "[info] RUNS_ROOT=$RUNS_ROOT"
  log "[info] RESULT_ROOT=$RESULT_ROOT"
  log "[info] CONFIG_PATH=$CONFIG_PATH"
  log "[info] CALL_INDICES=$CALL_INDICES"
  log "[info] LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"
  log "[info] START_TASK=${START_TASK:-<none>} END_TASK=${END_TASK:-<none>}"

  if [ ! -d "$RUNS_ROOT" ]; then
    log "[fatal] runs root does not exist: $RUNS_ROOT"
    exit 1
  fi

  if [ ! -f scripts/accounting/replay_frequency.py ]; then
    log "[fatal] missing accounting script: scripts/accounting/replay_frequency.py"
    exit 1
  fi

  python -m py_compile scripts/accounting/replay_frequency.py || exit 1

  : > "$SUCCESS_TSV"
  : > "$FAILED_TSV"
  : > "$ALL_TSV"

  mapfile -t run_dirs < <(
    find "$RUNS_ROOT" -type f -name model_calls.jsonl -printf '%h\n' | sort
  )

  log "[info] discovered run dirs with model_calls.jsonl: ${#run_dirs[@]}"

  if [ "${#run_dirs[@]}" -eq 0 ]; then
    log "[fatal] no model_calls.jsonl found under $RUNS_ROOT"
    exit 1
  fi

  for run_dir in "${run_dirs[@]}"; do
    task_num="$(get_task_num "$run_dir")"

    if ! task_in_range "$task_num"; then
      continue
    fi

    run_one "$run_dir" "$task_num"
  done

  log "[done] accounting loop finished"
  log "[done] master log: $MASTER_LOG"

  log "[summary] successful:"
  wc -l "$SUCCESS_TSV" 2>/dev/null || true

  log "[summary] failed:"
  if [ -s "$FAILED_TSV" ]; then
    cat "$FAILED_TSV"
  else
    echo "[none]"
  fi
}

main "$@"
