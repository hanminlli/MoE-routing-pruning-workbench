#!/usr/bin/env bash
set -uo pipefail

RUNS_ROOT="${RUNS_ROOT:-accounting/runs}"
RESULT_ROOT="${RESULT_ROOT:-accounting_response_activation_result}"
CONFIG_PATH="${CONFIG_PATH:-configs/run_config.json}"
CALL_INDICES="${CALL_INDICES:-all}"
RESUME="${RESUME:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
ACTIVATION_CHUNK_SIZE="${ACTIVATION_CHUNK_SIZE:-1024}"
COMPRESS_RESULTS="${COMPRESS_RESULTS:-1}"

# Inclusive lower bound and exclusive upper bound.
START_TASK="${START_TASK:-}"
END_TASK="${END_TASK:-}"

SCRIPT="scripts/accounting/replay_response_activation_metrics.py"

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/manifests"

MASTER_LOG="$RESULT_ROOT/master_response_activation.log"
SUCCESS_TSV="$RESULT_ROOT/manifests/successful_response_activation.tsv"
FAILED_TSV="$RESULT_ROOT/manifests/failed_response_activation.tsv"
ALL_TSV="$RESULT_ROOT/manifests/response_activation_manifest.tsv"
DISCOVERED_TSV="$RESULT_ROOT/manifests/discovered_successful_runs.tsv"

exec > >(tee -a "$MASTER_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

metadata_is_good() {
  local meta="$1"

  python - "$meta" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    m = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

ok = (
    m.get("mode") == "response_activation_nine_statistics_v1"
    and m.get("response_count_matches_expected_total") is True
    and m.get("gate_sum_matches_expected_total") is True
    and m.get("all_layer_count_checks_pass") is True
    and m.get("all_layer_gate_checks_pass") is True
    and m.get("all_scores_finite") is True
    and m.get("all_scores_nonnegative") is True
)
raise SystemExit(0 if ok else 1)
PY
}

verify_model_calls() {
  local model_calls="$1"

  python - "$model_calls" <<'PY'
import json
import sys
from pathlib import Path
from typing import Any


def escape_raw_control_characters_inside_json_strings(text: str) -> tuple[str, int]:
    out = []
    in_string = False
    escaped = False
    repaired = 0
    short = {"\b": r"\b", "\t": r"\t", "\n": r"\n", "\f": r"\f", "\r": r"\r"}

    for ch in text:
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
                escaped = False
            continue

        if escaped:
            out.append(ch)
            escaped = False
        elif ch == "\\":
            out.append(ch)
            escaped = True
        elif ch == '"':
            out.append(ch)
            in_string = False
        elif ord(ch) < 0x20:
            out.append(short.get(ch, f"\\u{ord(ch):04x}"))
            repaired += 1
        else:
            out.append(ch)

    if in_string:
        raise ValueError("file ends inside an unterminated JSON string")
    return "".join(out), repaired


def decode_object_stream(text: str, source: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    calls = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError as exc:
            left = max(0, exc.pos - 120)
            right = min(len(text), exc.pos + 120)
            context = text[left:right].replace("\n", r"\n").replace("\r", r"\r")
            raise ValueError(
                f"cannot parse {source} at character {exc.pos}: {exc.msg}; nearby={context!r}"
            ) from exc
        if not isinstance(obj, dict):
            raise ValueError(f"expected JSON object, got {type(obj).__name__} at character {i}")
        calls.append(obj)
        i = end
    return calls


def load_calls(path: Path) -> tuple[list[dict[str, Any]], int, str]:
    raw = path.read_text(encoding="utf-8-sig")
    try:
        calls = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if all(isinstance(call, dict) for call in calls):
            return calls, 0, "strict_jsonl"
    except json.JSONDecodeError:
        pass

    repaired_text, repaired_count = escape_raw_control_characters_inside_json_strings(raw)
    calls = decode_object_stream(repaired_text, str(path))
    return calls, repaired_count, "compatibility_stream"


p = Path(sys.argv[1])
bad = []

try:
    calls, repaired_count, parser_mode = load_calls(p)
except Exception as exc:
    print("[bad model_calls]")
    print(f"JSON parse error: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

seen_call_indices = set()
for ordinal, c in enumerate(calls, start=1):
    ci = c.get("call_index", ordinal)
    if ci in seen_call_indices:
        bad.append(f"duplicate call_index: {ci}")
    seen_call_indices.add(ci)

    response = c.get("response")
    usage = response.get("token_usage", {}) if isinstance(response, dict) else {}

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
    bad.append("no JSON calls found")

print(f"num_calls={len(calls)}")
print(f"json_parser_mode={parser_mode}")
print(f"repaired_raw_control_characters={repaired_count}")

if bad:
    print("[bad model_calls]")
    for item in bad[:30]:
        print(item)
    if len(bad) > 30:
        print(f"... plus {len(bad) - 30} more")
    raise SystemExit(1)

print("[ok] model_calls token IDs are usable")
PY
}
task_in_range() {
  local task_num="$1"
  local task_int=$((10#$task_num))

  if [ -n "$START_TASK" ] && [ "$task_int" -lt "$START_TASK" ]; then
    return 1
  fi
  if [ -n "$END_TASK" ] && [ "$task_int" -ge "$END_TASK" ]; then
    return 1
  fi
  return 0
}

discover_successful_runs() {
  python - "$RUNS_ROOT" "$DISCOVERED_TSV" <<'PY'
import json
import re
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out_path = Path(sys.argv[2])

candidates = {}
skipped_nonfinished = []

for model_calls in runs_root.rglob("model_calls.jsonl"):
    run_dir = model_calls.parent
    match = re.search(r"gdpval_task_(\d{4})", run_dir.name)
    task_num = match.group(1) if match else None

    status_path = run_dir / "status.json"
    status = None
    row_index = None
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            status = data.get("status")
            if data.get("row_index") is not None:
                row_index = int(data["row_index"])
        except Exception:
            pass

    if task_num is None and row_index is not None:
        task_num = f"{row_index:04d}"
    if task_num is None:
        continue

    # If status is present, require a finished task. Missing status is accepted
    # because some copied accounting/runs folders contain only the replay inputs.
    if status is not None and status != "finished":
        skipped_nonfinished.append((task_num, str(run_dir), str(status)))
        continue

    mtime = model_calls.stat().st_mtime
    previous = candidates.get(task_num)
    if previous is None or mtime > previous[0]:
        candidates[task_num] = (mtime, run_dir, status or "status_missing")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for task_num in sorted(candidates):
        _, run_dir, status = candidates[task_num]
        f.write(f"{task_num}\t{run_dir}\t{status}\n")

print(f"selected_successful_or_status_missing_runs={len(candidates)}")
print(f"skipped_explicit_nonfinished_runs={len(skipped_nonfinished)}")
if skipped_nonfinished:
    print("first_skipped_nonfinished:")
    for row in skipped_nonfinished[:20]:
        print("\t".join(row))
PY
}

run_one() {
  local task_num="$1"
  local run_dir="$2"

  local model_calls="$run_dir/model_calls.jsonl"
  local run_base
  run_base="$(basename "$run_dir")"

  local out_dir="$RESULT_ROOT/task_${task_num}__${run_base}"
  local task_log="$RESULT_ROOT/logs/task_${task_num}__${run_base}.log"

  mkdir -p "$out_dir"

  if [ "$RESUME" = "1" ] && [ -f "$out_dir/metadata.json" ] && metadata_is_good "$out_dir/metadata.json"; then
    log "[skip task $task_num] existing successful activation accounting: $out_dir"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$task_num" "$run_dir" "$out_dir" "skipped_existing_success" "0" >> "$ALL_TSV"
    return 0
  fi

  log "================================================================"
  log "[activation task $task_num] run_dir=$run_dir"
  log "[activation task $task_num] out_dir=$out_dir"
  log "================================================================"

  if [ ! -f "$model_calls" ]; then
    log "[fail task $task_num] missing model_calls.jsonl: $model_calls"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$task_num" "$run_dir" "$out_dir" "missing_model_calls" "1" >> "$FAILED_TSV"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$task_num" "$run_dir" "$out_dir" "missing_model_calls" "1" >> "$ALL_TSV"
    return 0
  fi

  log "[task $task_num] verifying exact token IDs"
  if ! verify_model_calls "$model_calls" 2>&1 | tee "$out_dir/token_id_check.log"; then
    log "[fail task $task_num] token-ID verification failed"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$task_num" "$run_dir" "$out_dir" "bad_model_calls" "1" >> "$FAILED_TSV"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$task_num" "$run_dir" "$out_dir" "bad_model_calls" "1" >> "$ALL_TSV"
    return 0
  fi

  local args=(
    --config "$CONFIG_PATH"
    --model-calls "$model_calls"
    --call-indices "$CALL_INDICES"
    --activation-chunk-size "$ACTIVATION_CHUNK_SIZE"
    --out-dir "$out_dir"
  )

  if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    args+=(--local-files-only)
  fi

  local start_epoch end_epoch duration_sec exit_code
  start_epoch="$(date +%s)"

  set +e
  PYTHONPATH=. python "$SCRIPT" "${args[@]}" 2>&1 | tee "$task_log"
  exit_code=${PIPESTATUS[0]}
  set -u

  end_epoch="$(date +%s)"
  duration_sec=$((end_epoch - start_epoch))

  if [ "$exit_code" -eq 0 ] && [ -f "$out_dir/metadata.json" ] && metadata_is_good "$out_dir/metadata.json"; then
    log "[ok task $task_num] response activation accounting passed duration_sec=$duration_sec"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$task_num" "$run_dir" "$out_dir" "finished" "0" >> "$SUCCESS_TSV"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$task_num" "$run_dir" "$out_dir" "finished" "0" >> "$ALL_TSV"
  else
    log "[fail task $task_num] activation accounting failed exit_code=$exit_code duration_sec=$duration_sec"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$task_num" "$run_dir" "$out_dir" "accounting_failed" "$exit_code" >> "$FAILED_TSV"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$task_num" "$run_dir" "$out_dir" "accounting_failed" "$exit_code" >> "$ALL_TSV"
  fi
}


# RESPONSE_ACTIVATION_AUTO_ARCHIVE_V1
compress_results() {
  if [ "$COMPRESS_RESULTS" != "1" ]; then
    log "[archive] COMPRESS_RESULTS is not 1; skipping compression"
    return 0
  fi

  local clean_root="${RESULT_ROOT%/}"
  local parent_dir
  local result_name
  local archive_path
  local pointer_path

  if [ ! -d "$clean_root" ]; then
    log "[archive] result directory does not exist: $clean_root"
    return 1
  fi

  parent_dir="$(dirname "$clean_root")"
  result_name="$(basename "$clean_root")"
  archive_path="${parent_dir}/${result_name}.tar.gz"
  pointer_path="${parent_dir}/latest_response_activation_archive_path.txt"

  log "[archive] creating: $archive_path"

  rm -f "$archive_path"

  if ! tar -C "$parent_dir" -czf "$archive_path" "$result_name"; then
    log "[archive] compression failed"
    return 1
  fi

  printf '%s\n' "$archive_path" > "$pointer_path"

  log "[archive] compression completed"
  log "[archive] archive path: $archive_path"
  log "[archive] pointer path: $pointer_path"

  ls -lh "$archive_path"
}

main() {
  log "[info] current dir: $PWD"
  log "[info] RUNS_ROOT=$RUNS_ROOT"
  log "[info] RESULT_ROOT=$RESULT_ROOT"
  log "[info] CONFIG_PATH=$CONFIG_PATH"
  log "[info] CALL_INDICES=$CALL_INDICES"
  log "[info] ACTIVATION_CHUNK_SIZE=$ACTIVATION_CHUNK_SIZE"
  log "[info] LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"
  log "[info] COMPRESS_RESULTS=$COMPRESS_RESULTS"
  log "[info] START_TASK=${START_TASK:-<none>} END_TASK=${END_TASK:-<none>}"

  if [ ! -d "$RUNS_ROOT" ]; then
    log "[fatal] runs root does not exist: $RUNS_ROOT"
    exit 1
  fi
  if [ ! -f "$SCRIPT" ]; then
    log "[fatal] missing replay script: $SCRIPT"
    exit 1
  fi
  if [ ! -f "$CONFIG_PATH" ]; then
    log "[fatal] missing config: $CONFIG_PATH"
    exit 1
  fi

  python -m py_compile "$SCRIPT" || exit 1

  : > "$SUCCESS_TSV"
  : > "$FAILED_TSV"
  : > "$ALL_TSV"

  discover_successful_runs

  local selected_count=0
  while IFS=$'\t' read -r task_num run_dir status_note; do
    [ -n "${task_num:-}" ] || continue
    if ! task_in_range "$task_num"; then
      continue
    fi
    selected_count=$((selected_count + 1))
    run_one "$task_num" "$run_dir"
  done < "$DISCOVERED_TSV"

  log "[done] selected tasks in requested range: $selected_count"
  log "[done] response activation accounting loop finished"
  log "[done] master log: $MASTER_LOG"

  log "[summary] newly successful:"
  wc -l "$SUCCESS_TSV" 2>/dev/null || true

  log "[summary] failed:"
  if [ -s "$FAILED_TSV" ]; then
    cat "$FAILED_TSV"
  else
    echo "[none]"
  fi

  compress_results || return 1
}

# RESPONSE_ACTIVATION_REPEAT_FOLDERS_WRAPPER_V2
#
# Outer invocation:
#   creates one parent result directory and five independent pass folders.
#
# Child invocation:
#   runs the original accounting main function once.
#
# Every pass uses RESUME=0 and therefore recomputes its own complete results.
if [ "${RESPONSE_ACTIVATION_REPEAT_CHILD:-0}" != "1" ]; then
  REPEAT_RUNS="${REPEAT_RUNS:-5}"
  GROUP_RESULT_ROOT="${RESULT_ROOT%/}"
  ARCHIVE_GROUP_RESULTS="${COMPRESS_RESULTS:-0}"

  mkdir -p "$GROUP_RESULT_ROOT"

  printf '%s\n' "$GROUP_RESULT_ROOT" \
    > ../outputs/latest_response_activation_repeat_group_dir.txt

  echo "================================================================"
  echo "[repeat] group result root: $GROUP_RESULT_ROOT"
  echo "[repeat] number of passes: $REPEAT_RUNS"
  echo "================================================================"

  for ((REPEAT_INDEX=1; REPEAT_INDEX<=REPEAT_RUNS; REPEAT_INDEX++)); do
    printf -v PASS_NAME "pass_%02d" "$REPEAT_INDEX"
    PASS_RESULT_ROOT="${GROUP_RESULT_ROOT}/${PASS_NAME}"

    mkdir -p "$PASS_RESULT_ROOT"

    printf '%s\n' "$PASS_RESULT_ROOT" \
      > ../outputs/latest_response_activation_result_dir.txt

    echo
    echo "================================================================"
    echo "[repeat] accounting pass ${REPEAT_INDEX}/${REPEAT_RUNS}"
    echo "[repeat] result root: $PASS_RESULT_ROOT"
    echo "================================================================"

    RESPONSE_ACTIVATION_REPEAT_CHILD=1 \
    REPEAT_RUNS=1 \
    RESULT_ROOT="$PASS_RESULT_ROOT" \
    RESUME=0 \
    COMPRESS_RESULTS=0 \
    bash "$0" "$@"

    PASS_EXIT=$?

    if [ "$PASS_EXIT" -ne 0 ]; then
      echo "[fatal] pass ${REPEAT_INDEX}/${REPEAT_RUNS} failed with exit code $PASS_EXIT"
      exit "$PASS_EXIT"
    fi

    echo "[repeat] pass ${REPEAT_INDEX}/${REPEAT_RUNS} completed"
  done

  printf '%s\n' "$GROUP_RESULT_ROOT" \
    > ../outputs/latest_response_activation_repeat_group_dir.txt

  if [ "$ARCHIVE_GROUP_RESULTS" = "1" ]; then
    GROUP_PARENT="$(dirname "$GROUP_RESULT_ROOT")"
    GROUP_NAME="$(basename "$GROUP_RESULT_ROOT")"
    GROUP_ARCHIVE="${GROUP_RESULT_ROOT}.tar.gz"

    echo "[archive] creating group archive: $GROUP_ARCHIVE"

    rm -f "$GROUP_ARCHIVE"
    tar -C "$GROUP_PARENT" -czf "$GROUP_ARCHIVE" "$GROUP_NAME"

    printf '%s\n' "$GROUP_ARCHIVE" \
      > ../outputs/latest_response_activation_repeat_group_archive.txt

    ls -lh "$GROUP_ARCHIVE"
  fi

  echo
  echo "[done] all ${REPEAT_RUNS} accounting passes completed"
  echo "[done] group result root: $GROUP_RESULT_ROOT"
  exit 0
fi

main "$@"
