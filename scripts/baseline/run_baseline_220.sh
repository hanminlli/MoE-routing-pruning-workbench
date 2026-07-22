#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# Final GDPval 220-task execution + accounting driver
#
# Key behavior:
# - Writes a master terminal log.
# - Uses existing self-contained artifacts/tasks when SKIP_TASK_EXPORT=1.
# - Optional export if SKIP_TASK_EXPORT=0 or auto decides tasks are missing.
# - Verifies task.json files are sanitized and self-contained.
# - Hides GDPval_data before execution.
# - Runs one task at a time.
# - Each task is retried with a max-turn schedule, default: 80 -> 50 -> 120.
# - Each attempt has TASK_TIMEOUT_MIN minutes, default 90.
# - If a task succeeds, moves directly to next task.
# - If all attempts fail, records failure and moves to next task.
# - Stops vLLM before accounting if requested.
# - Does not create partial archives. By default, run directories are copied to
#   ../outputs only once at the end, then one final archive is created.
###############################################################################

START="${START:-0}"
END="${END:-220}"
# Retry schedule for failed task attempts.
# Default means:
#   attempt 1 -> --max-turns 80
#   attempt 2 -> --max-turns 50
#   attempt 3 -> --max-turns 120
# You can override with, for example:
#   MAX_TURNS_SCHEDULE="80 50 120" bash scripts/baseline/run_baseline_220.sh
MAX_TURNS_SCHEDULE="${MAX_TURNS_SCHEDULE:-80 50 120}"
read -r -a MAX_TURNS_VALUES <<< "$MAX_TURNS_SCHEDULE"
TASK_MAX_ATTEMPTS_DEFAULT="${#MAX_TURNS_VALUES[@]}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

TASK_MAX_ATTEMPTS="${TASK_MAX_ATTEMPTS:-$TASK_MAX_ATTEMPTS_DEFAULT}"
TASK_TIMEOUT_MIN="${TASK_TIMEOUT_MIN:-90}"
TASK_TIMEOUT_SEC=$((TASK_TIMEOUT_MIN * 60))

# Keep this final-save runner quiet during execution: no partial archives.
# A single final archive is created at the end.
ARCHIVE_EVERY="${ARCHIVE_EVERY:-0}"
KEEP_PARTIAL_ARCHIVES="${KEEP_PARTIAL_ARCHIVES:-0}"

# By default, do not rsync each attempt's run directory into ../outputs during
# execution. The final copy happens once after execution/accounting.
COPY_RUNS_TO_OUTPUTS_DURING_EXECUTION="${COPY_RUNS_TO_OUTPUTS_DURING_EXECUTION:-0}"
FINAL_COPY_RUNS_TO_OUTPUTS="${FINAL_COPY_RUNS_TO_OUTPUTS:-1}"

RUN_ACCOUNTING="${RUN_ACCOUNTING:-0}"
STOP_VLLM_BEFORE_ACCOUNTING="${STOP_VLLM_BEFORE_ACCOUNTING:-0}"

# auto:
#   skip export if artifacts/tasks has enough task_*/task.json files.
# 1:
#   always skip export and use existing artifacts/tasks.
# 0:
#   always run scripts/data/export_gdpval_tasks.py.
SKIP_TASK_EXPORT="${SKIP_TASK_EXPORT:-auto}"

# Recommended after artifacts/tasks are self-contained.
HIDE_GDPVAL_DATA_BEFORE_EXECUTION="${HIDE_GDPVAL_DATA_BEFORE_EXECUTION:-1}"

CLEAR_STIRRUP_CACHE_BETWEEN_ATTEMPTS="${CLEAR_STIRRUP_CACHE_BETWEEN_ATTEMPTS:-1}"

VLLM_PROCESS_PATTERN="${VLLM_PROCESS_PATTERN:-vllm serve Qwen/Qwen3.6-35B-A3B}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://localhost:8000/v1/models}"

# Detect project directory robustly.
# If launched from project/, use that.
# If launched from the parent wd/, use wd/project.
if [ -f "$(pwd -P)/scripts/baseline/run_gdpval.py" ]; then
  PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
elif [ -f "$(pwd -P)/project/scripts/baseline/run_gdpval.py" ]; then
  PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)/project}"
else
  echo "[fatal] cannot find scripts/baseline/run_gdpval.py from:"
  echo "        $(pwd -P)"
  echo "        Please run this from project/ or set PROJECT_DIR=/path/to/project"
  exit 1
fi

cd "$PROJECT_DIR" || exit 1

RUNS_DIR="${RUNS_DIR:-artifacts/runs_220_${RUN_STAMP}}"
COUNT_ROOT="${COUNT_ROOT:-artifacts/expert_counts_220_${RUN_STAMP}}"
OUT_DIR="${OUT_DIR:-../outputs/gdpval_220_${RUN_STAMP}}"

TASKS_DIR="${TASKS_DIR:-artifacts/tasks}"
CONFIG_PATH="${CONFIG_PATH:-configs/run_config.json}"

mkdir -p "$OUT_DIR"
mkdir -p "$RUNS_DIR"
mkdir -p "$COUNT_ROOT"

mkdir -p "$OUT_DIR/task_logs"
mkdir -p "$OUT_DIR/count_logs"
mkdir -p "$OUT_DIR/runs"
mkdir -p "$OUT_DIR/expert_counts"
mkdir -p "$OUT_DIR/project_snapshot"
mkdir -p "$OUT_DIR/checkpoints"
mkdir -p "$OUT_DIR/reference_checks"
mkdir -p "$OUT_DIR/token_checks"
mkdir -p "$OUT_DIR/count_checks"

MASTER_LOG="$OUT_DIR/master.log"
TASK_ATTEMPT_MANIFEST="$OUT_DIR/task_attempt_manifest.jsonl"
TASK_FINAL_MANIFEST="$OUT_DIR/task_final_manifest.jsonl"
SUCCESSFUL_RUNS_TSV="$OUT_DIR/successful_run_dirs.tsv"
FAILED_TASKS_TSV="$OUT_DIR/failed_tasks.tsv"

# Capture all terminal output into master log.
exec > >(tee -a "$MASTER_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

json_append_attempt() {
  local task_index="$1"
  local task_num="$2"
  local attempt="$3"
  local max_turns="$4"
  local start_ts="$5"
  local end_ts="$6"
  local duration_sec="$7"
  local timeout_sec="$8"
  local exit_code="$9"
  local status="${10}"
  local run_dir="${11}"
  local trial_log="${12}"

  python - "$TASK_ATTEMPT_MANIFEST" \
    "$task_index" "$task_num" "$attempt" "$max_turns" "$start_ts" "$end_ts" \
    "$duration_sec" "$timeout_sec" "$exit_code" "$status" "$run_dir" "$trial_log" <<'PY'
import json
import sys

manifest, task_index, task_num, attempt, max_turns, start_ts, end_ts, duration_sec, timeout_sec, exit_code, status, run_dir, trial_log = sys.argv[1:]

row = {
    "task_index": int(task_index),
    "task_num": task_num,
    "attempt": int(attempt),
    "max_turns": int(max_turns),
    "start_ts": start_ts,
    "end_ts": end_ts,
    "duration_sec": float(duration_sec),
    "timeout_sec": int(timeout_sec),
    "exit_code": int(exit_code),
    "status": status,
    "run_dir": run_dir,
    "trial_log": trial_log,
}

with open(manifest, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

json_append_final() {
  local task_index="$1"
  local task_num="$2"
  local final_status="$3"
  local attempts_used="$4"
  local successful_run_dir="$5"
  local total_duration_sec="$6"

  python - "$TASK_FINAL_MANIFEST" \
    "$task_index" "$task_num" "$final_status" "$attempts_used" "$successful_run_dir" "$total_duration_sec" <<'PY'
import json
import sys

manifest, task_index, task_num, final_status, attempts_used, successful_run_dir, total_duration_sec = sys.argv[1:]

row = {
    "task_index": int(task_index),
    "task_num": task_num,
    "final_status": final_status,
    "attempts_used": int(attempts_used),
    "successful_run_dir": successful_run_dir,
    "total_duration_sec": float(total_duration_sec),
}

with open(manifest, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

activate_conda_env() {
  local env_name="$1"

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
  else
    log "[warn] conda command not found; assuming current environment is already correct"
  fi
}

make_archive() {
  local label="$1"
  local archive_path="${OUT_DIR}_${label}.tar.gz"

  log "[archive] creating $archive_path"

  tar -C "$(dirname "$OUT_DIR")" \
    -czf "$archive_path" \
    "$(basename "$OUT_DIR")"

  echo "$archive_path" > ../outputs/latest_220_archive_path.txt

  if [[ "$label" == partial_* ]]; then
    echo "$archive_path" > ../outputs/latest_220_partial_archive_path.txt

    if [ "$KEEP_PARTIAL_ARCHIVES" = "0" ]; then
      find ../outputs -maxdepth 1 \
        -name "$(basename "$OUT_DIR")_partial_*.tar.gz" \
        ! -name "$(basename "$archive_path")" \
        -type f \
        -delete 2>/dev/null || true
    fi
  else
    echo "$archive_path" > ../outputs/latest_220_final_archive_path.txt
  fi
}

print_environment_info() {
  log "[info] PROJECT_DIR: $PROJECT_DIR"
  log "[info] project pwd: $(pwd)"
  log "[info] output dir: $OUT_DIR"
  log "[info] runs dir: $RUNS_DIR"
  log "[info] count root: $COUNT_ROOT"
  log "[info] master log: $MASTER_LOG"

  log "[info] START=$START END=$END"
  log "[info] MAX_TURNS_SCHEDULE=$MAX_TURNS_SCHEDULE"
  log "[info] TASK_MAX_ATTEMPTS=$TASK_MAX_ATTEMPTS TASK_TIMEOUT_MIN=$TASK_TIMEOUT_MIN"
  log "[info] COPY_RUNS_TO_OUTPUTS_DURING_EXECUTION=$COPY_RUNS_TO_OUTPUTS_DURING_EXECUTION FINAL_COPY_RUNS_TO_OUTPUTS=$FINAL_COPY_RUNS_TO_OUTPUTS"
  log "[info] SKIP_TASK_EXPORT=$SKIP_TASK_EXPORT"
  log "[info] RUN_ACCOUNTING=$RUN_ACCOUNTING STOP_VLLM_BEFORE_ACCOUNTING=$STOP_VLLM_BEFORE_ACCOUNTING"
  log "[info] HIDE_GDPVAL_DATA_BEFORE_EXECUTION=$HIDE_GDPVAL_DATA_BEFORE_EXECUTION"

  log "[info] hostname: $(hostname)"
  log "[info] user: $(whoami)"
  log "[info] shell pwd: $PWD"

  log "[info] python:"
  python --version || true
  which python || true

  log "[info] conda env:"
  echo "${CONDA_DEFAULT_ENV:-<none>}"

  log "[info] git status:"
  git status --short 2>/dev/null || true

  log "[info] GPU snapshot:"
  nvidia-smi 2>/dev/null || true

  log "[info] disk snapshot:"
  df -h . ../outputs 2>/dev/null || true
}

check_timeout_command() {
  if ! command -v timeout >/dev/null 2>&1; then
    log "[fatal] GNU timeout command not found"
    exit 1
  fi
}

task_json_count() {
  find "$TASKS_DIR" -maxdepth 2 -path "$TASKS_DIR/task_*/task.json" -type f 2>/dev/null | wc -l | tr -d ' '
}

should_skip_export() {
  if [ "$SKIP_TASK_EXPORT" = "1" ]; then
    return 0
  fi

  if [ "$SKIP_TASK_EXPORT" = "0" ]; then
    return 1
  fi

  local expected=$((END - START))
  local count
  count="$(task_json_count)"

  if [ "$count" -ge "$expected" ]; then
    return 0
  fi

  return 1
}

export_tasks_if_needed() {
  activate_conda_env stirrup-py312

  if should_skip_export; then
    log "[info] skipping task export; using existing $TASKS_DIR"
    local count
    count="$(task_json_count)"
    log "[info] existing task.json count: $count"
  else
    log "[info] exporting GDPval tasks"

    rm -rf "$TASKS_DIR"

    PYTHONPATH=. python scripts/data/export_gdpval_tasks.py \
      --config "$CONFIG_PATH" \
      --start 0 \
      --end 220 \
      --download-missing-files \
      2>&1 | tee "$OUT_DIR/export_tasks.log"

    local export_exit=${PIPESTATUS[0]}
    if [ "$export_exit" -ne 0 ]; then
      log "[fatal] task export failed with exit code $export_exit"
      exit "$export_exit"
    fi
  fi

  rsync -a "$TASKS_DIR" "$OUT_DIR/project_snapshot"/ 2>/dev/null || true
}

verify_agent_facing_tasks() {
  log "[info] verifying exported tasks are sanitized and self-contained"

  python - "$TASKS_DIR" "$START" "$END" "$OUT_DIR/reference_checks/task_safety_check.json" <<'PY'
import json
import sys
from pathlib import Path

tasks_dir = Path(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
out_json = Path(sys.argv[4])

bad = []
summary = {
    "tasks_dir": str(tasks_dir),
    "start": start,
    "end": end,
    "num_tasks_checked": 0,
    "num_local_reference_files": 0,
    "bad_count": 0,
}

answer_markers = ("deliverable", "gold", "answer", "solution", "ground_truth", "reference_answer")

for i in range(start, end):
    task_num = f"{i:04d}"
    p = tasks_dir / f"task_{task_num}" / "task.json"

    if not p.exists():
        bad.append({
            "task": task_num,
            "problem": "missing_task_json",
            "path": str(p),
        })
        continue

    summary["num_tasks_checked"] += 1

    try:
        task = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        bad.append({
            "task": task_num,
            "problem": "could_not_read_task_json",
            "path": str(p),
            "error": repr(e),
        })
        continue

    def find_forbidden_keys(obj, prefix=""):
        found = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                current = f"{prefix}.{key}" if prefix else str(key)
                lowered = str(key).lower()
                if any(marker in lowered for marker in answer_markers) or lowered.startswith("rubric") or lowered == "hidden_reference_files" or lowered == "raw_row":
                    found.append(current)
                found.extend(find_forbidden_keys(value, current))
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                found.extend(find_forbidden_keys(value, f"{prefix}[{index}]"))
        return found

    for key_path in find_forbidden_keys(task):
        bad.append({
            "task": task_num,
            "problem": "answer_side_key_present",
            "key": key_path,
            "path": str(p),
        })

    refs = task.get("local_reference_files") or []
    if not isinstance(refs, list):
        bad.append({
            "task": task_num,
            "problem": "local_reference_files_not_list",
            "path": str(p),
            "type": type(refs).__name__,
        })
        refs = []

    for x in refs:
        summary["num_local_reference_files"] += 1
        sx = str(x)
        px = Path(sx)

        if not px.exists():
            bad.append({
                "task": task_num,
                "problem": "missing_reference_file",
                "reference": sx,
                "path": str(p),
            })

        if "GDPval_data" in sx or "gdpval_data" in sx or "GDPVal_data" in sx or "GDPVAL_data" in sx:
            bad.append({
                "task": task_num,
                "problem": "reference_still_points_to_raw_dataset",
                "reference": sx,
                "path": str(p),
            })

        if "/inputs/" not in sx.replace("\\", "/"):
            bad.append({
                "task": task_num,
                "problem": "reference_not_under_task_inputs",
                "reference": sx,
                "path": str(p),
            })

summary["bad_count"] = len(bad)

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps({"summary": summary, "bad": bad}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))

if bad:
    print("[fatal] task safety check failed; first 50 problems:")
    for item in bad[:50]:
        print(json.dumps(item, ensure_ascii=False))
    raise SystemExit(1)

print("[ok] task safety check passed")
PY

  local check_exit=$?
  if [ "$check_exit" -ne 0 ]; then
    log "[fatal] agent-facing task verification failed"
    exit "$check_exit"
  fi
}

check_task_prompt_filelist_if_available() {
  if [ -f scripts/data/check_task_filelist.py ]; then
    log "[info] running prompt file-list checker"

    PYTHONPATH=. python scripts/data/check_task_filelist.py \
      --tasks-dir "$TASKS_DIR" \
      --runner scripts/baseline/run_gdpval.py \
      2>&1 | tee "$OUT_DIR/reference_checks/check_task_prompt_filelist.log"

    local checker_exit=${PIPESTATUS[0]}
    if [ "$checker_exit" -ne 0 ]; then
      log "[fatal] prompt file-list checker failed with exit code $checker_exit"
      exit "$checker_exit"
    fi
  else
    log "[warn] scripts/data/check_task_filelist.py not found; skipping prompt file-list checker"
  fi
}

hide_gdpval_data_before_execution() {
  if [ "$HIDE_GDPVAL_DATA_BEFORE_EXECUTION" != "1" ]; then
    log "[info] HIDE_GDPVAL_DATA_BEFORE_EXECUTION is not 1; leaving raw dataset visible"
    return 0
  fi

  local hidden_root="../outputs/hidden_gdpval_data_${RUN_STAMP}"
  mkdir -p "$hidden_root"

  local found_any=0

  for d in GDPval_data gdpval_data GDPVal_data GDPVAL_data; do
    if [ -e "$d" ]; then
      found_any=1
      local dest="$hidden_root/$d"

      if [ -e "$dest" ]; then
        dest="${dest}_$(date +%Y%m%d_%H%M%S)"
      fi

      log "[info] hiding raw dataset before execution: $d -> $dest"
      mv "$d" "$dest"
      echo "$dest" >> "$OUT_DIR/hidden_gdpval_data_paths.txt"
      echo "$dest" >> ../outputs/latest_hidden_gdpval_data_paths.txt
    fi
  done

  if [ "$found_any" = "0" ]; then
    log "[info] no GDPval_data directory found under project to hide"
  fi

  log "[info] remaining GDPval-like directories under project:"
  find . -maxdepth 3 -type d -iname '*gdp*data*' -print 2>/dev/null || true
}

check_vllm_health() {
  log "[info] checking exact vLLM model at: $VLLM_HEALTH_URL"

  if ! python - "$VLLM_HEALTH_URL" "$CONFIG_PATH" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

url, config_path = sys.argv[1:]
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
expected = str(config["model"])

try:
    with urllib.request.urlopen(url, timeout=15) as response:
        payload = json.load(response)
except Exception as exc:
    print(f"[fatal] vLLM endpoint is unavailable: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

served = [str(item.get("id")) for item in payload.get("data", [])]
print("[info] expected model:", expected)
print("[info] served models:", served)
if expected not in served:
    print("[fatal] vLLM is not serving the model specified by CONFIG_PATH")
    raise SystemExit(1)
print("[ok] vLLM serves the expected baseline model")
PY
  then
    log "[fatal] vLLM validation failed; stop before running any GDPval task"
    exit 1
  fi
}
latest_run_dir_for_task() {
  local task_num="$1"

  find "$RUNS_DIR" -maxdepth 1 -type d -name "gdpval_task_${task_num}__*" \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

read_status_from_run_dir() {
  local run_dir="$1"

  if [ -z "$run_dir" ]; then
    echo "no_run_dir"
    return 0
  fi

  if [ ! -f "$run_dir/status.json" ]; then
    echo "missing_status_json"
    return 0
  fi

  python - "$run_dir/status.json" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])

try:
    data = json.loads(p.read_text(encoding="utf-8"))
    print(data.get("status", "unknown"))
except Exception:
    print("status_read_error")
PY
}

copy_run_to_outputs() {
  local run_dir="$1"

  if [ "$COPY_RUNS_TO_OUTPUTS_DURING_EXECUTION" != "1" ]; then
    return 0
  fi

  if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    local base
    base="$(basename "$run_dir")"
    rsync -a "$run_dir/" "$OUT_DIR/runs/$base/" 2>/dev/null || true
  fi
}

copy_all_runs_to_outputs_final() {
  if [ "$FINAL_COPY_RUNS_TO_OUTPUTS" != "1" ]; then
    log "[info] FINAL_COPY_RUNS_TO_OUTPUTS is not 1; skipping final run-dir copy to outputs"
    return 0
  fi

  if [ ! -d "$RUNS_DIR" ]; then
    log "[warn] RUNS_DIR does not exist; cannot final-copy run dirs: $RUNS_DIR"
    return 0
  fi

  log "[final-save] copying run directories to $OUT_DIR/runs"
  mkdir -p "$OUT_DIR/runs"
  rsync -a "$RUNS_DIR/" "$OUT_DIR/runs/" 2>/dev/null || true
}

max_turns_for_attempt() {
  local attempt="$1"
  local idx=$((attempt - 1))
  local n="${#MAX_TURNS_VALUES[@]}"

  if [ "$n" -eq 0 ]; then
    echo "80"
    return 0
  fi

  if [ "$idx" -lt "$n" ]; then
    echo "${MAX_TURNS_VALUES[$idx]}"
  else
    echo "${MAX_TURNS_VALUES[$((n - 1))]}"
  fi
}

run_one_task_attempt() {
  local task_index="$1"
  local task_num="$2"
  local attempt="$3"

  local max_turns_this_attempt
  max_turns_this_attempt="$(max_turns_for_attempt "$attempt")"

  local trial_log="$OUT_DIR/task_logs/task_${task_num}_attempt_${attempt}_maxturns_${max_turns_this_attempt}.log"

  log "[task $task_num] attempt $attempt/$TASK_MAX_ATTEMPTS starts with max_turns=$max_turns_this_attempt"
  log "[task $task_num] timeout: ${TASK_TIMEOUT_MIN} minutes"

  if [ "$CLEAR_STIRRUP_CACHE_BETWEEN_ATTEMPTS" = "1" ]; then
    rm -rf ~/.cache/stirrup 2>/dev/null || true
  fi

  local start_epoch
  local end_epoch
  local duration_sec
  local start_ts
  local end_ts

  start_epoch="$(date +%s)"
  start_ts="$(date -Iseconds)"

  set +e
  (
    cd "$PROJECT_DIR" || exit 99
    PYTHONPATH=. timeout --kill-after=60s "${TASK_TIMEOUT_SEC}s" \
      python scripts/baseline/run_gdpval.py \
        --config "$CONFIG_PATH" \
        --tasks-dir "$TASKS_DIR" \
        --runs-dir "$RUNS_DIR" \
        --start "$task_index" \
        --end "$((task_index + 1))" \
        --max-turns "$max_turns_this_attempt"
  ) 2>&1 | tee -a "$trial_log"

  local run_exit=${PIPESTATUS[0]}
  set -u

  end_epoch="$(date +%s)"
  end_ts="$(date -Iseconds)"
  duration_sec=$((end_epoch - start_epoch))

  local run_dir
  run_dir="$(latest_run_dir_for_task "$task_num")"

  local status
  status="$(read_status_from_run_dir "$run_dir")"

  if [ "$run_exit" -eq 124 ]; then
    status="timeout"
  elif [ "$run_exit" -eq 137 ]; then
    status="killed"
  elif [ "$run_exit" -eq 99 ]; then
    status="project_dir_cd_failed"
  fi

  if [ "$status" = "no_run_dir" ] && [ "$run_exit" -eq 2 ]; then
    if grep -q "can't open file .*scripts/baseline/run_gdpval.py" "$trial_log" 2>/dev/null; then
      status="infrastructure_missing_runner_path"
    fi
  fi

  copy_run_to_outputs "$run_dir"

  json_append_attempt \
    "$task_index" "$task_num" "$attempt" "$max_turns_this_attempt" \
    "$start_ts" "$end_ts" "$duration_sec" "$TASK_TIMEOUT_SEC" \
    "$run_exit" "$status" "$run_dir" "$trial_log"

  log "[task $task_num] attempt $attempt finished: max_turns=$max_turns_this_attempt status=$status exit_code=$run_exit duration_sec=$duration_sec run_dir=$run_dir"

  if [ "$status" = "infrastructure_missing_runner_path" ] || [ "$status" = "project_dir_cd_failed" ]; then
    log "[fatal] infrastructure/path failure detected; stopping instead of burning retries"
    exit 88
  fi

  if [ "$status" = "finished" ]; then
    echo "$task_index"$'\t'"$task_num"$'\t'"$attempt"$'\t'"$duration_sec"$'\t'"$run_dir" >> "$SUCCESSFUL_RUNS_TSV"
    return 0
  fi

  return 1
}

run_execution_phase() {
  activate_conda_env stirrup-py312
  check_vllm_health

  log "[phase 1] GDPval execution starts"

  : > "$SUCCESSFUL_RUNS_TSV"
  : > "$FAILED_TASKS_TSV"

  local completed_count=0

  for ((i=START; i<END; i++)); do
    local task_num
    printf -v task_num "%04d" "$i"

    log "================================================================"
    log "[task $task_num] starts"
    log "================================================================"

    local task_start_epoch
    task_start_epoch="$(date +%s)"

    local success=0
    local attempts_used=0
    local successful_run_dir=""

    for ((attempt=1; attempt<=TASK_MAX_ATTEMPTS; attempt++)); do
      attempts_used="$attempt"

      if run_one_task_attempt "$i" "$task_num" "$attempt"; then
        success=1
        successful_run_dir="$(latest_run_dir_for_task "$task_num")"
        log "[task $task_num] succeeded on attempt $attempt"
        break
      fi

      if [ "$attempt" -lt "$TASK_MAX_ATTEMPTS" ]; then
        log "[task $task_num] attempt $attempt failed; retrying"
      else
        log "[task $task_num] all $TASK_MAX_ATTEMPTS attempts failed; moving to next task"
      fi
    done

    local task_end_epoch
    task_end_epoch="$(date +%s)"
    local task_duration_sec=$((task_end_epoch - task_start_epoch))

    if [ "$success" = "1" ]; then
      json_append_final "$i" "$task_num" "finished" "$attempts_used" "$successful_run_dir" "$task_duration_sec"
    else
      echo "$i"$'\t'"$task_num"$'\t'"$attempts_used"$'\t'"$task_duration_sec" >> "$FAILED_TASKS_TSV"
      json_append_final "$i" "$task_num" "failed_after_retries" "$attempts_used" "" "$task_duration_sec"
    fi

    completed_count=$((completed_count + 1))

    if [ "$ARCHIVE_EVERY" -gt 0 ] && [ $((completed_count % ARCHIVE_EVERY)) -eq 0 ]; then
      make_archive "partial_after_${completed_count}_tasks"
    fi
  done

  log "[phase 1] GDPval execution completed"
}

stop_vllm_server() {
  if [ "$STOP_VLLM_BEFORE_ACCOUNTING" != "1" ]; then
    log "[info] STOP_VLLM_BEFORE_ACCOUNTING is not 1; leaving vLLM running"
    return 0
  fi

  log "[info] stopping vLLM before accounting"
  log "[info] process pattern: $VLLM_PROCESS_PATTERN"

  pkill -TERM -f "$VLLM_PROCESS_PATTERN" 2>/dev/null || true

  for _ in $(seq 1 60); do
    if ! pgrep -f "$VLLM_PROCESS_PATTERN" >/dev/null 2>&1; then
      log "[info] vLLM process stopped"
      nvidia-smi 2>/dev/null || true
      return 0
    fi
    sleep 1
  done

  log "[warn] vLLM still alive after TERM; sending KILL"
  pkill -KILL -f "$VLLM_PROCESS_PATTERN" 2>/dev/null || true
  sleep 5

  nvidia-smi 2>/dev/null || true
}

accounting_help_contains() {
  local pattern="$1"
  python scripts/accounting/replay_frequency.py --help 2>&1 | grep -q -- "$pattern"
}

run_accounting_for_run() {
  local task_num="$1"
  local run_dir="$2"

  if [ ! -f scripts/accounting/replay_frequency.py ]; then
    log "[warn] accounting script not found; skipping accounting for task $task_num"
    return 0
  fi

  local model_calls="$run_dir/model_calls.jsonl"
  if [ ! -f "$model_calls" ]; then
    log "[warn] missing model_calls.jsonl for task $task_num: $model_calls"
    return 0
  fi

  local count_dir="$COUNT_ROOT/task_${task_num}__$(basename "$run_dir")"
  local count_log="$OUT_DIR/count_logs/task_${task_num}.log"
  mkdir -p "$count_dir"

  log "[accounting task $task_num] starts: $run_dir"

  local args=()

  if accounting_help_contains "--config"; then
    args+=(--config "$CONFIG_PATH")
  fi

  if accounting_help_contains "--model-calls"; then
    args+=(--model-calls "$model_calls")
  elif accounting_help_contains "--model_calls"; then
    args+=(--model_calls "$model_calls")
  elif accounting_help_contains "--log-path"; then
    args+=(--log-path "$model_calls")
  elif accounting_help_contains "--log_path"; then
    args+=(--log_path "$model_calls")
  else
    log "[warn] could not detect model-calls argument from --help; trying --model-calls"
    args+=(--model-calls "$model_calls")
  fi

  if accounting_help_contains "--out-dir"; then
    args+=(--out-dir "$count_dir")
  elif accounting_help_contains "--out_dir"; then
    args+=(--out_dir "$count_dir")
  elif accounting_help_contains "--output-dir"; then
    args+=(--output-dir "$count_dir")
  else
    log "[warn] could not detect output-dir argument from --help; trying --out-dir"
    args+=(--out-dir "$count_dir")
  fi

  local start_epoch
  local end_epoch
  local duration_sec
  start_epoch="$(date +%s)"

  set +e
  PYTHONPATH=. python scripts/accounting/replay_frequency.py "${args[@]}" \
    2>&1 | tee -a "$count_log"
  local count_exit=${PIPESTATUS[0]}
  set -u

  end_epoch="$(date +%s)"
  duration_sec=$((end_epoch - start_epoch))

  rsync -a "$count_dir/" "$OUT_DIR/expert_counts/$(basename "$count_dir")/" 2>/dev/null || true

  python - "$OUT_DIR/accounting_manifest.jsonl" "$task_num" "$run_dir" "$model_calls" "$count_dir" "$count_log" "$count_exit" "$duration_sec" <<'PY'
import json
import sys

manifest, task_num, run_dir, model_calls, count_dir, count_log, count_exit, duration_sec = sys.argv[1:]

row = {
    "task_num": task_num,
    "run_dir": run_dir,
    "model_calls": model_calls,
    "count_dir": count_dir,
    "count_log": count_log,
    "exit_code": int(count_exit),
    "duration_sec": float(duration_sec),
}

with open(manifest, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

  log "[accounting task $task_num] finished: exit_code=$count_exit duration_sec=$duration_sec"

  return 0
}

run_accounting_phase() {
  if [ "$RUN_ACCOUNTING" != "1" ]; then
    log "[info] RUN_ACCOUNTING is not 1; skipping accounting"
    return 0
  fi

  stop_vllm_server

  activate_conda_env routing-hf-py312

  log "[phase 2] accounting starts"

  if [ ! -s "$SUCCESSFUL_RUNS_TSV" ]; then
    log "[warn] no successful runs found; skipping accounting"
    return 0
  fi

  while IFS=$'\t' read -r task_index task_num attempt duration_sec run_dir; do
    [ -n "${task_num:-}" ] || continue
    run_accounting_for_run "$task_num" "$run_dir"
  done < "$SUCCESSFUL_RUNS_TSV"

  log "[phase 2] accounting completed"
}

write_latest_pointers() {
  echo "$OUT_DIR" > artifacts/latest_220_output_dir.txt
  echo "$OUT_DIR" > ../outputs/latest_220_output_dir.txt
  echo "$RUNS_DIR" > "$OUT_DIR/latest_runs_dir.txt"
  echo "$COUNT_ROOT" > "$OUT_DIR/latest_count_root.txt"
}

main() {
  check_timeout_command
  write_latest_pointers
  print_environment_info

  log "[setup] compiling key scripts"
  python -m py_compile scripts/baseline/run_gdpval.py || exit 1

  if [ -f scripts/data/export_gdpval_tasks.py ]; then
    python -m py_compile scripts/data/export_gdpval_tasks.py || exit 1
    python -m py_compile scripts/data/export_gdpval_tasks_raw.py || exit 1
  fi

  if [ -f scripts/data/make_tasks_self_contained.py ]; then
    python -m py_compile scripts/data/make_tasks_self_contained.py || exit 1
  fi

  export_tasks_if_needed
  verify_agent_facing_tasks
  check_task_prompt_filelist_if_available
  hide_gdpval_data_before_execution

  run_execution_phase
  run_accounting_phase
  copy_all_runs_to_outputs_final
  make_archive "final"

  log "[done] all phases completed"
  log "[done] OUT_DIR=$OUT_DIR"
  log "[done] MASTER_LOG=$MASTER_LOG"

  log "[summary] successful tasks:"
  if [ -s "$SUCCESSFUL_RUNS_TSV" ]; then
    wc -l "$SUCCESSFUL_RUNS_TSV"
  else
    echo "0"
  fi

  log "[summary] failed tasks after retries:"
  if [ -s "$FAILED_TASKS_TSV" ]; then
    cat "$FAILED_TASKS_TSV"
  else
    echo "[none]"
  fi
}

main "$@"
