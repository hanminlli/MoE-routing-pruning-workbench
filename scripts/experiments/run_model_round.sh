#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# One GDPval model round for an expert-bank experiment.
#
# Turn budgets are an ordered fallback schedule, default:
#   task 0000: 80 turns; if unsuccessful, 50; if still unsuccessful, 120
#   task 0001: 80 turns; if unsuccessful, 50; if still unsuccessful, 120
#   ...
#
# By default STOP_AFTER_SUCCESS=1, so later budgets are skipped immediately
# after the first status="finished" trial for that task. Set
# STOP_AFTER_SUCCESS=0 only when every budget must be run independently.
#
# There are no same-budget automatic retries. Each attempted trial has the
# repository-standard 90-minute timeout by default.
#
# Required environment variables:
#   MODEL_TAG       e.g. keep192
#   CONFIG_PATH     e.g. configs/run_config_keep192.json
#
# Optional:
#   EXPERIMENT_NAME=first_experiment
#   START=0
#   END=220
#   TURN_BUDGETS="80 50 120"
#   STOP_AFTER_SUCCESS=1
#   TASK_TIMEOUT_MIN=90
#   RESUME=1
###############################################################################

EXPERIMENT_NAME="${EXPERIMENT_NAME:-first_experiment}"
MODEL_TAG="${MODEL_TAG:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
START="${START:-0}"
END="${END:-220}"
TURN_BUDGETS="${TURN_BUDGETS:-80 50 120}"
STOP_AFTER_SUCCESS="${STOP_AFTER_SUCCESS:-1}"
TASK_TIMEOUT_MIN="${TASK_TIMEOUT_MIN:-90}"
TASK_TIMEOUT_SEC=$((TASK_TIMEOUT_MIN * 60))
RESUME="${RESUME:-1}"
TASK_LIST_FILE="${TASK_LIST_FILE:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

BASELINE_RUNNER="scripts/baseline/run_gdpval.py"
BASELINE_CONFIG="configs/run_config.json"
TASKS_DIR="artifacts/tasks"
BASELINE_TEMPLATE="qwen36_chat_template.jinja"
EXPECTED_PROMPT_TRACK="optionB_prompt_v3_80turn_general_budget_prompt"
EXPECTED_RUNNER="$BASELINE_RUNNER"
VLLM_MODELS_URL="${VLLM_MODELS_URL:-http://localhost:8000/v1/models}"

if [ -f "$(pwd -P)/$BASELINE_RUNNER" ]; then
  PROJECT_DIR="$(pwd -P)"
elif [ -f "$(pwd -P)/project/$BASELINE_RUNNER" ]; then
  PROJECT_DIR="$(pwd -P)/project"
else
  echo "[fatal] run from project/ or its parent directory" >&2
  exit 1
fi
cd "$PROJECT_DIR" || exit 1

if [ -z "$MODEL_TAG" ]; then
  echo "[fatal] MODEL_TAG is required, e.g. MODEL_TAG=keep192" >&2
  exit 1
fi
if [ -z "$CONFIG_PATH" ]; then
  echo "[fatal] CONFIG_PATH is required" >&2
  exit 1
fi
if ! [[ "$START" =~ ^[0-9]+$ && "$END" =~ ^[0-9]+$ && "$TASK_TIMEOUT_MIN" =~ ^[0-9]+$ ]]; then
  echo "[fatal] START, END, and TASK_TIMEOUT_MIN must be integers" >&2
  exit 1
fi
if [ "$START" -lt 0 ] || [ "$END" -gt 220 ] || [ "$START" -ge "$END" ]; then
  echo "[fatal] expected 0 <= START < END <= 220" >&2
  exit 1
fi
if [ "$STOP_AFTER_SUCCESS" != "0" ] && [ "$STOP_AFTER_SUCCESS" != "1" ]; then
  echo "[fatal] STOP_AFTER_SUCCESS must be 0 or 1" >&2
  exit 1
fi
if [ -z "${TURN_BUDGETS//[[:space:]]/}" ]; then
  echo "[fatal] TURN_BUDGETS must contain at least one integer" >&2
  exit 1
fi
for turns in $TURN_BUDGETS; do
  if ! [[ "$turns" =~ ^[1-9][0-9]*$ ]]; then
    echo "[fatal] invalid turn budget: $turns" >&2
    exit 1
  fi
done
if ! command -v timeout >/dev/null 2>&1; then
  echo "[fatal] GNU timeout is required" >&2
  exit 1
fi

RESULT_ROOT="../outputs/${EXPERIMENT_NAME}/${MODEL_TAG}_${RUN_STAMP}"
RUNS_ROOT="$RESULT_ROOT/runs"
LOG_ROOT="$RESULT_ROOT/task_logs"
SNAPSHOT_ROOT="$RESULT_ROOT/experiment_snapshot"
MANIFEST="$RESULT_ROOT/trial_manifest.jsonl"
SUMMARY_JSON="$RESULT_ROOT/summary.json"
SUMMARY_TXT="$RESULT_ROOT/summary.txt"
MASTER_LOG="$RESULT_ROOT/master.log"
ARCHIVE="${RESULT_ROOT}.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"

mkdir -p "$RUNS_ROOT" "$LOG_ROOT" "$SNAPSHOT_ROOT"
exec > >(tee -a "$MASTER_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

activate_stirrup() {
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate stirrup-py312
}

validate_contract() {
  log "[validate] exact baseline contract"

  for p in "$BASELINE_RUNNER" "$BASELINE_CONFIG" "$CONFIG_PATH" "$BASELINE_TEMPLATE"; do
    if [ ! -f "$p" ]; then
      log "[fatal] missing required file: $p"
      exit 1
    fi
  done

  python -m py_compile "$BASELINE_RUNNER"

  python - "$BASELINE_CONFIG" "$CONFIG_PATH" "$EXPECTED_PROMPT_TRACK" "$EXPECTED_RUNNER" "$BASELINE_TEMPLATE" <<'PY'
import json
import sys
from pathlib import Path

base_path, cfg_path, expected_track, expected_runner, expected_template = map(Path, sys.argv[1:])
base = json.loads(base_path.read_text(encoding="utf-8"))
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

assert base.get("prompt_track") == str(expected_track), (base.get("prompt_track"), expected_track)
assert cfg.get("prompt_track") == str(expected_track), (cfg.get("prompt_track"), expected_track)
assert base.get("runner") == str(expected_runner), (base.get("runner"), expected_runner)
assert cfg.get("runner") == str(expected_runner), (cfg.get("runner"), expected_runner)
assert base.get("chat_template_path") == str(expected_template), (base.get("chat_template_path"), expected_template)
assert cfg.get("chat_template_path") == str(expected_template), (cfg.get("chat_template_path"), expected_template)

allowed = {"model", "tokenizer"}
diffs = sorted(k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k))
assert set(diffs).issubset(allowed), f"unexpected config differences: {diffs}"
assert cfg.get("model") == cfg.get("tokenizer"), "model and tokenizer paths differ"
model = Path(cfg["model"])
assert model.is_dir(), f"model directory missing: {model}"
assert (model / "config.json").is_file(), f"model config missing: {model / 'config.json'}"
assert (model / "pruning_manifest.json").is_file(), f"pruning manifest missing: {model / 'pruning_manifest.json'}"

print("[ok] prompt_track:", cfg["prompt_track"])
print("[ok] runner:", cfg["runner"])
print("[ok] chat template:", cfg["chat_template_path"])
print("[ok] max_tokens_per_turn:", cfg["max_tokens_per_turn"])
print("[ok] config differences from baseline base:", diffs)
print("[ok] model:", cfg["model"])
PY

  local task_count
  task_count="$(find "$TASKS_DIR" -maxdepth 2 -path "$TASKS_DIR/task_*/task.json" -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$task_count" -lt 220 ]; then
    log "[fatal] found only $task_count task.json files; expected at least 220"
    exit 1
  fi
  log "[ok] task count: $task_count"
}

validate_vllm_model() {
  log "[validate] vLLM endpoint and served model"

  python - "$VLLM_MODELS_URL" "$CONFIG_PATH" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

url, config_path = sys.argv[1:]
cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
expected = str(Path(cfg["model"]).resolve())

with urllib.request.urlopen(url, timeout=15) as response:
    payload = json.load(response)
served = [str(x.get("id")) for x in payload.get("data", [])]

print("[info] expected model:", expected)
print("[info] served models:", served)
assert expected in served, "vLLM is not serving the model from CONFIG_PATH"
print("[ok] vLLM serves the expected checkpoint")
PY
}

write_snapshot() {
  cp -p "$BASELINE_RUNNER" "$SNAPSHOT_ROOT/"
  cp -p "$BASELINE_CONFIG" "$SNAPSHOT_ROOT/"
  cp -p "$CONFIG_PATH" "$SNAPSHOT_ROOT/"
  cp -p "$BASELINE_TEMPLATE" "$SNAPSHOT_ROOT/"
  cp -p "$0" "$SNAPSHOT_ROOT/$(basename "$0")"
  if [ -n "$TASK_LIST_FILE" ]; then
    cp -p "$TASK_LIST_FILE" "$SNAPSHOT_ROOT/$(basename "$TASK_LIST_FILE")"
  fi

  sha256sum \
    "$BASELINE_RUNNER" \
    "$BASELINE_CONFIG" \
    "$CONFIG_PATH" \
    "$BASELINE_TEMPLATE" \
    "$0" \
    > "$SNAPSHOT_ROOT/sha256sums.txt"

  python - "$RESULT_ROOT/experiment_spec.json" "$EXPERIMENT_NAME" "$MODEL_TAG" "$CONFIG_PATH" "$START" "$END" "$TURN_BUDGETS" "$STOP_AFTER_SUCCESS" "$TASK_TIMEOUT_MIN" "$TASK_LIST_FILE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(out, experiment, model_tag, config_path, start, end, budgets,
 stop_after_success, timeout_min, task_list_file) = sys.argv[1:]
cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
stop_after_success = bool(int(stop_after_success))
if task_list_file:
    payload = json.loads(Path(task_list_file).read_text(encoding="utf-8"))
    values = payload.get("task_indices", payload) if isinstance(payload, dict) else payload
    task_indices = [int(x) for x in values]
else:
    task_indices = list(range(int(start), int(end)))
obj = {
    "experiment_name": experiment,
    "model_tag": model_tag,
    "model": cfg["model"],
    "tokenizer": cfg["tokenizer"],
    "config_path": config_path,
    "runner": "scripts/baseline/run_gdpval.py",
    "chat_template_path": cfg["chat_template_path"],
    "prompt_track": cfg["prompt_track"],
    "max_tokens_per_turn": int(cfg["max_tokens_per_turn"]),
    "task_start_inclusive": int(start),
    "task_end_exclusive": int(end),
    "task_list_file": task_list_file or None,
    "task_indices": task_indices,
    "expected_tasks": len(task_indices),
    "turn_budgets": [int(x) for x in budgets.split()],
    "stop_after_success": stop_after_success,
    "trial_policy": (
        "ordered fallback; stop after first finished trial"
        if stop_after_success
        else "run every turn budget independently"
    ),
    "same_budget_retry_policy": "none",
    "loop_order": "task-major, then ordered turn budget",
    "task_timeout_minutes": int(timeout_min),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
Path(out).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

trial_status() {
  local task_index="$1"
  local turns="$2"
  [ -f "$MANIFEST" ] || { printf '%s\n' "__MISSING__"; return 0; }

  python - "$MANIFEST" "$task_index" "$turns" <<'PY'
import json
import sys
from pathlib import Path

p, task, turns = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
status = None
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("task_index") == task and row.get("max_turns") == turns:
        status = str(row.get("status", "unknown"))
print(status if status is not None else "__MISSING__")
PY
}

task_has_success() {
  local task_index="$1"
  [ -f "$MANIFEST" ] || return 1

  python - "$MANIFEST" "$task_index" <<'PY'
import json
import sys
from pathlib import Path

p, task = Path(sys.argv[1]), int(sys.argv[2])
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("task_index") == task and row.get("status") == "finished":
        raise SystemExit(0)
raise SystemExit(1)
PY
}

latest_run_dir() {
  local runs_dir="$1"
  local task_num="$2"
  find "$runs_dir" -maxdepth 1 -type d -name "gdpval_task_${task_num}__*" \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

read_status() {
  local run_dir="$1"
  if [ -z "$run_dir" ] || [ ! -f "$run_dir/status.json" ]; then
    echo "missing_status"
    return
  fi
  python - "$run_dir/status.json" <<'PY'
import json
import sys
from pathlib import Path
try:
    print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("status", "unknown"))
except Exception:
    print("status_read_error")
PY
}

append_manifest() {
  local task_index="$1"
  local task_num="$2"
  local turns="$3"
  local start_ts="$4"
  local end_ts="$5"
  local duration="$6"
  local exit_code="$7"
  local status="$8"
  local run_dir="$9"
  local trial_log="${10}"

  python - "$MANIFEST" "$MODEL_TAG" "$task_index" "$task_num" "$turns" "$start_ts" "$end_ts" "$duration" "$TASK_TIMEOUT_SEC" "$exit_code" "$status" "$run_dir" "$trial_log" <<'PY'
import json
import sys

(manifest, model_tag, task_index, task_num, turns, start_ts, end_ts,
 duration, timeout_sec, exit_code, status, run_dir, trial_log) = sys.argv[1:]
row = {
    "model_tag": model_tag,
    "task_index": int(task_index),
    "task_num": task_num,
    "max_turns": int(turns),
    "start_ts": start_ts,
    "end_ts": end_ts,
    "duration_sec": int(duration),
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

run_trial() {
  local task_index="$1"
  local turns="$2"
  local task_num existing_status
  printf -v task_num '%04d' "$task_index"

  existing_status="$(trial_status "$task_index" "$turns")"
  if [ "$RESUME" = "1" ] && [ "$existing_status" != "__MISSING__" ]; then
    log "[resume skip] task=$task_num turns=$turns already recorded status=$existing_status"
    [ "$existing_status" = "finished" ]
    return
  fi

  local runs_dir="$RUNS_ROOT/turns_$(printf '%03d' "$turns")"
  local trial_log="$LOG_ROOT/task_${task_num}_turns_$(printf '%03d' "$turns").log"
  mkdir -p "$runs_dir"

  rm -rf ~/.cache/stirrup 2>/dev/null || true

  local start_epoch end_epoch duration start_ts end_ts exit_code run_dir status
  start_epoch="$(date +%s)"
  start_ts="$(date -Iseconds)"

  log "[trial] task=$task_num turns=$turns timeout=${TASK_TIMEOUT_MIN}m"

  set +e
  timeout --kill-after=60s "${TASK_TIMEOUT_SEC}s" \
    bash -lc "cd '$PROJECT_DIR' && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate stirrup-py312 && PYTHONPATH=. python '$BASELINE_RUNNER' --config '$CONFIG_PATH' --tasks-dir '$TASKS_DIR' --runs-dir '$runs_dir' --start '$task_index' --end '$((task_index + 1))' --max-turns '$turns'" \
    2>&1 | tee -a "$trial_log"
  exit_code=${PIPESTATUS[0]}
  set -u

  end_epoch="$(date +%s)"
  end_ts="$(date -Iseconds)"
  duration=$((end_epoch - start_epoch))

  run_dir="$(latest_run_dir "$runs_dir" "$task_num")"
  status="$(read_status "$run_dir")"
  if [ "$exit_code" -eq 124 ]; then
    status="timeout"
  elif [ "$exit_code" -eq 137 ]; then
    status="killed"
  fi

  append_manifest "$task_index" "$task_num" "$turns" "$start_ts" "$end_ts" "$duration" "$exit_code" "$status" "$run_dir" "$trial_log"
  log "[trial done] task=$task_num turns=$turns status=$status exit=$exit_code duration=${duration}s run_dir=$run_dir"

  [ "$status" = "finished" ]
}

write_summary() {
  python - "$MANIFEST" "$SUMMARY_JSON" "$SUMMARY_TXT" "$START" "$END" "$TURN_BUDGETS" "$STOP_AFTER_SUCCESS" "$TASK_LIST_FILE" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

(manifest, summary_json, summary_txt, start, end, budgets,
 stop_after_success, task_list_file) = sys.argv[1:]
start = int(start)
end = int(end)
expected_budgets = [int(x) for x in budgets.split()]
stop_after_success = bool(int(stop_after_success))

if task_list_file:
    payload = json.loads(Path(task_list_file).read_text(encoding="utf-8"))
    values = payload.get("task_indices", payload) if isinstance(payload, dict) else payload
    tasks = [int(x) for x in values]
else:
    tasks = list(range(start, end))
if not tasks or len(tasks) != len(set(tasks)):
    raise ValueError("task list must be non-empty and contain unique indices")

raw_rows = []
p = Path(manifest)
if p.exists():
    raw_rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

latest = {}
for row in raw_rows:
    key = (int(row["task_index"]), int(row["max_turns"]))
    latest[key] = row
rows = list(latest.values())
by_budget = defaultdict(Counter)
for row in rows:
    by_budget[int(row["max_turns"])][str(row.get("status", "unknown"))] += 1
overall = Counter(str(row.get("status", "unknown")) for row in rows)
success_pairs = {
    (int(row["task_index"]), int(row["max_turns"]))
    for row in raw_rows
    if str(row.get("status", "unknown")) == "finished"
}
success_by_budget = Counter()
completed_tasks = successful_tasks = exhausted_tasks = 0
incomplete_tasks = []
exhausted_task_indices = []
for task in tasks:
    statuses = {
        budget: str(latest[(task, budget)].get("status", "unknown"))
        for budget in expected_budgets
        if (task, budget) in latest
    }
    success_budget = next(
        (budget for budget in expected_budgets if (task, budget) in success_pairs), None
    )
    all_recorded = all(budget in statuses for budget in expected_budgets)
    terminal = all_recorded or (stop_after_success and success_budget is not None)
    if terminal:
        completed_tasks += 1
    else:
        incomplete_tasks.append(task)
    if success_budget is not None:
        successful_tasks += 1
        success_by_budget[success_budget] += 1
    elif all_recorded:
        exhausted_tasks += 1
        exhausted_task_indices.append(task)
expected_tasks = len(tasks)
maximum_trials = expected_tasks * len(expected_budgets)
minimum_trials = expected_tasks if stop_after_success else maximum_trials
complete = completed_tasks == expected_tasks
summary = {
    "task_indices": tasks,
    "turn_budgets": expected_budgets,
    "stop_after_success": stop_after_success,
    "trial_policy": (
        "ordered fallback; stop after first finished trial"
        if stop_after_success else "run every turn budget independently"
    ),
    "expected_tasks": expected_tasks,
    "minimum_trials_if_all_first_attempts_succeed": minimum_trials,
    "maximum_trials": maximum_trials,
    "raw_manifest_rows": len(raw_rows),
    "recorded_trials": len(rows),
    "completed_tasks": completed_tasks,
    "successful_tasks": successful_tasks,
    "exhausted_tasks": exhausted_tasks,
    "incomplete_task_count": len(incomplete_tasks),
    "incomplete_task_indices": incomplete_tasks,
    "exhausted_task_indices": exhausted_task_indices,
    "complete": complete,
    "success_by_budget": {str(b): success_by_budget[b] for b in expected_budgets},
    "by_budget": {str(b): dict(by_budget[b]) for b in expected_budgets},
    "overall_status_counts": dict(overall),
}
Path(summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
lines = [
    f"Policy: {summary['trial_policy']}",
    f"Turn order: {' -> '.join(map(str, expected_budgets))}",
    f"Expected tasks: {expected_tasks}",
    f"Completed tasks: {completed_tasks}",
    f"Successful tasks: {successful_tasks}",
    f"Exhausted tasks: {exhausted_tasks}",
    f"Recorded trials: {summary['recorded_trials']} (maximum {maximum_trials})",
    f"Complete: {complete}",
    "",
]
for budget in expected_budgets:
    lines.append(
        f"Turns {budget}: {dict(by_budget[budget])}; first-success tasks={success_by_budget[budget]}"
    )
lines += [
    "",
    f"Overall trial statuses: {summary['overall_status_counts']}",
    f"Incomplete tasks: {incomplete_tasks}",
    f"Exhausted tasks: {exhausted_task_indices}",
]
Path(summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

summary_is_complete() {
  python - "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path
obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ok = bool(obj.get("complete")) and int(obj.get("completed_tasks", -1)) == int(obj.get("expected_tasks", -2))
raise SystemExit(0 if ok else 1)
PY
}

make_archive() {
  log "[archive] creating $ARCHIVE"
  tar -C "$(dirname "$RESULT_ROOT")" -czf "$ARCHIVE" "$(basename "$RESULT_ROOT")"
  sha256sum "$ARCHIVE" > "$ARCHIVE_SHA"
  printf '%s\n' "$(readlink -f "$ARCHIVE")" > "../outputs/${EXPERIMENT_NAME}/latest_${MODEL_TAG}_archive_path.txt"
  printf '%s\n' "$(readlink -f "$RESULT_ROOT")" > "../outputs/${EXPERIMENT_NAME}/latest_${MODEL_TAG}_result_dir.txt"
  log "[archive] $(du -h "$ARCHIVE" | awk '{print $1}') $ARCHIVE"
  log "[archive] checksum: $ARCHIVE_SHA"
}

main() {
  activate_stirrup
  log "[info] experiment=$EXPERIMENT_NAME model_tag=$MODEL_TAG"
  local -a task_array
  if [ -n "$TASK_LIST_FILE" ]; then
    if [ ! -f "$TASK_LIST_FILE" ]; then
      log "[fatal] TASK_LIST_FILE does not exist: $TASK_LIST_FILE"
      return 1
    fi
    mapfile -t task_array < <(python - "$TASK_LIST_FILE" <<'PY'
import json
import sys
from pathlib import Path
obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
values = obj.get("task_indices", obj) if isinstance(obj, dict) else obj
for value in values:
    print(int(value))
PY
)
  else
    mapfile -t task_array < <(seq "$START" "$((END - 1))")
  fi
  local expected_count="${#task_array[@]}"
  log "[info] selected tasks=$expected_count ordered budgets=$TURN_BUDGETS"
  log "[info] task_list_file=${TASK_LIST_FILE:-<range [$START,$END)>}"
  log "[info] stop_after_success=$STOP_AFTER_SUCCESS"
  log "[info] at least $expected_count and at most $((expected_count * $(wc -w <<< "$TURN_BUDGETS" | tr -d ' '))) trials"
  log "[info] timeout per attempted trial=${TASK_TIMEOUT_MIN} minutes"
  log "[info] result root=$RESULT_ROOT"

  validate_contract
  validate_vllm_model
  write_snapshot

  local task turns task_num
  for task in "${task_array[@]}"; do
    printf -v task_num '%04d' "$task"

    if [ "$STOP_AFTER_SUCCESS" = "1" ] && task_has_success "$task"; then
      log "[resume skip task] task=$task_num already has a finished trial"
      write_summary
      continue
    fi

    for turns in $TURN_BUDGETS; do
      if run_trial "$task" "$turns"; then
        if [ "$STOP_AFTER_SUCCESS" = "1" ]; then
          log "[early stop] task=$task_num succeeded at turns=$turns; skipping later budgets"
          break
        fi
      fi
    done
    write_summary
  done

  write_summary
  make_archive

  if ! summary_is_complete; then
    log "[fatal] model round ended with incomplete task states"
    return 1
  fi

  log "[done] model round completed"
  log "[done] result=$RESULT_ROOT"
  log "[done] archive=$ARCHIVE"
}

main "$@"
