#!/usr/bin/env bash
set -uo pipefail

###############################################################################
# Unattended multi-model experiment controller
#
# Automatically evaluates each model with an ordered fallback schedule:
#   80 turns -> if unsuccessful, 50 turns -> if unsuccessful, 120 turns.
# Later budgets are skipped after the first successful trial by default.
#
# For each model:
#   1. Stop any currently running vLLM server owned by this user.
#   2. Start vLLM with the exact checkpoint from the model-specific baseline config.
#   3. Wait until /v1/models reports that exact checkpoint.
#   4. Run scripts/experiments/run_model_round.sh.
#   5. Verify that every task reached a terminal outcome and archive it.
#   6. Stop vLLM before switching to the next checkpoint.
#
# Resume behavior:
# - All three model rounds share one SUITE_STAMP.
# - The child round receives RUN_STAMP=$SUITE_STAMP and RESUME=1.
# - If interrupted, rerunning this controller automatically resumes the latest
#   incomplete suite unless NEW_SUITE=1 or an explicit SUITE_STAMP is provided.
# - A trial already present in the child round manifest is not repeated.
#
# No accounting is performed here.
###############################################################################

EXPERIMENT_NAME="${EXPERIMENT_NAME:-experiment_suite}"
MODEL_ORDER="${MODEL_ORDER:-keep192 keep128 keep64}"
START="${START:-0}"
END="${END:-220}"
TURN_BUDGETS="${TURN_BUDGETS:-80 50 120}"
STOP_AFTER_SUCCESS="${STOP_AFTER_SUCCESS:-1}"
TASK_TIMEOUT_MIN="${TASK_TIMEOUT_MIN:-90}"
TASK_LIST_FILE="${TASK_LIST_FILE:-}"
RESUME="${RESUME:-1}"
CONTINUE_ON_MODEL_FAILURE="${CONTINUE_ON_MODEL_FAILURE:-1}"
ARCHIVE_PARTIAL_ON_FAILURE="${ARCHIVE_PARTIAL_ON_FAILURE:-1}"
NEW_SUITE="${NEW_SUITE:-0}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

VLLM_ENV="${VLLM_ENV:-vllm-env}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_STARTUP_TIMEOUT_MIN="${VLLM_STARTUP_TIMEOUT_MIN:-45}"
VLLM_SHUTDOWN_TIMEOUT_SEC="${VLLM_SHUTDOWN_TIMEOUT_SEC:-180}"
VLLM_MODELS_URL="http://localhost:${VLLM_PORT}/v1/models"

ROUND_SCRIPT="scripts/experiments/run_model_round.sh"
CHAT_TEMPLATE="qwen36_chat_template.jinja"

# Detect project directory.
if [ -f "$(pwd -P)/$ROUND_SCRIPT" ]; then
  PROJECT_DIR="$(pwd -P)"
elif [ -f "$(pwd -P)/project/$ROUND_SCRIPT" ]; then
  PROJECT_DIR="$(pwd -P)/project"
else
  echo "[fatal] run from project/ or its parent directory" >&2
  exit 1
fi
cd "$PROJECT_DIR" || exit 1

OUTPUTS_DIR="$(readlink -f ../outputs)"
mkdir -p "$OUTPUTS_DIR/$EXPERIMENT_NAME"

LATEST_SUITE_STAMP_FILE="$OUTPUTS_DIR/$EXPERIMENT_NAME/latest_controller_suite_stamp.txt"
LATEST_SUITE_ROOT_FILE="$OUTPUTS_DIR/$EXPERIMENT_NAME/latest_controller_suite_root.txt"

# Reuse the latest incomplete suite unless explicitly told to create a new one.
if [ -z "${SUITE_STAMP:-}" ]; then
  if [ "$NEW_SUITE" != "1" ] && [ -s "$LATEST_SUITE_STAMP_FILE" ]; then
    candidate_stamp="$(tr -d '[:space:]' < "$LATEST_SUITE_STAMP_FILE")"
    candidate_root="$OUTPUTS_DIR/${EXPERIMENT_NAME}_controller_${candidate_stamp}"
    if [ -n "$candidate_stamp" ] && [ -d "$candidate_root" ] && [ ! -f "$candidate_root/COMPLETE" ]; then
      SUITE_STAMP="$candidate_stamp"
      echo "[resume] reusing latest incomplete SUITE_STAMP=$SUITE_STAMP"
    else
      SUITE_STAMP="$(date +%Y%m%d_%H%M%S)"
    fi
  else
    SUITE_STAMP="$(date +%Y%m%d_%H%M%S)"
  fi
fi

SUITE_ROOT="$OUTPUTS_DIR/${EXPERIMENT_NAME}_controller_${SUITE_STAMP}"
VLLM_LOG_ROOT="$SUITE_ROOT/vllm_logs"
STATE_ROOT="$SUITE_ROOT/state"
CONTROLLER_LOG="$SUITE_ROOT/controller.log"
MODEL_STATE_JSONL="$STATE_ROOT/model_state.jsonl"
SUITE_SPEC_JSON="$SUITE_ROOT/suite_spec.json"
SUITE_SUMMARY_JSON="$SUITE_ROOT/suite_summary.json"
SUITE_SUMMARY_TXT="$SUITE_ROOT/suite_summary.txt"

mkdir -p "$VLLM_LOG_ROOT" "$STATE_ROOT"
printf '%s\n' "$SUITE_STAMP" > "$LATEST_SUITE_STAMP_FILE"
printf '%s\n' "$SUITE_ROOT" > "$LATEST_SUITE_ROOT_FILE"

exec > >(tee -a "$CONTROLLER_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

CONFIG_FOR_keep192="${CONFIG_FOR_keep192:-configs/run_config_keep192.json}"
CONFIG_FOR_keep128="${CONFIG_FOR_keep128:-configs/run_config_keep128.json}"
CONFIG_FOR_keep64="${CONFIG_FOR_keep64:-configs/run_config_keep64.json}"

MANAGED_VLLM_PID=""
MANAGED_VLLM_TAG=""
VLLM_MANAGEMENT_STARTED=0

config_for_tag() {
  local tag="$1"
  case "$tag" in
    keep192) printf '%s\n' "$CONFIG_FOR_keep192" ;;
    keep128) printf '%s\n' "$CONFIG_FOR_keep128" ;;
    keep64)  printf '%s\n' "$CONFIG_FOR_keep64" ;;
    *)
      log "[fatal] unsupported model tag: $tag"
      return 1
      ;;
  esac
}

model_from_config() {
  local config_path="$1"
  python - "$config_path" <<'PY'
import json
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(Path(cfg["model"]).resolve()))
PY
}

port_is_listening() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$VLLM_PORT$"
}

stop_vllm() {
  local reason="${1:-model switch}"
  log "[vllm stop] reason=$reason"

  # Stop the process group created by this controller, if present.
  if [ -n "$MANAGED_VLLM_PID" ] && kill -0 "$MANAGED_VLLM_PID" 2>/dev/null; then
    log "[vllm stop] TERM managed process group -$MANAGED_VLLM_PID ($MANAGED_VLLM_TAG)"
    kill -TERM -- "-$MANAGED_VLLM_PID" 2>/dev/null || true
  fi

  # Also stop a manually launched vLLM frontend from before the controller.
  pkill -TERM -u "$USER" -f '[v]llm serve' 2>/dev/null || true

  local deadline=$(( $(date +%s) + VLLM_SHUTDOWN_TIMEOUT_SEC ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! pgrep -u "$USER" -f '[v]llm serve' >/dev/null 2>&1 && ! port_is_listening; then
      MANAGED_VLLM_PID=""
      MANAGED_VLLM_TAG=""
      log "[vllm stop] server stopped and port $VLLM_PORT is free"
      return 0
    fi
    sleep 5
  done

  log "[vllm stop] TERM timeout; sending KILL"
  if [ -n "$MANAGED_VLLM_PID" ]; then
    kill -KILL -- "-$MANAGED_VLLM_PID" 2>/dev/null || true
  fi
  pkill -KILL -u "$USER" -f '[v]llm serve' 2>/dev/null || true
  sleep 10

  if port_is_listening; then
    log "[fatal] port $VLLM_PORT remains occupied after stopping vLLM"
    ss -ltnp 2>/dev/null | grep ":$VLLM_PORT" || true
    return 1
  fi

  MANAGED_VLLM_PID=""
  MANAGED_VLLM_TAG=""
  log "[vllm stop] forced stop completed"
}

wait_for_exact_model() {
  local expected_model="$1"
  local server_pid="$2"
  local server_log="$3"
  local deadline=$(( $(date +%s) + VLLM_STARTUP_TIMEOUT_MIN * 60 ))
  local next_progress=$(( $(date +%s) + 60 ))

  log "[vllm wait] expected model: $expected_model"
  log "[vllm wait] startup timeout: ${VLLM_STARTUP_TIMEOUT_MIN} minutes"

  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      log "[fatal] vLLM process exited during startup: pid=$server_pid"
      tail -100 "$server_log" 2>/dev/null || true
      return 1
    fi

    if python - "$VLLM_MODELS_URL" "$expected_model" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request
url, expected = sys.argv[1:]
with urllib.request.urlopen(url, timeout=10) as response:
    payload = json.load(response)
served = [str(x.get("id")) for x in payload.get("data", [])]
raise SystemExit(0 if expected in served else 1)
PY
    then
      log "[vllm ready] exact checkpoint is being served"
      python - "$VLLM_MODELS_URL" <<'PY'
import json
import sys
import urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=10) as response:
    print(json.dumps(json.load(response), indent=2))
PY
      return 0
    fi

    if [ "$(date +%s)" -ge "$next_progress" ]; then
      log "[vllm wait] still loading; recent server log:"
      tail -20 "$server_log" 2>/dev/null || true
      next_progress=$(( $(date +%s) + 60 ))
    fi

    sleep 10
  done

  log "[fatal] vLLM did not serve the expected model within ${VLLM_STARTUP_TIMEOUT_MIN} minutes"
  tail -100 "$server_log" 2>/dev/null || true
  return 1
}

start_vllm() {
  local tag="$1"
  local config_path="$2"
  local model_path="$3"
  local server_log="$VLLM_LOG_ROOT/${tag}.log"
  local pid_file="$STATE_ROOT/${tag}.vllm.pid"

  VLLM_MANAGEMENT_STARTED=1
  stop_vllm "prepare $tag" || return 1

  log "[vllm start] tag=$tag"
  log "[vllm start] model=$model_path"
  log "[vllm start] log=$server_log"

  # setsid creates a dedicated process group so the frontend and worker
  # descendants can be stopped together after the model round.
  setsid bash -lc "
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate '$VLLM_ENV'
    cd '$PROJECT_DIR'
    exec vllm serve '$model_path' \\
      --served-model-name '$model_path' \\
      --host '$VLLM_HOST' \\
      --port '$VLLM_PORT' \\
      --tensor-parallel-size '$VLLM_TENSOR_PARALLEL_SIZE' \\
      --max-model-len '$VLLM_MAX_MODEL_LEN' \\
      --gpu-memory-utilization '$VLLM_GPU_MEMORY_UTILIZATION' \\
      --reasoning-parser qwen3 \\
      --enable-auto-tool-choice \\
      --tool-call-parser qwen3_coder \\
      --chat-template '$CHAT_TEMPLATE' \\
      --trust-remote-code
  " >> "$server_log" 2>&1 &

  MANAGED_VLLM_PID="$!"
  MANAGED_VLLM_TAG="$tag"
  printf '%s\n' "$MANAGED_VLLM_PID" > "$pid_file"

  log "[vllm start] pid=$MANAGED_VLLM_PID"

  wait_for_exact_model "$model_path" "$MANAGED_VLLM_PID" "$server_log"
}

round_result_root() {
  local tag="$1"
  printf '%s/%s/%s_%s\n' "$OUTPUTS_DIR" "$EXPERIMENT_NAME" "$tag" "$SUITE_STAMP"
}

round_archive_path() {
  local tag="$1"
  printf '%s.tar.gz\n' "$(round_result_root "$tag")"
}

round_is_complete() {
  local tag="$1"
  local result_root archive summary
  result_root="$(round_result_root "$tag")"
  archive="$(round_archive_path "$tag")"
  summary="$result_root/summary.json"

  [ -f "$summary" ] && [ -f "$archive" ] && [ -f "${archive}.sha256" ] || return 1

  python - "$summary" <<'PY'
import json
import sys
from pathlib import Path
obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if "completed_tasks" in obj:
    ok = bool(obj.get("complete")) and int(obj.get("completed_tasks", -1)) == int(obj.get("expected_tasks", -2))
else:
    ok = bool(obj.get("complete"))
raise SystemExit(0 if ok else 1)
PY
}

append_model_state() {
  local tag="$1"
  local config_path="$2"
  local model_path="$3"
  local started_at="$4"
  local ended_at="$5"
  local status="$6"
  local exit_code="$7"
  local result_root="$8"
  local archive="$9"

  python - "$MODEL_STATE_JSONL" "$tag" "$config_path" "$model_path" "$started_at" "$ended_at" "$status" "$exit_code" "$result_root" "$archive" <<'PY'
import json
import sys
(manifest, tag, config_path, model_path, started_at, ended_at,
 status, exit_code, result_root, archive) = sys.argv[1:]
row = {
    "model_tag": tag,
    "config_path": config_path,
    "model": model_path,
    "started_at": started_at,
    "ended_at": ended_at,
    "status": status,
    "exit_code": int(exit_code),
    "result_root": result_root,
    "archive": archive,
}
with open(manifest, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

make_partial_archive() {
  local tag="$1"
  local result_root partial_archive
  result_root="$(round_result_root "$tag")"
  partial_archive="${result_root}_partial_${SUITE_STAMP}.tar.gz"

  if [ "$ARCHIVE_PARTIAL_ON_FAILURE" != "1" ] || [ ! -d "$result_root" ]; then
    return 0
  fi

  log "[partial archive] creating $partial_archive"
  tar -C "$(dirname "$result_root")" -czf "$partial_archive" "$(basename "$result_root")"
  sha256sum "$partial_archive" > "${partial_archive}.sha256"
  log "[partial archive] completed: $(du -h "$partial_archive" | awk '{print $1}')"
}

write_suite_spec() {
  python - "$SUITE_SPEC_JSON" "$EXPERIMENT_NAME" "$SUITE_STAMP" "$MODEL_ORDER" "$START" "$END" "$TURN_BUDGETS" "$STOP_AFTER_SUCCESS" "$TASK_TIMEOUT_MIN" "$VLLM_STARTUP_TIMEOUT_MIN" "$TASK_LIST_FILE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
(out, experiment, stamp, order, start, end, budgets, stop_after_success,
 timeout_min, startup_timeout_min, task_list_file) = sys.argv[1:]
if task_list_file:
    payload = json.loads(Path(task_list_file).read_text(encoding="utf-8"))
    values = payload.get("task_indices", payload) if isinstance(payload, dict) else payload
    task_indices = [int(x) for x in values]
else:
    task_indices = list(range(int(start), int(end)))
num_tasks = len(task_indices)
num_budgets = len(budgets.split())
num_models = len(order.split())
stop_after_success = bool(int(stop_after_success))
obj = {
    "experiment_name": experiment,
    "suite_stamp": stamp,
    "model_order": order.split(),
    "task_start_inclusive": int(start),
    "task_end_exclusive": int(end),
    "task_list_file": task_list_file or None,
    "task_indices": task_indices,
    "expected_tasks_per_model": num_tasks,
    "turn_budgets": [int(x) for x in budgets.split()],
    "stop_after_success": stop_after_success,
    "trial_policy": (
        "ordered fallback; stop after first finished trial"
        if stop_after_success else "run every turn budget independently"
    ),
    "minimum_trials_per_model": num_tasks if stop_after_success else num_tasks * num_budgets,
    "maximum_trials_per_model": num_tasks * num_budgets,
    "total_minimum_trials": (num_tasks if stop_after_success else num_tasks * num_budgets) * num_models,
    "total_maximum_trials": num_tasks * num_budgets * num_models,
    "task_timeout_minutes": int(timeout_min),
    "vllm_startup_timeout_minutes": int(startup_timeout_min),
    "runner": "scripts/experiments/run_model_round.sh",
    "baseline_runner": "scripts/baseline/run_gdpval.py",
    "chat_template": "qwen36_chat_template.jinja",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
Path(out).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

write_suite_summary() {
  python - "$SUITE_SUMMARY_JSON" "$SUITE_SUMMARY_TXT" "$OUTPUTS_DIR" "$EXPERIMENT_NAME" "$SUITE_STAMP" "$MODEL_ORDER" "$TURN_BUDGETS" "$STOP_AFTER_SUCCESS" "$SUITE_SPEC_JSON" <<'PY'
import json
import sys
from pathlib import Path
(out_json, out_txt, outputs_dir, experiment, stamp, order,
 budgets, stop_after_success, suite_spec_path) = sys.argv[1:]
spec = json.loads(Path(suite_spec_path).read_text(encoding="utf-8"))
expected_tasks = int(spec["expected_tasks_per_model"])
maximum_trials = expected_tasks * len(budgets.split())
stop_after_success = bool(int(stop_after_success))
models = []
all_complete = True
for tag in order.split():
    root = Path(outputs_dir) / experiment / f"{tag}_{stamp}"
    summary_path = root / "summary.json"
    archive = Path(str(root) + ".tar.gz")
    archive_sha = Path(str(archive) + ".sha256")
    record = {
        "model_tag": tag,
        "result_root": str(root),
        "summary_path": str(summary_path),
        "archive": str(archive),
        "archive_sha256": str(archive_sha),
        "expected_tasks": expected_tasks,
        "maximum_trials": maximum_trials,
        "recorded_trials": 0,
        "completed_tasks": 0,
        "successful_tasks": 0,
        "complete": False,
    }
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            record["recorded_trials"] = int(summary.get("recorded_trials", 0))
            record["completed_tasks"] = int(summary.get("completed_tasks", 0))
            record["successful_tasks"] = int(summary.get("successful_tasks", 0))
            record["status_counts"] = summary.get("overall_status_counts", {})
            summary_complete = (
                bool(summary.get("complete"))
                and record["completed_tasks"] == expected_tasks
                and int(summary.get("expected_tasks", expected_tasks)) == expected_tasks
            )
            record["complete"] = summary_complete and archive.is_file() and archive_sha.is_file()
        except Exception as exc:
            record["summary_error"] = repr(exc)
    all_complete = all_complete and record["complete"]
    models.append(record)
obj = {
    "experiment_name": experiment,
    "suite_stamp": stamp,
    "turn_budgets": [int(x) for x in budgets.split()],
    "stop_after_success": stop_after_success,
    "expected_tasks_per_model": expected_tasks,
    "maximum_trials_per_model": maximum_trials,
    "all_complete": all_complete,
    "models": models,
}
Path(out_json).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
lines = [
    f"Experiment: {experiment}",
    f"Suite stamp: {stamp}",
    f"Turn order: {' -> '.join(budgets.split())}",
    f"Stop after success: {stop_after_success}",
    f"Expected tasks per model: {expected_tasks}",
    f"Maximum trials per model: {maximum_trials}",
    f"All complete: {all_complete}",
    "",
]
for model in models:
    lines.append(
        f"{model['model_tag']}: tasks={model['completed_tasks']}/{expected_tasks}, "
        f"successful={model['successful_tasks']}, trials={model['recorded_trials']}/{maximum_trials} max, "
        f"complete={model['complete']}, archive={model['archive']}"
    )
Path(out_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

validate_inputs() {
  log "[validate] project=$PROJECT_DIR"
  log "[validate] outputs=$OUTPUTS_DIR"
  log "[validate] suite_stamp=$SUITE_STAMP"
  log "[validate] model_order=$MODEL_ORDER"
  log "[validate] task range=[$START,$END)"
  log "[validate] ordered turn budgets=$TURN_BUDGETS"
  log "[validate] stop after success=$STOP_AFTER_SUCCESS"
  log "[validate] task timeout=${TASK_TIMEOUT_MIN} minutes"

  if ! [[ "$START" =~ ^[0-9]+$ && "$END" =~ ^[0-9]+$ && "$TASK_TIMEOUT_MIN" =~ ^[0-9]+$ ]]; then
    log "[fatal] START, END, and TASK_TIMEOUT_MIN must be integers"
    return 1
  fi
  if [ "$START" -lt 0 ] || [ "$END" -gt 220 ] || [ "$START" -ge "$END" ]; then
    log "[fatal] expected 0 <= START < END <= 220"
    return 1
  fi
  if [ "$STOP_AFTER_SUCCESS" != "0" ] && [ "$STOP_AFTER_SUCCESS" != "1" ]; then
    log "[fatal] STOP_AFTER_SUCCESS must be 0 or 1"
    return 1
  fi

  for cmd in python curl ss setsid timeout tar sha256sum conda; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      log "[fatal] required command not found: $cmd"
      return 1
    fi
  done

  for p in "$ROUND_SCRIPT" "$CHAT_TEMPLATE" \
           scripts/baseline/run_gdpval.py \
           configs/run_config.json; do
    if [ ! -f "$p" ]; then
      log "[fatal] missing required file: $p"
      return 1
    fi
  done

  bash -n "$ROUND_SCRIPT" || return 1

  local tag cfg model
  for tag in $MODEL_ORDER; do
    cfg="$(config_for_tag "$tag")" || return 1
    if [ ! -f "$cfg" ]; then
      log "[fatal] missing model config: $cfg"
      return 1
    fi
    model="$(model_from_config "$cfg")" || return 1
    if [ ! -d "$model" ] || [ ! -f "$model/config.json" ] || [ ! -f "$model/pruning_manifest.json" ]; then
      log "[fatal] incomplete checkpoint for $tag: $model"
      return 1
    fi
    log "[ok] $tag config=$cfg model=$model"
  done

  local task_count
  task_count="$(find artifacts/tasks -maxdepth 2 -path 'artifacts/tasks/task_*/task.json' -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$task_count" -lt 220 ]; then
    log "[fatal] found only $task_count GDPval task files; expected at least 220"
    return 1
  fi
  log "[ok] GDPval task count=$task_count"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$VLLM_MANAGEMENT_STARTED" = "1" ]; then
    stop_vllm "controller exit rc=$rc" || true
  fi
  write_suite_summary || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

main() {
  validate_inputs || exit 1
  write_suite_spec
  write_suite_summary

  if [ "$VALIDATE_ONLY" = "1" ]; then
    log "[validate only] all controller inputs passed; no vLLM process was stopped or started"
    return 0
  fi

  local tag config_path model_path result_root archive start_ts end_ts round_rc round_status

  for tag in $MODEL_ORDER; do
    config_path="$(config_for_tag "$tag")" || exit 1
    model_path="$(model_from_config "$config_path")" || exit 1
    result_root="$(round_result_root "$tag")"
    archive="$(round_archive_path "$tag")"

    log "================================================================"
    log "[model round] $tag"
    log "[model round] task selection is recorded in $SUITE_SPEC_JSON; turn budgets=$TURN_BUDGETS"
    log "================================================================"

    if round_is_complete "$tag"; then
      log "[skip model] $tag is already complete with archive and checksum"
      append_model_state "$tag" "$config_path" "$model_path" "$(date -Iseconds)" "$(date -Iseconds)" "already_complete" 0 "$result_root" "$archive"
      write_suite_summary
      continue
    fi

    start_ts="$(date -Iseconds)"

    if ! start_vllm "$tag" "$config_path" "$model_path"; then
      end_ts="$(date -Iseconds)"
      append_model_state "$tag" "$config_path" "$model_path" "$start_ts" "$end_ts" "vllm_start_failed" 1 "$result_root" "$archive"
      stop_vllm "failed startup $tag" || true
      write_suite_summary
      if [ "$CONTINUE_ON_MODEL_FAILURE" = "1" ]; then
        log "[continue] moving to next model after vLLM startup failure"
        continue
      fi
      exit 1
    fi

    log "[round start] invoking $ROUND_SCRIPT"
    set +e
    EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    MODEL_TAG="$tag" \
    CONFIG_PATH="$config_path" \
    START="$START" \
    END="$END" \
    TURN_BUDGETS="$TURN_BUDGETS" \
    STOP_AFTER_SUCCESS="$STOP_AFTER_SUCCESS" \
    TASK_TIMEOUT_MIN="$TASK_TIMEOUT_MIN" \
    TASK_LIST_FILE="$TASK_LIST_FILE" \
    RESUME="$RESUME" \
    RUN_STAMP="$SUITE_STAMP" \
    VLLM_MODELS_URL="$VLLM_MODELS_URL" \
    bash "$ROUND_SCRIPT"
    round_rc=$?
    set -u

    stop_vllm "completed child process for $tag" || true
    end_ts="$(date -Iseconds)"

    if [ "$round_rc" -eq 0 ] && round_is_complete "$tag"; then
      round_status="complete"
      log "[round complete] $tag archive=$archive"
    else
      round_status="failed_or_incomplete"
      log "[round incomplete] $tag child_exit=$round_rc"
      make_partial_archive "$tag" || true
    fi

    append_model_state "$tag" "$config_path" "$model_path" "$start_ts" "$end_ts" "$round_status" "$round_rc" "$result_root" "$archive"
    write_suite_summary

    if [ "$round_status" != "complete" ] && [ "$CONTINUE_ON_MODEL_FAILURE" != "1" ]; then
      log "[fatal] stopping after incomplete model round: $tag"
      exit 1
    fi
  done

  stop_vllm "all model rounds processed" || true
  write_suite_summary

  if python - "$SUITE_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path
obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if obj.get("all_complete") else 1)
PY
  then
    touch "$SUITE_ROOT/COMPLETE"
    log "[done] all three model rounds are complete"
    log "[done] all requested tasks reached terminal outcomes; actual trials depend on early success"
    log "[done] suite summary=$SUITE_SUMMARY_JSON"
  else
    log "[warning] controller finished, but one or more model rounds are incomplete"
    return 1
  fi
}

main "$@"
