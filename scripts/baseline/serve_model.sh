#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${CONFIG_PATH:-configs/run_config.json}"
MODEL="$(python -c 'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text())["model"])' "$CONFIG_PATH")"
vllm serve "$MODEL" \
  --served-model-name "$MODEL" --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 --max-model-len 262144 \
  --gpu-memory-utilization 0.85 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --chat-template qwen36_chat_template.jinja --trust-remote-code
