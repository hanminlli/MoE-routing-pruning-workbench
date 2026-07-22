#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stirrup-py312

STIRRUP_ROOT="$(python -c 'from pathlib import Path; import stirrup; print(Path(stirrup.__file__).resolve().parent)')"
TARGET="$STIRRUP_ROOT/clients/chat_completions_client.py"

if ! grep -q "MODEL_OUTPUT_TRUNCATED_BY_MAX_TOKENS" "$TARGET" 2>/dev/null; then
  cp -p "$TARGET" "${TARGET}.bak_routecat_$(date +%Y%m%d_%H%M%S)"
  cp -f patches/chat_completions_client.py "$TARGET"
  python -m py_compile "$TARGET"
else
  echo "[ok] token-id/truncation patch already installed"
fi

python patches/patch_stirrup_web_guard.py
python patches/patch_stirrup_agent_empty_code_exec.py
python patches/patch_stirrup_shell_timeout.py
python scripts/setup/verify_stirrup_runtime.py

echo "[done] Stirrup runtime patched and verified"
