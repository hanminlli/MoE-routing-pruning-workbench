#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PINNED = {
    "chat_template_path": "qwen36_chat_template.jinja",
    "vllm_base_url": "http://localhost:8000/v1",
    "vllm_api_key": "EMPTY",
    "tool_call_parser": "qwen3_coder",
    "reasoning_parser": "qwen3",
    "temperature": 0,
    "max_model_len": 262144,
    "max_tokens_per_turn": 32768,
    "num_gdpval_tasks": 220,
    "prompt_track": "optionB_prompt_v3_80turn_general_budget_prompt",
    "runner": "scripts/baseline/run_gdpval.py",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=[])
    ap.add_argument("--base", default="configs/run_config.json")
    args = ap.parse_args()
    base_path = Path(args.base)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    for k, expected in PINNED.items():
        if base.get(k) != expected:
            raise SystemExit(f"[fatal] base config mismatch: {k}={base.get(k)!r}, expected={expected!r}")
    candidates = [Path(x) for x in args.configs]
    if not candidates:
        candidates = sorted(Path("configs").glob("experiment_*/run_config_keep*.json"))
    allowed = {"model", "tokenizer"}
    for p in candidates:
        if not p.is_file():
            raise SystemExit(f"[fatal] config missing: {p}")
        cfg = json.loads(p.read_text(encoding="utf-8"))
        diffs = sorted(k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k))
        if not set(diffs).issubset(allowed):
            raise SystemExit(f"[fatal] {p} differs from baseline in forbidden fields: {diffs}")
        if cfg.get("model") != cfg.get("tokenizer"):
            raise SystemExit(f"[fatal] {p}: model and tokenizer differ")
        if cfg.get("max_tokens_per_turn") != 32768:
            raise SystemExit(f"[fatal] {p}: max_tokens_per_turn is not 32768")
        print(f"[ok] {p}: only model/tokenizer differ")
    print("[ok] baseline contract pinned at max_tokens_per_turn=32768")


if __name__ == "__main__":
    main()
