# Build and validation report

## Project lineage

This repository was built from two deliberately separated sources:

1. The validated Qwen3.6, Stirrup, vLLM, GDPval reconstruction guide is the operational source of truth for Phase I. Its task export, exact token logging, causal replay alignment, accounting, checkpoint surgery, and evaluation contract were retained and reorganized around the research experiments.
2. The uploaded activation-steering repository was inspected only for Phase-II rationale and architectural reference. Its implementation was not copied into Phase I, and it was not imported wholesale into this repository.

The result is a working research project rather than a recovery snapshot. Runtime data, model weights, benchmark content, logs, and machine-specific environment captures are excluded.

## Implemented scope

### Phase I

- Exact vLLM prompt and generated token-ID preservation.
- Stirrup agent execution with bounded failure-recovery guards.
- Generated-token causal replay through Hugging Face.
- Per-task, per-layer, per-expert ordinary routing accounting.
- Optional token-type and advanced expert-output accounting.
- Four pruning-plan families:
  1. global weighted routing mass;
  2. task-normalized weighted routing mass;
  3. unweighted top-8 count;
  4. sector-conditioned weighted routing mass.
- Keep-192, keep-128, and keep-64 plan generation.
- Streaming Qwen-style Safetensors expert-bank pruning.
- Model-specific evaluation-config generation.
- Full-task and non-contiguous sector-task evaluation controllers.
- Stop-after-first-success fallback policy: 80, then 50, then 120 turns.

### Phase II

- Qwen-style transformer-layer resolution.
- Residual activation capture with last-token and masked-mean pooling.
- Paired contrastive activation addition and difference-in-means directions.
- L2-normalized, checkpoint-specific steering artifacts.
- Residual hooks with all, last, prefill, prefill-last, and decode policies.
- Single-coefficient application and symmetric coefficient sweeps.

## Validation performed in the build environment

### Repository checks

- Python source compilation passed.
- Shell syntax validation passed for every checked-in shell script.
- Checked-in JSON and YAML parsing passed.
- Repository-relative config-path validation passed.
- Local Markdown-link validation passed.
- Public-tree secret/path/large-file audit passed.
- Six synthetic unit tests passed.
- Editable installation succeeded with the existing local build toolchain and build isolation disabled.
- CLI help smoke tests passed for the plan generator and both steering commands.

Ruff was not available from the build environment's package mirror, so lint execution was left to the checked-in GitHub Actions workflow. The CI workflow installs Ruff before running the configured lint scope.

### Actual accounting compatibility

The compact-accounting importer and compactor were exercised on the real 219-task ordinary accounting schema used by the project:

- 219 calibration tasks were recovered;
- 2,242,560 task-layer-expert rows were reconstructed;
- task 0188 remained excluded;
- row count, task/layer/expert keys, selected counts, weighted counts, response-token counts, and top-k values matched the source table under order-independent signatures.

The unified plan generator then produced keep-192, keep-128, and keep-64 plans for all four Phase-I criteria. The sector test selected the expected 25 Finance and Insurance tasks and produced a separate checkpoint-compatible plan family.

## What was not executed here

The build environment did not contain the full Qwen3.6 checkpoint, an eight-GPU serving node, or the private GDPval runtime artifacts. Therefore the following were not rerun during packaging:

- full vLLM model serving;
- all 220 Stirrup tasks;
- Hugging Face replay of every model call;
- physical pruning of the complete sharded model;
- full GDPval evaluation of generated checkpoints;
- Phase-II steering on an actual pruned Qwen3.6 checkpoint.

Those operations are represented by the same workflow and contracts used in the validated project, but they still require the authorized compute and data environment described in the root README.
