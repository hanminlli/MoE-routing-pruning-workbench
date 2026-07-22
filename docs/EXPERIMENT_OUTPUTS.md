# Expected experiment outputs

## Phase I

Each pruning-plan directory should contain:

```text
global_expert_scores.csv
plans/*.json
plan_summary.csv
run_metadata.json
```

Each checkpoint should contain:

```text
config.json
model.safetensors.index.json
model-*.safetensors
pruning_manifest.json
```

Each evaluation round should contain:

```text
experiment_spec.json
trial_manifest.jsonl
summary.json
summary.txt
master.log
runs/
task_logs/
experiment_snapshot/
```

## Phase II

Direction discovery produces a `.pt` steering artifact. Application produces JSONL records containing the prompt, completion, checkpoint, artifact path, layer, coefficient, and position mode.
