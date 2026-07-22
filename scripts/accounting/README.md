# Accounting scripts

- `replay_frequency.py` is the ordinary exact-token replay used for the primary pruning experiments.
- `run_ordinary_accounting_from_baseline.sh` applies it to accepted baseline runs.
- `compact_ordinary_accounting.py` validates and combines task outputs into the table used by all four plan generators.
- `replay_by_token_type.py` provides prompt/generated token-type diagnostics.
- `replay_response_activation_metrics.py` and its launcher implement the optional advanced expert-output observer.
- Import, restore, and packaging helpers support moving accepted accounting outside Git without changing the project schema.

Do not combine ordinary and partial advanced accounting in one plan-generation run.
