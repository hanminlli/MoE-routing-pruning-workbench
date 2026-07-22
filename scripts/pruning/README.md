# Pruning scripts

## Primary Phase-I path

1. `generate_compact_accounting_plans.py` consumes the accepted compact per-task accounting table and emits plans for all four experiment criteria.
2. `build_checkpoint_family.sh` builds keep-192, keep-128, and keep-64 checkpoints from one plan family.
3. `prune_checkpoint.py` performs CPU-streamed Safetensors surgery.
4. `configure_pruned_configs.py` derives model-specific evaluation configs from the fixed baseline contract.

The wrappers under `scripts/experiments/` call this path.

## Reconstruction-compatible utilities

`prepare_pruning_plans.py`, `prepare_experiment_variant_plans.py`, `build_pruned_checkpoints.sh`, and `build_variant_checkpoints.sh` are retained for compatibility with earlier per-task accounting layouts. They are not the recommended entry point for new runs. New experiments should use the compact-accounting path above so every criterion shares one validated input table and plan schema.
