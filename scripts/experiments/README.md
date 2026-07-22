# Experiment controllers

The four top-level wrappers preserve one evaluation contract:

- `run_experiment_1.sh`: global weighted routing mass;
- `run_experiment_2.sh`: task-normalized weighted routing mass;
- `run_experiment_3.sh`: unweighted top-8 count;
- `run_experiment_4.sh`: one sector-conditioned weighted-mass family.

`run_experiment_family.sh` binds model-specific configs, `run_experiment_suite.sh` manages vLLM across keep sizes, and `run_model_round.sh` executes the task-major fallback policy `80 -> 50 -> 120` with early stopping after the first successful trial.
