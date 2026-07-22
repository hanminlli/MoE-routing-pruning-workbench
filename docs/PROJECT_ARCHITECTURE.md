# Project architecture

## Design principles

1. **The agent trajectory is the calibration object.** Routing is measured on actual generated response tokens from a tool-using agent, not on isolated short prompts.
2. **Serving and accounting are decoupled.** vLLM produces trajectories and exact token IDs; Hugging Face replay exposes router and expert modules.
3. **Pruning plans are immutable artifacts.** Every checkpoint is traceable to a JSON plan and source model.
4. **The evaluation contract is fixed.** Only the model/tokenizer path changes between baseline and pruned runs.
5. **Phase II consumes Phase-I outputs.** Activation steering never participates in expert selection.
6. **Public code and private artifacts are separated.** The repository is functional without embedding benchmark or model data.

## Phase boundaries

### Phase I output contract

A Phase-I checkpoint must contain:

- a valid Hugging Face model configuration;
- a rebuilt `model.safetensors.index.json`;
- sharded tensors with the expected expert axis;
- a `pruning_manifest.json`;
- the source plan checksum;
- consistent routed-expert count in all 40 language-model MoE layers.

### Phase II input contract

A Phase-II run requires:

- a validated full or pruned checkpoint;
- a contrastive JSONL dataset;
- a concrete transformer layer index;
- a pooling rule;
- a direction-discovery method;
- a coefficient schedule and intervention-position policy.

The steering artifact records these choices so an intervention is reproducible.
