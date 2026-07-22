# RouteCat: MoE Pruning and Activation Steering

A two-phase project for Mixture-of-Experts pruning and post-pruning activation steering on long-horizon tool-using language-model agents. The repository is built around the operational workflow that was validated for `Qwen/Qwen3.6-35B-A3B`, vLLM, Stirrup, and GDPval on a 8 $\times$ A100 80 GB node. 

## 1. Context

Large sparse Mixture-of-Experts models activate only a small subset of routed experts for each token, yet serving them typically requires the full expert bank to remain resident in GPU memory. Consequently, their per-token computation is sparse, but their parameter-memory footprint remains large.  

This project investigates whether expert-routing statistics collected from representative agent trajectories can identify a smaller expert bank that preserves useful model behavior. The primary objective is to reduce GPU memory consumption and make large MoE models easier to serve on constrained hardware, while maintaining agent-level task performance. Reduced checkpoint size and faster model loading are additional benefits.

Because the number of activated experts per token remains fixed, expert-bank pruning should be viewed primarily as a parameter-memory optimization rather than a direct reduction in per-token compute.

Specifically, this project asks two connected questions:

1. **Phase I — MoE pruning:** Can we use routing statistics collected from realistic agent trajectories to remove low-importance routed experts?

2. **Phase II — Activation steering:** After reducing the expert bank, does the pruned model still expose stable residual-stream directions that can control behavior at inference time?

The first phase produces pruned checkpoints. The second phase treats a selected pruned checkpoint as its input. The phases are intentionally separate: pruning changes model capacity and expert topology; activation steering changes hidden states at inference time without modifying weights.

## 2. Target system

The validated reference configuration is:

| Component | Reference setting |
|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B` |
| Routed MoE layers | 40 |
| Routed experts per layer | 256 |
| Experts selected per token | 8 |
| Shared experts | Preserved unchanged |
| Retained banks | 192, 128, and 64 experts per layer |
| Agent harness | Stirrup |
| Inference server | vLLM, OpenAI-compatible endpoint |
| Tensor parallelism | 8 |
| Benchmark | GDPval, 220 tasks |
| Evaluation tasks | All 220 for Experiments 1–3; sector-local tasks for Experiment 4 |
| Calibration bucket | Generated-output prediction positions |
| Turn fallback | 80 → 50 → 120 |
| Maximum tokens per turn | 32,768 |
| Maximum model length | 262,144 |
| Per-trial timeout | 90 minutes |


## 3. System overview

```text
GDPval task
  ↓
Stirrup agent
  ↓
OpenAI-compatible vLLM endpoint
  ↓
Qwen3.6-35B-A3B
  ↓ return_token_ids=True
Exact prompt/generated token IDs
  ↓
Hugging Face replay with MoE hooks
  ↓
Per-task, per-layer, per-expert routing statistics
  ↓
Four pruning criteria
  ↓
Layer-wise top-K expert plans
  ↓
Streaming Safetensors checkpoint surgery
  ↓
Pruned 192 / 128 / 64 checkpoints
  ↓
GDPval evaluation under the fixed contract
  ↓
Selected pruned checkpoint
  ↓
Contrastive activation collection
  ↓
Steering-direction discovery and coefficient sweep
```

## 4. Phase I methodology: MoE pruning

### 4.1 Exact causal accounting

For one model call, let:

- $N$ be the number of prompt tokens;
- $T$ be the number of generated tokens.

The replay uses two conceptual buckets:

- Prompt/input routing positions: $0,\ldots,N-1$.
- Generated-output prediction positions: $N-1,\ldots,N+T-2$.

The final prompt position appears in both conceptual buckets because it is part of the prompt and predicts the first generated token. The pruning experiments use only Generated_output prediction positions.

For every task $q$, generated position $t$, layer $\ell$, and expert $e$, let $p_{q,t,\ell,e}$ be the router probability renormalized over the selected top-8 experts.

### 4.2 Experiment 1 — global weighted routing mass

$$
S^{(1)}_{\ell,e}=\sum_q\sum_{t=1}^{T_q}p_{q,t,\ell,e}\mathbf{1}\{e\in\text{Top8}_{q,t,\ell}\}.
$$

Longer responses contribute more total mass. This is the primary global weighted-mass criterion.

### 4.3 Experiment 2 — task-normalized weighted routing mass

$$
S^{(2)}_{\ell,e}=\sum_q\frac{1}{T_q}\sum_{t=1}^{T_q}p_{q,t,\ell,e}\mathbf{1}\{e\in\text{Top8}_{q,t,\ell}\}.
$$

Each task contributes approximately equal total mass per layer, reducing domination by unusually long agent trajectories.

### 4.4 Experiment 3 — unweighted top-8 appearance count

$$
S^{(3)}_{\ell,e}=\sum_q\sum_{t=1}^{T_q}\mathbf{1}\{e\in\text{Top8}_{q,t,\ell}\}.
$$

Every selected appearance has equal weight. Router-probability magnitude is ignored.

### 4.5 Experiment 4 — sector-conditioned weighted routing mass

For sector $c$:

$$
S^{(4,c)}_{\ell,e}=\sum_{q\in\mathcal{Q}_c}\sum_{t=1}^{T_q}p_{q,t,\ell,e}\mathbf{1}\{e\in\text{Top8}_{q,t,\ell}\}.
$$

A separate expert ranking is built for each sector. The current largest-sector study uses `Finance and Insurance` as an example, containing 25 tasks under the current task taxonomy. 

Unlike Experiments 1–3, the main evaluation for Experiment 4 is sector-local: a sector-conditioned checkpoint is first tested on the same sector’s task set. Cross-sector transfer can then be measured as a separate analysis.

### 4.6 Plan construction

For each experiment and each layer:

1. validate the accepted task accounting;
2. aggregate the selected score;
3. sort by score descending and expert ID ascending;
4. retain the top $K\in\{192,128,64\}$ experts;
5. sort retained original IDs before tensor slicing;
6. write original-to-new and new-to-original expert maps;
7. compute retained-score coverage;
8. emit a checkpoint-compatible JSON plan.

Every layer is ranked independently. We do not average the same expert ID across different layers.

### 4.7 Checkpoint surgery

The pruning exporter streams the sharded Safetensors checkpoint on CPU. It slices only:

- routed expert `gate_up_proj` tensors;
- routed expert `down_proj` tensors;
- router `gate.weight` rows.

All attention blocks, normalization layers, embeddings, output heads, shared experts, and non-MoE tensors are copied unchanged. The shard index and model configuration are regenerated, and a pruning manifest records the plan and tensor shapes.

### 4.8 Optional advanced activation accounting

The repository also preserves the advanced response-activation observer. For a selected expert event with gate $g$ and raw expert-output norm $r$, it records:

$$
A_{\ell,e}(\alpha,\beta)=\sum_t\mathbf{1}\{e\text{ selected}\}g^{\alpha}r^{\beta},\qquad \alpha,\beta\in\{0,1,2\}.
$$

This supports frequency, SEER-like, EAN, REAP, MAN, and MSAN-style scores. It is an on going extension.

## 5. Phase II methodology: activation steering on a pruned model

Phase II begins only after selecting a validated pruned checkpoint from Phase I.

### 5.1 Contrastive direction discovery

Given positive and negative examples, collect residual activations $h^+_i$ and $h^-_i$ at a selected transformer layer.

Paired contrastive activation addition uses:

$$
v=\frac{1}{m}\sum_{i=1}^{m}(h_i^+-h_i^-).
$$

Difference in means uses:

$$
v=\frac{1}{m_+}\sum_i h_i^+-\frac{1}{m_-}\sum_j h_j^-.
$$

The saved artifact contains the normalized direction, layer index, pooling rule, source checkpoint, method, sample counts, and discovery metadata. 

There are many other ways of finding such a direction and they are still being implemented and tested.

### 5.2 Residual-Stream Intervention

After estimating a steering direction $v \in \mathbb{R}^{d_{\mathrm{model}}}$, we apply it at a selected transformer layer $\ell$ during inference. Let $h_{\ell,t}$ denote the residual-stream representation at token position $t$. The steered representation is

$$
\widetilde{h}_{\ell,t} = h_{\ell,t} + \alpha m_t v,
$$

where:

- $v$ is the steering direction;
- $\alpha \in \mathbb{R}$ controls the intervention strength and sign;
- $m_t \in \{0,1\}$ determines whether position $t$ is modified.

The direction is L2-normalized before use: $v \leftarrow \frac{v}{\lVert v \rVert_2}.$ This makes $\lvert \alpha \rvert$ directly control the perturbation magnitude. The setting $\alpha=0$ recovers the original unsteered model and is used as the baseline. The intervention is applied dynamically during the forward pass and does not modify the model parameters. All layers after layer $\ell$ process the modified hidden representation.

#### Intervention stages

Decoder-only generation contains two stages:

1. **Prefill:** the full prompt is processed and the key-value cache is initialized.
2. **Decode:** generated tokens are processed autoregressively, typically one new token per forward call.

The intervention mode determines whether steering is applied during prefill, decoding, or both.

#### Supported modes

- **`all`**

  Apply steering to every token position during both prefill and decoding.

  This is the strongest and most persistent mode because it modifies the full prompt representation and every subsequent decoding step.

- **`last`**

  Apply steering only to the final position of each forward call.

  During prefill, this means the final prompt token. During cached decoding, the newly processed token is usually also the final position, so steering is applied at every decoding step.

  Therefore, `last` does not mean steering is applied only once.

- **`prefill`**

  Apply steering to all prompt positions during prefill, but not during decoding.

  This changes how the prompt is encoded and can influence later generation indirectly through the modified key-value cache.

- **`prefill_last`**

  Apply steering only to the final prompt position during prefill.

  No intervention is applied to earlier prompt positions or during decoding. This provides a localized intervention directly before prediction of the first generated token.

- **`decode`**

  Leave the prompt prefill unchanged and apply steering only during autoregressive decoding.

  This isolates the effect of steering the evolving generation process without modifying the model's initial prompt representation.

#### Mode comparison

| Mode | Prompt positions | Final prompt position | Decode steps |
|---|---:|---:|---:|
| `all` | All | Yes | Yes |
| `last` | Final only | Yes | Yes |
| `prefill` | All | Yes | No |
| `prefill_last` | Final only | Yes | No |
| `decode` | None | No | Yes |

#### Coefficient sweep

Each steering direction should be evaluated over a symmetric range of coefficients, for example $\alpha \in \{-8,-4,-2,-1,0,1,2,4,8\}.$

### 5.3 Phase-II experimental design

For a selected pruning experiment and bank size:

1. discover directions on the pruned checkpoint itself;
2. sweep symmetric coefficients, including $\alpha=0$;
3. compare task quality, behavior-control metrics, and generation stability;
4. compare the same direction-discovery protocol across full, keep192, keep128, and keep64 checkpoints;
5. measure whether pruning changes direction alignment, effective coefficient range, or controllability.

The steering code is intentionally independent of the GDPval/vLLM agent runner. Initial steering studies run through Hugging Face so hidden-state hooks remain explicit and auditable. A serving integration should be added only after validating the offline intervention semantics.

## 6. Repository layout

```text
.
├── README.md
├── configs/
│   ├── run_config.json
│   ├── phase1/                  # Four experiment contracts
│   ├── phase2/                  # Direction discovery/application examples
│   └── models/                  # Generated model-specific configs
├── data/
│   ├── contrastive/             # Public examples only
│   └── task_metadata/           # Sector taxonomy template
├── src/
│   ├── stirrup_logging_client.py
│   └── routecat_moe_steering/
│       ├── phase1/              # Scoring and plan construction
│       └── phase2/              # Activation capture, directions, hooks
├── scripts/
│   ├── setup/                   # Three environments and Stirrup patches
│   ├── data/                    # GDPval download/export/sanitization
│   ├── baseline/                # vLLM serving and Stirrup execution
│   ├── accounting/              # Exact and advanced HF replay
│   ├── pruning/                 # Four criteria and checkpoint surgery
│   ├── experiments/             # Fixed evaluation contract
│   ├── steering/                # Phase-II discovery and application
│   └── validation/
├── patches/                     # Reproducible Stirrup runtime patches
├── artifacts/                   # Runtime tasks, runs, and accounting outputs
├── accounting_result/           # Accepted ordinary accounting
├── advanced_accounting_result/  # Optional advanced accounting
├── pruning_info/                # Generated plans
├── models/                      # Generated pruned checkpoints; Git-ignored
└── results/                     # Generated evaluations; Git-ignored
```

## 7. Complete step-by-step workflow

All commands below are run from the repository root.

### Step 0 — Clone and initialize

```bash
git clone <YOUR_REPOSITORY_URL>
cd routecat-moe-steering
bash scripts/setup/initialize_project.sh
```

### Step 1 — Create the three environments

The workflow separates serving, agent execution, and Hugging Face replay because their CUDA and framework requirements differ.

```bash
bash scripts/setup/create_environments.sh
```

This creates:

- `vllm-env` for serving;
- `stirrup-py312` for GDPval agent runs;
- `routing-hf-py312` for replay, plan construction, pruning, and steering.

### Step 2 — Install and verify Stirrup patches

```bash
bash scripts/setup/install_stirrup_patches.sh
```

The patches preserve exact vLLM token IDs, handle deterministic max-token truncation, prevent repeated empty `code_exec{}` calls, limit repeated web-fetch loops, and set a bounded shell timeout.

### Step 3 — Download GDPval

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stirrup-py312
bash scripts/data/download_gdpval.sh
```

### Step 4 — Export self-contained tasks

```bash
PYTHONPATH=. python scripts/data/export_gdpval_tasks.py \
  --config configs/run_config.json \
  --start 0 \
  --end 220 \
  --download-missing-files

python scripts/validation/validate_tasks.py --tasks-dir artifacts/tasks
```

The exporter copies reference inputs into each task folder and removes answer-side fields. The model prompt sees only local basenames, never absolute host paths.

### Step 5 — Start the baseline model

In a dedicated terminal:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-env
CONFIG_PATH=configs/run_config.json bash scripts/baseline/serve_model.sh
```

### Step 6 — Run one smoke-test task

In another terminal:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stirrup-py312

PYTHONPATH=. python scripts/baseline/run_gdpval.py \
  --config configs/run_config.json \
  --tasks-dir artifacts/tasks \
  --runs-dir artifacts/runs \
  --start 0 \
  --end 1 \
  --max-turns 80
```

Verify that the run contains `status.json`, `model_calls.jsonl`, and at least one output file.

### Step 7 — Verify exact token IDs

The logger requests `return_token_ids=True` from vLLM and checks:

```text
len(prompt_token_ids) == response.token_usage.input
len(generated_token_ids) == response.token_usage.answer + response.token_usage.reasoning
```

Inspect a run with the validation utilities before launching the full baseline.

### Step 8 — Run the full baseline

```bash
bash scripts/baseline/run_baseline_220.sh
```

The baseline should use the same agent prompt, tools, turn policy, timeout, maximum tokens, and chat template later used for every pruned model.

### Step 9 — Stop vLLM and run ordinary routing accounting

Hugging Face replay requires the GPUs occupied by vLLM, so stop the server first.

```bash
bash scripts/accounting/run_ordinary_accounting_from_baseline.sh
python scripts/validation/validate_accounting_input.py --accounting-root accounting_result
```

Create the compact table consumed by the four plan generators:

```bash
python scripts/accounting/compact_ordinary_accounting.py \
  --accounting-root "$(cat ../outputs/latest_routecat_ordinary_accounting_dir.txt)" \
  --output accounting_result/ordinary_response_routing_by_task.csv.gz \
  --exclude-tasks 188
```

The accepted compact table retains, per task/layer/expert:

```text
task_num, task_id, source_run, layer, module_name, expert,
selected_count, weighted_count, response_tokens, top_k
```

The core conservation check is:

$$
\sum_e \text{selected\_count}_{q,\ell,e}=T_q\times 8
$$

for every task and layer.

### Step 10 — Prepare task-sector metadata

Create a local CSV using:

```text
data/task_metadata/task_sectors.example.csv
```

Required columns:

```text
task_num,sector
```

The real taxonomy file is intentionally not committed if it contains internal annotations.

### Step 11 — Generate all four pruning-plan families

```bash
ACCOUNTING=accounting_result/ordinary_response_routing_by_task.csv.gz \
TASK_METADATA=data/task_metadata/task_sectors.csv \
bash scripts/experiments/prepare_all_phase1_plans.sh
```

Expected roots:

```text
pruning_info/experiment_1_weighted_mass/
pruning_info/experiment_2_task_normalized_weighted_mass/
pruning_info/experiment_3_unweighted_count/
pruning_info/experiment_4_sector_weighted_mass/
```

Experiment 4 also writes deterministic JSON task lists under:

```text
pruning_info/experiment_4_sector_weighted_mass/task_selection/
```

### Step 12 — Build Experiment 1 checkpoints

```bash
EXPERIMENT_NAME=experiment_1_weighted_mass \
PLAN_ROOT=pruning_info/experiment_1_weighted_mass/global \
PLAN_PREFIX=weighted_mass_global \
MODEL_ROOT=models/experiment_1_weighted_mass \
CONFIG_ROOT=configs/models/experiment_1_weighted_mass \
SOURCE_MODEL=Qwen/Qwen3.6-35B-A3B \
bash scripts/pruning/build_checkpoint_family.sh
```

### Step 13 — Build Experiment 2 checkpoints

```bash
EXPERIMENT_NAME=experiment_2_task_normalized_weighted_mass \
PLAN_ROOT=pruning_info/experiment_2_task_normalized_weighted_mass/global \
PLAN_PREFIX=task_normalized_weighted_mass_global \
MODEL_ROOT=models/experiment_2_task_normalized_weighted_mass \
CONFIG_ROOT=configs/models/experiment_2_task_normalized_weighted_mass \
SOURCE_MODEL=Qwen/Qwen3.6-35B-A3B \
bash scripts/pruning/build_checkpoint_family.sh
```

### Step 14 — Build Experiment 3 checkpoints

```bash
EXPERIMENT_NAME=experiment_3_unweighted_count \
PLAN_ROOT=pruning_info/experiment_3_unweighted_count/global \
PLAN_PREFIX=unweighted_count_global \
MODEL_ROOT=models/experiment_3_unweighted_count \
CONFIG_ROOT=configs/models/experiment_3_unweighted_count \
SOURCE_MODEL=Qwen/Qwen3.6-35B-A3B \
bash scripts/pruning/build_checkpoint_family.sh
```

### Step 15 — Build the Finance and Insurance Experiment 4 checkpoints

```bash
EXPERIMENT_NAME=experiment_4_sector_weighted_mass_finance_and_insurance \
PLAN_ROOT=pruning_info/experiment_4_sector_weighted_mass/finance_and_insurance \
PLAN_PREFIX=sector_weighted_mass_finance_and_insurance \
MODEL_ROOT=models/experiment_4_sector_weighted_mass/finance_and_insurance \
CONFIG_ROOT=configs/models/experiment_4_sector_weighted_mass/finance_and_insurance \
SOURCE_MODEL=Qwen/Qwen3.6-35B-A3B \
bash scripts/pruning/build_checkpoint_family.sh
```

Replace the sector slug to build another sector.

### Step 16 — Validate every checkpoint family

```bash
python scripts/validation/validate_pruned_family.py \
  --model-root models/experiment_1_weighted_mass \
  --config-dir configs/models/experiment_1_weighted_mass \
  --expected-criterion weighted_mass \
  --check-tensor-headers
```

Repeat for each family. Validation must check model configuration, shard index, pruning manifest, number of layers, retained expert count, router rows, expert tensor axes, and preserved non-prunable tensors.

### Step 17 — Evaluate Experiments 1–3

The suite controller starts one checkpoint at a time, waits for the exact served-model ID, evaluates tasks with the ordered fallback, archives the round, stops vLLM, and advances to the next checkpoint.

```bash
bash scripts/experiments/run_experiment_1.sh
bash scripts/experiments/run_experiment_2.sh
bash scripts/experiments/run_experiment_3.sh
```

The evaluation policy is task-major:

```text
for each task:
  try 80 turns
  if successful: stop
  otherwise try 50 turns
  if successful: stop
  otherwise try 120 turns
```

The three budgets are fallbacks, not three mandatory independent evaluations.

### Step 18 — Evaluate Experiment 4 locally

```bash
SECTOR_SLUG=finance_and_insurance \
bash scripts/experiments/run_experiment_4.sh
```

`run_model_round.sh` accepts a `TASK_LIST_FILE`, so non-contiguous sector task IDs are evaluated without changing the global benchmark indexing.

### Step 19 — Optional advanced activation accounting

```bash
bash scripts/accounting/run_advanced_accounting_from_baseline.sh
python scripts/validation/validate_advanced_accounting.py \
  --root advanced_accounting_result
```

Do not mix partial advanced accounting with the accepted ordinary accounting used by Experiments 1–4.

### Step 20 — Select a Phase-II checkpoint

Choose a checkpoint only after Phase-I validation and evaluation. A reasonable progression is:

1. full model as a steering reference;
2. keep192 for low-compression comparison;
3. keep128 as the main compression/controllability trade-off;
4. keep64 only as an aggressive-pruning stress test.

### Step 21 — Prepare Contrastive Steering Data

Contrastive steering data must be created separately from the MoE routing statistics. Routing statistics describe expert usage, but they do not directly provide the positive and negative behavioral examples needed for steering-direction discovery.

Create JSONL rows using either:

    {"positive": "...", "negative": "..."}

or:

    {"matching": "...", "not_matching": "..."}

Each pair should differ mainly in the target behavior while remaining similar in topic, length, and information content. For this project, the recommended initial target is **execution-oriented agent behavior**.

Positive examples may represent:

- decisive task execution;
- valid tool usage;
- stopping research once sufficient evidence is available;
- output verification;
- successful task completion.

Negative examples may represent:

- repeated browsing;
- repeated invalid tool calls;
- excessive planning without execution;
- failure to verify deliverables;
- failure to terminate the task correctly.

Example:

    {
      "pair_id": "train-001",
      "behavior": "stop_browsing_and_execute",
      "positive": "The available evidence is sufficient. I will now create and verify the requested deliverable.",
      "negative": "I will continue trying additional sources before creating the requested deliverable."
    }

Suitable data sources include:

1. **Manually curated pairs**

   Write positive and negative versions of the same underlying response. This provides the cleanest behavioral contrast.

2. **Phase-I agent trajectories**

   Use successful and failed trajectories to identify useful behavioral contrasts, then convert them into matched or minimally edited pairs. Raw successful and failed trajectories should not be used directly because they may differ in many unrelated ways.

3. **Public behavioral datasets**

   Public datasets may be used when their licenses permit redistribution and their labels match the intended steering behavior.

A directory structure usable is:

    data/
    └── contrastive/
        └── execution_oriented/
            ├── train_pairs.jsonl
            ├── validation_pairs.jsonl
            ├── evaluation_prompts.jsonl
            └── README.md

Example training rows:

    {"pair_id":"train-001","positive":"Create the requested spreadsheet, verify that it opens, and return the file path.","negative":"Continue considering alternative spreadsheet formats before creating anything."}
    {"pair_id":"train-002","positive":"The available evidence is sufficient. Produce the report and verify the citations.","negative":"Search for more sources even though the required evidence has already been collected."}

Evaluation prompts should remain neutral:

    {"prompt_id":"eval-001","prompt":"Prepare a short project status report from the supplied notes."}
    {"prompt_id":"eval-002","prompt":"Inspect the input files and create the requested deliverable."}

Do not reuse the same examples for both direction discovery and final evaluation. A reasonable split is:

    70% direction discovery
    15% validation and coefficient selection
    15% held-out evaluation

For the initial experiment, approximately 100–500 carefully controlled pairs are preferable to a much larger but noisy dataset. Use the public example at `data/contrastive/behavior_pairs.example.jsonl` as a schema reference.

### Step 22 — Discover a steering direction

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate routing-hf-py312

python scripts/steering/discover_direction.py \
  --model models/experiment_1_weighted_mass/keep128 \
  --dataset data/contrastive/behavior_pairs.jsonl \
  --layer 20 \
  --method paired_caa \
  --pooling last \
  --output results/phase2_steering/directions/keep128_layer20.pt \
  --trust-remote-code
```

Directions should be discovered separately on each checkpoint when comparing pruning levels. Reusing a full-model vector on a pruned checkpoint is a separate transfer experiment, not the default protocol. Both steering commands use the checkpoint tokenizer's chat template by default. Use `--no-use-chat-template` only for already-rendered text.

### Step 23 — Apply one steering coefficient

```bash
python scripts/steering/apply_steering.py \
  --model models/experiment_1_weighted_mass/keep128 \
  --artifact results/phase2_steering/directions/keep128_layer20.pt \
  --coefficient 2 \
  --position-mode last \
  --prompts-jsonl data/contrastive/evaluation_prompts.jsonl \
  --output results/phase2_steering/keep128_alpha_2.jsonl \
  --trust-remote-code
```

### Step 24 — Run a symmetric coefficient sweep

```bash
MODEL=models/experiment_1_weighted_mass/keep128 \
ARTIFACT=results/phase2_steering/directions/keep128_layer20.pt \
PROMPTS_JSONL=data/contrastive/evaluation_prompts.jsonl \
COEFFICIENTS="-4 -2 -1 0 1 2 4" \
bash scripts/steering/compare_coefficients.sh
```

Always include zero as the within-run baseline and use the same decoding parameters for every coefficient.

### Step 25 — Run repository validation

```bash
make check
```

The check compiles Python sources, validates shell syntax and checked-in config paths, checks local Markdown links, runs the publication audit and unit tests, and runs Ruff when it is installed.