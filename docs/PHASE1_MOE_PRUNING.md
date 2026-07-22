# Phase I: MoE pruning details

## Routing quantities

For selected experts, `weighted_count` is the sum of top-k-renormalized router weights. `selected_count` is the number of top-k appearances. `response_tokens` is the number of generated tokens for the task.

The ordinary compact accounting table is sufficient for all four main experiments. The full replay outputs are retained only when call-level analysis or re-accounting is required.

## Experiment comparison

| Experiment | Score | Main question |
|---|---|---|
| 1 | Global weighted mass | Which experts receive the most total router mass? |
| 2 | Task-normalized weighted mass | Which experts are important when each task contributes equally? |
| 3 | Unweighted count | Which experts are selected most frequently regardless of gate magnitude? |
| 4 | Sector-conditioned weighted mass | Which experts are most important for one task sector? |

## Coverage interpretation

Coverage is computed using the full-model accounting and the retained original expert IDs. It answers how much of the original routing statistic falls inside the selected set. It does not measure how the pruned model reroutes tokens after checkpoint surgery.

## Why uniform bank sizes

The current exporter and vLLM serving path assume a common number of routed experts across layers. Each layer may retain a different set of original experts, but every layer keeps the same bank size: 192, 128, or 64. Heterogeneous per-layer bank sizes require model and serving-kernel changes and are outside the current validated path.
