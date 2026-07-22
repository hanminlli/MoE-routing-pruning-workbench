# Phase II: Activation steering details

## Why steering follows pruning

Pruning modifies the parameterized expert bank. Steering modifies the residual stream during inference. Running pruning first makes the causal question clean: after capacity reduction, does the remaining network retain controllable behavioral directions?

## Discovery protocol

For each checkpoint under comparison:

1. use the same contrastive examples;
2. render examples with each checkpoint tokenizer's chat template unless the dataset is already serialized;
3. capture activations at the same nominal layer index;
4. use the same pooling rule;
5. discover and L2-normalize a checkpoint-specific direction;
6. save the artifact before running any coefficient sweep.

Checkpoint-specific discovery is the default because pruning may rotate or distort the hidden representation. Applying a direction learned on another checkpoint is treated as a transfer experiment.

## Intervention protocol

Always include an unsteered coefficient of zero. Use symmetric positive and negative coefficients. Hold generation parameters fixed. Record failures, repetition, truncation, and refusal behavior alongside the target behavior metric.

## Interpretation

A change in output at large coefficients is not enough. A useful steering direction should show a monotonic or at least structured response over a coefficient range while preserving fluency and task competence.

## Prompt rendering

Discovery and application default to the checkpoint tokenizer's `apply_chat_template` path. Pass `--no-use-chat-template` only when JSONL text is already a complete serialized conversation. Use the same system prompt and rendering policy for every checkpoint and coefficient.
