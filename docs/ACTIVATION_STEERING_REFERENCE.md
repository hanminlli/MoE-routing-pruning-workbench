# Activation-steering reference boundary

The Phase-II rationale was informed by the uploaded `steer-MOE_extraction` project and standard contrastive activation addition methodology. The uploaded repository was not used as the Phase-I codebase and was not copied wholesale into this project.

The implementation here is deliberately narrower:

- one explicit model-structure resolver;
- residual activation capture;
- paired CAA and difference-in-means directions;
- immutable steering artifacts;
- explicit residual hooks and position policies;
- coefficient sweeps on a selected pruned checkpoint.

More elaborate methods such as probes, CAVs, principal components, feature dictionaries, modular composition, and vLLM-native steering remain future extensions. This keeps Phase II auditable and prevents activation-steering infrastructure from obscuring the primary MoE-pruning experiment.
