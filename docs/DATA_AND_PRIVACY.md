# Data and privacy

## Commit to Git

- source code;
- public documentation;
- synthetic tests;
- example schemas and tiny synthetic examples;
- configuration templates without private paths.

## Do not commit

- benchmark prompts or reference files unless redistribution is explicitly permitted;
- model calls, reasoning traces, tool outputs, or generated deliverables;
- ordinary or advanced accounting tables;
- sector/occupation annotations that are internal;
- model checkpoints, tokenizer snapshots, or pruned weights;
- cloud job IDs, subscription/workspace names, hostnames, usernames, absolute scratch paths, GPU UUIDs, or environment-variable dumps;
- credentials, tokens, cookies, or private package indexes.

Use approved artifact storage for large or restricted files and document their expected local paths in private deployment notes rather than in the public repository.
