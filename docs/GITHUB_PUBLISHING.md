# Publishing to GitHub

Review `SECURITY.md`, `THIRD_PARTY.md`, and `docs/DATA_AND_PRIVACY.md` before publishing.

From a clean extracted source directory:

```bash
make check
git init -b main
git add .
git commit -m "Initial RouteCat MoE pruning and steering project"
```

Then create an empty GitHub repository and add it as the remote:

```bash
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Keep these outside ordinary Git history:

- `GDPval_data/`;
- `artifacts/tasks/` and `artifacts/runs/`;
- accounting CSV/CSV.GZ files;
- model weights and pruned checkpoints;
- evaluation outputs;
- credentials, cloud metadata, and absolute compute paths.

Use an approved artifact store or Git LFS only after confirming ownership and distribution terms. The default `.gitignore` blocks the known large and sensitive paths.
