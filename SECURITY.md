# Security

This package is intended for local model training and inference workflows.

## Reporting Issues

If you find a security issue, report it privately to the project maintainer
instead of opening a public issue.

## Data Handling

- Do not commit private match footage, annotations, checkpoints, or credentials.
- Keep `.env` files and local config overrides out of version control.
- Review generated prediction files before sharing them if they may contain
  private video paths or customer data.

## Model Artifacts

Checkpoints can contain learned information from private datasets. Treat them
as sensitive artifacts unless they were trained only on public data.
