# Data directory

Only `eval_dataset.json` is intended for version control. It contains synthetic evaluation cases.

Runtime files such as checkpoints, encrypted session logs, encryption keys, and generated reports are local-only and ignored by Git. Do not commit real patient records or exported conversations.

The checked-in evaluation cases are synthetic development fixtures. Their scores are regression signals only; they are not clinical accuracy, real-user outcomes, or production benchmarks. A sanitized public summary is maintained in `docs/evaluation-baseline.md`.
