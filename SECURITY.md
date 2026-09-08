# Security and privacy

This project handles potentially sensitive health information. It is an educational demo, not a medical device.

## Reporting a vulnerability

Open a private security advisory in the GitHub repository. Do not include real patient information, API keys, logs, or exploit data in a public issue.

## Data handling

- Session logging is disabled by default.
- Uploaded text is stored under a thread-scoped directory with a generic filename and common direct identifiers redacted.
- Runtime uploads, indexes, checkpoints, encryption keys, and encrypted logs are excluded from version control.
- Redaction is a safety layer, not a guarantee of full anonymization. Review every file before publication.

Rotate any credential immediately if it has ever been committed or shared.
