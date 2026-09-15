# Contributing

1. Create a local environment with Python 3.11 or newer.
2. Install `requirements-dev.txt`.
3. Copy `.env.example` to `.env` and use test credentials only.
4. Run the public-readiness check, formatter, linter, and unit tests before opening a pull request.

```powershell
python scripts/check_public_ready.py
ruff format --check .
ruff check .
python -m unittest discover -s tests -v
```

Never submit real patient data, generated runtime databases, model credentials, local filesystem paths, or copied proprietary datasets.
