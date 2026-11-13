# Local Development and Testing

Developers use Python 3.11 or later and create a virtual environment before installing dependencies.
Run `python -m pip install -e ".[dev]"` from the repository root to install the project and development tools.
Run unit tests locally with `python -m pytest -q`.
Run linting locally with `ruff check .`.
The CI workflow runs Ruff, pytest, the deterministic retrieval evaluation and the offline demo.

## Configuration

For offline retrieval, no API key is needed. To enable live chat answers, set `OPENAI_API_KEY` in your environment. The `.env.example` file lists optional configuration values. Never commit `.env` or service credentials.
