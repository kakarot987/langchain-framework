"""Small, explicit environment-based configuration (no secrets committed)."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    model: str = "gpt-4.1-mini"
    embedding: str = "hash"
    timeout_seconds: float = 20.0
    prices_per_million: tuple[float, float] | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        timeout = float(os.getenv("DOCQA_TIMEOUT_SECONDS", "20"))
        if not 1 <= timeout <= 120:
            raise ValueError("DOCQA_TIMEOUT_SECONDS must be between 1 and 120")
        embedding = os.getenv("DOCQA_EMBEDDING", "hash").lower()
        if embedding not in {"hash", "openai"}:
            raise ValueError("DOCQA_EMBEDDING must be 'hash' or 'openai'")
        raw_input = os.getenv("DOCQA_INPUT_USD_PER_MILLION", "").strip()
        raw_output = os.getenv("DOCQA_OUTPUT_USD_PER_MILLION", "").strip()
        if bool(raw_input) != bool(raw_output):
            raise ValueError("Both pricing environment variables must be set together")
        prices = (float(raw_input), float(raw_output)) if raw_input else None
        if prices and any(value < 0 for value in prices):
            raise ValueError("Token prices cannot be negative")
        return cls(
            docs_dir=Path(os.getenv("DOCQA_DOCS", "docs/sample")),
            model=os.getenv("DOCQA_MODEL", "gpt-4.1-mini"),
            embedding=embedding,
            timeout_seconds=timeout,
            prices_per_million=prices,
        )
