from pathlib import Path

import pytest

from docqa_lab.corpus import KnowledgeBase
from docqa_lab.embeddings import TokenHashEmbeddings


@pytest.fixture
def kb(tmp_path: Path) -> KnowledgeBase:
    (tmp_path / "build.md").write_text(
        "# Build\nRun pytest to test the build.\nUse ruff for linting.", encoding="utf-8"
    )
    (tmp_path / "deploy.md").write_text(
        "# Deploy\nRollback the release using a reviewed revert commit.", encoding="utf-8"
    )
    return KnowledgeBase(tmp_path, TokenHashEmbeddings())
