from pathlib import Path

import pytest

from docqa_lab.corpus import KnowledgeBase
from docqa_lab.embeddings import TokenHashEmbeddings


def test_retrieves_appropriate_document(kb):
    hits = kb.search("pytest test build")
    assert hits
    assert hits[0].source == "build.md"
    assert hits[0].score > 0


def test_lexical_embedding_stable():
    model = TokenHashEmbeddings()
    assert model.embed_query("hello world") == model.embed_query("hello world")
    assert abs(sum(x * x for x in model.embed_query("hello world")) - 1) < 1e-8


def test_exact_filename_only(kb):
    with pytest.raises(ValueError, match="Unknown document"):
        kb.read_excerpt("../secrets.env")


def test_symlink_is_not_loaded(tmp_path: Path):
    outside = tmp_path.parent / "outside.md"
    outside.write_text("Sensitive", encoding="utf-8")
    (tmp_path / "real.md").write_text("Allowed", encoding="utf-8")
    (tmp_path / "linked.md").symlink_to(outside)
    index = KnowledgeBase(tmp_path, TokenHashEmbeddings())
    assert list(index.files) == ["real.md"]


def test_empty_corpus_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="No nonempty"):
        KnowledgeBase(tmp_path, TokenHashEmbeddings())
