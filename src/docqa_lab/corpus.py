"""Bounded Markdown ingestion, chunking, and LangChain vector retrieval."""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_FILES = 100
MAX_FILE_BYTES = 200_000


@dataclass(frozen=True)
class Hit:
    source: str
    chunk: int
    text: str
    score: float


class KnowledgeBase:
    def __init__(self, root: Path, embedding: Embeddings):
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Document path must be a directory")
        self.files: dict[str, Path] = {}
        documents = []
        for path in sorted(self.root.rglob("*.md")):
            if path.is_symlink() or any(
                parent.is_symlink() for parent in path.parents if parent != self.root
            ):
                continue
            if not path.is_file():
                continue
            if len(self.files) >= MAX_FILES:
                raise ValueError(f"At most {MAX_FILES} Markdown files are supported")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"Document too large: {path.name}")
            source = path.relative_to(self.root).as_posix()
            self.files[source] = path
            text = path.read_text(encoding="utf-8")
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": source}))
        if not documents:
            raise ValueError("No nonempty Markdown documents found")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_documents(documents)
        counter: dict[str, int] = {}
        for chunk in chunks:
            source = chunk.metadata["source"]
            counter[source] = counter.get(source, 0) + 1
            chunk.metadata["chunk"] = counter[source]
        self.chunk_count = len(chunks)
        self.store = InMemoryVectorStore(embedding)
        self.store.add_documents(chunks)

    def search(self, query: str, k: int = 3, minimum_score: float = 0.12) -> list[Hit]:
        if not query.strip() or not 1 <= k <= 10:
            raise ValueError("Query must be nonempty and k between 1 and 10")
        hits = self.store.similarity_search_with_score(query, k=k)
        return [
            Hit(doc.metadata["source"], doc.metadata["chunk"], doc.page_content, float(score))
            for doc, score in hits if float(score) >= minimum_score
        ]

    def read_excerpt(self, source: str, max_chars: int = 1600) -> str:
        # Exact key lookup prevents traversal and arbitrary reads.
        path = self.files.get(source)
        if path is None:
            raise ValueError("Unknown document name")
        text = path.read_text(encoding="utf-8")
        return text[:max_chars] + ("\n[truncated]" if len(text) > max_chars else "")
