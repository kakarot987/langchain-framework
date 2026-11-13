"""Offline lexical hash vectors; NOT a substitute for semantic embeddings."""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "our", "the", "to", "we",
    "what", "when", "where", "which", "who", "with", "you", "your",
}


class TokenHashEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing, for tiny offline demonstrations.

    Collisions and lack of semantics mean this must not be used as a quality benchmark
    for real semantic RAG. Use OpenAIEmbeddings to compare semantic retrieval.
    """

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be >= 32")
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9_-]{2,}", text.lower())
        for token in tokens:
            if token in STOP_WORDS:
                continue
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest, "big") % self.dimensions
            values[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]
