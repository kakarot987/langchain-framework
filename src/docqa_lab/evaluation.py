"""Deterministic retrieval evaluation; not a model accuracy benchmark."""

import json
from pathlib import Path

from .corpus import KnowledgeBase


def evaluate_retrieval(kb: KnowledgeBase, dataset: Path, k: int = 3) -> dict:
    examples = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError("Evaluation dataset must be a nonempty JSON list")
    results = []
    for example in examples:
        sources = [hit.source for hit in kb.search(example["question"], k=k)]
        expected = example["expected_source"]
        results.append({
            "question": example["question"],
            "expected_source": expected,
            "retrieved": sources,
            "hit": expected in sources,
        })
    return {
        "hit_at_k": sum(item["hit"] for item in results) / len(results),
        "k": k,
        "count": len(results),
        "examples": results,
    }
