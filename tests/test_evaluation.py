import json

from docqa_lab.evaluation import evaluate_retrieval


def test_retrieval_hit_at_k(kb, tmp_path):
    dataset = tmp_path / "examples.json"
    dataset.write_text(json.dumps([
        {"question": "Run pytest for build tests", "expected_source": "build.md"},
        {"question": "Rollback deploy release", "expected_source": "deploy.md"},
    ]), encoding="utf-8")
    report = evaluate_retrieval(kb, dataset, k=1)
    assert report["hit_at_k"] == 1.0
    assert report["count"] == 2
