import json
from pathlib import Path

import pytest

from evaluation.golden import load_golden, missing_docs

FIXTURE = Path(__file__).parent / "fixtures" / "golden.jsonl"


def _write(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_load_fixture():
    items = load_golden(FIXTURE)
    assert len(items) == 5
    assert items[0].relevant_docs == {"quantization.md"}
    assert items[0].reference_answer == "Use 4-bit quantization."
    assert [i.answerable for i in items].count(False) == 1


def test_rejects_wrong_schema_version(tmp_path):
    path = _write(tmp_path, {"schema_version": 2, "id": "a", "question": "?", "relevant_docs": []})
    with pytest.raises(ValueError, match="schema_version"):
        load_golden(path)


def test_rejects_duplicate_ids(tmp_path):
    row = {"schema_version": 1, "id": "a", "question": "?", "relevant_docs": []}
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(_write(tmp_path, row, row))


def test_missing_docs():
    items = load_golden(FIXTURE)
    assert missing_docs(items, {"quantization.md", "guides/trainer.md"}) == {
        "guides/pipelines.md",
        "tokenizers.md",
    }
