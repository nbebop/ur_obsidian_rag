"""Golden set: versioned JSONL of questions with their expected source documents.

One JSON object per line:

    {"schema_version": 1, "id": "q001", "question": "...",
     "relevant_docs": ["tasks/asr.md"], "reference_answer": "..."}

`relevant_docs` are doc ids (paths relative to the corpus root). An empty list
marks a question the corpus cannot answer; those are skipped by retrieval
metrics and later used to test "no answer" behaviour. `reference_answer` is
optional.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    relevant_docs: frozenset[str]
    reference_answer: str | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.relevant_docs)


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            row = json.loads(line)
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{where}: expected schema_version {SCHEMA_VERSION}")
            for key in ("id", "question", "relevant_docs"):
                if key not in row:
                    raise ValueError(f"{where}: missing '{key}'")
            if not isinstance(row["relevant_docs"], list):
                raise ValueError(f"{where}: 'relevant_docs' must be a list")
            if row["id"] in seen:
                raise ValueError(f"{where}: duplicate id {row['id']!r}")
            seen.add(row["id"])
            items.append(
                GoldenItem(
                    id=row["id"],
                    question=row["question"],
                    relevant_docs=frozenset(row["relevant_docs"]),
                    reference_answer=row.get("reference_answer"),
                )
            )
    return items


def missing_docs(items: Iterable[GoldenItem], corpus_doc_ids: set[str]) -> set[str]:
    """Labelled documents that don't exist in the corpus (stale or mistyped labels)."""
    return {d for item in items for d in item.relevant_docs} - corpus_doc_ids
