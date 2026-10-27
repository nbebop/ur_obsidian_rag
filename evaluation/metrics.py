"""Retrieval metrics with binary, document-level relevance.

All functions take `ranked`, a list of document ids in retrieval order with
duplicates already removed (see `dedupe`), and `relevant`, the set of ids that
answer the question.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def dedupe(ids: Iterable[str]) -> list[str]:
    """Collapse chunk-level hits to documents, keeping each document's best rank."""
    return list(dict.fromkeys(ids))


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without relevant documents")
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("nDCG is undefined without relevant documents")
    dcg = sum(1.0 / math.log2(i + 2) for i, doc_id in enumerate(ranked[:k]) if doc_id in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal
