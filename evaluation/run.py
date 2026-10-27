"""Run the retrieval evaluation for one experiment config and log it to MLflow.

Usage:  python -m evaluation.run --config evaluation/experiments/baseline.yaml [--rebuild]

Tracking goes to MLflow's default store unless MLFLOW_TRACKING_URI is set;
browse results with `mlflow ui`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import yaml
from llama_index.core import VectorStoreIndex

from obsidian_rag.config import PROJECT_ROOT
from obsidian_rag.indexer import (
    IndexSpec,
    get_or_build_index,
    load_documents,
    source_fingerprint,
)

from .golden import SCHEMA_VERSION, GoldenItem, load_golden, missing_docs
from .metrics import dedupe, ndcg_at_k, recall_at_k, reciprocal_rank

log = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "rag-retrieval"


@dataclass(frozen=True)
class Experiment:
    name: str
    corpus_dir: Path
    golden_path: Path
    index: IndexSpec
    # Chunks fetched per query; metrics@k count distinct documents, so fetch
    # more chunks than the largest k to fill the document ranking.
    chunk_depth: int = 20
    ks: tuple[int, ...] = (1, 3, 5, 10)
    chroma_path: Path = PROJECT_ROOT / "storage" / "eval"

    @classmethod
    def from_yaml(cls, path: Path) -> Experiment:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        idx = raw["index"]
        return cls(
            name=raw["name"],
            corpus_dir=PROJECT_ROOT / raw["corpus_dir"],
            golden_path=PROJECT_ROOT / raw["golden_path"],
            index=IndexSpec(idx["embedding_model"], idx["chunk_size"], idx["chunk_overlap"]),
            chunk_depth=raw.get("chunk_depth", cls.chunk_depth),
            ks=tuple(raw.get("ks", cls.ks)),
            chroma_path=PROJECT_ROOT / raw.get("chroma_path", "storage/eval"),
        )


def evaluate_retrieval(
    index: VectorStoreIndex, items: list[GoldenItem], chunk_depth: int, ks: tuple[int, ...]
) -> tuple[dict[str, float], list[dict]]:
    """Score answerable golden items. Returns (mean metrics, per-question rows)."""
    retriever = index.as_retriever(similarity_top_k=chunk_depth)
    rows: list[dict] = []
    for item in (i for i in items if i.answerable):
        start = time.perf_counter()
        hits = retriever.retrieve(item.question)
        latency_ms = (time.perf_counter() - start) * 1000
        ranked = dedupe(h.node.metadata["doc_id"] for h in hits)
        row = {
            "id": item.id,
            "question": item.question,
            "relevant_docs": sorted(item.relevant_docs),
            "ranked_docs": ranked[: max(ks)],
            "mrr": reciprocal_rank(ranked, item.relevant_docs),
            "latency_ms": latency_ms,
        }
        for k in ks:
            row[f"recall_at_{k}"] = recall_at_k(ranked, item.relevant_docs, k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(ranked, item.relevant_docs, k)
        rows.append(row)

    if not rows:
        raise ValueError("golden set has no answerable questions")
    metric_keys = ["mrr"] + [f"{m}_at_{k}" for k in ks for m in ("recall", "ndcg")]
    summary = {key: statistics.fmean(r[key] for r in rows) for key in metric_keys}
    latencies = sorted(r["latency_ms"] for r in rows)
    summary["latency_p50_ms"] = statistics.median(latencies)
    summary["latency_p95_ms"] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    return summary, rows


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" or "unknown"
    except OSError:
        return "unknown"


def run_experiment(exp: Experiment, *, rebuild: bool = False) -> dict[str, float]:
    items = load_golden(exp.golden_path)
    corpus_ids = {d.metadata["doc_id"] for d in load_documents(exp.corpus_dir)}
    if missing := missing_docs(items, corpus_ids):
        raise ValueError(f"golden set references docs not in the corpus: {sorted(missing)}")

    start = time.perf_counter()
    index, built = get_or_build_index(
        exp.corpus_dir, exp.index, exp.chroma_path, prefix="eval", rebuild=rebuild
    )
    index_seconds = time.perf_counter() - start

    summary, rows = evaluate_retrieval(index, items, exp.chunk_depth, exp.ks)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=exp.name):
        mlflow.log_params(
            {
                **asdict(exp.index),
                "chunk_depth": exp.chunk_depth,
                "golden_set": exp.golden_path.name,
                "n_questions": len(rows),
                "n_unanswerable_skipped": sum(not i.answerable for i in items),
            }
        )
        mlflow.set_tags(
            {
                "git_sha": _git_sha(),
                "golden_schema_version": SCHEMA_VERSION,
                "golden_sha256": hashlib.sha256(exp.golden_path.read_bytes()).hexdigest()[:16],
                "corpus_fingerprint": source_fingerprint(exp.corpus_dir),
                "index_rebuilt": built,
            }
        )
        mlflow.log_metrics({**summary, "index_load_or_build_s": index_seconds})
        mlflow.log_text("\n".join(json.dumps(r) for r in rows), "per_question.jsonl")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true", help="force re-embedding the corpus")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    summary = run_experiment(Experiment.from_yaml(args.config), rebuild=args.rebuild)
    for key, value in summary.items():
        print(f"{key:>20}  {value:.3f}")


if __name__ == "__main__":
    main()
