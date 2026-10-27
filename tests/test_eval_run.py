"""End-to-end retrieval eval on a 4-document fixture corpus with a small real embedding model."""

from pathlib import Path

import mlflow
import pytest

from evaluation.run import MLFLOW_EXPERIMENT, Experiment, run_experiment
from obsidian_rag.indexer import IndexSpec

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.slow
def test_run_experiment_logs_to_mlflow(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    exp = Experiment(
        name="fixture",
        corpus_dir=FIXTURES / "corpus",
        golden_path=FIXTURES / "golden.jsonl",
        index=IndexSpec("sentence-transformers/all-MiniLM-L6-v2", chunk_size=128, chunk_overlap=16),
        chunk_depth=4,
        ks=(1, 3),
        chroma_path=tmp_path / "chroma",
    )

    summary = run_experiment(exp)

    # Each question is written to be answered by exactly one of 4 distinct docs.
    assert summary["recall_at_3"] == 1.0
    assert 0.75 <= summary["mrr"] <= 1.0
    assert all(0.0 <= summary[k] <= 1.0 for k in ("recall_at_1", "ndcg_at_1", "ndcg_at_3"))

    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["n_questions"] == "4"
    assert run.data.params["n_unanswerable_skipped"] == "1"
    assert run.data.tags["index_rebuilt"] == "True"
    assert run.data.metrics["recall_at_3"] == 1.0

    # Second run reuses the persisted collection.
    run_experiment(exp)
    latest = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT], output_format="list", order_by=["start_time DESC"]
    )[0]
    assert latest.data.tags["index_rebuilt"] == "False"


def test_unknown_golden_doc_is_rejected(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"schema_version": 1, "id": "x", "question": "?", "relevant_docs": ["nope.md"]}',
        encoding="utf-8",
    )
    exp = Experiment(
        name="bad",
        corpus_dir=FIXTURES / "corpus",
        golden_path=golden,
        index=IndexSpec("unused", 128, 16),
        chroma_path=tmp_path / "chroma",
    )
    with pytest.raises(ValueError, match="nope.md"):
        run_experiment(exp)
