import math

import pytest

from evaluation.metrics import dedupe, ndcg_at_k, recall_at_k, reciprocal_rank


def test_dedupe_keeps_first_occurrence_order():
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall_at_k():
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(ranked, {"a", "b"}, k=1) == 0.0
    assert recall_at_k(ranked, {"a", "b"}, k=2) == 0.5
    assert recall_at_k(ranked, {"a", "b"}, k=4) == 1.0
    assert recall_at_k(ranked, {"a", "b"}, k=10) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y", "a"], {"a", "y"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_and_missing():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_ndcg_hand_computed():
    # One relevant doc at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1 / math.log2(3))
    # Relevant at ranks 1 and 3 of 2 relevant: DCG = 1 + 1/log2(4) = 1.5,
    # IDCG = 1 + 1/log2(3).
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(1.5 / ideal)


def test_ndcg_ideal_is_capped_at_k():
    # 3 relevant docs but k=1: one hit at rank 1 is already ideal.
    assert ndcg_at_k(["a"], {"a", "b", "c"}, k=1) == pytest.approx(1.0)


@pytest.mark.parametrize("fn", [recall_at_k, ndcg_at_k])
def test_empty_relevant_set_is_an_error(fn):
    with pytest.raises(ValueError):
        fn(["a"], set(), 1)
