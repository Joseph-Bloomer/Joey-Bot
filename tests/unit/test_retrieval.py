"""Unit tests for HybridRetriever._reciprocal_rank_fusion().

_reciprocal_rank_fusion() merges a dense-vector result list and a BM25 sparse
result list into a single dict using Reciprocal Rank Fusion (RRF). The formula
for each entry is:

    fused_score = sum(1 / (k + rank))   for each list the candidate appears in

where k=60 (the standard RRF constant) and rank starts at 1.

This method is pure computation — it only reads the input lists and
self.RRF_K. No vector store or BM25 index is touched.

Run from the project root:
    python -m pytest tests/unit/test_retrieval.py -v
"""

import pytest

from app.services.retrieval import HybridRetriever

# ---------------------------------------------------------------------------
# RRF constant — matches HybridRetriever.RRF_K
# ---------------------------------------------------------------------------
_K = HybridRetriever.RRF_K  # 60


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------

class _StubStore:
    """Minimal stub that satisfies HybridRetriever's constructor.

    _reciprocal_rank_fusion() does not call self.store at all, so the
    store just needs to exist as an attribute.
    """


def make_dense(memory_id: str, text: str = "Dense memory.", dense_score: float = 0.9) -> dict:
    """Build a result dict in the format _dense_search() returns."""
    return {
        "memory_id": memory_id,
        "text": text,
        "metadata": {},
        "dense_score": dense_score,
    }


def make_sparse(memory_id: str, text: str = "Sparse memory.", sparse_score: float = 0.7) -> dict:
    """Build a result dict in the format _sparse_search() returns."""
    return {
        "memory_id": memory_id,
        "text": text,
        "metadata": {},
        "sparse_score": sparse_score,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever() -> HybridRetriever:
    """HybridRetriever with a stub store — sufficient for testing RRF fusion."""
    return HybridRetriever(_StubStore())


# ---------------------------------------------------------------------------
# Empty-list edge cases
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    """Degenerate cases where one or both input lists are empty."""

    def test_both_empty_returns_empty_dict(self, retriever):
        """Fusing two empty lists should produce an empty result dict."""
        result = retriever._reciprocal_rank_fusion([], [])
        assert result == {}

    def test_empty_sparse_returns_dense_results(self, retriever):
        """When sparse is empty, every dense candidate should appear in output."""
        dense = [make_dense("id-1"), make_dense("id-2")]
        result = retriever._reciprocal_rank_fusion(dense, [])
        assert set(result.keys()) == {"id-1", "id-2"}

    def test_empty_dense_returns_sparse_results(self, retriever):
        """When dense is empty, every sparse candidate should appear in output."""
        sparse = [make_sparse("id-A"), make_sparse("id-B")]
        result = retriever._reciprocal_rank_fusion([], sparse)
        assert set(result.keys()) == {"id-A", "id-B"}

    def test_single_dense_result(self, retriever):
        """A single dense result with no sparse should yield exactly one entry."""
        result = retriever._reciprocal_rank_fusion([make_dense("solo")], [])
        assert len(result) == 1
        assert "solo" in result

    def test_single_sparse_result(self, retriever):
        """A single sparse result with no dense should yield exactly one entry."""
        result = retriever._reciprocal_rank_fusion([], [make_sparse("solo")])
        assert len(result) == 1
        assert "solo" in result


# ---------------------------------------------------------------------------
# Score formula: 1 / (k + rank)
# ---------------------------------------------------------------------------

class TestScoreFormula:
    """Verify that fused_score values match the RRF formula exactly."""

    def test_rank_1_dense_score(self, retriever):
        """First dense result (rank 1) should receive a fused_score of 1/(k+1)."""
        result = retriever._reciprocal_rank_fusion([make_dense("id-1")], [])
        expected = 1.0 / (_K + 1)
        assert result["id-1"]["fused_score"] == pytest.approx(expected)

    def test_rank_2_dense_score(self, retriever):
        """Second dense result (rank 2) should receive a fused_score of 1/(k+2)."""
        dense = [make_dense("id-1"), make_dense("id-2")]
        result = retriever._reciprocal_rank_fusion(dense, [])
        expected_rank2 = 1.0 / (_K + 2)
        assert result["id-2"]["fused_score"] == pytest.approx(expected_rank2)

    def test_rank_1_sparse_score(self, retriever):
        """First sparse result (rank 1) should receive a fused_score of 1/(k+1)."""
        result = retriever._reciprocal_rank_fusion([], [make_sparse("id-A")])
        expected = 1.0 / (_K + 1)
        assert result["id-A"]["fused_score"] == pytest.approx(expected)

    def test_rank_1_higher_than_rank_2(self, retriever):
        """Within the same list, rank 1 must yield a higher fused_score than rank 2
        because 1/(k+1) > 1/(k+2) for any positive k.
        """
        dense = [make_dense("first"), make_dense("second")]
        result = retriever._reciprocal_rank_fusion(dense, [])
        assert result["first"]["fused_score"] > result["second"]["fused_score"]


# ---------------------------------------------------------------------------
# Merging behaviour
# ---------------------------------------------------------------------------

class TestMerging:
    """Tests for how separate dense and sparse result lists are combined."""

    def test_non_overlapping_lists_are_merged(self, retriever):
        """Dense and sparse results with different IDs should both appear in output."""
        dense  = [make_dense("d-1"), make_dense("d-2")]
        sparse = [make_sparse("s-1"), make_sparse("s-2")]
        result = retriever._reciprocal_rank_fusion(dense, sparse)
        assert set(result.keys()) == {"d-1", "d-2", "s-1", "s-2"}

    def test_overlapping_id_appears_once_with_combined_score(self, retriever):
        """A memory present in both lists should appear exactly once with its
        scores summed, not duplicated.
        """
        shared_id = "shared"
        dense  = [make_dense(shared_id)]
        sparse = [make_sparse(shared_id)]
        result = retriever._reciprocal_rank_fusion(dense, sparse)
        assert len(result) == 1
        assert shared_id in result

    def test_overlap_score_equals_sum_of_both_contributions(self, retriever):
        """A shared candidate at rank 1 in both lists should have
        fused_score = 1/(k+1) + 1/(k+1) = 2/(k+1).
        """
        shared = "shared"
        dense  = [make_dense(shared)]
        sparse = [make_sparse(shared)]
        result = retriever._reciprocal_rank_fusion(dense, sparse)

        expected = (1.0 / (_K + 1)) + (1.0 / (_K + 1))
        assert result[shared]["fused_score"] == pytest.approx(expected)

    def test_overlap_score_exceeds_single_source_score(self, retriever):
        """A candidate appearing in both dense and sparse should score higher than
        one appearing in only one list, when both are at rank 1 in their lists.
        This is the fundamental property of RRF that rewards multi-source agreement.
        """
        dense  = [make_dense("dense-only"), make_dense("both")]
        sparse = [make_sparse("both")]
        result = retriever._reciprocal_rank_fusion(dense, sparse)

        # "both" benefits from two contributions; "dense-only" has just one
        assert result["both"]["fused_score"] > result["dense-only"]["fused_score"]

    def test_dense_score_preserved_in_output(self, retriever):
        """The dense_score from the original search result should be stored
        in the fused entry for diagnostic purposes.
        """
        result = retriever._reciprocal_rank_fusion([make_dense("id-1", dense_score=0.88)], [])
        assert result["id-1"]["dense_score"] == pytest.approx(0.88)

    def test_sparse_score_preserved_in_output(self, retriever):
        """The sparse_score from the BM25 result should be stored in the fused entry."""
        result = retriever._reciprocal_rank_fusion([], [make_sparse("id-A", sparse_score=4.2)])
        assert result["id-A"]["sparse_score"] == pytest.approx(4.2)

    def test_dense_only_result_has_zero_sparse_score(self, retriever):
        """A result that came only from dense search should have sparse_score=0.0."""
        result = retriever._reciprocal_rank_fusion([make_dense("id-1")], [])
        assert result["id-1"]["sparse_score"] == 0.0

    def test_sparse_only_result_has_zero_dense_score(self, retriever):
        """A result that came only from sparse search should have dense_score=0.0."""
        result = retriever._reciprocal_rank_fusion([], [make_sparse("id-A")])
        assert result["id-A"]["dense_score"] == 0.0

    def test_memory_id_preserved_in_output_entry(self, retriever):
        """Each fused entry's memory_id field should match its dict key."""
        result = retriever._reciprocal_rank_fusion([make_dense("my-id")], [])
        assert result["my-id"]["memory_id"] == "my-id"

    def test_text_preserved_in_output_entry(self, retriever):
        """The text from the original result should be stored in the fused entry."""
        result = retriever._reciprocal_rank_fusion(
            [make_dense("id-1", text="Important fact about Python.")], []
        )
        assert result["id-1"]["text"] == "Important fact about Python."
