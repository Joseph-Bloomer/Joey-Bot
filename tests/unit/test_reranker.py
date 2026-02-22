"""Unit tests for HeuristicReranker.

HeuristicReranker scores retrieval candidates using a weighted combination of
five signals already present in the candidate metadata. It makes no LLM calls,
no database queries, and has no I/O — so every test here runs in milliseconds
with no external dependencies.

Run from the project root:
    python -m pytest tests/unit/test_reranker.py -v
"""

import math
from datetime import datetime, timedelta

import pytest

from app.services.reranker import HeuristicReranker


# ---------------------------------------------------------------------------
# Helper: build a candidate dict
# ---------------------------------------------------------------------------

def make_candidate(
    fused_score: float = 0.5,
    importance: float = 0.5,
    access_count: int = 0,
    days_old: int = 0,
    memory_type: str = "semantic",
    text: str = "A test memory.",
    memory_id: str = "test-id-001",
) -> dict:
    """Build a candidate dict in the format HybridRetriever returns.

    HeuristicReranker reads these specific fields:
      - fused_score              (from RRF fusion)
      - metadata.importance      (float 0-1)
      - metadata.access_count    (int)
      - metadata.created_at      (ISO datetime string)
      - metadata.memory_type     (string: "semantic", "procedural", etc.)

    All other fields (dense_score, sparse_score) are present for realism
    but are ignored by the reranker.

    Args:
        fused_score:  The RRF-fused retrieval score. Higher = better match.
        importance:   Memory importance, 0-1.
        access_count: How many times this memory has been retrieved before.
        days_old:     How many days ago the memory was created.
        memory_type:  "semantic", "procedural", "preference", etc.
        text:         The memory text (not used in scoring, but required).
        memory_id:    Unique identifier.
    """
    created_at = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
    return {
        "memory_id": memory_id,
        "text": text,
        "fused_score": fused_score,
        "dense_score": 0.5,
        "sparse_score": 0.5,
        "metadata": {
            "importance": importance,
            "access_count": access_count,
            "created_at": created_at,
            "memory_type": memory_type,
            "strength": 0.5,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reranker() -> HeuristicReranker:
    """Return a fresh HeuristicReranker for each test.

    HeuristicReranker holds no mutable state between calls, so this is
    equivalent to a module-scoped fixture. We use function scope (the
    default) anyway to keep tests fully independent.
    """
    return HeuristicReranker()


# A query_context that forces retrieval_relevance = 1.0 for any candidate.
# When fused_min == fused_max the normalisation formula short-circuits to 1.0,
# which lets us isolate other scoring signals without list-level normalisation.
_FLAT_CTX = {"fused_min": 0.5, "fused_max": 0.5}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions that must not crash or return unexpected lengths."""

    def test_empty_candidates_returns_empty_list(self, reranker):
        """rerank() with no input should return an empty list, not raise."""
        result = reranker.rerank([], query_context={})
        assert result == []

    def test_single_candidate_is_returned(self, reranker):
        """A single candidate should pass through rerank() and be returned."""
        result = reranker.rerank([make_candidate()], query_context={})
        assert len(result) == 1

    def test_missing_metadata_does_not_crash(self, reranker):
        """Candidates with an empty metadata dict must not raise.

        Missing fields should fall back to the defaults baked into score():
          importance → 0.5, access_count → 0, created_at → neutral 0.5,
          memory_type → no boost.
        """
        bare = {
            "memory_id": "bare-id",
            "text": "Bare candidate with no metadata.",
            "fused_score": 0.5,
            "metadata": {},  # every field missing
        }
        result = reranker.rerank([bare], query_context={})
        assert len(result) == 1
        # Score must still be present and in range
        assert "rerank_score" in result[0]
        assert 0.0 <= result[0]["rerank_score"] <= 1.0

    def test_top_k_limits_result_count(self, reranker):
        """rerank(top_k=2) should return exactly 2 results from a larger list."""
        candidates = [
            make_candidate(fused_score=float(i), memory_id=f"id-{i}")
            for i in range(5)
        ]
        result = reranker.rerank(candidates, query_context={}, top_k=2)
        assert len(result) == 2

    def test_top_k_larger_than_input_returns_all(self, reranker):
        """Asking for more results than candidates should return what's available."""
        candidates = [make_candidate(memory_id="only-one")]
        result = reranker.rerank(candidates, query_context={}, top_k=10)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Score formula correctness
# ---------------------------------------------------------------------------

class TestScoreFormula:
    """Verify the weighted composite formula using score() directly.

    We call reranker.score() with _FLAT_CTX so that retrieval_relevance is
    always 1.0 and we can vary only the signal we're testing.

    Composite = retrieval_relevance(0.45) + recency(0.20) + importance(0.20)
                + usage(0.10) + type_boost(0.05)
    """

    def test_high_relevance_beats_low_relevance(self, reranker):
        """The candidate with the higher fused_score should rank first.

        rerank() min-max normalises fused_scores across the candidate list:
          highest  → retrieval_relevance = 1.0 (contributes 0.45 to composite)
          lowest   → retrieval_relevance = 0.0 (contributes 0.00 to composite)
        All other metadata is identical, so the high-relevance candidate wins.
        """
        high = make_candidate(fused_score=1.0, memory_id="high")
        low  = make_candidate(fused_score=0.0, memory_id="low")

        results = reranker.rerank([high, low], query_context={})

        assert results[0]["memory_id"] == "high"
        assert results[0]["rerank_score"] > results[1]["rerank_score"]

    def test_retrieval_relevance_dominates_all_other_signals(self, reranker):
        """A candidate with perfect retrieval relevance beats one with zero
        retrieval relevance even when the latter has maxed-out importance,
        usage, recency, and type boost.

        This confirms the 0.45 weight makes retrieval_relevance the single
        most important scoring signal.

        Expected scores (approximate):
          high_retrieval: 0.45*1.0 + 0.20*1.0 + 0.20*0.5 + 0.10*0.0 + 0.05*0.0 = 0.75
          low_retrieval:  0.45*0.0 + 0.20*1.0 + 0.20*1.0 + 0.10*1.0 + 0.05*0.10 = 0.505
        """
        # fused_score=1.0 → retrieval_relevance=1.0 after normalisation
        high_retrieval = make_candidate(
            fused_score=1.0,
            importance=0.5,
            access_count=0,
            days_old=0,
            memory_type="semantic",   # no type boost
            memory_id="high-retrieval",
        )
        # fused_score=0.0 → retrieval_relevance=0.0, but everything else maxed
        low_retrieval = make_candidate(
            fused_score=0.0,
            importance=1.0,
            access_count=5,           # hits USAGE_CAP=5.0 → usage_signal=1.0
            days_old=0,               # exp(-0.03*0) = 1.0 → perfect recency
            memory_type="procedural", # highest type boost (+0.10)
            memory_id="low-retrieval",
        )

        results = reranker.rerank([high_retrieval, low_retrieval], query_context={})

        assert results[0]["memory_id"] == "high-retrieval"

    def test_importance_raises_score(self, reranker):
        """A candidate with higher importance should score higher when all
        other signals are equal.

        Both get retrieval_relevance=1.0 via _FLAT_CTX.
        Difference = 0.20 * (1.0 - 0.0) = 0.20.
        """
        high_imp = make_candidate(importance=1.0)
        low_imp  = make_candidate(importance=0.0)

        score_high = reranker.score(high_imp, _FLAT_CTX)
        score_low  = reranker.score(low_imp,  _FLAT_CTX)

        assert score_high > score_low
        # The difference must equal exactly the importance weight contribution
        assert score_high - score_low == pytest.approx(
            HeuristicReranker.WEIGHTS["importance"] * (1.0 - 0.0)
        )

    def test_usage_cap_prevents_score_above_max_usage_signal(self, reranker):
        """Access counts above USAGE_CAP should be treated the same as USAGE_CAP.

        usage_signal = min(access_count / USAGE_CAP, 1.0), so 100 accesses
        should give the same score as 5 (the cap).
        """
        at_cap  = make_candidate(access_count=int(HeuristicReranker.USAGE_CAP))
        over_cap = make_candidate(access_count=9999)

        score_at   = reranker.score(at_cap,   _FLAT_CTX)
        score_over = reranker.score(over_cap, _FLAT_CTX)

        assert score_at == pytest.approx(score_over)

    def test_score_breakdown_contains_all_expected_keys(self, reranker):
        """After rerank(), every result must carry a score_breakdown dict with
        all five component keys plus the composite total.

        The dashboard reads these keys to render per-candidate diagnostics.
        """
        result = reranker.rerank([make_candidate()], query_context={})[0]
        breakdown = result["score_breakdown"]

        expected_keys = {"retrieval_relevance", "recency", "importance",
                         "usage", "type_boost", "composite"}
        assert set(breakdown.keys()) == expected_keys

    def test_composite_score_clamped_to_unit_interval(self, reranker):
        """rerank_score must always be in [0.0, 1.0].

        Test with absurdly large field values to probe the clamping logic.
        """
        extreme = make_candidate(
            fused_score=999.0,
            importance=999.0,
            access_count=99999,
            days_old=0,
            memory_type="procedural",
        )
        result = reranker.rerank([extreme], query_context={})[0]
        assert 0.0 <= result["rerank_score"] <= 1.0


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------

class TestRecencyDecay:
    """Tests for the exponential recency component: exp(-RECENCY_DECAY * days).

    We call _compute_recency() directly to isolate this signal from the full
    composite score.
    """

    def test_memory_from_today_has_recency_one(self, reranker):
        """A memory created right now should yield recency = 1.0.

        Formula: exp(-0.03 * 0) = exp(0) = 1.0.
        """
        today = datetime.utcnow().isoformat()
        recency = reranker._compute_recency(today)
        assert recency == pytest.approx(1.0)

    def test_recent_memory_has_higher_recency_than_old(self, reranker):
        """A memory from today should have higher recency than one from 30 days ago.

        At 30 days: exp(-0.03 * 30) = exp(-0.9) ≈ 0.41 — a substantial drop.
        """
        today          = datetime.utcnow().isoformat()
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()

        recency_today = reranker._compute_recency(today)
        recency_old   = reranker._compute_recency(thirty_days_ago)

        assert recency_today > recency_old

    def test_thirty_day_old_memory_matches_formula(self, reranker):
        """Verify the exact recency value at 30 days matches exp(-0.03 * 30).

        This pins down the formula so any accidental change to RECENCY_DECAY
        or the exponential_recency helper will be caught immediately.
        """
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        recency = reranker._compute_recency(thirty_days_ago)

        expected = math.exp(-HeuristicReranker.RECENCY_DECAY * 30)
        assert recency == pytest.approx(expected, abs=1e-6)

    def test_missing_created_at_returns_neutral_recency(self, reranker):
        """When created_at is None or an empty string the reranker should return
        0.5 (a neutral default) rather than crashing or scoring the memory as
        brand-new or completely stale.
        """
        assert reranker._compute_recency(None) == 0.5
        assert reranker._compute_recency("")   == 0.5

    def test_malformed_created_at_returns_neutral_recency(self, reranker):
        """A corrupted timestamp string should fall back to the 0.5 default."""
        assert reranker._compute_recency("not-a-date") == 0.5


# ---------------------------------------------------------------------------
# Type boosts
# ---------------------------------------------------------------------------

class TestTypeBoosts:
    """Tests for the per-memory-type bonus applied to the composite score.

    TYPE_BOOSTS = {"procedural": 0.10, "preference": 0.05}
    These are multiplied by the type_boost weight (0.05), so the actual score
    contributions are 0.005 and 0.0025 respectively.

    We use _FLAT_CTX so retrieval_relevance=1.0 for all candidates, isolating
    the type_boost signal.
    """

    def test_procedural_beats_semantic(self, reranker):
        """Procedural memories (TYPE_BOOSTS=0.10) should score higher than
        semantic ones (no boost) when all other signals are identical.

        Score difference = 0.05 (weight) * 0.10 (boost) = 0.005.
        """
        procedural = make_candidate(memory_type="procedural")
        semantic   = make_candidate(memory_type="semantic")

        score_proc = reranker.score(procedural, _FLAT_CTX)
        score_sem  = reranker.score(semantic,   _FLAT_CTX)

        assert score_proc > score_sem

    def test_preference_beats_semantic(self, reranker):
        """Preference memories (TYPE_BOOSTS=0.05) should score higher than
        semantic ones when all other signals are identical.

        Score difference = 0.05 (weight) * 0.05 (boost) = 0.0025.
        """
        preference = make_candidate(memory_type="preference")
        semantic   = make_candidate(memory_type="semantic")

        score_pref = reranker.score(preference, _FLAT_CTX)
        score_sem  = reranker.score(semantic,   _FLAT_CTX)

        assert score_pref > score_sem

    def test_procedural_boost_larger_than_preference_boost(self, reranker):
        """procedural (boost=0.10) should outscore preference (boost=0.05)."""
        procedural = make_candidate(memory_type="procedural")
        preference = make_candidate(memory_type="preference")

        score_proc = reranker.score(procedural, _FLAT_CTX)
        score_pref = reranker.score(preference, _FLAT_CTX)

        assert score_proc > score_pref

    def test_procedural_boost_exact_value(self, reranker):
        """The procedural type_boost contribution should be exactly
        TYPE_BOOSTS['procedural'] * WEIGHTS['type_boost'] = 0.10 * 0.05 = 0.005.
        """
        procedural = make_candidate(memory_type="procedural")
        semantic   = make_candidate(memory_type="semantic")

        score_proc = reranker.score(procedural, _FLAT_CTX)
        score_sem  = reranker.score(semantic,   _FLAT_CTX)

        expected_diff = (
            HeuristicReranker.TYPE_BOOSTS["procedural"]
            * HeuristicReranker.WEIGHTS["type_boost"]
        )
        assert score_proc - score_sem == pytest.approx(expected_diff, abs=1e-9)

    def test_unknown_type_gets_no_boost(self, reranker):
        """An unrecognised memory_type should receive 0.0 boost — same as
        'semantic'. The reranker must not crash on unknown types.
        """
        unknown  = make_candidate(memory_type="totally_unknown_type")
        semantic = make_candidate(memory_type="semantic")

        score_unk = reranker.score(unknown,  _FLAT_CTX)
        score_sem = reranker.score(semantic, _FLAT_CTX)

        assert score_unk == pytest.approx(score_sem)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

class TestSorting:
    """rerank() must return results in descending order of composite score."""

    def test_results_sorted_descending_by_rerank_score(self, reranker):
        """Each result's rerank_score must be >= the next one in the list."""
        candidates = [
            make_candidate(fused_score=0.1, memory_id="low"),
            make_candidate(fused_score=0.9, memory_id="high"),
            make_candidate(fused_score=0.5, memory_id="mid"),
        ]
        results = reranker.rerank(candidates, query_context={})
        scores = [r["rerank_score"] for r in results]

        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_highest_fused_score_appears_first(self, reranker):
        """The candidate with the highest fused_score gets retrieval_relevance=1.0
        after normalisation and should be the top result when everything else
        is equal.
        """
        candidates = [
            make_candidate(fused_score=0.1, memory_id="low"),
            make_candidate(fused_score=0.9, memory_id="high"),
        ]
        results = reranker.rerank(candidates, query_context={})

        assert results[0]["memory_id"] == "high"

    def test_rerank_score_added_to_each_result(self, reranker):
        """Every result dict must have a 'rerank_score' key added by rerank()."""
        candidates = [
            make_candidate(memory_id="a"),
            make_candidate(memory_id="b"),
        ]
        results = reranker.rerank(candidates, query_context={})

        for result in results:
            assert "rerank_score" in result
