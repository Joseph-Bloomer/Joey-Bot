"""Unit tests for MemoryLifecycle.calculate_strength().

calculate_strength() computes a long-term retention score for a memory using
a weighted combination of three signals:

    strength = importance * 0.40 + recency * 0.35 + access_score * 0.25

where:
  - importance   is from metadata, clamped to [0, 1]
  - recency      is exp(-DECAY_RATE * days_old), using DECAY_RATE=0.05
  - access_score is min(access_count / ACCESS_CAP, 1.0), capped at ACCESS_CAP=10

This is distinct from the reranker's score — it measures "should this memory be
kept?" rather than "is this relevant to the current query?".

calculate_strength() only reads class attributes (STRENGTH_WEIGHTS, DECAY_RATE,
ACCESS_CAP) and calls self._compute_recency(). It does not touch self.store or
self.memory_service, so MemoryLifecycle can be safely instantiated with None
for both constructor arguments.

Run from the project root:
    python -m pytest tests/unit/test_memory_lifecycle.py -v
"""

import math
from datetime import datetime, timedelta

import pytest

from app.services.memory_lifecycle import MemoryLifecycle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_meta(
    importance: float = 0.5,
    days_old: int = 0,
    access_count: int = 0,
) -> dict:
    """Build a metadata dict for a single memory.

    Args:
        importance:   Memory importance score, 0.0-1.0.
        days_old:     How many days ago the memory was created.
        access_count: Number of times this memory has been retrieved.
    """
    created_at = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
    return {
        "importance": importance,
        "created_at": created_at,
        "access_count": access_count,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lifecycle() -> MemoryLifecycle:
    """MemoryLifecycle with stub store and memory_service.

    calculate_strength() and _compute_recency() use only class-level constants
    and stdlib datetime — neither store nor memory_service is accessed.
    """
    return MemoryLifecycle(store=None, memory_service=None)


# ---------------------------------------------------------------------------
# Overall strength range
# ---------------------------------------------------------------------------

class TestStrengthRange:
    """strength must always be in [0.0, 1.0] regardless of input values."""

    def test_perfect_signals_give_high_strength(self, lifecycle):
        """importance=1.0, brand-new, at access cap should yield strength ≈ 1.0."""
        cap = int(MemoryLifecycle.ACCESS_CAP)
        meta = make_meta(importance=1.0, days_old=0, access_count=cap)
        strength = lifecycle.calculate_strength(meta)
        # 1.0*0.40 + 1.0*0.35 + 1.0*0.25 = 1.0
        assert strength == pytest.approx(1.0)

    def test_zero_signals_give_near_zero_strength(self, lifecycle):
        """importance=0.0, 365 days old, never accessed should yield strength ≈ 0.0.

        Recency at 365 days: exp(-0.05 * 365) ≈ 3.7e-9 — essentially zero.
        """
        meta = make_meta(importance=0.0, days_old=365, access_count=0)
        strength = lifecycle.calculate_strength(meta)
        assert strength == pytest.approx(0.0, abs=1e-5)

    def test_strength_clamped_to_0_when_negative_would_result(self, lifecycle):
        """Extreme or malformed metadata should never produce a negative strength."""
        meta = {"importance": -99.0, "access_count": -10, "created_at": None}
        strength = lifecycle.calculate_strength(meta)
        assert strength >= 0.0

    def test_strength_clamped_to_1_when_above_would_result(self, lifecycle):
        """Extreme values should never push strength above 1.0."""
        meta = make_meta(importance=999.0, days_old=0, access_count=99999)
        strength = lifecycle.calculate_strength(meta)
        assert strength <= 1.0


# ---------------------------------------------------------------------------
# Weight verification: importance (0.40)
# ---------------------------------------------------------------------------

class TestImportanceWeight:
    """Verify the 0.40 weight on the importance signal."""

    def test_importance_weight_is_0_40(self, lifecycle):
        """Changing importance by 1.0 while holding recency and access constant
        should change strength by exactly 0.40.

        Both candidates are brand-new (recency=1.0) with no access,
        so the only difference is the importance component.
        """
        today = datetime.utcnow().isoformat()
        low  = lifecycle.calculate_strength({"importance": 0.0, "created_at": today, "access_count": 0})
        high = lifecycle.calculate_strength({"importance": 1.0, "created_at": today, "access_count": 0})

        assert (high - low) == pytest.approx(MemoryLifecycle.STRENGTH_WEIGHTS["importance"])

    def test_higher_importance_gives_higher_strength(self, lifecycle):
        """importance=0.9 should score higher than importance=0.2, all else equal."""
        meta_low  = make_meta(importance=0.2)
        meta_high = make_meta(importance=0.9)

        assert lifecycle.calculate_strength(meta_high) > lifecycle.calculate_strength(meta_low)


# ---------------------------------------------------------------------------
# Weight verification: recency (0.35)
# ---------------------------------------------------------------------------

class TestRecencyWeight:
    """Verify the 0.35 weight on the recency signal."""

    def test_brand_new_memory_recency_component_is_0_35(self, lifecycle):
        """A brand-new memory (0 days old) with importance=0 and access=0 should
        have strength = recency * 0.35 = 1.0 * 0.35 = 0.35.

        This is the cleanest isolation of the recency weight.
        """
        meta = make_meta(importance=0.0, days_old=0, access_count=0)
        strength = lifecycle.calculate_strength(meta)
        assert strength == pytest.approx(0.35)

    def test_recent_memory_stronger_than_old(self, lifecycle):
        """With importance and access equal, a newer memory must score higher."""
        new_meta = make_meta(importance=0.5, days_old=0,   access_count=2)
        old_meta = make_meta(importance=0.5, days_old=180, access_count=2)

        assert lifecycle.calculate_strength(new_meta) > lifecycle.calculate_strength(old_meta)

    def test_very_old_memory_has_near_zero_recency(self, lifecycle):
        """A 365-day-old memory should have recency ≈ exp(-0.05*365) ≈ 3.7e-9.

        With importance=0 and access=0, strength ≈ 0.
        """
        meta = make_meta(importance=0.0, days_old=365, access_count=0)
        strength = lifecycle.calculate_strength(meta)
        # recency ≈ 0, so strength ≈ 0 * 0.40 + ~0 * 0.35 + 0 * 0.25 ≈ 0
        assert strength == pytest.approx(0.0, abs=1e-5)

    def test_recency_at_0_days_equals_decay_formula(self, lifecycle):
        """The recency value for a 0-day-old memory should exactly match
        exp(-DECAY_RATE * 0) = 1.0.
        """
        today = datetime.utcnow().isoformat()
        recency = lifecycle._compute_recency(today)
        assert recency == pytest.approx(math.exp(-MemoryLifecycle.DECAY_RATE * 0))

    def test_recency_at_30_days_matches_formula(self, lifecycle):
        """Verify the exact recency value at 30 days against exp(-0.05 * 30)."""
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        recency = lifecycle._compute_recency(thirty_days_ago)
        expected = math.exp(-MemoryLifecycle.DECAY_RATE * 30)
        assert recency == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Weight verification: access (0.25)
# ---------------------------------------------------------------------------

class TestAccessWeight:
    """Verify the 0.25 weight on the access signal."""

    def test_access_weight_is_0_25(self, lifecycle):
        """Changing access from 0 to ACCESS_CAP while holding everything else
        constant should change strength by exactly 0.25.

        Both candidates are brand-new (recency=1.0) to keep recency identical.
        """
        today = datetime.utcnow().isoformat()
        cap = int(MemoryLifecycle.ACCESS_CAP)

        no_access = lifecycle.calculate_strength({"importance": 0.5, "created_at": today, "access_count": 0})
        at_cap    = lifecycle.calculate_strength({"importance": 0.5, "created_at": today, "access_count": cap})

        assert (at_cap - no_access) == pytest.approx(MemoryLifecycle.STRENGTH_WEIGHTS["access"])

    def test_frequently_accessed_beats_never_accessed(self, lifecycle):
        """A memory accessed many times should score higher than one never used."""
        never  = make_meta(importance=0.5, days_old=0, access_count=0)
        often  = make_meta(importance=0.5, days_old=0, access_count=5)

        assert lifecycle.calculate_strength(often) > lifecycle.calculate_strength(never)

    def test_access_cap_clamps_contribution(self, lifecycle):
        """Access counts above ACCESS_CAP should produce the same strength as
        exactly ACCESS_CAP, because access_score = min(count / CAP, 1.0).
        """
        today = datetime.utcnow().isoformat()
        cap = int(MemoryLifecycle.ACCESS_CAP)

        meta_at_cap   = {"importance": 0.5, "created_at": today, "access_count": cap}
        meta_over_cap = {"importance": 0.5, "created_at": today, "access_count": cap * 10}

        assert lifecycle.calculate_strength(meta_at_cap) == pytest.approx(
            lifecycle.calculate_strength(meta_over_cap)
        )

    def test_access_count_1_is_fractional_contribution(self, lifecycle):
        """access_count=1 contributes (1 / ACCESS_CAP) * 0.25 to the score,
        which is strictly less than the full 0.25.
        """
        today = datetime.utcnow().isoformat()
        cap = MemoryLifecycle.ACCESS_CAP

        one_access = lifecycle.calculate_strength({"importance": 0.0, "created_at": today, "access_count": 1})
        # strength = 0*0.40 + 1.0*0.35 + (1/cap)*0.25
        expected_access = (1.0 / cap) * MemoryLifecycle.STRENGTH_WEIGHTS["access"]
        expected_total  = MemoryLifecycle.STRENGTH_WEIGHTS["recency"] + expected_access
        assert one_access == pytest.approx(expected_total)


# ---------------------------------------------------------------------------
# Robustness / edge cases
# ---------------------------------------------------------------------------

class TestRobustness:
    """Malformed or missing metadata must not crash the method."""

    def test_empty_metadata_uses_defaults(self, lifecycle):
        """An empty metadata dict should not raise; defaults kick in for
        importance (0.5), access_count (0), and created_at (neutral 0.5 recency).
        """
        result = lifecycle.calculate_strength({})
        assert 0.0 <= result <= 1.0

    def test_missing_created_at_returns_neutral_recency(self, lifecycle):
        """When created_at is absent, _compute_recency returns 0.5 (neutral).
        The overall strength should still be a valid float.
        """
        meta = {"importance": 0.5, "access_count": 0}  # no created_at
        result = lifecycle.calculate_strength(meta)
        # recency=0.5 (default), so strength = 0.5*0.40 + 0.5*0.35 + 0*0.25 = 0.375
        assert result == pytest.approx(0.375)

    def test_importance_clamped_when_out_of_range(self, lifecycle):
        """importance values outside [0, 1] must be clamped before scoring."""
        high = lifecycle.calculate_strength(make_meta(importance=5.0,  days_old=0, access_count=0))
        low  = lifecycle.calculate_strength(make_meta(importance=-1.0, days_old=0, access_count=0))
        normal_max = lifecycle.calculate_strength(make_meta(importance=1.0, days_old=0, access_count=0))
        normal_min = lifecycle.calculate_strength(make_meta(importance=0.0, days_old=0, access_count=0))

        # Clamped values should be identical to in-range extremes
        assert high == pytest.approx(normal_max)
        assert low  == pytest.approx(normal_min)
