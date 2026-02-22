"""Unit tests for utils.math_helpers.exponential_recency().

exponential_recency() implements the formula:
    exp(-decay_rate * max(0.0, days_old))

It is used in two places with intentionally different decay rates:
  - HeuristicReranker._compute_recency():  decay_rate=0.03 (slow decay)
  - MemoryLifecycle._compute_recency():    decay_rate=0.05 (faster decay)

This file tests the shared utility function in isolation, verifying its
mathematical properties independently of either caller.

Run from the project root:
    python -m pytest tests/unit/test_math_helpers.py -v
"""

import math

import pytest

from utils.math_helpers import exponential_recency


# ---------------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------------

class TestBoundaryValues:
    """Verify the function at the edges of its input domain."""

    def test_zero_days_returns_one(self):
        """exp(-rate * 0) = exp(0) = 1.0 for any decay_rate."""
        assert exponential_recency(days_old=0.0, decay_rate=0.03) == pytest.approx(1.0)

    def test_zero_days_any_rate_returns_one(self):
        """The 0-days result must be 1.0 regardless of decay_rate."""
        for rate in [0.0, 0.01, 0.03, 0.05, 0.5, 10.0]:
            assert exponential_recency(0.0, rate) == pytest.approx(1.0), \
                f"Failed for decay_rate={rate}"

    def test_negative_days_clamped_to_zero(self):
        """Negative days_old should be clamped to 0.0, returning 1.0.

        The docstring guarantees 'Clamped to >= 0', so a memory from the
        'future' should behave the same as a brand-new one.
        """
        assert exponential_recency(days_old=-10.0, decay_rate=0.03) == pytest.approx(1.0)

    def test_negative_days_same_as_zero_days(self):
        """exponential_recency(-5, rate) must equal exponential_recency(0, rate)."""
        rate = 0.05
        assert exponential_recency(-5.0, rate) == pytest.approx(exponential_recency(0.0, rate))

    def test_zero_decay_rate_always_returns_one(self):
        """decay_rate=0 means no decay: exp(-0 * days) = exp(0) = 1.0 always."""
        for days in [0, 1, 30, 365, 10000]:
            assert exponential_recency(float(days), 0.0) == pytest.approx(1.0), \
                f"Failed for days_old={days}"

    def test_very_large_days_approaches_zero(self):
        """exp(-decay * 1000) should be effectively zero for any meaningful decay rate."""
        result = exponential_recency(days_old=1000.0, decay_rate=0.1)
        # exp(-0.1 * 1000) = exp(-100) ≈ 3.7e-44
        assert result == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Monotonicity and ordering
# ---------------------------------------------------------------------------

class TestMonotonicity:
    """Score must decrease as days_old increases (holding decay_rate fixed)."""

    def test_recent_beats_old(self):
        """Score for 1 day should be higher than score for 30 days."""
        assert exponential_recency(1.0, 0.03) > exponential_recency(30.0, 0.03)

    def test_score_decreases_monotonically_with_age(self):
        """Scores at [1, 7, 30, 90, 365] days must be strictly decreasing."""
        rate = 0.03
        ages = [1.0, 7.0, 30.0, 90.0, 365.0]
        scores = [exponential_recency(d, rate) for d in ages]
        assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))

    def test_higher_decay_rate_gives_lower_score(self):
        """For the same days_old, a higher decay_rate must produce a lower score.

        This validates the two callers use intentionally different rates:
          - Reranker uses 0.03 (older memories stay more relevant longer)
          - Lifecycle uses 0.05 (older memories weaken faster)
        """
        days = 30.0
        score_slow = exponential_recency(days, decay_rate=0.03)
        score_fast = exponential_recency(days, decay_rate=0.05)
        assert score_slow > score_fast


# ---------------------------------------------------------------------------
# Result range
# ---------------------------------------------------------------------------

class TestResultRange:
    """Result must always be in (0, 1] for positive inputs."""

    @pytest.mark.parametrize("days,rate", [
        (0.0,   0.03),
        (1.0,   0.03),
        (30.0,  0.03),
        (365.0, 0.03),
        (0.0,   0.05),
        (7.0,   0.05),
        (180.0, 0.05),
        (0.0,   0.0),
        (100.0, 0.5),
    ])
    def test_result_in_0_1(self, days, rate):
        """exponential_recency must always return a value in [0, 1]."""
        result = exponential_recency(days, rate)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Known values (formula spot-checks)
# ---------------------------------------------------------------------------

class TestKnownValues:
    """Spot-check specific (days_old, decay_rate) pairs against the formula."""

    def test_reranker_30_days(self):
        """Reranker decay rate (0.03) at 30 days: exp(-0.03 * 30) = exp(-0.9)."""
        result   = exponential_recency(30.0, 0.03)
        expected = math.exp(-0.03 * 30)
        assert result == pytest.approx(expected)

    def test_lifecycle_7_days(self):
        """Lifecycle decay rate (0.05) at 7 days: exp(-0.05 * 7) = exp(-0.35)."""
        result   = exponential_recency(7.0, 0.05)
        expected = math.exp(-0.05 * 7)
        assert result == pytest.approx(expected)

    def test_reranker_decay_comment_1_day(self):
        """The reranker docstring states: 1 day = 0.97.
        Verify exp(-0.03 * 1) ≈ 0.97.
        """
        result = exponential_recency(1.0, 0.03)
        assert result == pytest.approx(math.exp(-0.03), rel=1e-4)

    def test_reranker_decay_comment_90_days(self):
        """The reranker docstring states: 90 days = 0.07.
        Verify exp(-0.03 * 90) ≈ 0.07.
        """
        result = exponential_recency(90.0, 0.03)
        assert result == pytest.approx(math.exp(-2.7), rel=1e-4)

    def test_arbitrary_value_matches_formula(self):
        """A completely arbitrary (days, rate) pair should match exp(-rate*days)."""
        days, rate = 42.0, 0.07
        assert exponential_recency(days, rate) == pytest.approx(math.exp(-rate * days))
