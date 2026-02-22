"""Shared math utilities for Joey-Bot."""

import math


def exponential_recency(days_old: float, decay_rate: float) -> float:
    """Compute exponential recency score.

    Returns a value between 0 and 1, where 1 means 'just now'
    and values decay towards 0 as days_old increases.

    Args:
        days_old: Number of days since the event. Clamped to >= 0.
        decay_rate: Controls how fast the score decays.
            Higher = faster decay. Common values:
            - 0.03 (reranker scoring — slow decay, older memories stay relevant)
            - 0.05 (lifecycle strength — faster decay, old memories weaken)
    """
    return math.exp(-decay_rate * max(0.0, days_old))
