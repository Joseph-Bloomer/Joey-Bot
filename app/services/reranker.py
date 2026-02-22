"""Heuristic reranker for scoring retrieval candidates without LLM calls."""

from datetime import datetime
from typing import Dict, Any, List, Optional

from utils.logger import get_logger
from utils.math_helpers import exponential_recency

logger = get_logger()


class HeuristicReranker:
    """
    Scores retrieval candidates using a weighted combination of signals
    already present in the candidate metadata. Pure computation — no LLM calls.

    Composite score = weighted sum of:
      - retrieval_relevance (fused_score, min-max normalised)
      - recency_boost (exponential decay from created_at)
      - importance_score (from metadata, 0-1)
      - usage_signal (access_count, capped)
      - type_boost (bonus for procedural/preference memories)
    """

    # Tunable weights — must sum to 1.0
    WEIGHTS = {
        "retrieval_relevance": 0.45,
        "recency": 0.20,
        "importance": 0.20,
        "usage": 0.10,
        "type_boost": 0.05,
    }

    # Bonus scores by memory_type
    TYPE_BOOSTS = {
        "procedural": 0.10,
        "preference": 0.05,
    }

    # Recency decay: exp(-RECENCY_DECAY * days_old)
    # 1 day=0.97, 7 days=0.81, 30 days=0.41, 90 days=0.07
    RECENCY_DECAY = 0.03

    # Usage signal caps at this many accesses
    USAGE_CAP = 5.0

    def score(self, candidate: Dict[str, Any], query_context: Dict[str, Any]) -> float:
        """
        Compute composite score for a single candidate.

        Args:
            candidate: Dict with fused_score, metadata (importance, access_count,
                       created_at, memory_type, strength).
            query_context: Dict with fused_min, fused_max for normalisation.

        Returns:
            Composite score clamped to 0.0-1.0.
        """
        meta = candidate.get("metadata", {})

        # a) Retrieval relevance — min-max normalise fused_score across candidate set
        fused = candidate.get("fused_score", 0.0)
        fused_min = query_context.get("fused_min", 0.0)
        fused_max = query_context.get("fused_max", 0.0)
        if fused_max > fused_min:
            retrieval_relevance = (fused - fused_min) / (fused_max - fused_min)
        else:
            retrieval_relevance = 1.0  # single candidate or all equal

        # b) Recency boost — exponential decay from created_at
        recency_boost = self._compute_recency(meta.get("created_at"))

        # c) Importance — already 0-1 in metadata
        importance_score = float(meta.get("importance", 0.5))
        importance_score = max(0.0, min(1.0, importance_score))

        # d) Usage signal — capped at USAGE_CAP accesses
        access_count = int(meta.get("access_count", 0))
        usage_signal = min(access_count / self.USAGE_CAP, 1.0)

        # e) Type boost — bonus for procedural/preference
        memory_type = meta.get("memory_type", "")
        type_boost = self.TYPE_BOOSTS.get(memory_type, 0.0)

        # Composite weighted sum
        w = self.WEIGHTS
        composite = (
            retrieval_relevance * w["retrieval_relevance"]
            + recency_boost * w["recency"]
            + importance_score * w["importance"]
            + usage_signal * w["usage"]
            + type_boost * w["type_boost"]
        )

        return max(0.0, min(1.0, composite))

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        query_context: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Score, sort, and trim candidates.

        Args:
            candidates: List of retrieval result dicts from HybridRetriever.
            query_context: Additional context (currently unused beyond normalisation).
            top_k: Maximum number of results to return.

        Returns:
            Sorted list of candidates with rerank_score and score_breakdown added.
        """
        if not candidates:
            return []

        # Compute fused_score min/max for normalisation
        fused_scores = [c.get("fused_score", 0.0) for c in candidates]
        query_context = {
            **query_context,
            "fused_min": min(fused_scores),
            "fused_max": max(fused_scores),
        }

        scored = []
        for candidate in candidates:
            meta = candidate.get("metadata", {})

            # Compute individual components for breakdown
            fused = candidate.get("fused_score", 0.0)
            fused_min = query_context["fused_min"]
            fused_max = query_context["fused_max"]
            if fused_max > fused_min:
                retrieval_relevance = (fused - fused_min) / (fused_max - fused_min)
            else:
                retrieval_relevance = 1.0

            recency_boost = self._compute_recency(meta.get("created_at"))
            importance_score = max(0.0, min(1.0, float(meta.get("importance", 0.5))))
            access_count = int(meta.get("access_count", 0))
            usage_signal = min(access_count / self.USAGE_CAP, 1.0)
            memory_type = meta.get("memory_type", "")
            type_boost = self.TYPE_BOOSTS.get(memory_type, 0.0)

            composite = self.score(candidate, query_context)

            result = {**candidate}
            result["rerank_score"] = composite
            result["score_breakdown"] = {
                "retrieval_relevance": round(retrieval_relevance, 3),
                "recency": round(recency_boost, 3),
                "importance": round(importance_score, 3),
                "usage": round(usage_signal, 3),
                "type_boost": round(type_boost, 3),
                "composite": round(composite, 3),
            }
            scored.append(result)

        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored[:top_k]

    def explain(self, candidate: Dict[str, Any]) -> str:
        """
        Return a human-readable explanation of a candidate's score.

        Requires the candidate to have been processed by rerank() first
        (i.e. has rerank_score and score_breakdown).
        """
        breakdown = candidate.get("score_breakdown")
        if not breakdown:
            return "No score breakdown available — run rerank() first."

        meta = candidate.get("metadata", {})
        parts = [f"Score {breakdown['composite']:.2f}:"]

        # Retrieval relevance
        rr = breakdown["retrieval_relevance"]
        if rr >= 0.7:
            parts.append(f"high retrieval relevance ({rr:.2f})")
        elif rr >= 0.4:
            parts.append(f"moderate retrieval relevance ({rr:.2f})")
        else:
            parts.append(f"low retrieval relevance ({rr:.2f})")

        # Recency
        days_old = self._days_old(meta.get("created_at"))
        if days_old is not None:
            if days_old < 7:
                parts.append(f"recent ({days_old} days)")
            elif days_old < 30:
                parts.append(f"moderately old ({days_old} days)")
            else:
                parts.append(f"old ({days_old} days)")
        else:
            parts.append("unknown age")

        # Importance
        imp = breakdown["importance"]
        if imp >= 0.7:
            parts.append(f"high importance ({imp:.1f})")
        elif imp >= 0.4:
            parts.append(f"moderate importance ({imp:.1f})")
        else:
            parts.append(f"low importance ({imp:.1f})")

        # Usage
        access_count = int(meta.get("access_count", 0))
        if access_count > 0:
            parts.append(f"accessed {access_count} times")

        # Type boost
        memory_type = meta.get("memory_type", "")
        if memory_type in self.TYPE_BOOSTS:
            parts.append(f"{memory_type} type (+{self.TYPE_BOOSTS[memory_type]:.2f})")

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_recency(self, created_at: Optional[str]) -> float:
        """Compute recency boost from created_at ISO timestamp."""
        days = self._days_old(created_at)
        if days is None:
            return 0.5  # unknown age, neutral default
        return exponential_recency(days, self.RECENCY_DECAY)

    @staticmethod
    def _days_old(created_at: Optional[str]) -> Optional[int]:
        """Parse created_at and return days since creation, or None."""
        if not created_at:
            return None
        try:
            created = datetime.fromisoformat(created_at)
            delta = datetime.utcnow() - created
            return max(0, delta.days)
        except (ValueError, TypeError):
            return None
