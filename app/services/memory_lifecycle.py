"""Memory lifecycle management: strength decay, consolidation, and pruning."""

import json
import math
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
from qdrant_client import models as qdrant_models

from app.data.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger()

# Module-level timestamp for last lifecycle run
_last_lifecycle_run: Optional[str] = None


def get_last_lifecycle_run() -> Optional[str]:
    return _last_lifecycle_run


class MemoryLifecycle:
    """
    Background maintenance for semantic memories.

    Manages three operations that run on schedules (NOT during chat):
      - Strength decay: recalculates strength for all active memories
      - Consolidation: clusters weak related memories and merges them via LLM
      - Pruning: removes very weak or already-consolidated memories
    """

    # Strength formula weights
    STRENGTH_WEIGHTS = {
        "importance": 0.40,
        "recency": 0.35,
        "access": 0.25,
    }

    # Recency decay rate: exp(-DECAY_RATE * days)
    # 1d=0.95, 7d=0.70, 30d=0.22, 60d=0.05
    DECAY_RATE = 0.05

    # Usage cap for access_score component
    ACCESS_CAP = 10.0

    # Consolidation thresholds
    CONSOLIDATION_STRENGTH_THRESHOLD = 0.3
    CONSOLIDATION_SIMILARITY_THRESHOLD = 0.8
    CONSOLIDATION_MIN_CLUSTER_SIZE = 3
    CONSOLIDATION_MAX_TOKENS = 200

    # Pruning thresholds
    PRUNE_CONSOLIDATED_STRENGTH = 0.1
    PRUNE_UNUSED_STRENGTH = 0.05
    PRUNE_IMPORTANCE_PROTECT = 0.9

    def __init__(self, store: VectorStore, memory_service):
        """
        Args:
            store: VectorStore instance (Qdrant wrapper).
            memory_service: MemoryService instance (for LLM calls and store_facts).
        """
        self.store = store
        self.memory_service = memory_service

    # ------------------------------------------------------------------
    # Strength calculation
    # ------------------------------------------------------------------

    def calculate_strength(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate long-term retention strength for a memory.

        Different from the reranker's score — this measures "is this memory
        worth keeping?" not "is this relevant to the current query?"

        Returns:
            Strength clamped to 0.0-1.0.
        """
        importance = float(metadata.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))

        recency = self._compute_recency(metadata.get("created_at"))

        access_count = int(metadata.get("access_count", 0))
        access_score = min(access_count / self.ACCESS_CAP, 1.0)

        w = self.STRENGTH_WEIGHTS
        strength = (
            importance * w["importance"]
            + recency * w["recency"]
            + access_score * w["access"]
        )
        return max(0.0, min(1.0, strength))

    def _compute_recency(self, created_at: Optional[str]) -> float:
        """Exponential recency decay from created_at ISO timestamp."""
        if not created_at:
            return 0.5
        try:
            created = datetime.fromisoformat(created_at)
            days = max(0, (datetime.utcnow() - created).days)
            return math.exp(-self.DECAY_RATE * days)
        except (ValueError, TypeError):
            return 0.5

    # ------------------------------------------------------------------
    # Strength decay (every 6 hours)
    # ------------------------------------------------------------------

    def decay_all_strengths(self) -> Dict[str, Any]:
        """
        Recalculate and update strength for all active memories.

        Returns:
            Stats dict with total_processed, strong/medium/weak counts.
        """
        global _last_lifecycle_run
        t0 = time.perf_counter()

        all_data = self._scroll_all_memories(with_vectors=False)

        strong = medium = weak = updated = 0

        for point_id, _text, metadata in all_data:
            # Skip consolidated memories — they're managed separately
            if metadata.get("consolidated"):
                continue

            new_strength = self.calculate_strength(metadata)
            old_strength = float(metadata.get("strength", 0.5))

            # Only write if changed meaningfully (avoid needless Qdrant writes)
            if abs(new_strength - old_strength) > 0.01:
                self.store.client.set_payload(
                    collection_name=self.store.collection_name,
                    payload={"strength": round(new_strength, 4)},
                    points=[point_id],
                )
                updated += 1

            # Tally by tier
            if new_strength > 0.7:
                strong += 1
            elif new_strength >= 0.3:
                medium += 1
            else:
                weak += 1

        elapsed = (time.perf_counter() - t0) * 1000
        total = strong + medium + weak
        _last_lifecycle_run = datetime.utcnow().isoformat()

        logger.info(
            f"[LIFECYCLE:DECAY] {total} memories processed ({updated} updated) "
            f"| strong={strong} medium={medium} weak={weak} ({elapsed:.1f}ms)"
        )
        return {
            "total_processed": total,
            "updated": updated,
            "strong_count": strong,
            "medium_count": medium,
            "weak_count": weak,
            "elapsed_ms": round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    # Consolidation (every 24 hours)
    # ------------------------------------------------------------------

    def consolidate(self) -> Dict[str, Any]:
        """
        Cluster weak, related memories and merge them via LLM.

        Returns:
            Stats dict with clusters_found, memories_consolidated, orphan_weak.
        """
        global _last_lifecycle_run
        t0 = time.perf_counter()

        # Gather weak, non-consolidated, non-procedural memories with embeddings
        all_data = self._scroll_all_memories(with_vectors=True)
        weak_memories = []
        for point_id, text, metadata, embedding in all_data:
            if metadata.get("consolidated"):
                continue
            if metadata.get("memory_type") == "procedural":
                continue
            strength = float(metadata.get("strength", 0.5))
            if strength >= self.CONSOLIDATION_STRENGTH_THRESHOLD:
                continue
            weak_memories.append({
                "point_id": point_id,
                "text": text,
                "metadata": metadata,
                "embedding": embedding,
            })

        if not weak_memories:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"[LIFECYCLE:CONSOLIDATE] No weak memories to consolidate ({elapsed:.1f}ms)")
            return {"clusters_found": 0, "memories_consolidated": 0, "orphan_weak_memories": 0}

        # Cluster by embedding cosine similarity
        clusters = self._greedy_cluster(weak_memories)

        memories_consolidated = 0
        clustered_ids = set()

        for cluster in clusters:
            try:
                success = self._merge_cluster(cluster)
                if success:
                    memories_consolidated += len(cluster)
                    for mem in cluster:
                        clustered_ids.add(mem["point_id"])
            except Exception as e:
                logger.warning(f"[LIFECYCLE:CONSOLIDATE] Cluster merge error: {e}")

        orphan_count = len(weak_memories) - len(clustered_ids)
        elapsed = (time.perf_counter() - t0) * 1000
        _last_lifecycle_run = datetime.utcnow().isoformat()

        logger.info(
            f"[LIFECYCLE:CONSOLIDATE] {len(clusters)} clusters, "
            f"{memories_consolidated} memories consolidated, "
            f"{orphan_count} orphan weak memories ({elapsed:.1f}ms)"
        )
        return {
            "clusters_found": len(clusters),
            "memories_consolidated": memories_consolidated,
            "orphan_weak_memories": orphan_count,
            "elapsed_ms": round(elapsed, 1),
        }

    def _greedy_cluster(self, memories: List[Dict]) -> List[List[Dict]]:
        """
        Greedy clustering by pairwise cosine similarity.

        Builds clusters where every member has similarity > threshold
        to the seed memory. Only returns clusters with 3+ members.
        """
        if len(memories) < self.CONSOLIDATION_MIN_CLUSTER_SIZE:
            return []

        embeddings = np.array([m["embedding"] for m in memories], dtype=np.float32)
        # Normalise for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        assigned = set()
        clusters = []

        for i in range(len(memories)):
            if i in assigned:
                continue

            # Find all unassigned memories similar to seed i
            group = [i]
            for j in range(i + 1, len(memories)):
                if j in assigned:
                    continue
                if sim_matrix[i, j] > self.CONSOLIDATION_SIMILARITY_THRESHOLD:
                    group.append(j)

            if len(group) >= self.CONSOLIDATION_MIN_CLUSTER_SIZE:
                clusters.append([memories[idx] for idx in group])
                assigned.update(group)

        return clusters

    def _merge_cluster(self, cluster: List[Dict]) -> bool:
        """
        Merge a cluster of weak memories into a single consolidated memory.

        Returns True if consolidation succeeded.
        """
        texts = [m["text"] for m in cluster]
        memories_text = "\n".join(f"- {t}" for t in texts)

        prompt = self.memory_service.prompts.get("memory_consolidation", "")
        if not prompt:
            logger.warning("[LIFECYCLE:CONSOLIDATE] No memory_consolidation prompt configured")
            return False

        prompt = prompt.format(memories_text=memories_text)

        # Generate consolidated text (non-streaming, limited tokens)
        consolidated_text = self.memory_service.llm.generate(
            prompt, stream=False, temperature=0.3
        )
        if not consolidated_text or len(consolidated_text.strip()) < 10:
            logger.warning("[LIFECYCLE:CONSOLIDATE] LLM returned empty consolidation")
            return False

        consolidated_text = consolidated_text.strip()

        # Get embedding for the consolidated text
        embedding = self.memory_service.llm.get_embedding(consolidated_text)
        if not embedding:
            logger.warning("[LIFECYCLE:CONSOLIDATE] Failed to embed consolidated text")
            return False

        # Build metadata for consolidated memory
        max_importance = max(
            float(m["metadata"].get("importance", 0.5)) for m in cluster
        )
        source_ids = [str(m["point_id"]) for m in cluster]
        now = datetime.utcnow().isoformat()
        metadata = {
            "memory_type": "consolidated",
            "importance": max_importance,
            "created_at": now,
            "last_accessed": now,
            "access_count": 0,
            "source_conversation_id": "",
            "consolidated": False,
            "strength": max_importance,
            "message_ids": "[]",
            "source_memory_ids": json.dumps(source_ids),
        }

        # Store the consolidated memory
        fact_id = self.memory_service.compute_fact_hash(consolidated_text)
        self.store.add_memory(
            memory_id=fact_id,
            text=consolidated_text,
            embedding=embedding,
            metadata=metadata,
        )

        # Mark originals as consolidated
        for mem in cluster:
            try:
                self.store.client.set_payload(
                    collection_name=self.store.collection_name,
                    payload={"consolidated": True},
                    points=[mem["point_id"]],
                )
            except Exception as e:
                logger.warning(f"[LIFECYCLE:CONSOLIDATE] Failed to mark {mem['point_id']} as consolidated: {e}")

        # Rebuild BM25 index to include the new consolidated memory
        self.memory_service.retriever.build_bm25_index()

        logger.info(
            f"[LIFECYCLE:CONSOLIDATE] Merged {len(cluster)} memories into: "
            f"{consolidated_text[:80]}..."
        )
        return True

    # ------------------------------------------------------------------
    # Pruning (weekly)
    # ------------------------------------------------------------------

    def prune(self) -> Dict[str, Any]:
        """
        Delete very weak or already-consolidated original memories.

        Never prunes procedural memories or high-importance (>=0.9) memories.

        Returns:
            Stats dict with pruned counts and protected count.
        """
        global _last_lifecycle_run
        t0 = time.perf_counter()

        all_data = self._scroll_all_memories(with_vectors=False)

        pruned_consolidated = 0
        pruned_unused = 0
        protected = 0
        to_delete = []

        for point_id, text, metadata in all_data:
            memory_type = metadata.get("memory_type", "")
            importance = float(metadata.get("importance", 0.5))
            strength = float(metadata.get("strength", 0.5))
            access_count = int(metadata.get("access_count", 0))
            is_consolidated = metadata.get("consolidated", False)

            # Protection rules
            if memory_type == "procedural":
                protected += 1
                continue
            if importance >= self.PRUNE_IMPORTANCE_PROTECT:
                protected += 1
                continue

            # Prune: consolidated originals with very low strength
            if is_consolidated and strength < self.PRUNE_CONSOLIDATED_STRENGTH:
                to_delete.append((point_id, text, "consolidated+weak"))
                pruned_consolidated += 1
                continue

            # Prune: never-accessed memories with extremely low strength
            if strength < self.PRUNE_UNUSED_STRENGTH and access_count == 0:
                to_delete.append((point_id, text, "unused+very_weak"))
                pruned_unused += 1
                continue

        # Execute deletions
        for point_id, text, reason in to_delete:
            try:
                self.store.client.delete(
                    collection_name=self.store.collection_name,
                    points_selector=qdrant_models.PointIdsList(points=[point_id]),
                )
                logger.info(f"[LIFECYCLE:PRUNE] Deleted ({reason}): {text[:60]}...")
            except Exception as e:
                logger.warning(f"[LIFECYCLE:PRUNE] Failed to delete {point_id}: {e}")

        # Rebuild BM25 index if anything was pruned
        if to_delete:
            self.memory_service.retriever.build_bm25_index()

        elapsed = (time.perf_counter() - t0) * 1000
        _last_lifecycle_run = datetime.utcnow().isoformat()

        total_pruned = pruned_consolidated + pruned_unused
        logger.info(
            f"[LIFECYCLE:PRUNE] {total_pruned} pruned "
            f"(consolidated={pruned_consolidated}, unused={pruned_unused}), "
            f"{protected} protected ({elapsed:.1f}ms)"
        )
        return {
            "pruned_consolidated": pruned_consolidated,
            "pruned_unused": pruned_unused,
            "protected_count": protected,
            "elapsed_ms": round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def run_full_cycle(self) -> Dict[str, Any]:
        """Run decay → consolidate → prune in sequence."""
        logger.info("[LIFECYCLE] Starting full cycle")
        decay_stats = self.decay_all_strengths()
        consolidation_stats = self.consolidate()
        prune_stats = self.prune()
        logger.info("[LIFECYCLE] Full cycle complete")
        return {
            "decay": decay_stats,
            "consolidation": consolidation_stats,
            "prune": prune_stats,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scroll_all_memories(self, with_vectors: bool = False):
        """
        Scroll through all Qdrant points, yielding (point_id, text, metadata[, embedding]).

        Uses store.client directly to access UUIDs and optional vectors.
        """
        results_list = []
        offset = None

        while True:
            points, next_offset = self.store.client.scroll(
                collection_name=self.store.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )

            for point in points:
                payload = dict(point.payload)
                text = payload.pop("text", "")
                payload.pop("original_id", None)

                if with_vectors:
                    results_list.append((point.id, text, payload, point.vector))
                else:
                    results_list.append((point.id, text, payload))

            if next_offset is None:
                break
            offset = next_offset

        return results_list
