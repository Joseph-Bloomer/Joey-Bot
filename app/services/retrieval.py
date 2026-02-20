"""Hybrid retrieval combining dense vector search with sparse BM25 keyword search."""

from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from app.data.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger()


class HybridRetriever:
    """Fuses Qdrant vector similarity with BM25 keyword matching via Reciprocal Rank Fusion."""

    RRF_K = 60  # Standard RRF constant

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_ids: List[str] = []      # BM25 position -> memory UUID
        self._corpus_texts: List[str] = []    # BM25 position -> raw text
        self._corpus_meta: List[Dict] = []    # BM25 position -> metadata
        self._index_built = False

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Lazily build the BM25 index on first use."""
        if not self._index_built:
            self.build_bm25_index()

    def build_bm25_index(self) -> None:
        """(Re)build the BM25 index from all memories in the vector store."""
        all_memories = self.store.get_all()
        ids = all_memories["ids"]
        docs = all_memories["documents"]
        metas = all_memories["metadatas"]

        if not docs:
            self._bm25 = None
            self._corpus_ids = []
            self._corpus_texts = []
            self._corpus_meta = []
            self._index_built = True
            logger.info("HybridRetriever: BM25 index built (0 documents)")
            return

        tokenized = [self._tokenize(doc) for doc in docs]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus_ids = list(ids)
        self._corpus_texts = list(docs)
        self._corpus_meta = list(metas)
        self._index_built = True
        logger.info(f"HybridRetriever: BM25 index built ({len(docs)} documents)")

    def add_to_index(self, memory_id: str, text: str, metadata: Dict[str, Any] = None) -> None:
        """Add a single memory to the BM25 index by triggering a full rebuild.

        BM25Okapi precomputes IDF statistics across the whole corpus, so
        appending a single document would leave those statistics stale.
        A full rebuild is the simplest correct approach and is fast for
        the expected corpus size (hundreds to low thousands of memories).
        """
        self.build_bm25_index()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        query_embedding: List[float],
        n_results: int = 10,
        where_filter=None,
    ) -> List[Dict[str, Any]]:
        """Run hybrid search and fuse results with Reciprocal Rank Fusion.

        Args:
            query_text: Raw query string (used for BM25).
            query_embedding: Pre-computed embedding vector (used for dense search).
            n_results: Number of final results to return.
            where_filter: Optional Qdrant filter passed to dense search.

        Returns:
            List of dicts, each with memory_id, text, metadata,
            dense_score, sparse_score, and fused_score.
        """
        self._ensure_index()
        dense_results = self._dense_search(query_embedding, n_results=20, where_filter=where_filter)
        sparse_results = self._sparse_search(query_text, n_results=20)

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Sort descending by fused score, take top n
        fused_sorted = sorted(fused.values(), key=lambda r: r["fused_score"], reverse=True)
        top = fused_sorted[:n_results]

        self._log_diagnostics(top, dense_results, sparse_results)

        return top

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_search(
        self,
        query_embedding: List[float],
        n_results: int = 20,
        where_filter=None,
    ) -> List[Dict[str, Any]]:
        """Run vector similarity search via Qdrant."""
        if self.store.count() == 0:
            return []

        raw = self.store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where_filter=where_filter,
        )

        results = []
        for mid, doc, meta, score in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["scores"][0]
        ):
            results.append({
                "memory_id": mid,
                "text": doc,
                "metadata": meta,
                "dense_score": score,
            })
        return results

    def _sparse_search(self, query_text: str, n_results: int = 20) -> List[Dict[str, Any]]:
        """Run BM25 keyword search over the in-memory corpus."""
        if self._bm25 is None or not self._corpus_ids:
            return []

        tokenized_query = self._tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-n indices by score
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:n_results]

        results = []
        for idx in scored_indices:
            if scores[idx] <= 0:
                break
            results.append({
                "memory_id": self._corpus_ids[idx],
                "text": self._corpus_texts[idx],
                "metadata": self._corpus_meta[idx],
                "sparse_score": float(scores[idx]),
            })
        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Merge two ranked lists using RRF: score = sum(1 / (k + rank))."""
        fused: Dict[str, Dict[str, Any]] = {}
        k = self.RRF_K

        for rank, result in enumerate(dense_results, start=1):
            mid = result["memory_id"]
            if mid not in fused:
                fused[mid] = {
                    "memory_id": mid,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "dense_score": result.get("dense_score", 0.0),
                    "sparse_score": 0.0,
                    "fused_score": 0.0,
                }
            fused[mid]["dense_score"] = result.get("dense_score", 0.0)
            fused[mid]["fused_score"] += 1.0 / (k + rank)

        for rank, result in enumerate(sparse_results, start=1):
            mid = result["memory_id"]
            if mid not in fused:
                fused[mid] = {
                    "memory_id": mid,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "dense_score": 0.0,
                    "sparse_score": result.get("sparse_score", 0.0),
                    "fused_score": 0.0,
                }
            fused[mid]["sparse_score"] = result.get("sparse_score", 0.0)
            fused[mid]["fused_score"] += 1.0 / (k + rank)

        return fused

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log_diagnostics(
        self,
        fused_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
    ) -> None:
        """Log which retrieval path contributed each result."""
        dense_ids = {r["memory_id"] for r in dense_results}
        sparse_ids = {r["memory_id"] for r in sparse_results}

        both = []
        dense_only = []
        sparse_only = []

        for r in fused_results:
            mid = r["memory_id"]
            in_dense = mid in dense_ids
            in_sparse = mid in sparse_ids

            if in_dense and in_sparse:
                both.append(mid)
            elif in_dense:
                dense_only.append(mid)
            else:
                sparse_only.append(mid)

        logger.info(
            f"HybridRetriever diagnostics: "
            f"{len(fused_results)} results | "
            f"both={len(both)} dense_only={len(dense_only)} sparse_only={len(sparse_only)}"
        )

        if sparse_only:
            texts = [r["text"][:60] for r in fused_results if r["memory_id"] in set(sparse_only)]
            logger.info(f"  BM25-only hits: {texts}")

        if dense_only:
            texts = [r["text"][:60] for r in fused_results if r["memory_id"] in set(dense_only)]
            logger.info(f"  Dense-only hits: {texts}")
