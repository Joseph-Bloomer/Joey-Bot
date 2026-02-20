"""Semantic memory service for vector store operations and fact management."""

import json
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.models.base import BaseLLM
from app.data.vector_store import VectorStore
from app.services.retrieval import HybridRetriever
from app.prompts import format_fact_extraction_prompt
import config


class MemoryService:
    """
    Handles semantic memory operations (Tier 2 memory).

    Manages the Qdrant vector store for long-term fact storage,
    including embedding generation, deduplication, and retrieval.
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompts: Dict[str, Any],
        vector_store_path: str = None,
        similarity_threshold: float = None,
        results_count: int = None
    ):
        self.llm = llm
        self.prompts = prompts
        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        self.results_count = results_count or config.SEMANTIC_RESULTS_COUNT
        self.enabled = config.SEMANTIC_MEMORY_ENABLED

        # Initialize Qdrant-backed vector store
        self.store = VectorStore(
            persist_dir=config.QDRANT_PERSIST_DIR,
            collection_name=config.QDRANT_COLLECTION_NAME,
            vector_size=config.QDRANT_VECTOR_SIZE,
        )

        # Hybrid retriever (dense + BM25 sparse search with RRF fusion)
        # Index is built lazily on first search to avoid Qdrant file-lock
        # conflicts with Flask's debug reloader.
        self.retriever = HybridRetriever(self.store)

    @staticmethod
    def normalize_fact(fact_text: str) -> str:
        """Normalize fact text for deduplication."""
        return ' '.join(fact_text.lower().strip().split())

    @staticmethod
    def compute_fact_hash(fact_text: str) -> str:
        """Compute SHA256 hash of normalized fact."""
        normalized = MemoryService.normalize_fact(fact_text)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _build_metadata(
        self,
        conversation_id: int,
        message_ids: List[int],
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        """Build the standard metadata dict for a new memory."""
        now = datetime.utcnow().isoformat()
        return {
            "memory_type": "semantic",
            "importance": importance,
            "created_at": now,
            "last_accessed": now,
            "access_count": 0,
            "source_conversation_id": str(conversation_id) if conversation_id else "",
            "consolidated": False,
            "strength": importance,
            "message_ids": json.dumps(message_ids),
        }

    def _is_duplicate(self, embedding: List[float]) -> bool:
        """Check if a similar memory already exists via Qdrant search."""
        if self.store.count() == 0:
            return False

        results = self.store.search(query_embedding=embedding, n_results=1)
        if results["scores"] and results["scores"][0]:
            # Qdrant cosine score is similarity directly (higher = more similar)
            best_similarity = results["scores"][0][0]
            return best_similarity > self.similarity_threshold
        return False

    def extract_semantic_facts(self, messages_text: str) -> List[str]:
        """Extract permanent facts from messages using LLM."""
        prompt = format_fact_extraction_prompt(self.prompts, messages_text)
        try:
            response = self.llm.generate_json(prompt)
            if response:
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except Exception as e:
            print(f"Fact extraction error: {e}")
        return []

    def store_facts(
        self,
        facts: List[str],
        conversation_id: int,
        message_ids: List[int]
    ) -> int:
        """Store extracted facts with deduplication."""
        if not facts:
            return 0

        stored = 0
        for fact in facts:
            if not fact or len(fact) < config.MIN_FACT_LENGTH:
                continue

            embedding = self.llm.get_embedding(fact)
            if not embedding:
                continue

            if self._is_duplicate(embedding):
                continue

            fact_id = self.compute_fact_hash(fact)
            metadata = self._build_metadata(conversation_id, message_ids)

            self.store.add_memory(
                memory_id=fact_id,
                text=fact,
                embedding=embedding,
                metadata=metadata,
            )
            stored += 1

        # Rebuild BM25 index once after all new facts are stored
        if stored > 0:
            self.retriever.add_to_index("", "")

        return stored

    def process_semantic_memory(
        self,
        conversation_id: int,
        messages_text: str,
        message_ids: List[int]
    ) -> int:
        """Main entry point: extract and store facts."""
        if not self.enabled:
            return 0
        facts = self.extract_semantic_facts(messages_text)
        if facts:
            return self.store_facts(facts, conversation_id, message_ids)
        return 0

    def get_relevant_facts(
        self,
        query_text: str,
        n_results: int = None,
        extra_keywords: List[str] = None,
    ) -> str:
        """Retrieve relevant facts using hybrid dense + BM25 search.

        Args:
            query_text: The query to search for.
            n_results: Max number of results.
            extra_keywords: Additional keyword terms (e.g. gatekeeper retrieval_keys)
                            appended to the BM25 query for better keyword coverage.
        """
        if not query_text or not self.enabled:
            return ''

        n_results = n_results or self.results_count

        try:
            if self.store.count() == 0:
                return ''

            query_embedding = self.llm.get_embedding(query_text)
            if not query_embedding:
                return ''

            # Append extra keywords to give BM25 more signal
            bm25_query = query_text
            if extra_keywords:
                bm25_query = query_text + " " + " ".join(extra_keywords)

            results = self.retriever.search(
                query_text=bm25_query,
                query_embedding=query_embedding,
                n_results=n_results,
            )

            if not results:
                return ''

            # Update access metadata for each returned memory
            now = datetime.utcnow().isoformat()
            for r in results:
                existing = self.store.get_memory(r["memory_id"])
                if existing:
                    meta = existing["metadata"]
                    new_count = meta.get("access_count", 0) + 1
                    self.store.update_metadata(r["memory_id"], {
                        "access_count": new_count,
                        "last_accessed": now,
                    })

            return '\n'.join(f"- {r['text']}" for r in results)

        except Exception as e:
            print(f"Memory retrieval error: {e}")
            return ''

    def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics from Qdrant."""
        total = self.store.count()

        stats = {
            'enabled': self.enabled,
            'total_facts': total,
            'embedding_model': config.EMBEDDING_MODEL,
            'storage_backend': 'qdrant',
        }

        if total > 0:
            all_memories = self.store.get_all()
            metadatas = all_memories.get("metadatas", [])

            # Count by memory_type
            type_counts = {}
            importance_sum = 0.0
            for meta in metadatas:
                mtype = meta.get("memory_type", "unknown")
                type_counts[mtype] = type_counts.get(mtype, 0) + 1
                importance_sum += meta.get("importance", 0.0)

            stats["count_by_type"] = type_counts
            stats["avg_importance"] = round(importance_sum / total, 3)

        return stats
