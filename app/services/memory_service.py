"""Semantic memory service for vector store operations and fact management."""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

from app.models.base import BaseLLM
from app.prompts import format_fact_extraction_prompt
import config


class MemoryService:
    """
    Handles semantic memory operations (Tier 2 memory).

    Manages the vector store for long-term fact storage,
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
        """
        Initialize memory service.

        Args:
            llm: LLM provider for embeddings and fact extraction
            prompts: Loaded prompt templates
            vector_store_path: Path to vector store JSON file
            similarity_threshold: Threshold for duplicate detection (0.0-1.0)
            results_count: Number of results to return from retrieval
        """
        self.llm = llm
        self.prompts = prompts
        self.vector_store_path = vector_store_path or config.VECTOR_STORE_PATH
        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        self.results_count = results_count or config.SEMANTIC_RESULTS_COUNT
        self.enabled = config.SEMANTIC_MEMORY_ENABLED

    def load_vector_store(self) -> Dict[str, Any]:
        """Load the vector store from JSON file."""
        if os.path.exists(self.vector_store_path):
            try:
                with open(self.vector_store_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading vector store: {e}")
        return {"facts": []}

    def save_vector_store(self, store: Dict[str, Any]) -> None:
        """Save the vector store to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
            with open(self.vector_store_path, 'w', encoding='utf-8') as f:
                json.dump(store, f, indent=2)
        except Exception as e:
            print(f"Error saving vector store: {e}")

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    @staticmethod
    def normalize_fact(fact_text: str) -> str:
        """Normalize fact text for deduplication."""
        return ' '.join(fact_text.lower().strip().split())

    @staticmethod
    def compute_fact_hash(fact_text: str) -> str:
        """Compute SHA256 hash of normalized fact."""
        normalized = MemoryService.normalize_fact(fact_text)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def is_duplicate_fact(
        self,
        store: Dict[str, Any],
        new_embedding: List[float]
    ) -> bool:
        """Check if a similar fact already exists using cosine similarity."""
        for fact_entry in store.get("facts", []):
            existing_embedding = fact_entry.get("embedding")
            if existing_embedding:
                similarity = self.cosine_similarity(new_embedding, existing_embedding)
                if similarity > self.similarity_threshold:
                    return True
        return False

    def extract_semantic_facts(self, messages_text: str) -> List[str]:
        """
        Extract permanent facts from messages using LLM.

        Args:
            messages_text: Formatted messages text

        Returns:
            List of extracted fact strings
        """
        prompt = format_fact_extraction_prompt(self.prompts, messages_text)

        try:
            response = self.llm.generate_json(prompt)
            if response:
                # Parse JSON array from response
                import re
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
        """
        Store extracted facts with deduplication.

        Args:
            facts: List of fact strings to store
            conversation_id: Associated conversation ID
            message_ids: IDs of messages facts were extracted from

        Returns:
            Number of facts successfully stored
        """
        if not facts:
            return 0

        store = self.load_vector_store()
        stored = 0

        for fact in facts:
            if not fact or len(fact) < config.MIN_FACT_LENGTH:
                continue

            embedding = self.llm.get_embedding(fact)
            if not embedding:
                continue

            if self.is_duplicate_fact(store, embedding):
                continue

            fact_id = self.compute_fact_hash(fact)
            store["facts"].append({
                "id": fact_id,
                "text": fact,
                "embedding": embedding,
                "conversation_id": conversation_id or 0,
                "timestamp": datetime.utcnow().isoformat(),
                "message_ids": message_ids
            })
            stored += 1

        if stored > 0:
            self.save_vector_store(store)

        return stored

    def process_semantic_memory(
        self,
        conversation_id: int,
        messages_text: str,
        message_ids: List[int]
    ) -> int:
        """
        Main entry point: extract and store facts.

        Args:
            conversation_id: Conversation ID
            messages_text: Formatted messages for fact extraction
            message_ids: Message IDs involved

        Returns:
            Number of facts stored
        """
        if not self.enabled:
            return 0
        facts = self.extract_semantic_facts(messages_text)
        if facts:
            return self.store_facts(facts, conversation_id, message_ids)
        return 0

    def get_relevant_facts(
        self,
        query_text: str,
        n_results: int = None
    ) -> str:
        """
        Retrieve semantically relevant facts from vector store.

        Args:
            query_text: Query to find relevant facts for
            n_results: Number of results to return (default: self.results_count)

        Returns:
            Formatted string of relevant facts, or empty string
        """
        if not query_text or not self.enabled:
            return ''

        n_results = n_results or self.results_count

        try:
            store = self.load_vector_store()
            facts_list = store.get("facts", [])

            if not facts_list:
                return ''

            query_embedding = self.llm.get_embedding(query_text)
            if not query_embedding:
                return ''

            # Calculate similarity scores for all facts
            scored_facts = []
            for fact_entry in facts_list:
                fact_embedding = fact_entry.get("embedding")
                if fact_embedding:
                    similarity = self.cosine_similarity(query_embedding, fact_embedding)
                    scored_facts.append((similarity, fact_entry["text"]))

            # Sort by similarity (descending) and take top n_results
            scored_facts.sort(key=lambda x: x[0], reverse=True)
            top_facts = scored_facts[:n_results]

            if not top_facts:
                return ''

            # Format as bullet points
            return '\n'.join(f"- {fact}" for _, fact in top_facts)

        except Exception as e:
            print(f"Memory retrieval error: {e}")
            return ''

    def get_stats(self) -> Dict[str, Any]:
        """
        Get semantic memory statistics.

        Returns:
            Dictionary with memory stats
        """
        store = self.load_vector_store()
        facts = store.get("facts", [])

        # Count facts by conversation
        conversation_counts = {}
        for fact in facts:
            conv_id = fact.get("conversation_id", 0)
            conversation_counts[conv_id] = conversation_counts.get(conv_id, 0) + 1

        return {
            'enabled': self.enabled,
            'total_facts': len(facts),
            'conversations_with_facts': len(conversation_counts),
            'embedding_model': config.EMBEDDING_MODEL,
            'storage_path': self.vector_store_path
        }
