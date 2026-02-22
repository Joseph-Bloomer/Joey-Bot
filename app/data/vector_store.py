"""Qdrant-backed vector store for semantic memory."""

import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when a Qdrant operation fails."""


class VectorStore:
    """Wraps Qdrant local mode for persistent vector storage with metadata.

    Uses lazy initialization to avoid file-lock conflicts with Flask's
    debug reloader (which spawns two processes importing the same module).
    """

    def __init__(self, persist_dir: str, collection_name: str, vector_size: int = 768):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            try:
                self._client = QdrantClient(path=self._persist_dir)
                if not self._client.collection_exists(self._collection_name):
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=models.VectorParams(
                            size=self._vector_size,
                            distance=models.Distance.COSINE,
                        ),
                    )
            except Exception as e:
                self._client = None
                logger.error("Failed to initialize Qdrant client at '%s': %s", self._persist_dir, e)
                raise VectorStoreError(
                    f"Qdrant initialization failed for path '{self._persist_dir}': {e}"
                ) from e
        return self._client

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @staticmethod
    def _to_uuid(memory_id: str) -> str:
        """Convert a SHA256 hex string to a valid UUID for Qdrant."""
        return str(uuid.UUID(memory_id[:32]))

    def add_memory(
        self,
        memory_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Store a single memory with its embedding and metadata."""
        point_id = self._to_uuid(memory_id)
        payload = {**metadata, "text": text, "original_id": memory_id}
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(id=point_id, vector=embedding, payload=payload)],
            )
        except UnexpectedResponse as e:
            logger.error("add_memory: Qdrant upsert failed for memory_id='%s': %s", memory_id, e)
            raise VectorStoreError(f"Failed to store memory '{memory_id}': {e}") from e
        except Exception as e:
            logger.error("add_memory: unexpected error for memory_id='%s': %s", memory_id, e)
            raise VectorStoreError(f"Failed to store memory '{memory_id}': {e}") from e

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where_filter: Optional[models.Filter] = None,
    ) -> Dict[str, Any]:
        """Similarity search. Returns dict with ids, documents, metadatas, scores."""
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=n_results,
                query_filter=where_filter,
                with_payload=True,
            )
        except UnexpectedResponse as e:
            logger.error("search: Qdrant query_points failed (n_results=%d): %s", n_results, e)
            raise VectorStoreError(f"Vector search failed: {e}") from e
        except Exception as e:
            logger.error("search: unexpected error (n_results=%d): %s", n_results, e)
            raise VectorStoreError(f"Vector search failed: {e}") from e

        ids = []
        documents = []
        metadatas = []
        scores = []
        for point in results.points:
            ids.append(str(point.id))
            payload = dict(point.payload)
            documents.append(payload.pop("text", ""))
            payload.pop("original_id", None)
            metadatas.append(payload)
            scores.append(point.score)

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "scores": [scores],
        }

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single memory by ID. Returns None if not found."""
        point_id = self._to_uuid(memory_id)
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True,
            )
        except UnexpectedResponse as e:
            logger.error("get_memory: Qdrant retrieve failed for memory_id='%s': %s", memory_id, e)
            raise VectorStoreError(f"Failed to retrieve memory '{memory_id}': {e}") from e
        except Exception as e:
            logger.error("get_memory: unexpected error for memory_id='%s': %s", memory_id, e)
            raise VectorStoreError(f"Failed to retrieve memory '{memory_id}': {e}") from e
        if not results:
            return None
        point = results[0]
        payload = dict(point.payload)
        text = payload.pop("text", "")
        payload.pop("original_id", None)
        return {
            "id": str(point.id),
            "text": text,
            "metadata": payload,
            "embedding": point.vector,
        }

    def update_metadata(self, memory_id: str, metadata_updates: Dict[str, Any]) -> None:
        """Update metadata fields on an existing memory."""
        point_id = self._to_uuid(memory_id)
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=metadata_updates,
                points=[point_id],
            )
        except UnexpectedResponse as e:
            logger.error("update_metadata: Qdrant set_payload failed for memory_id='%s': %s", memory_id, e)
            raise VectorStoreError(f"Failed to update metadata for memory '{memory_id}': {e}") from e
        except Exception as e:
            logger.error("update_metadata: unexpected error for memory_id='%s': %s", memory_id, e)
            raise VectorStoreError(f"Failed to update metadata for memory '{memory_id}': {e}") from e

    def delete_memory(self, memory_id: str) -> None:
        """Remove a memory by ID."""
        point_id = self._to_uuid(memory_id)
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[point_id]),
            )
        except UnexpectedResponse as e:
            logger.error("delete_memory: Qdrant delete failed for memory_id='%s': %s", memory_id, e)
            raise VectorStoreError(f"Failed to delete memory '{memory_id}': {e}") from e
        except Exception as e:
            logger.error("delete_memory: unexpected error for memory_id='%s': %s", memory_id, e)
            raise VectorStoreError(f"Failed to delete memory '{memory_id}': {e}") from e

    def get_all(self, where_filter: Optional[models.Filter] = None) -> Dict[str, Any]:
        """List all memories, optionally filtered."""
        all_ids = []
        all_documents = []
        all_metadatas = []

        offset = None
        while True:
            try:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=where_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )
            except UnexpectedResponse as e:
                logger.error("get_all: Qdrant scroll failed (offset=%s): %s", offset, e)
                raise VectorStoreError(f"Failed to scroll memories: {e}") from e
            except Exception as e:
                logger.error("get_all: unexpected error (offset=%s): %s", offset, e)
                raise VectorStoreError(f"Failed to scroll memories: {e}") from e
            for point in results:
                payload = dict(point.payload)
                all_ids.append(str(point.id))
                all_documents.append(payload.pop("text", ""))
                payload.pop("original_id", None)
                all_metadatas.append(payload)

            if next_offset is None:
                break
            offset = next_offset

        return {"ids": all_ids, "documents": all_documents, "metadatas": all_metadatas}

    def count(self) -> int:
        """Total number of stored memories."""
        try:
            return self.client.count(collection_name=self.collection_name).count
        except UnexpectedResponse as e:
            logger.error("count: Qdrant count failed: %s", e)
            raise VectorStoreError(f"Failed to count memories: {e}") from e
        except Exception as e:
            logger.error("count: unexpected error: %s", e)
            raise VectorStoreError(f"Failed to count memories: {e}") from e
