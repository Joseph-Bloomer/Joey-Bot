"""Migrate semantic_memory.json into Qdrant.

Usage:
    python scripts/migrate_json_to_qdrant.py
"""

import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from app.data.vector_store import VectorStore


def migrate():
    json_path = config.VECTOR_STORE_PATH
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    facts = data.get("facts", [])
    print(f"Found {len(facts)} facts in JSON file")

    if not facts:
        print("Nothing to migrate.")
        return

    store = VectorStore(
        persist_dir=config.QDRANT_PERSIST_DIR,
        collection_name=config.QDRANT_COLLECTION_NAME,
        vector_size=config.QDRANT_VECTOR_SIZE,
    )

    migrated = 0
    skipped = 0

    for fact in facts:
        fact_id = fact.get("id")
        text = fact.get("text", "")
        embedding = fact.get("embedding")

        if not fact_id or not text or not embedding:
            skipped += 1
            continue

        # Check if already migrated
        existing = store.get_memory(fact_id)
        if existing:
            skipped += 1
            continue

        timestamp = fact.get("timestamp", "")
        conversation_id = fact.get("conversation_id", 0)
        message_ids = fact.get("message_ids", [])

        metadata = {
            "memory_type": "semantic",
            "importance": 0.5,
            "created_at": timestamp,
            "last_accessed": timestamp,
            "access_count": 0,
            "source_conversation_id": str(conversation_id) if conversation_id else "",
            "consolidated": False,
            "strength": 0.5,
            "message_ids": json.dumps(message_ids),
        }

        store.add_memory(
            memory_id=fact_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )
        migrated += 1

    total_in_qdrant = store.count()
    print(f"Migrated: {migrated}")
    print(f"Skipped (duplicate/invalid): {skipped}")
    print(f"Total in Qdrant: {total_in_qdrant}")
    print(f"Total in JSON: {len(facts)}")

    if total_in_qdrant == len(facts):
        print("SUCCESS: Counts match.")
    else:
        print(f"WARNING: Count mismatch (Qdrant={total_in_qdrant}, JSON={len(facts)})")


if __name__ == "__main__":
    migrate()
