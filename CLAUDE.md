# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Joey-Bot is a Flask-based AI chatbot with a three-tier semantic memory system. It provides a web interface for conversing with a local Gemma3 4B model via Ollama, featuring conversation persistence, token tracking, and intelligent context management.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with required models
ollama serve                      # Start Ollama (port 11434)
ollama pull gemma3:4b             # AI model
ollama pull nomic-embed-text      # Embedding model

# Start the application
python main.py                    # Runs on http://localhost:5000
```

The application auto-creates `instance/joeybot.db` (SQLite) and `instance/vectordb/` (Qdrant vector store) on first run.

## Architecture

### Project Structure

```
Joey-Bot/
├── main.py                      # Flask app entry point, routes
├── config.py                    # Centralized configuration
├── app/
│   ├── models/
│   │   ├── base.py              # BaseLLM abstract base class
│   │   └── ollama_wrapper.py    # LiteLLM-based Ollama wrapper
│   ├── prompts/
│   │   └── chat_templates.yaml  # All prompt templates
│   ├── services/
│   │   ├── chat_service.py      # Chat orchestration, context building
│   │   ├── memory_service.py    # Semantic memory, vector store ops
│   │   ├── retrieval.py         # Hybrid dense+BM25 search with RRF fusion
│   │   └── gatekeeper.py        # Memory need classifier (skip unnecessary retrieval)
│   └── data/
│       ├── database.py          # SQLAlchemy models
│       └── vector_store.py      # Qdrant vector store wrapper
├── scripts/
│   └── migrate_json_to_qdrant.py  # Migration from legacy JSON to Qdrant
├── utils/
│   └── logger.py                # Logging with token tracking
├── static/                      # Frontend assets
├── templates/                   # Jinja2 templates
└── instance/                    # Runtime data (DB, vectordb/)
```

### Backend Services

**LLM Layer (`app/models/`):**
- `BaseLLM` - Abstract interface for LLM providers
- `OllamaWrapper` - LiteLLM-based implementation (easily swap to OpenAI/Anthropic)

**Services (`app/services/`):**
- `ChatService` - Orchestrates chat: context building, streaming responses, auto-summarization
- `MemoryService` - Vector store operations, fact extraction, semantic retrieval
- `HybridRetriever` - Combines Qdrant dense vector search with BM25 keyword search, fuses results via Reciprocal Rank Fusion (k=60). BM25 index is built lazily on first search and rebuilt when new memories are stored. Logs retrieval diagnostics (dense-only, sparse-only, both).
- `MemoryGatekeeper` - Classifies incoming messages to decide if memory retrieval is needed (NONE, RECENT, SEMANTIC, PROFILE, MULTI). Fail-open: defaults to SEMANTIC on error. Returns `retrieval_keys` passed as extra BM25 keywords.

**Data Layer (`app/data/`):**
- `database.py` - SQLAlchemy models: `Conversation`, `Message`, `MemoryMarker`, `UserProfile`, `TokenUsage`, `ThreadView`
- `vector_store.py` - `VectorStore` class wrapping Qdrant local mode with lazy initialization. Methods: `add_memory`, `search`, `get_memory`, `update_metadata`, `delete_memory`, `get_all`, `count`. Converts SHA256 fact hashes to UUIDs for Qdrant point IDs.

**Configuration (`config.py`):**
- Model settings, thresholds, database URI, paths

**Prompts (`app/prompts/chat_templates.yaml`):**
- `fact_extraction` - Extract permanent facts from conversations
- `rolling_summary` - Update conversation summaries
- `title_summary` - Generate chat titles
- `modes` - Concise/logic response prefixes

### Three-Tier Memory System

1. **Tier 1 - Recent Messages:** Last 8 raw messages for immediate context
2. **Tier 2 - Semantic Memory:** Hybrid retrieval over Qdrant vector store. Dense search (cosine similarity, `nomic-embed-text`, 768 dims) and sparse BM25 keyword search run in parallel, fused via RRF. Similarity threshold 0.92 for deduplication, returns top 3 matches. Each memory carries metadata: `memory_type`, `importance`, `access_count`, `strength`, `created_at`, `last_accessed`, `source_conversation_id`, `consolidated`.
3. **Tier 3 - Rolling Summary:** Auto-generated 150-300 word summary after 8+ unsummarized messages

### API Endpoints (main.py)

- `POST /chat` - Main chat endpoint with SSE streaming
- `GET/POST /user-profile` - User profile management
- `GET/DELETE /conversation/<id>` - Conversation CRUD
- `POST /save-chat` - Save with AI-generated title/summary
- `POST /message` - Save message, triggers auto-summarization
- `GET /semantic-memory-stats` - Vector store statistics
- `GET /usage-stats` - Token usage analytics

### Frontend (static/script.js, templates/index.html)

**State:**
- `currentConversationId` - null for unsaved chats
- `conversationHistory` - In-memory message array
- `currentMode` - normal/concise/logic response modes

**Key Functions:**
- `sendMessage()` - Handles chat with SSE streaming
- `appendStreamingMessage()` / `appendToStreamingMessage()` - Real-time token display
- `setMode()` - Switches response style prefixes
- Token tracking functions for analytics

## External Dependencies

- **Ollama API:** `http://localhost:11434/api/` for model inference and embeddings
- **LiteLLM:** Provider abstraction layer (can switch to OpenAI, Anthropic, etc.)
- **Qdrant:** Local-mode vector database (`qdrant-client`, no server needed). Data persisted to `instance/vectordb/`
- **rank-bm25:** BM25Okapi sparse keyword search, used alongside dense vectors in `HybridRetriever`
- **Models Required:** `gemma3:4b` (chat), `nomic-embed-text` (embeddings)

## File Notes

- `static/stlye.css` - Note the typo in filename (stlye vs style)
- `instance/vectordb/` - Qdrant persistent storage (auto-created on first run)
- `instance/semantic_memory.json` - Legacy JSON vector store (kept as backup, no longer used at runtime)
- `app.py.bak` - Backup of original monolithic implementation
- `logs/` - Application logs (gitignored)

## Migration

To migrate existing memories from the legacy JSON format to Qdrant:
```bash
python scripts/migrate_json_to_qdrant.py
```
