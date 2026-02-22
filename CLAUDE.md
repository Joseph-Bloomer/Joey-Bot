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
│   │   ├── orchestrator.py      # 6-stage chat pipeline orchestrator
│   │   ├── reranker.py          # Heuristic scoring of retrieval candidates
│   │   ├── memory_lifecycle.py  # Background maintenance: decay, consolidation, pruning
│   │   ├── chat_service.py      # Prompt assembly, LLM generation, conversation management
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
├── static/
│   ├── script.js                # Chat UI logic
│   ├── stlye.css                # Chat UI styles (note typo in name)
│   ├── dashboard.js             # Pipeline dashboard polling and rendering
│   └── dashboard.css            # Pipeline dashboard styles
├── templates/
│   ├── index.html               # Main chat interface
│   └── dashboard.html           # Pipeline observability dashboard
└── instance/                    # Runtime data (DB, vectordb/)
```

### Backend Services

**LLM Layer (`app/models/`):**
- `BaseLLM` - Abstract interface for LLM providers
- `OllamaWrapper` - LiteLLM-based implementation (easily swap to OpenAI/Anthropic)

**Services (`app/services/`):**
- `ChatOrchestrator` - Manages the chat pipeline end-to-end. Receives a user message from the `/chat` route, runs it through 6 sequential stages, and yields SSE tokens. Delegates classification to `MemoryGatekeeper`, retrieval to `HybridRetriever`, prompt assembly to `ChatService.build_prompt()`, and generation to the LLM. Each stage has isolated error handling and `time.perf_counter()` timing. Exposes `get_pipeline_metadata()` for diagnostics (timings, errors, classification, candidate counts).
- `ChatService` - Generation service: prompt assembly (`build_prompt()`), auto-summarization (`auto_summarize()`), and conversation save (`save_chat_with_metadata()`). Has a legacy `generate_response()` wrapper for backwards compat (logs a deprecation warning). No longer owns retrieval or gatekeeper logic — those moved to `ChatOrchestrator`.
- `HeuristicReranker` - Scores retrieval candidates using a weighted composite of signals already in the metadata — no LLM calls, pure computation (<1ms). Composite = retrieval_relevance (0.45) + recency (0.20) + importance (0.20) + usage (0.10) + type_boost (0.05). Recency uses exponential decay `exp(-0.03 * days_old)`. Usage caps at 5 accesses. Procedural memories get +0.10 boost, preferences +0.05. Weights stored as class attributes for easy tuning. Each scored candidate gets a `score_breakdown` dict for diagnostics.
- `MemoryLifecycle` - Background maintenance for semantic memories (never runs during chat). Three operations on schedules: **Strength decay** (every 6h) recalculates strength = importance(0.40) + recency(0.35) + access(0.25) where recency = exp(-0.05 * days_old). **Consolidation** (every 24h) clusters weak memories (strength < 0.3) by cosine similarity > 0.8, merges clusters of 3+ via LLM into a single consolidated memory, marks originals. Procedural memories are never consolidated. **Pruning** (weekly) deletes consolidated originals with strength < 0.1 and never-accessed memories with strength < 0.05. Protects procedural and high-importance (>= 0.9) memories. Manual trigger via `POST /memory-lifecycle` with action: decay/consolidate/prune/full.
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
- `memory_consolidation` - Merge related weak memories into one
- `modes` - Concise/logic response prefixes

### Pipeline Flow (`ChatOrchestrator.process_message`)

The `/chat` route loads recent messages then delegates to the orchestrator, which runs 6 stages sequentially:

1. **CLASSIFY** — `MemoryGatekeeper.classify()` determines memory need (NONE, RECENT, SEMANTIC, PROFILE, MULTI). If gatekeeper is disabled or errors, defaults to SEMANTIC (fail-open).
2. **RETRIEVE** — Skipped when memory_need is NONE/RECENT/PROFILE. Otherwise builds a query from `retrieval_keys` (or falls back to the raw message), gets an embedding, and calls `HybridRetriever.search()` for top-3 candidates.
3. **SCORE** — `HeuristicReranker.rerank()` scores each candidate with a weighted composite (retrieval relevance 0.45, recency 0.20, importance 0.20, usage 0.10, type boost 0.05), sorts descending, and trims to top-k. Logs per-candidate breakdowns. Falls back to passthrough if `RERANKER_ENABLED=False` or on error.
4. **BUILD_CONTEXT** — Assembles the LLM prompt via `ChatService.build_prompt()` from: mode prefix, user profile, memory candidates, rolling summary, recent messages, and current turn. Falls back to a minimal `"User: ...\nAssistant:"` prompt on error.
5. **GENERATE** — Streams tokens from the LLM as SSE events (`{"token": "..."}`) and accumulates the full response. Yields `{"done": true}` at end.
6. **POST_PROCESS** — Fire-and-forget. Updates `access_count`/`last_accessed` on retrieved memories, then extracts new facts from the exchange via `MemoryService.process_semantic_memory()`. Errors here never affect the user response.

Each stage logs timing: `[CLASSIFY] SEMANTIC (conf=0.80, 2005.3ms)`, etc. A summary line is logged at the end: `[PIPELINE] Total=5200.0ms | classify=... | retrieve=... | score=... | build_context=... | generate=... | post_process=...`

### Three-Tier Memory System

1. **Tier 1 - Recent Messages:** Last 8 raw messages for immediate context
2. **Tier 2 - Semantic Memory:** Hybrid retrieval over Qdrant vector store. Dense search (cosine similarity, `nomic-embed-text`, 768 dims) and sparse BM25 keyword search run in parallel, fused via RRF. Similarity threshold 0.92 for deduplication, returns top 3 matches. Each memory carries metadata: `memory_type`, `importance`, `access_count`, `strength`, `created_at`, `last_accessed`, `source_conversation_id`, `consolidated`.
3. **Tier 3 - Rolling Summary:** Auto-generated 150-300 word summary after 8+ unsummarized messages

### API Endpoints (main.py)

- `POST /chat` - Main chat endpoint with SSE streaming (records pipeline run after streaming)
- `GET /dashboard` - Pipeline observability dashboard page
- `GET /api/pipeline-runs` - Recent pipeline run history (up to 10, in-memory)
- `GET /api/pipeline-latest` - Most recent pipeline run (or null)
- `GET/POST /user-profile` - User profile management
- `GET/DELETE /conversation/<id>` - Conversation CRUD
- `POST /save-chat` - Save with AI-generated title/summary
- `POST /message` - Save message, triggers auto-summarization
- `GET /semantic-memory-stats` - Vector store statistics (includes strength tiers, consolidated count)
- `POST /memory-lifecycle` - Manual lifecycle trigger (action: decay/consolidate/prune/full)
- `GET /usage-stats` - Token usage analytics

### Pipeline Observability Dashboard (`/dashboard`)

Standalone page for debugging and demonstrating the pipeline. Three sections:

1. **Pipeline Inspector** — Shows the 6 stage cards (classify, retrieve, score, build_context, generate, post_process) from the most recent run. Each card displays timing (ms), status badge (success/error/skipped), and detail text. Color-coded borders: green=success, red=error, gray=skipped.
2. **Memory Health** — Stat cards for Total, Strong, Medium, Weak, and Consolidated memory counts (from `/semantic-memory-stats`). Lifecycle action buttons (Decay, Consolidate, Prune, Full Cycle) trigger `POST /memory-lifecycle`.
3. **Recent Pipeline Runs** — Table of up to 10 recent runs with timestamp, message preview, classification badge, candidate count, total time, and error count. Classification badges color-coded per category (NONE=gray, RECENT=blue, SEMANTIC=green, PROFILE=amber, MULTI=purple).

Auto-refreshes every 5 seconds (toggleable). Pipeline history is in-memory only — not persisted across restarts.

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
- **APScheduler:** Background task scheduling for memory lifecycle (decay every 6h, consolidation every 24h, pruning weekly)
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
