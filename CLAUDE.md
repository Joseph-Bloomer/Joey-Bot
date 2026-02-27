# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Joey-Bot is a Flask-based AI chatbot with a three-tier semantic memory system. It supports hybrid local/cloud model routing — a local Gemma3 4B model via Ollama handles pipeline tasks (classification, embeddings, fact extraction), while the user can select any configured model (local or cloud) for chat generation. Features conversation persistence, per-model token tracking, and intelligent context management.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with required models
ollama serve                      # Start Ollama (port 11434)
ollama pull gemma3:4b             # AI model
ollama pull nomic-embed-text      # Embedding model

# (Optional) Configure API keys for cloud models and web search
cp .env.example .env              # Then edit .env with your API keys

# Start the application
python main.py                    # Runs on http://localhost:5000
```

The application auto-creates `instance/joeybot.db` (SQLite) and `instance/vectordb/` (Qdrant vector store) on first run.

**Cloud models** require API keys in `.env` (e.g. `GEMINI_API_KEY` for Gemini 2.0 Flash). See `.env.example` for the template. Cloud models are optional — the app works fully with just the local Ollama model. Only models with valid API keys appear in the model selector.

**Web search** requires a Tavily API key (`TAVILY_API_KEY` in `.env`). The feature is optional — the app works normally without it, and the search toggle will be non-functional.

## Architecture

### Project Structure

```
Joey-Bot/
├── main.py                      # Flask app entry point, routes
├── config.py                    # Centralized configuration
├── app/
│   ├── models/
│   │   ├── base.py              # BaseLLM abstract base class
│   │   ├── ollama_wrapper.py    # LiteLLM-based Ollama wrapper (local)
│   │   └── cloud_wrapper.py     # LiteLLM-based cloud provider wrapper
│   ├── prompts/
│   │   └── chat_templates.yaml  # All prompt templates
│   ├── services/
│   │   ├── orchestrator.py      # 7-stage chat pipeline orchestrator
│   │   ├── reranker.py          # Heuristic scoring of retrieval candidates
│   │   ├── memory_lifecycle.py  # Background maintenance: decay, consolidation, pruning
│   │   ├── chat_service.py      # Prompt assembly, LLM generation, conversation management
│   │   ├── memory_service.py    # Semantic memory, vector store ops
│   │   ├── retrieval.py         # Hybrid dense+BM25 search with RRF fusion
│   │   ├── web_search.py        # Tavily web search (ephemeral, results not stored)
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
- `BaseLLM` - Abstract interface for LLM providers (`generate`, `get_embedding`, `generate_json`)
- `OllamaWrapper` - LiteLLM-based local model implementation. Handles chat generation, embeddings, and JSON generation. Used as the local LLM for all pipeline stages (classification, embeddings, fact extraction, summarization) regardless of which model the user selects for chat.
- `CloudWrapper` - LiteLLM-based cloud provider wrapper. Implements `generate()` (streaming and non-streaming) via `litellm.completion()`. Raises `CloudGenerationError` (not caught internally) with typed `error_type` (auth_error, rate_limit, network_error, unknown) so the orchestrator can emit structured SSE error events. `get_embedding()` and `generate_json()` raise `NotImplementedError` — cloud models are only used for chat generation, never for pipeline tasks.

**Services (`app/services/`):**
- `ChatOrchestrator` - Manages the chat pipeline end-to-end. Constructor takes `local_llm` (for pipeline stages) and `model_registry` (dict mapping display names to BaseLLM instances). `process_message()` accepts an optional `model` param — `_resolve_llm(model)` looks up the registry and falls back to `local_llm` with a warning if the name is unknown. Stages 1-5 and 7 always use the local LLM (via `chat_service.llm` / `memory_service.llm`); only stage 6 (GENERATE) uses the user-selected model. Catches `CloudGenerationError` in GENERATE to yield a `{"cloud_error": true, "error_type": "...", "message": "...", "model": "..."}` SSE event and sets `ctx["cloud_failed"] = True` to skip POST_PROCESS. Each stage has isolated error handling and `time.perf_counter()` timing. Exposes `get_pipeline_metadata()` for diagnostics.
- `ChatService` - Generation service: prompt assembly (`build_prompt()`), auto-summarization (`auto_summarize()`), and conversation save (`save_chat_with_metadata()`). Has a legacy `generate_response()` wrapper for backwards compat (logs a deprecation warning). No longer owns retrieval or gatekeeper logic — those moved to `ChatOrchestrator`.
- `HeuristicReranker` - Scores retrieval candidates using a weighted composite of signals already in the metadata — no LLM calls, pure computation (<1ms). Composite = retrieval_relevance (0.45) + recency (0.20) + importance (0.20) + usage (0.10) + type_boost (0.05). Recency uses exponential decay `exp(-0.03 * days_old)`. Usage caps at 5 accesses. Procedural memories get +0.10 boost, preferences +0.05. Weights stored as class attributes for easy tuning. Each scored candidate gets a `score_breakdown` dict for diagnostics.
- `MemoryLifecycle` - Background maintenance for semantic memories (never runs during chat). Three operations on schedules: **Strength decay** (every 6h) recalculates strength = importance(0.40) + recency(0.35) + access(0.25) where recency = exp(-0.05 * days_old). **Consolidation** (every 24h) clusters weak memories (strength < 0.3) by cosine similarity > 0.8, merges clusters of 3+ via LLM into a single consolidated memory, marks originals. Procedural memories are never consolidated. **Pruning** (weekly) deletes consolidated originals with strength < 0.1 and never-accessed memories with strength < 0.05. Protects procedural and high-importance (>= 0.9) memories. Manual trigger via `POST /memory-lifecycle` with action: decay/consolidate/prune/full.
- `MemoryService` - Vector store operations, fact extraction, semantic retrieval
- `HybridRetriever` - Combines Qdrant dense vector search with BM25 keyword search, fuses results via Reciprocal Rank Fusion (k=60). BM25 index is built lazily on first search and rebuilt when new memories are stored. Logs retrieval diagnostics (dense-only, sparse-only, both).
- `WebSearchService` - Ephemeral web search via Tavily API. Results are used for one response then discarded — never stored in Qdrant or SQLite. Methods: `search(query)` returns a structured dict (results, error, count) and never raises (fail-open). `format_for_prompt(results)` produces a `[Web Search Results]` text block for injection into the LLM prompt. `format_sources_for_response(results)` produces markdown source links. Respects `WEB_SEARCH_MAX_CHARS_PER_RESULT` and `WEB_SEARCH_MAX_TOTAL_CHARS` truncation limits.
- `MemoryGatekeeper` - Classifies incoming messages to decide if memory retrieval is needed (NONE, RECENT, SEMANTIC, PROFILE, MULTI, WEB_SEARCH). Fail-open: defaults to SEMANTIC on error. Returns `retrieval_keys` passed as extra BM25 keywords. When classifying as WEB_SEARCH, retrieval_keys contain concise search engine keywords.

**Data Layer (`app/data/`):**
- `database.py` - SQLAlchemy models: `Conversation`, `Message`, `MemoryMarker`, `UserProfile`, `TokenUsage`, `ThreadView`
- `vector_store.py` - `VectorStore` class wrapping Qdrant local mode with lazy initialization. Methods: `add_memory`, `search`, `get_memory`, `update_metadata`, `delete_memory`, `get_all`, `count`. Converts SHA256 fact hashes to UUIDs for Qdrant point IDs.

**Configuration (`config.py`):**
- Model settings, thresholds, database URI, paths
- Cloud model settings: `CLOUD_MODELS` (dict of display_name → provider, model, api_key_env, cost_per_1k_input, cost_per_1k_output), `CLOUD_ENABLED` (true if any cloud API key is set), `LOCAL_MODEL_DISPLAY_NAME` ("Gemma 3 4B (Local)")
- Web search settings: `TAVILY_API_KEY`, `WEB_SEARCH_ENABLED`, `WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_MAX_CHARS_PER_RESULT`, `WEB_SEARCH_MAX_TOTAL_CHARS`, `WEB_SEARCH_TIMEOUT`, `WEB_SEARCH_DEPTH`
- Uses `python-dotenv` to load `.env` file (API keys should never be committed; `.env` is in `.gitignore`)

**Prompts (`app/prompts/chat_templates.yaml`):**
- `fact_extraction` - Extract permanent facts from conversations
- `rolling_summary` - Update conversation summaries
- `title_summary` - Generate chat titles
- `memory_consolidation` - Merge related weak memories into one
- `modes` - Concise/logic response prefixes

### Pipeline Flow (`ChatOrchestrator.process_message`)

The `/chat` route loads recent messages then delegates to the orchestrator, which runs 7 stages sequentially:

1. **CLASSIFY** — `MemoryGatekeeper.classify()` determines memory need (NONE, RECENT, SEMANTIC, PROFILE, MULTI, WEB_SEARCH). If gatekeeper is disabled or errors, defaults to SEMANTIC (fail-open).
2. **WEB_SEARCH** — Controlled by `search_mode` parameter ("off"/"auto"/"on"). In "auto" mode, runs only when gatekeeper classifies as WEB_SEARCH. Builds a search query from `retrieval_keys` (or falls back to the raw message), calls `WebSearchService.search()`, and yields a `{"searching": true}` SSE event before the HTTP request so the frontend can show an indicator. Results are ephemeral — used for prompt injection only, never stored. Fail-open: errors set web_results to None and the pipeline continues.
3. **RETRIEVE** — Skipped when memory_need is NONE/RECENT/PROFILE. Otherwise builds a query from `retrieval_keys` (or falls back to the raw message), gets an embedding, and calls `HybridRetriever.search()` for top-3 candidates. Runs even when web search results exist (web provides current facts, memories provide personal context).
4. **SCORE** — `HeuristicReranker.rerank()` scores each candidate with a weighted composite (retrieval relevance 0.45, recency 0.20, importance 0.20, usage 0.10, type boost 0.05), sorts descending, and trims to top-k. Logs per-candidate breakdowns. Falls back to passthrough if `RERANKER_ENABLED=False` or on error.
5. **BUILD_CONTEXT** — Assembles the LLM prompt via `ChatService.build_prompt()` from: mode prefix, user profile, web search results (if any, with citation instruction), memory candidates, rolling summary, recent messages, and current turn. Falls back to a minimal `"User: ...\nAssistant:"` prompt on error.
6. **GENERATE** — Resolves the user-selected model via `_resolve_llm(model)` (falls back to local if unknown). Streams tokens from the resolved LLM as SSE events (`{"token": "..."}`) and accumulates the full response. Yields `{"done": true}` at end. If the LLM raises `CloudGenerationError`, yields a `{"cloud_error": true, "error_type": "...", "message": "...", "model": "..."}` SSE event instead, sets `ctx["cloud_failed"] = True`, and skips POST_PROCESS. The frontend shows a fallback prompt offering retry or switch-to-local options.
7. **POST_PROCESS** — Fire-and-forget. Skipped entirely when `cloud_failed` is set. Updates `access_count`/`last_accessed` on retrieved memories, then extracts new facts from the exchange via `MemoryService.process_semantic_memory()`. Web search results are NOT passed to post_process — they are ephemeral and never stored.

Each stage logs timing: `[CLASSIFY] SEMANTIC (conf=0.80, 2005.3ms)`, etc. A summary line is logged at the end: `[PIPELINE] Total=5200.0ms | classify=... | web_search=... | retrieve=... | score=... | build_context=... | generate=... | post_process=...`

### Three-Tier Memory System

1. **Tier 1 - Recent Messages:** Last 8 raw messages for immediate context
2. **Tier 2 - Semantic Memory:** Hybrid retrieval over Qdrant vector store. Dense search (cosine similarity, `nomic-embed-text`, 768 dims) and sparse BM25 keyword search run in parallel, fused via RRF. Similarity threshold 0.92 for deduplication, returns top 3 matches. Each memory carries metadata: `memory_type`, `importance`, `access_count`, `strength`, `created_at`, `last_accessed`, `source_conversation_id`, `consolidated`.
3. **Tier 3 - Rolling Summary:** Auto-generated 150-300 word summary after 8+ unsummarized messages

### API Endpoints (main.py)

- `POST /chat` - Main chat endpoint with SSE streaming. Accepts `model` (display name, default local), `search_mode` ("off"/"auto"/"on", default "auto") alongside existing params. SSE events: `{"token": "..."}`, `{"searching": true}`, `{"done": true}`, `{"error": "..."}`, `{"cloud_error": true, "error_type": "...", "message": "...", "model": "..."}`. Records pipeline run after streaming.
- `GET /api/available-models` - Lists all models in the registry (local + cloud with valid API keys). Each entry has `name` (display name) and `is_local` (boolean).
- `GET /dashboard` - Pipeline observability dashboard page
- `GET /api/pipeline-runs` - Recent pipeline run history (up to 10, in-memory)
- `GET /api/pipeline-latest` - Most recent pipeline run (or null)
- `GET/POST /user-profile` - User profile management
- `GET/DELETE /conversation/<id>` - Conversation CRUD
- `POST /save-chat` - Save with AI-generated title/summary
- `POST /message` - Save message, triggers auto-summarization
- `GET /semantic-memory-stats` - Vector store statistics (includes strength tiers, consolidated count)
- `POST /memory-lifecycle` - Manual lifecycle trigger (action: decay/consolidate/prune/full)
- `POST /token-usage` - Log token usage after AI response. Accepts `model_name` (defaults to local model display name).
- `GET /usage-stats` - Token usage analytics with per-model breakdown. Returns `per_model` dict keyed by display name with `tokens`, `requests`, `avg_speed`, `is_local`, `cost_per_1k_output`, `estimated_cost`.

### Pipeline Observability Dashboard (`/dashboard`)

Standalone page for debugging and demonstrating the pipeline. Three sections:

1. **Pipeline Inspector** — Shows the 7 stage cards (classify, web_search, retrieve, score, build_context, generate, post_process) from the most recent run. Each card displays timing (ms), status badge (success/error/skipped), and detail text. Color-coded borders: green=success, red=error, gray=skipped.
2. **Memory Health** — Stat cards for Total, Strong, Medium, Weak, and Consolidated memory counts (from `/semantic-memory-stats`). Lifecycle action buttons (Decay, Consolidate, Prune, Full Cycle) trigger `POST /memory-lifecycle`.
3. **Recent Pipeline Runs** — Table of up to 10 recent runs with timestamp, message preview, classification badge, candidate count, total time, and error count. Runs that used web search show a magnifying glass icon. Classification badges color-coded per category (NONE=gray, RECENT=blue, SEMANTIC=green, PROFILE=amber, MULTI=purple, WEB_SEARCH=sky blue).

Auto-refreshes every 5 seconds (toggleable). Pipeline history is in-memory only — not persisted across restarts.

### Frontend (static/script.js, templates/index.html)

**State:**
- `currentConversationId` - null for unsaved chats
- `conversationHistory` - In-memory message array
- `currentMode` - normal/concise/logic response modes
- `currentModel` - currently selected model display name (set on page load from `/models`)
- `currentSearchMode` - "off"/"auto"/"on" web search toggle (default "auto")

**Key Functions:**
- `sendMessage()` - Handles chat with SSE streaming, passes `search_mode` and `model` to `/chat`. Handles `cloud_error` SSE events.
- `appendStreamingMessage()` / `appendToStreamingMessage()` - Real-time token display
- `showSearchingIndicator()` / `removeSearchingIndicator()` - "Searching the web..." indicator shown on `{"searching": true}` SSE event, replaced by first token
- `setMode()` - Switches response style prefixes
- `setSearchMode()` - Switches web search toggle (Off/Auto/On)
- `logTokenUsage()` - Sends token usage to `/token-usage` including `model_name: currentModel`
- `loadUsageStats()` - Fetches `/usage-stats` and renders per-model breakdown cards with tokens, requests, avg speed, and cost label
- `formatText()` - Renders bold, italic, lists, and markdown links (`[text](url)` → clickable `<a>` tags)

**UI Elements:**
- Model selector — dropdown in the input footer (leftmost). Lists local Ollama models.
- Search toggle (Off/Auto/On) — segmented button group in the input footer, between mode selector and token speed. Controls whether web search is used per message.
- Per-model usage breakdown — in Settings > Usage section, shows a card per model with tokens, requests, avg tok/s, and cost (Free (local) / Free tier / actual cost).

## External Dependencies

- **Ollama API:** `http://localhost:11434/api/` for local model inference and embeddings. Always required — the local model handles all pipeline tasks (classification, embeddings, fact extraction, summarization) even when a cloud model is selected for chat.
- **LiteLLM:** Provider abstraction layer. Routes to Ollama for local models and to cloud APIs (Gemini, OpenAI, Anthropic, etc.) for cloud models via `litellm.completion()`.
- **google-generativeai:** Required by LiteLLM for Gemini model support. Installed as a dependency.
- **Qdrant:** Local-mode vector database (`qdrant-client`, no server needed). Data persisted to `instance/vectordb/`
- **rank-bm25:** BM25Okapi sparse keyword search, used alongside dense vectors in `HybridRetriever`
- **APScheduler:** Background task scheduling for memory lifecycle (decay every 6h, consolidation every 24h, pruning weekly)
- **Gemini API:** Cloud chat generation via `GEMINI_API_KEY` env var. Optional — app works fully with just the local Ollama model. Configured in `config.CLOUD_MODELS`.
- **Tavily API:** `https://api.tavily.com/search` for web search. Requires `TAVILY_API_KEY` env var. Optional — app works without it. Results are ephemeral (used once, never stored).
- **python-dotenv:** Loads `.env` file for API keys and other environment configuration
- **Models Required:** `gemma3:4b` (local chat + pipeline), `nomic-embed-text` (embeddings)

## File Notes

- `static/stlye.css` - Note the typo in filename (stlye vs style)
- `instance/vectordb/` - Qdrant persistent storage (auto-created on first run)
- `instance/semantic_memory.json` - Legacy JSON vector store (kept as backup, no longer used at runtime)
- `app.py.bak` - Backup of original monolithic implementation
- `.env` - Environment variables including `GEMINI_API_KEY`, `TAVILY_API_KEY` (gitignored, never committed)
- `.env.example` - Template for `.env` with placeholder values for all API keys
- `logs/` - Application logs (gitignored)

## Migration

To migrate existing memories from the legacy JSON format to Qdrant:
```bash
python scripts/migrate_json_to_qdrant.py
```
