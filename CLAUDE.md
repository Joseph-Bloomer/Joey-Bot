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

The application auto-creates `instance/joeybot.db` (SQLite) and `instance/semantic_memory.json` (vector store) on first run.

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
│   │   └── memory_service.py    # Semantic memory, vector store ops
│   └── data/
│       └── database.py          # SQLAlchemy models
├── utils/
│   └── logger.py                # Logging with token tracking
├── static/                      # Frontend assets
├── templates/                   # Jinja2 templates
└── instance/                    # Runtime data (DB, vector store)
```

### Backend Services

**LLM Layer (`app/models/`):**
- `BaseLLM` - Abstract interface for LLM providers
- `OllamaWrapper` - LiteLLM-based implementation (easily swap to OpenAI/Anthropic)

**Services (`app/services/`):**
- `ChatService` - Orchestrates chat: context building, streaming responses, auto-summarization
- `MemoryService` - Vector store operations, fact extraction, semantic retrieval

**Data Layer (`app/data/database.py`):**
- `Conversation` - Chat sessions with rolling summaries
- `Message` - Individual messages with role (user/assistant)
- `MemoryMarker` - Summarization checkpoints
- `UserProfile` - Persistent user context (name, details)
- `TokenUsage` - Analytics (tokens, speed, duration)
- `ThreadView` - Multi-user/view support

**Configuration (`config.py`):**
- Model settings, thresholds, database URI, paths

**Prompts (`app/prompts/chat_templates.yaml`):**
- `fact_extraction` - Extract permanent facts from conversations
- `rolling_summary` - Update conversation summaries
- `title_summary` - Generate chat titles
- `modes` - Concise/logic response prefixes

### Three-Tier Memory System

1. **Tier 1 - Recent Messages:** Last 8 raw messages for immediate context
2. **Tier 2 - Semantic Memory:** Vector store with fact embeddings (`nomic-embed-text`), similarity threshold 0.92, returns top 3 matches
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
- **Models Required:** `gemma3:4b` (chat), `nomic-embed-text` (embeddings)

## File Notes

- `static/stlye.css` - Note the typo in filename (stlye vs style)
- `instance/semantic_memory.json` - Can grow large; contains all extracted facts with embeddings
- `app.py.bak` - Backup of original monolithic implementation
- `logs/` - Application logs (gitignored)
