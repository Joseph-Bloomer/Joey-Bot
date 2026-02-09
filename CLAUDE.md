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
python app.py                     # Runs on http://localhost:5000
```

The application auto-creates `instance/joeybot.db` (SQLite) and `instance/semantic_memory.json` (vector store) on first run.

## Architecture

### Backend (app.py)

**Database Models (SQLAlchemy):**
- `Conversation` - Chat sessions with rolling summaries
- `Message` - Individual messages with role (user/assistant)
- `MemoryMarker` - Summarization checkpoints
- `UserProfile` - Persistent user context (name, details)
- `TokenUsage` - Analytics (tokens, speed, duration)

**Three-Tier Memory System:**
1. **Tier 1 - Recent Messages:** Last 8 raw messages for immediate context
2. **Tier 2 - Semantic Memory:** Vector store with fact embeddings (`nomic-embed-text`), similarity threshold 0.92, returns top 3 matches
3. **Tier 3 - Rolling Summary:** Auto-generated 150-300 word summary after 8+ unsummarized messages

**Key Functions:**
- `build_context_v2()` - Assembles full prompt from all memory tiers
- `auto_summarize_conversation()` - Triggers rolling summary updates
- `process_semantic_memory()` - Extracts and stores permanent facts
- `get_embedding()` / `get_long_term_memory()` - Vector operations via Ollama

**API Endpoints:**
- `POST /chat` - Main chat endpoint with SSE streaming
- `GET/POST /user-profile` - User profile management
- `GET/DELETE /conversation/<id>` - Conversation CRUD
- `POST /save-chat` - Save with AI-generated title/summary
- `POST /message` - Save message, triggers auto-summarization

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
- **Models Required:** `gemma3:4b` (chat), `nomic-embed-text` (embeddings)

## File Notes

- `static/stlye.css` - Note the typo in filename (stlye vs style)
- `instance/semantic_memory.json` - Can grow large; contains all extracted facts with embeddings
