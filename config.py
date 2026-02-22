"""Centralized configuration for Joey-Bot."""

import os

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434/api"

# Model configuration
CHAT_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"

# LiteLLM model strings (prefixed for provider routing)
LITELLM_CHAT_MODEL = f"ollama/{CHAT_MODEL}"
LITELLM_EMBEDDING_MODEL = f"ollama/{EMBEDDING_MODEL}"

# Semantic memory configuration
SEMANTIC_MEMORY_ENABLED = True
SEMANTIC_RESULTS_COUNT = 3
SIMILARITY_THRESHOLD = 0.92

# Database configuration
DATABASE_URI = "sqlite:///joeybot.db"

# Instance path (for runtime-generated files)
INSTANCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
VECTOR_STORE_PATH = os.path.join(INSTANCE_PATH, 'semantic_memory.json')  # Legacy JSON (kept for migration)

# Qdrant configuration
QDRANT_PERSIST_DIR = os.path.join(INSTANCE_PATH, 'vectordb')
QDRANT_COLLECTION_NAME = "semantic_memory"
QDRANT_VECTOR_SIZE = 768  # nomic-embed-text embedding dimension

# Context building
MAX_RECENT_MESSAGES = 8
UNSUMMARIZED_THRESHOLD = 8  # Trigger summarization after this many unsummarized messages
MIN_MESSAGES_TO_SUMMARIZE = 5

# Summary constraints
SUMMARY_MIN_WORDS = 150
SUMMARY_MAX_WORDS = 300

# Fact extraction
MIN_FACT_LENGTH = 10

# Reranker configuration
RERANKER_ENABLED = True

# Memory lifecycle configuration
LIFECYCLE_ENABLED = True
LIFECYCLE_DECAY_HOURS = 6          # Run strength decay every N hours
LIFECYCLE_CONSOLIDATE_HOURS = 24   # Run consolidation every N hours
LIFECYCLE_PRUNE_DAYS = 7           # Run pruning every N days

# Gatekeeper configuration
GATEKEEPER_ENABLED = True
GATEKEEPER_TIMEOUT = 3.0       # seconds
GATEKEEPER_MAX_TOKENS = 100    # limit classification response length

# Embedding model filter patterns (models to exclude from chat model dropdown)
EMBEDDING_MODEL_PATTERNS = ['nomic-embed', 'all-minilm', 'bge-', 'e5-']

# Server configuration
DEBUG = True
PORT = 5000

# --- Memory Retrieval ---
GATEKEEPER_CONTEXT_WINDOW = 3          # Recent messages sent to gatekeeper for classification
                                        # Also used by orchestrator for context building

# --- Memory Lifecycle ---
LIFECYCLE_DECAY_RATE = 0.05            # Exponential decay rate for strength calculation
LIFECYCLE_ACCESS_CAP = 10.0            # Maximum access count for strength scoring
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.8  # Cosine similarity threshold for clustering

# --- Qdrant ---
QDRANT_SCROLL_BATCH_SIZE = 100         # Batch size for scrolling through all memories

# --- Pipeline ---
MAX_PIPELINE_HISTORY = 10              # Pipeline runs kept in memory for dashboard

# --- Logging ---
LOG_LEVEL = "INFO"                     # Application log level

# --- LLM Generation ---
JSON_GENERATION_TEMPERATURE = 0.3      # Temperature for structured JSON output tasks
