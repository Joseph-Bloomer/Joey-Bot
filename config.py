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
VECTOR_STORE_PATH = os.path.join(INSTANCE_PATH, 'semantic_memory.json')

# Context building
MAX_RECENT_MESSAGES = 8
UNSUMMARIZED_THRESHOLD = 8  # Trigger summarization after this many unsummarized messages
MIN_MESSAGES_TO_SUMMARIZE = 5

# Summary constraints
SUMMARY_MIN_WORDS = 150
SUMMARY_MAX_WORDS = 300

# Fact extraction
MIN_FACT_LENGTH = 10

# Server configuration
DEBUG = True
PORT = 5000
