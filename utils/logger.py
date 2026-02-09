"""Logging configuration for Joey-Bot."""

import logging
import os
from datetime import datetime


def setup_logging(log_dir: str = 'logs', level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with file and console handlers.

    Args:
        log_dir: Directory to store log files
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('joeybot')
    logger.setLevel(level)

    # Prevent duplicate handlers on multiple calls
    if logger.handlers:
        return logger

    # Custom formatter with timestamps
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (daily rotation by filename)
    log_filename = os.path.join(log_dir, f'joeybot_{datetime.now():%Y%m%d}.log')
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'joeybot') -> logging.Logger:
    """Get a logger instance by name."""
    return logging.getLogger(name)


def log_token_usage(
    logger: logging.Logger,
    tokens_output: int,
    tokens_per_second: float,
    duration_ms: int,
    conversation_id: int = None
) -> None:
    """
    Log token usage metrics.

    Args:
        logger: Logger instance
        tokens_output: Number of tokens generated
        tokens_per_second: Generation speed
        duration_ms: Time taken in milliseconds
        conversation_id: Optional conversation ID
    """
    conv_str = f"conv={conversation_id}" if conversation_id else "conv=unsaved"
    logger.info(
        f"Token usage: {tokens_output} tokens | "
        f"{tokens_per_second:.1f} tok/s | "
        f"{duration_ms}ms | {conv_str}"
    )


def log_memory_operation(
    logger: logging.Logger,
    operation: str,
    facts_count: int = 0,
    conversation_id: int = None
) -> None:
    """
    Log semantic memory operations.

    Args:
        logger: Logger instance
        operation: Type of operation (extract, store, retrieve)
        facts_count: Number of facts involved
        conversation_id: Optional conversation ID
    """
    conv_str = f"conv={conversation_id}" if conversation_id else ""
    logger.info(f"Memory [{operation}]: {facts_count} facts {conv_str}".strip())
