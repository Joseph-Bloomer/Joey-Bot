"""Memory gatekeeper service for classifying message memory needs."""

import json
import re
import time
from typing import Dict, Any, List, Optional

from app.models.base import BaseLLM
from app.prompts import format_memory_classification_prompt
from utils.logger import get_logger

logger = get_logger()


class MemoryGatekeeper:
    """
    Classifies incoming messages to determine if memory retrieval is needed.

    Prevents unnecessary vector store searches for messages like greetings
    or simple questions that don't benefit from semantic memory context.
    """

    VALID_CATEGORIES = {"NONE", "RECENT", "SEMANTIC", "PROFILE", "MULTI"}

    def __init__(
        self,
        llm: BaseLLM,
        prompts: Dict[str, Any],
        max_tokens: int = 100,
        timeout: float = 3.0
    ):
        self.llm = llm
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.timeout = timeout

    def classify(
        self,
        user_message: str,
        recent_messages: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Classify a user message's memory retrieval needs.

        Args:
            user_message: The current user message
            recent_messages: Last 2-3 messages for context

        Returns:
            Dict with memory_need, retrieval_keys, and confidence
        """
        recent_context = self._format_recent(recent_messages or [])

        prompt = format_memory_classification_prompt(
            self.prompts, user_message, recent_context
        )

        start = time.time()
        try:
            raw = self.llm.generate_json(prompt, max_tokens=self.max_tokens)
            elapsed_ms = (time.time() - start) * 1000

            if raw is None:
                logger.warning(f"Gatekeeper: LLM returned None ({elapsed_ms:.0f}ms), using fail-open default")
                return self._default_result()

            result = self._parse_response(raw)
            logger.info(
                f"Gatekeeper: {result['memory_need']} "
                f"(conf={result['confidence']:.2f}, {elapsed_ms:.0f}ms)"
            )
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.warning(f"Gatekeeper error ({elapsed_ms:.0f}ms): {e}, using fail-open default")
            return self._default_result()

    def _format_recent(self, messages: List[Dict[str, str]]) -> str:
        """Format recent messages into a context string."""
        if not messages:
            return "(no recent messages)"
        lines = []
        for msg in messages[-3:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)

    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Extract classification JSON from model response.

        Handles markdown code blocks and bare JSON.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r'```(?:json)?\s*', '', raw_response)
        cleaned = cleaned.strip().rstrip('`')

        # Find JSON object
        match = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
        if not match:
            logger.warning(f"Gatekeeper: no JSON found in response: {raw_response[:200]}")
            return self._default_result()

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning(f"Gatekeeper: invalid JSON in response: {raw_response[:200]}")
            return self._default_result()

        memory_need = parsed.get("memory_need", "SEMANTIC").upper()
        if memory_need not in self.VALID_CATEGORIES:
            memory_need = "SEMANTIC"

        retrieval_keys = parsed.get("retrieval_keys", [])
        if not isinstance(retrieval_keys, list):
            retrieval_keys = []

        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "memory_need": memory_need,
            "retrieval_keys": retrieval_keys,
            "confidence": confidence
        }

    def _default_result(self) -> Dict[str, Any]:
        """Fail-open default: assume SEMANTIC needed."""
        return {
            "memory_need": "SEMANTIC",
            "retrieval_keys": [],
            "confidence": 0.0
        }
