"""Integration tests for MemoryGatekeeper.

Requires Ollama running with gemma3:4b model.
Run: python -m pytest tests/test_gatekeeper.py -v
"""

import pytest
from unittest.mock import MagicMock

from app.services.gatekeeper import MemoryGatekeeper
from app.prompts import load_prompts


# ============================================================================
# Unit tests (no LLM required)
# ============================================================================

class TestParseResponse:
    """Test _parse_response with various model output formats."""

    def setup_method(self):
        self.gk = MemoryGatekeeper(
            llm=MagicMock(),
            prompts=load_prompts(),
        )

    def test_clean_json(self):
        raw = '{"memory_need": "NONE", "retrieval_keys": [], "confidence": 0.95}'
        result = self.gk._parse_response(raw)
        assert result["memory_need"] == "NONE"
        assert result["confidence"] == 0.95
        assert result["retrieval_keys"] == []

    def test_markdown_wrapped_json(self):
        raw = '```json\n{"memory_need": "SEMANTIC", "retrieval_keys": ["job"], "confidence": 0.8}\n```'
        result = self.gk._parse_response(raw)
        assert result["memory_need"] == "SEMANTIC"
        assert result["retrieval_keys"] == ["job"]
        assert result["confidence"] == 0.8

    def test_extra_text_around_json(self):
        raw = 'Here is my classification:\n{"memory_need": "PROFILE", "retrieval_keys": ["name"], "confidence": 0.9}\nThat is my answer.'
        result = self.gk._parse_response(raw)
        assert result["memory_need"] == "PROFILE"

    def test_invalid_category_falls_back_to_semantic(self):
        raw = '{"memory_need": "INVALID", "retrieval_keys": [], "confidence": 0.5}'
        result = self.gk._parse_response(raw)
        assert result["memory_need"] == "SEMANTIC"

    def test_no_json_returns_default(self):
        raw = "I cannot classify this message."
        result = self.gk._parse_response(raw)
        assert result["memory_need"] == "SEMANTIC"
        assert result["confidence"] == 0.0

    def test_confidence_clamped(self):
        raw = '{"memory_need": "NONE", "retrieval_keys": [], "confidence": 5.0}'
        result = self.gk._parse_response(raw)
        assert result["confidence"] == 1.0

    def test_negative_confidence_clamped(self):
        raw = '{"memory_need": "NONE", "retrieval_keys": [], "confidence": -1.0}'
        result = self.gk._parse_response(raw)
        assert result["confidence"] == 0.0


class TestDefaultResult:
    """Test fail-open default behavior."""

    def test_default_is_semantic(self):
        gk = MemoryGatekeeper(llm=MagicMock(), prompts=load_prompts())
        result = gk._default_result()
        assert result["memory_need"] == "SEMANTIC"
        assert result["retrieval_keys"] == []
        assert result["confidence"] == 0.0


class TestClassifyFailOpen:
    """Test that classify returns SEMANTIC on LLM failures."""

    def test_llm_returns_none(self):
        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = None
        gk = MemoryGatekeeper(llm=mock_llm, prompts=load_prompts())

        result = gk.classify("hello")
        assert result["memory_need"] == "SEMANTIC"

    def test_llm_raises_exception(self):
        mock_llm = MagicMock()
        mock_llm.generate_json.side_effect = Exception("connection refused")
        gk = MemoryGatekeeper(llm=mock_llm, prompts=load_prompts())

        result = gk.classify("hello")
        assert result["memory_need"] == "SEMANTIC"
        assert result["confidence"] == 0.0


class TestFormatRecent:
    """Test recent message formatting."""

    def test_empty_messages(self):
        gk = MemoryGatekeeper(llm=MagicMock(), prompts=load_prompts())
        assert gk._format_recent([]) == "(no recent messages)"

    def test_formats_messages(self):
        gk = MemoryGatekeeper(llm=MagicMock(), prompts=load_prompts())
        msgs = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = gk._format_recent(msgs)
        assert "User: Hi there" in result
        assert "Assistant: Hello!" in result

    def test_limits_to_three(self):
        gk = MemoryGatekeeper(llm=MagicMock(), prompts=load_prompts())
        msgs = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
            {"role": "user", "content": "msg5"},
        ]
        result = gk._format_recent(msgs)
        assert "msg1" not in result
        assert "msg2" not in result
        assert "msg3" in result
        assert "msg4" in result
        assert "msg5" in result


# ============================================================================
# Integration tests (require Ollama + gemma3:4b)
# ============================================================================

@pytest.fixture
def live_gatekeeper():
    """Create a gatekeeper with a real LLM connection."""
    try:
        from app.models.ollama_wrapper import OllamaWrapper
        llm = OllamaWrapper(
            model="ollama/gemma3:4b",
            embedding_model="ollama/nomic-embed-text",
            api_base="http://localhost:11434"
        )
        prompts = load_prompts()
        return MemoryGatekeeper(llm, prompts, max_tokens=100, timeout=10.0)
    except Exception:
        pytest.skip("Ollama not available")


@pytest.mark.integration
class TestLiveClassification:
    """Integration tests requiring Ollama running with gemma3:4b."""

    def test_greeting_classifies_none(self, live_gatekeeper):
        result = live_gatekeeper.classify("hello how are you")
        assert result["memory_need"] in ("NONE", "RECENT"), \
            f"Expected NONE/RECENT for greeting, got {result['memory_need']}"

    def test_memory_reference_classifies_semantic(self, live_gatekeeper):
        result = live_gatekeeper.classify("what did I tell you about my job")
        assert result["memory_need"] in ("SEMANTIC", "MULTI"), \
            f"Expected SEMANTIC/MULTI for memory reference, got {result['memory_need']}"

    def test_profile_query_classifies_profile(self, live_gatekeeper):
        result = live_gatekeeper.classify("what's my name")
        assert result["memory_need"] in ("PROFILE", "SEMANTIC", "MULTI"), \
            f"Expected PROFILE/SEMANTIC/MULTI for profile query, got {result['memory_need']}"

    def test_returns_valid_structure(self, live_gatekeeper):
        result = live_gatekeeper.classify("tell me a joke")
        assert "memory_need" in result
        assert "retrieval_keys" in result
        assert "confidence" in result
        assert isinstance(result["retrieval_keys"], list)
        assert 0.0 <= result["confidence"] <= 1.0
