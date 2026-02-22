"""Unit tests for MemoryGatekeeper._parse_response().

_parse_response() converts a raw LLM string into a structured classification
dict. It handles:
  - Bare JSON:          {"memory_need": "NONE", "confidence": 0.9, ...}
  - JSON with preamble: "Sure, here is the JSON: {...}"
  - Markdown fences:    ```json\n{...}\n```
  - Malformed output:   returns the fail-open SEMANTIC default

This method only uses self.VALID_CATEGORIES (a class attribute) and
self._default_result() — no LLM or prompts are called — so MemoryGatekeeper
can be instantiated with stub values for testing.

Run from the project root:
    python -m pytest tests/unit/test_gatekeeper.py -v
"""

import json

import pytest

from app.services.gatekeeper import MemoryGatekeeper

# ---------------------------------------------------------------------------
# Expected default (fail-open)
# ---------------------------------------------------------------------------

_DEFAULT = {"memory_need": "SEMANTIC", "retrieval_keys": [], "confidence": 0.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_json_str(**kwargs) -> str:
    """Produce a bare JSON string with the given fields."""
    return json.dumps(kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gatekeeper() -> MemoryGatekeeper:
    """MemoryGatekeeper with stub LLM and prompts.

    _parse_response() does not call self.llm or access self.prompts, so
    passing None and {} is safe and keeps the test isolated from any LLM.
    """
    return MemoryGatekeeper(llm=None, prompts={})


# ---------------------------------------------------------------------------
# Default / fallback behaviour
# ---------------------------------------------------------------------------

class TestDefaultFallback:
    """All bad or unrecognisable inputs must fall back to the SEMANTIC default."""

    def test_empty_string_returns_default(self, gatekeeper):
        """An empty response string has no JSON — should return the default."""
        result = gatekeeper._parse_response("")
        assert result == _DEFAULT

    def test_no_json_object_returns_default(self, gatekeeper):
        """A response with no curly-brace JSON object should return the default."""
        result = gatekeeper._parse_response("Hello, I am the assistant.")
        assert result == _DEFAULT

    def test_malformed_json_returns_default(self, gatekeeper):
        """JSON with a syntax error (missing closing brace) should fall back."""
        result = gatekeeper._parse_response('{"memory_need": "NONE"')
        assert result == _DEFAULT

    def test_invalid_memory_need_falls_back_to_semantic(self, gatekeeper):
        """An unrecognised memory_need value should be replaced with SEMANTIC."""
        raw = make_json_str(memory_need="UNKNOWN_CATEGORY", confidence=0.9)
        result = gatekeeper._parse_response(raw)
        assert result["memory_need"] == "SEMANTIC"

    def test_missing_memory_need_defaults_to_semantic(self, gatekeeper):
        """A JSON object that omits memory_need should default to SEMANTIC."""
        raw = make_json_str(confidence=0.8, retrieval_keys=[])
        result = gatekeeper._parse_response(raw)
        assert result["memory_need"] == "SEMANTIC"


# ---------------------------------------------------------------------------
# Valid memory_need values
# ---------------------------------------------------------------------------

class TestValidCategories:
    """Every value in VALID_CATEGORIES must be accepted as-is."""

    @pytest.mark.parametrize("category", ["NONE", "RECENT", "SEMANTIC", "PROFILE", "MULTI"])
    def test_each_valid_category_is_accepted(self, gatekeeper, category):
        """_parse_response() must return the exact category for each valid value."""
        raw = make_json_str(memory_need=category, confidence=0.8, retrieval_keys=[])
        result = gatekeeper._parse_response(raw)
        assert result["memory_need"] == category

    def test_lowercase_memory_need_is_normalised_to_upper(self, gatekeeper):
        """LLMs sometimes return lowercase. The parser should upper-case and accept it."""
        raw = make_json_str(memory_need="semantic", confidence=0.7, retrieval_keys=[])
        result = gatekeeper._parse_response(raw)
        assert result["memory_need"] == "SEMANTIC"

    def test_mixed_case_memory_need_normalised(self, gatekeeper):
        """'Semantic' (title case) should normalise to 'SEMANTIC'."""
        raw = make_json_str(memory_need="Semantic", confidence=0.6)
        result = gatekeeper._parse_response(raw)
        assert result["memory_need"] == "SEMANTIC"


# ---------------------------------------------------------------------------
# Confidence parsing and clamping
# ---------------------------------------------------------------------------

class TestConfidence:
    """Confidence must be a float in [0.0, 1.0]; anything outside is clamped."""

    def test_valid_confidence_parsed(self, gatekeeper):
        """A confidence value within [0, 1] should be returned unchanged."""
        raw = make_json_str(memory_need="SEMANTIC", confidence=0.75)
        result = gatekeeper._parse_response(raw)
        assert result["confidence"] == pytest.approx(0.75)

    def test_confidence_above_1_clamped_to_1(self, gatekeeper):
        """confidence=2.5 should be clamped to 1.0."""
        raw = make_json_str(memory_need="NONE", confidence=2.5)
        result = gatekeeper._parse_response(raw)
        assert result["confidence"] == pytest.approx(1.0)

    def test_confidence_below_0_clamped_to_0(self, gatekeeper):
        """confidence=-0.5 should be clamped to 0.0."""
        raw = make_json_str(memory_need="NONE", confidence=-0.5)
        result = gatekeeper._parse_response(raw)
        assert result["confidence"] == pytest.approx(0.0)

    def test_non_numeric_confidence_defaults_to_0(self, gatekeeper):
        """A non-numeric confidence string (e.g. 'high') should become 0.0."""
        raw = make_json_str(memory_need="SEMANTIC", confidence="high")
        result = gatekeeper._parse_response(raw)
        assert result["confidence"] == pytest.approx(0.0)

    def test_missing_confidence_defaults_to_0(self, gatekeeper):
        """When confidence is absent the default is 0.0."""
        raw = make_json_str(memory_need="NONE", retrieval_keys=[])
        result = gatekeeper._parse_response(raw)
        assert result["confidence"] == pytest.approx(0.0)

    def test_confidence_boundary_0_accepted(self, gatekeeper):
        """confidence=0.0 is on the boundary and should be kept."""
        raw = make_json_str(memory_need="NONE", confidence=0.0)
        result = gatekeeper._parse_response(raw)
        assert result["confidence"] == pytest.approx(0.0)

    def test_confidence_boundary_1_accepted(self, gatekeeper):
        """confidence=1.0 is on the boundary and should be kept."""
        raw = make_json_str(memory_need="SEMANTIC", confidence=1.0)
        result = gatekeeper._parse_response(raw)
        assert result["confidence"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# retrieval_keys parsing
# ---------------------------------------------------------------------------

class TestRetrievalKeys:
    """retrieval_keys should be a list; anything else is normalised to []."""

    def test_retrieval_keys_list_is_returned(self, gatekeeper):
        """A valid list of keyword strings should be returned as-is."""
        raw = make_json_str(memory_need="SEMANTIC", confidence=0.8,
                            retrieval_keys=["Python", "functions"])
        result = gatekeeper._parse_response(raw)
        assert result["retrieval_keys"] == ["Python", "functions"]

    def test_empty_retrieval_keys_list_returned(self, gatekeeper):
        """An empty list for retrieval_keys should remain an empty list."""
        raw = make_json_str(memory_need="NONE", confidence=0.9, retrieval_keys=[])
        result = gatekeeper._parse_response(raw)
        assert result["retrieval_keys"] == []

    def test_missing_retrieval_keys_defaults_to_empty_list(self, gatekeeper):
        """When retrieval_keys is absent the result must contain an empty list."""
        raw = make_json_str(memory_need="SEMANTIC", confidence=0.7)
        result = gatekeeper._parse_response(raw)
        assert result["retrieval_keys"] == []

    def test_non_list_retrieval_keys_normalised_to_empty(self, gatekeeper):
        """If the LLM returns retrieval_keys as a string rather than a list,
        it should be normalised to [] to avoid downstream errors.
        """
        raw = make_json_str(memory_need="SEMANTIC", confidence=0.6,
                            retrieval_keys="Python")
        result = gatekeeper._parse_response(raw)
        assert result["retrieval_keys"] == []


# ---------------------------------------------------------------------------
# JSON extraction from noisy LLM output
# ---------------------------------------------------------------------------

class TestNoisyInput:
    """LLMs often wrap JSON in prose or markdown. The parser must strip that."""

    def test_json_with_preamble_text(self, gatekeeper):
        """JSON preceded by a sentence (a common LLM habit) should still parse."""
        raw = 'Sure, here is the classification: {"memory_need": "NONE", "confidence": 0.9, "retrieval_keys": []}'
        result = gatekeeper._parse_response(raw)
        assert result["memory_need"] == "NONE"
        assert result["confidence"] == pytest.approx(0.9)

    def test_json_with_trailing_text(self, gatekeeper):
        """JSON followed by extra prose should still parse the JSON portion."""
        raw = '{"memory_need": "RECENT", "confidence": 0.7, "retrieval_keys": []} Hope that helps!'
        result = gatekeeper._parse_response(raw)
        assert result["memory_need"] == "RECENT"

    def test_json_in_markdown_code_fence(self, gatekeeper):
        """JSON wrapped in ```json ... ``` markdown should be parsed correctly."""
        raw = '```json\n{"memory_need": "PROFILE", "confidence": 0.85, "retrieval_keys": ["name"]}\n```'
        result = gatekeeper._parse_response(raw)
        assert result["memory_need"] == "PROFILE"
        assert result["retrieval_keys"] == ["name"]

    def test_json_in_plain_code_fence(self, gatekeeper):
        """JSON wrapped in ``` ... ``` (no language tag) should also parse."""
        raw = '```\n{"memory_need": "MULTI", "confidence": 0.6, "retrieval_keys": []}\n```'
        result = gatekeeper._parse_response(raw)
        assert result["memory_need"] == "MULTI"

    def test_result_dict_always_has_three_keys(self, gatekeeper):
        """Every parse result must contain exactly memory_need, retrieval_keys,
        and confidence — whether parsing succeeded or fell back to default.
        """
        for raw in [
            make_json_str(memory_need="NONE", confidence=0.5, retrieval_keys=[]),
            "not json at all",
            "",
        ]:
            result = gatekeeper._parse_response(raw)
            assert set(result.keys()) == {"memory_need", "retrieval_keys", "confidence"}
