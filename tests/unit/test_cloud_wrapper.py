"""Unit tests for CloudWrapper and CloudGenerationError.

CloudWrapper wraps cloud LLM providers via LiteLLM. All tests mock
litellm.completion() to avoid real API calls. CloudGenerationError
is tested for correct error classification.

Run from the project root:
    python -m pytest tests/unit/test_cloud_wrapper.py -v
"""

from unittest.mock import patch, MagicMock

import pytest

from app.models.cloud_wrapper import CloudWrapper, CloudGenerationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wrapper():
    """A CloudWrapper pointing at a fake Gemini model."""
    return CloudWrapper(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key="fake-key-123",
        display_name="Gemini 2.0 Flash",
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    """CloudWrapper stores provider, model, api_key, and builds the litellm string."""

    def test_litellm_model_string(self, wrapper):
        assert wrapper.litellm_model == "gemini/gemini-2.0-flash"

    def test_display_name_stored(self, wrapper):
        assert wrapper.display_name == "Gemini 2.0 Flash"

    def test_api_key_stored(self, wrapper):
        assert wrapper.api_key == "fake-key-123"

    def test_different_provider(self):
        w = CloudWrapper(provider="openai", model="gpt-4o", api_key="sk-test")
        assert w.litellm_model == "openai/gpt-4o"

    def test_default_display_name(self):
        w = CloudWrapper(provider="openai", model="gpt-4o", api_key="sk-test")
        assert w.display_name == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# NotImplementedError for unsupported methods
# ---------------------------------------------------------------------------

class TestNotImplemented:
    """get_embedding and generate_json must raise NotImplementedError."""

    def test_generate_json_raises(self, wrapper):
        with pytest.raises(NotImplementedError, match="JSON generation"):
            wrapper.generate_json("return JSON")

    def test_get_embedding_raises(self, wrapper):
        with pytest.raises(NotImplementedError, match="embeddings"):
            wrapper.get_embedding("test text")


# ---------------------------------------------------------------------------
# CloudGenerationError
# ---------------------------------------------------------------------------

class TestCloudGenerationError:
    """CloudGenerationError carries error_type and message."""

    def test_auth_error(self):
        e = CloudGenerationError("auth_error", "Invalid API key")
        assert e.error_type == "auth_error"
        assert "Invalid API key" in str(e)

    def test_rate_limit(self):
        e = CloudGenerationError("rate_limit", "429 Too Many Requests")
        assert e.error_type == "rate_limit"

    def test_network_error(self):
        e = CloudGenerationError("network_error", "Connection refused")
        assert e.error_type == "network_error"

    def test_unknown_type(self):
        e = CloudGenerationError("unknown", "Something went wrong")
        assert e.error_type == "unknown"

    def test_is_exception_subclass(self):
        assert issubclass(CloudGenerationError, Exception)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestErrorClassification:
    """_classify_error maps exception text to typed error categories."""

    @pytest.mark.parametrize("msg,expected", [
        ("AuthenticationError: invalid api_key", "auth_error"),
        ("401 Unauthorized", "auth_error"),
        ("invalid key provided", "auth_error"),
        ("Rate limit exceeded", "rate_limit"),
        ("429 Too Many Requests", "rate_limit"),
        ("quota exceeded for this billing period", "rate_limit"),
        ("Connection refused", "network_error"),
        ("Request timeout after 30s", "network_error"),
        ("Network unreachable", "network_error"),
        ("Unexpected server error", "unknown"),
        ("", "unknown"),
    ])
    def test_classify_keywords(self, msg, expected):
        error = Exception(msg)
        assert CloudWrapper._classify_error(error) == expected


# ---------------------------------------------------------------------------
# generate() — non-streaming
# ---------------------------------------------------------------------------

class TestGenerateNonStreaming:
    """Non-streaming generate() returns a string or raises CloudGenerationError."""

    @patch("app.models.cloud_wrapper.completion")
    def test_successful_generation(self, mock_completion, wrapper):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from the cloud!"
        mock_completion.return_value = mock_response

        result = wrapper.generate("Say hello", stream=False)

        assert result == "Hello from the cloud!"
        mock_completion.assert_called_once()

    @patch("app.models.cloud_wrapper.completion")
    def test_auth_error_raises_cloud_generation_error(self, mock_completion, wrapper):
        mock_completion.side_effect = Exception("AuthenticationError: invalid api_key")

        with pytest.raises(CloudGenerationError) as exc_info:
            wrapper.generate("Say hello", stream=False)
        assert exc_info.value.error_type == "auth_error"

    @patch("app.models.cloud_wrapper.completion")
    def test_rate_limit_error(self, mock_completion, wrapper):
        mock_completion.side_effect = Exception("429 Rate limit exceeded")

        with pytest.raises(CloudGenerationError) as exc_info:
            wrapper.generate("Say hello", stream=False)
        assert exc_info.value.error_type == "rate_limit"

    @patch("app.models.cloud_wrapper.completion")
    def test_network_error(self, mock_completion, wrapper):
        mock_completion.side_effect = Exception("Connection timeout")

        with pytest.raises(CloudGenerationError) as exc_info:
            wrapper.generate("Say hello", stream=False)
        assert exc_info.value.error_type == "network_error"

    @patch("app.models.cloud_wrapper.completion")
    def test_unknown_error(self, mock_completion, wrapper):
        mock_completion.side_effect = Exception("Unexpected server error")

        with pytest.raises(CloudGenerationError) as exc_info:
            wrapper.generate("Say hello", stream=False)
        assert exc_info.value.error_type == "unknown"


# ---------------------------------------------------------------------------
# generate() — streaming
# ---------------------------------------------------------------------------

class TestGenerateStreaming:
    """Streaming generate() yields tokens or raises CloudGenerationError."""

    @patch("app.models.cloud_wrapper.completion")
    def test_successful_streaming(self, mock_completion, wrapper):
        """Streaming yields delta.content strings."""
        chunks = []
        for text in ["Hello", " ", "world!"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = text
            chunks.append(chunk)

        mock_completion.return_value = iter(chunks)

        tokens = list(wrapper.generate("Say hello", stream=True))
        assert tokens == ["Hello", " ", "world!"]

    @patch("app.models.cloud_wrapper.completion")
    def test_skips_empty_deltas(self, mock_completion, wrapper):
        """Chunks with empty or None delta.content are skipped."""
        chunk_good = MagicMock()
        chunk_good.choices = [MagicMock()]
        chunk_good.choices[0].delta = MagicMock()
        chunk_good.choices[0].delta.content = "Hello"

        chunk_empty = MagicMock()
        chunk_empty.choices = [MagicMock()]
        chunk_empty.choices[0].delta = MagicMock()
        chunk_empty.choices[0].delta.content = ""

        chunk_none = MagicMock()
        chunk_none.choices = [MagicMock()]
        chunk_none.choices[0].delta = MagicMock()
        chunk_none.choices[0].delta.content = None

        mock_completion.return_value = iter([chunk_good, chunk_empty, chunk_none])

        tokens = list(wrapper.generate("test", stream=True))
        assert tokens == ["Hello"]

    @patch("app.models.cloud_wrapper.completion")
    def test_streaming_error_raises_cloud_generation_error(self, mock_completion, wrapper):
        """An error during streaming raises CloudGenerationError."""
        mock_completion.side_effect = Exception("Connection timeout")

        gen = wrapper.generate("test", stream=True)
        with pytest.raises(CloudGenerationError) as exc_info:
            list(gen)
        assert exc_info.value.error_type == "network_error"
