"""Unit tests for CloudWrapper and CloudGenerationError.

CloudWrapper wraps cloud LLM providers via LiteLLM. All tests mock
litellm.completion() to avoid real API calls.

Run from the project root:
    python -m pytest tests/test_cloud_wrapper.py -v
"""

from unittest.mock import patch, MagicMock

import pytest

from app.models.cloud_wrapper import CloudWrapper, CloudGenerationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wrapper():
    """A CloudWrapper with fake credentials (no real API calls)."""
    return CloudWrapper(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key="fake-api-key",
        display_name="Gemini 2.0 Flash",
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    """Test that the constructor builds correct LiteLLM model strings."""

    def test_litellm_model_string(self, wrapper):
        """Model string should be '{provider}/{model}'."""
        assert wrapper.litellm_model == "gemini/gemini-2.0-flash"

    def test_display_name_stored(self, wrapper):
        assert wrapper.display_name == "Gemini 2.0 Flash"

    def test_api_key_stored(self, wrapper):
        assert wrapper.api_key == "fake-api-key"

    def test_different_provider(self):
        """Verify model string for a different provider."""
        w = CloudWrapper(provider="openai", model="gpt-4o", api_key="key")
        assert w.litellm_model == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# NotImplementedError methods
# ---------------------------------------------------------------------------

class TestNotImplemented:
    """generate_json() and get_embedding() must raise NotImplementedError."""

    def test_generate_json_raises(self, wrapper):
        with pytest.raises(NotImplementedError, match="JSON generation"):
            wrapper.generate_json("Give me JSON")

    def test_get_embedding_raises(self, wrapper):
        with pytest.raises(NotImplementedError, match="embeddings"):
            wrapper.get_embedding("some text")


# ---------------------------------------------------------------------------
# CloudGenerationError
# ---------------------------------------------------------------------------

class TestCloudGenerationError:
    """Test the CloudGenerationError exception class."""

    def test_auth_error_type(self):
        err = CloudGenerationError("auth_error", "Invalid API key")
        assert err.error_type == "auth_error"
        assert err.message == "Invalid API key"
        assert str(err) == "Invalid API key"

    def test_rate_limit_error_type(self):
        err = CloudGenerationError("rate_limit", "429 Too Many Requests")
        assert err.error_type == "rate_limit"
        assert err.message == "429 Too Many Requests"

    def test_network_error_type(self):
        err = CloudGenerationError("network_error", "Connection refused")
        assert err.error_type == "network_error"

    def test_unknown_error_type(self):
        err = CloudGenerationError("unknown", "Something unexpected")
        assert err.error_type == "unknown"

    def test_is_exception_subclass(self):
        err = CloudGenerationError("unknown", "test")
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestErrorClassification:
    """Test _classify_error maps exceptions to the right error_type."""

    def test_auth_keywords(self):
        assert CloudWrapper._classify_error(Exception("AuthenticationError: invalid api_key")) == "auth_error"
        assert CloudWrapper._classify_error(Exception("401 Unauthorized")) == "auth_error"

    def test_rate_limit_keywords(self):
        assert CloudWrapper._classify_error(Exception("Rate limit exceeded")) == "rate_limit"
        assert CloudWrapper._classify_error(Exception("429 Too Many Requests")) == "rate_limit"
        assert CloudWrapper._classify_error(Exception("Quota exhausted")) == "rate_limit"

    def test_network_keywords(self):
        assert CloudWrapper._classify_error(Exception("Connection refused")) == "network_error"
        assert CloudWrapper._classify_error(Exception("Request timeout")) == "network_error"
        assert CloudWrapper._classify_error(Exception("Network unreachable")) == "network_error"

    def test_unknown_fallback(self):
        assert CloudWrapper._classify_error(Exception("Something totally weird")) == "unknown"


# ---------------------------------------------------------------------------
# generate() — non-streaming
# ---------------------------------------------------------------------------

class TestGenerateNonStreaming:
    """Test generate() with stream=False, mocking litellm.completion."""

    @patch("app.models.cloud_wrapper.completion")
    def test_successful_generation(self, mock_completion, wrapper):
        """A successful completion returns the message content."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from the cloud!"
        mock_completion.return_value = mock_response

        result = wrapper.generate("Say hello", stream=False)

        assert result == "Hello from the cloud!"
        mock_completion.assert_called_once_with(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            temperature=0.7,
            api_key="fake-api-key",
        )

    @patch("app.models.cloud_wrapper.completion")
    def test_auth_error_raises_cloud_generation_error(self, mock_completion, wrapper):
        """An auth error from litellm should raise CloudGenerationError."""
        mock_completion.side_effect = Exception("AuthenticationError: invalid api_key")

        with pytest.raises(CloudGenerationError) as exc_info:
            wrapper.generate("test", stream=False)

        assert exc_info.value.error_type == "auth_error"

    @patch("app.models.cloud_wrapper.completion")
    def test_rate_limit_raises_cloud_generation_error(self, mock_completion, wrapper):
        """A rate limit error raises CloudGenerationError with rate_limit type."""
        mock_completion.side_effect = Exception("429 Rate limit exceeded")

        with pytest.raises(CloudGenerationError) as exc_info:
            wrapper.generate("test", stream=False)

        assert exc_info.value.error_type == "rate_limit"

    @patch("app.models.cloud_wrapper.completion")
    def test_network_error_raises_cloud_generation_error(self, mock_completion, wrapper):
        """A network error raises CloudGenerationError with network_error type."""
        mock_completion.side_effect = Exception("Connection timeout")

        with pytest.raises(CloudGenerationError) as exc_info:
            wrapper.generate("test", stream=False)

        assert exc_info.value.error_type == "network_error"

    @patch("app.models.cloud_wrapper.completion")
    def test_unknown_error_raises_cloud_generation_error(self, mock_completion, wrapper):
        """An unclassifiable error raises CloudGenerationError with unknown type."""
        mock_completion.side_effect = Exception("Something completely unexpected")

        with pytest.raises(CloudGenerationError) as exc_info:
            wrapper.generate("test", stream=False)

        assert exc_info.value.error_type == "unknown"


# ---------------------------------------------------------------------------
# generate() — streaming
# ---------------------------------------------------------------------------

class TestGenerateStreaming:
    """Test generate() with stream=True, mocking litellm.completion."""

    @patch("app.models.cloud_wrapper.completion")
    def test_successful_streaming(self, mock_completion, wrapper):
        """Streaming generation should yield individual token chunks."""
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        mock_completion.return_value = [chunk1, chunk2]

        result = list(wrapper.generate("test", stream=True))

        assert result == ["Hello", " world"]
        mock_completion.assert_called_once_with(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            api_key="fake-api-key",
            stream=True,
        )

    @patch("app.models.cloud_wrapper.completion")
    def test_streaming_skips_empty_deltas(self, mock_completion, wrapper):
        """Chunks with None or empty content should be skipped."""
        chunk_with_content = MagicMock()
        chunk_with_content.choices = [MagicMock()]
        chunk_with_content.choices[0].delta.content = "token"

        chunk_empty = MagicMock()
        chunk_empty.choices = [MagicMock()]
        chunk_empty.choices[0].delta.content = None

        mock_completion.return_value = [chunk_empty, chunk_with_content]

        result = list(wrapper.generate("test", stream=True))

        assert result == ["token"]

    @patch("app.models.cloud_wrapper.completion")
    def test_streaming_error_raises_cloud_generation_error(self, mock_completion, wrapper):
        """An error during streaming raises CloudGenerationError."""
        mock_completion.side_effect = Exception("Connection timeout during stream")

        gen = wrapper.generate("test", stream=True)
        with pytest.raises(CloudGenerationError) as exc_info:
            list(gen)

        assert exc_info.value.error_type == "network_error"
