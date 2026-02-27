"""Unit tests for WebSearchService.

WebSearchService wraps the Tavily API for web search. All tests mock
requests.post() to avoid real HTTP calls. The service is designed to
never raise — errors are returned in the result dict (fail-open pattern).

Run from the project root:
    python -m pytest tests/unit/test_web_search.py -v
"""

from unittest.mock import patch, MagicMock

import pytest
import requests

from app.services.web_search import WebSearchService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tavily_response(results=None, status_code=200):
    """Build a mock requests.Response mimicking Tavily's API."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )
    mock_resp.json.return_value = {"results": results or []}
    return mock_resp


SAMPLE_RESULTS = [
    {
        "title": "BBC News",
        "url": "https://bbc.co.uk/news/1",
        "content": "Labour won the election with a large majority.",
    },
    {
        "title": "The Guardian",
        "url": "https://theguardian.com/politics/2",
        "content": "Final results confirm a reduced Labour majority.",
    },
    {
        "title": "Reuters",
        "url": "https://reuters.com/world/3",
        "content": "UK election results are now fully declared.",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    """WebSearchService with a fake API key configured."""
    with patch("app.services.web_search.config") as mock_config:
        mock_config.TAVILY_API_KEY = "tvly-test-key"
        mock_config.WEB_SEARCH_ENABLED = True
        mock_config.WEB_SEARCH_MAX_RESULTS = 3
        mock_config.WEB_SEARCH_MAX_CHARS_PER_RESULT = 500
        mock_config.WEB_SEARCH_MAX_TOTAL_CHARS = 1500
        mock_config.WEB_SEARCH_TIMEOUT = 10
        mock_config.WEB_SEARCH_DEPTH = "basic"
        svc = WebSearchService()
    return svc


@pytest.fixture
def no_key_service():
    """WebSearchService with no API key."""
    with patch("app.services.web_search.config") as mock_config:
        mock_config.TAVILY_API_KEY = ""
        mock_config.WEB_SEARCH_ENABLED = True
        mock_config.WEB_SEARCH_MAX_RESULTS = 3
        mock_config.WEB_SEARCH_MAX_CHARS_PER_RESULT = 500
        mock_config.WEB_SEARCH_MAX_TOTAL_CHARS = 1500
        mock_config.WEB_SEARCH_TIMEOUT = 10
        mock_config.WEB_SEARCH_DEPTH = "basic"
        svc = WebSearchService()
    return svc


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    """is_available depends on both the API key and the enabled flag."""

    def test_available_when_key_set_and_enabled(self, service):
        assert service.is_available is True

    def test_unavailable_when_key_empty(self, no_key_service):
        assert no_key_service.is_available is False

    def test_unavailable_when_disabled(self):
        with patch("app.services.web_search.config") as mock_config:
            mock_config.TAVILY_API_KEY = "tvly-test-key"
            mock_config.WEB_SEARCH_ENABLED = False
            mock_config.WEB_SEARCH_MAX_RESULTS = 3
            mock_config.WEB_SEARCH_MAX_CHARS_PER_RESULT = 500
            mock_config.WEB_SEARCH_MAX_TOTAL_CHARS = 1500
            mock_config.WEB_SEARCH_TIMEOUT = 10
            mock_config.WEB_SEARCH_DEPTH = "basic"
            svc = WebSearchService()
        assert svc.is_available is False


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

class TestSearch:
    """Tests for the search() method — always returns a dict, never raises."""

    @patch("app.services.web_search.requests.post")
    def test_successful_search(self, mock_post, service):
        """A 200 response with valid results returns structured data."""
        mock_post.return_value = _tavily_response(SAMPLE_RESULTS)

        result = service.search("UK election results")

        assert result["error"] is None
        assert result["result_count"] == 3
        assert result["query"] == "UK election results"
        assert len(result["results"]) == 3
        assert result["results"][0]["title"] == "BBC News"
        assert result["results"][0]["url"] == "https://bbc.co.uk/news/1"
        assert "Labour" in result["results"][0]["content"]

    @patch("app.services.web_search.requests.post")
    def test_timeout_returns_error_dict(self, mock_post, service):
        """A timeout should return an error dict, not raise."""
        mock_post.side_effect = requests.exceptions.Timeout("timed out")

        result = service.search("test query")

        assert result["error"] is not None
        assert "timed out" in result["error"].lower()
        assert result["results"] == []
        assert result["result_count"] == 0

    @patch("app.services.web_search.requests.post")
    def test_connection_error_returns_error_dict(self, mock_post, service):
        """A network error should return an error dict, not raise."""
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")

        result = service.search("test query")

        assert result["error"] is not None
        assert result["results"] == []
        assert result["result_count"] == 0

    def test_empty_api_key_returns_error_without_http_call(self, no_key_service):
        """With no API key, search() returns an error without any HTTP request."""
        with patch("app.services.web_search.requests.post") as mock_post:
            result = no_key_service.search("test query")
            mock_post.assert_not_called()

        assert result["error"] is not None
        assert result["results"] == []
        assert result["result_count"] == 0

    @patch("app.services.web_search.requests.post")
    def test_empty_results_from_tavily(self, mock_post, service):
        """Tavily returns 200 but with an empty results array."""
        mock_post.return_value = _tavily_response([])

        result = service.search("obscure query with no results")

        assert result["error"] is None
        assert result["result_count"] == 0
        assert result["results"] == []

    @patch("app.services.web_search.requests.post")
    def test_http_error_status_returns_error_dict(self, mock_post, service):
        """A 401 or 400 response returns an error dict, not raise."""
        mock_post.return_value = _tavily_response(status_code=401)

        result = service.search("test query")

        assert result["error"] is not None
        assert result["results"] == []
        assert result["result_count"] == 0

    @patch("app.services.web_search.requests.post")
    def test_per_result_content_truncation(self, mock_post, service):
        """Content longer than max_chars_per_result is truncated."""
        long_content = "x" * 1000
        mock_post.return_value = _tavily_response([
            {"title": "Long", "url": "https://example.com", "content": long_content}
        ])

        result = service.search("test")

        assert len(result["results"][0]["content"]) <= service.max_chars_per_result

    @patch("app.services.web_search.requests.post")
    def test_total_content_truncation(self, mock_post, service):
        """Total content across all results is capped at max_total_chars."""
        # Each result has 600 chars, 3 results = 1800 > 1500 max_total
        results_data = [
            {"title": f"Source {i}", "url": f"https://example.com/{i}", "content": "y" * 600}
            for i in range(3)
        ]
        mock_post.return_value = _tavily_response(results_data)

        result = service.search("test")

        total_chars = sum(len(r["content"]) for r in result["results"])
        assert total_chars <= service.max_total_chars


# ---------------------------------------------------------------------------
# format_for_prompt()
# ---------------------------------------------------------------------------

class TestFormatForPrompt:
    """Tests for formatting search results into an LLM prompt block."""

    def test_valid_results_contain_all_sources(self, service):
        """Output should contain all three source titles and content."""
        search_results = {
            "results": SAMPLE_RESULTS,
            "query": "test",
            "result_count": 3,
            "error": None,
        }
        output = service.format_for_prompt(search_results)

        assert "[Web Search Results]" in output
        assert "BBC News" in output
        assert "The Guardian" in output
        assert "Reuters" in output
        assert "Labour won" in output

    def test_error_results_return_empty_string(self, service):
        """When error is set, format_for_prompt returns empty string."""
        search_results = {
            "results": [],
            "query": "test",
            "result_count": 0,
            "error": "Search timed out",
        }
        assert service.format_for_prompt(search_results) == ""

    def test_no_results_return_empty_string(self, service):
        """When results list is empty (no error), returns empty string."""
        search_results = {
            "results": [],
            "query": "test",
            "result_count": 0,
            "error": None,
        }
        assert service.format_for_prompt(search_results) == ""


# ---------------------------------------------------------------------------
# format_sources_for_response()
# ---------------------------------------------------------------------------

class TestFormatSourcesForResponse:
    """Tests for formatting source links in markdown."""

    def test_valid_results_produce_markdown_links(self, service):
        """Output should contain markdown links for each source."""
        search_results = {
            "results": SAMPLE_RESULTS,
            "query": "test",
            "result_count": 3,
            "error": None,
        }
        output = service.format_sources_for_response(search_results)

        assert "**Sources:**" in output
        assert "[BBC News](https://bbc.co.uk/news/1)" in output
        assert "[The Guardian](https://theguardian.com/politics/2)" in output
        assert "[Reuters](https://reuters.com/world/3)" in output

    def test_error_results_return_empty_string(self, service):
        """When error is set, returns empty string."""
        search_results = {
            "results": [],
            "query": "test",
            "result_count": 0,
            "error": "Network error",
        }
        assert service.format_sources_for_response(search_results) == ""

    def test_empty_results_return_empty_string(self, service):
        """When results list is empty, returns empty string."""
        search_results = {
            "results": [],
            "query": "test",
            "result_count": 0,
            "error": None,
        }
        assert service.format_sources_for_response(search_results) == ""

    def test_special_characters_in_title(self, service):
        """Titles with special characters should not break formatting."""
        search_results = {
            "results": [
                {"title": "O'Brien & Sons — Latest", "url": "https://example.com", "content": "test"},
            ],
            "query": "test",
            "result_count": 1,
            "error": None,
        }
        output = service.format_sources_for_response(search_results)
        assert "[O'Brien & Sons — Latest](https://example.com)" in output
