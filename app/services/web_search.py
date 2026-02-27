"""Web search service using Tavily API.

Ephemeral search — results are used for one response then discarded.
Never stores results in memory or vector store.
"""

import time

import requests

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchService:
    """Handles web search via Tavily API."""

    def __init__(self):
        self.api_key = config.TAVILY_API_KEY
        self.enabled = config.WEB_SEARCH_ENABLED
        self.max_results = config.WEB_SEARCH_MAX_RESULTS
        self.max_chars_per_result = config.WEB_SEARCH_MAX_CHARS_PER_RESULT
        self.max_total_chars = config.WEB_SEARCH_MAX_TOTAL_CHARS
        self.timeout = config.WEB_SEARCH_TIMEOUT
        self.search_depth = config.WEB_SEARCH_DEPTH
        self.api_url = "https://api.tavily.com/search"

        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set — web search will be unavailable")

    @property
    def is_available(self) -> bool:
        """Check if web search is configured and enabled."""
        return self.enabled and bool(self.api_key)

    def search(self, query: str) -> dict:
        """Execute a web search and return formatted results.

        Args:
            query: Search query string.

        Returns:
            dict with keys: results, query, result_count, error
        """
        if not self.is_available:
            return {
                "results": [],
                "query": query,
                "result_count": 0,
                "error": "Web search not available (disabled or missing API key)",
            }

        start = time.perf_counter()
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": self.max_results,
                "search_depth": self.search_depth,
            }
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            raw_results = data.get("results", [])
            results = []
            total_chars = 0

            for item in raw_results:
                content = (item.get("content") or "")[:self.max_chars_per_result]
                if total_chars + len(content) > self.max_total_chars:
                    content = content[:self.max_total_chars - total_chars]
                total_chars += len(content)

                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": content,
                })

                if total_chars >= self.max_total_chars:
                    break

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                f"[WEB_SEARCH] query=\"{query}\" results={len(results)} "
                f"time={elapsed:.1f}ms"
            )

            return {
                "results": results,
                "query": query,
                "result_count": len(results),
                "error": None,
            }

        except requests.exceptions.Timeout:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(
                f"[WEB_SEARCH] Timeout after {elapsed:.1f}ms for query=\"{query}\""
            )
            return {
                "results": [],
                "query": query,
                "result_count": 0,
                "error": f"Search timed out after {self.timeout}s",
            }
        except requests.exceptions.RequestException as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(
                f"[WEB_SEARCH] Request error after {elapsed:.1f}ms: {e}"
            )
            return {
                "results": [],
                "query": query,
                "result_count": 0,
                "error": str(e),
            }

    def format_for_prompt(self, search_results: dict) -> str:
        """Format search results as a text block for injection into the LLM prompt.

        Returns empty string if there are no results or an error occurred.
        """
        if search_results.get("error") or not search_results.get("results"):
            return ""

        lines = ["[Web Search Results]"]
        for result in search_results["results"]:
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")
            lines.append(f"Source: {title} ({url})")
            lines.append(content)
            lines.append("")

        return "\n".join(lines).rstrip()

    def format_sources_for_response(self, search_results: dict) -> str:
        """Format source URLs for appending to the assistant response.

        Returns empty string if there are no results or an error occurred.
        """
        if search_results.get("error") or not search_results.get("results"):
            return ""

        lines = ["\n\n**Sources:**"]
        for result in search_results["results"]:
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            if url:
                lines.append(f"- [{title}]({url})")

        return "\n".join(lines) if len(lines) > 1 else ""
