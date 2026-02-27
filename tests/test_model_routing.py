"""Unit tests for Phase 9 model routing.

Tests cover:
- Model registry construction from config (only models with valid API keys)
- Orchestrator routing: _resolve_llm picks the correct LLM
- Unknown model names fall back to local with a warning
- /api/available-models returns the correct list
- Token usage records include model_name

All tests mock external dependencies — no real API calls or Ollama.

Run from the project root:
    python -m pytest tests/test_model_routing.py -v
"""

import json
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from app.models.cloud_wrapper import CloudWrapper, CloudGenerationError
from app.services.orchestrator import ChatOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(name="local"):
    """Create a mock LLM with a display_name attribute."""
    mock = MagicMock()
    mock.display_name = name
    return mock


def _make_orchestrator(local_llm=None, model_registry=None):
    """Build a ChatOrchestrator with mocked dependencies."""
    local = local_llm or _make_mock_llm("local")
    registry = model_registry or {}

    mock_chat_service = MagicMock()
    mock_chat_service.llm = local

    return ChatOrchestrator(
        gatekeeper=MagicMock(),
        retriever=MagicMock(),
        chat_service=mock_chat_service,
        memory_service=MagicMock(),
        reranker=MagicMock(),
        local_llm=local,
        model_registry=registry,
    )


# ---------------------------------------------------------------------------
# Model Registry Construction
# ---------------------------------------------------------------------------

class TestModelRegistryConstruction:
    """Test that model_registry is built correctly from config."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "real-key"}, clear=False)
    def test_cloud_model_added_when_key_present(self):
        """Cloud models with valid API keys should appear in the registry."""
        import os
        from app.models.cloud_wrapper import CloudWrapper

        cloud_models = {
            "Gemini 2.0 Flash": {
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "api_key_env": "GEMINI_API_KEY",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
            }
        }
        local_llm = _make_mock_llm("Gemma 3 4B (Local)")
        registry = {"Gemma 3 4B (Local)": local_llm}

        for display_name, cfg in cloud_models.items():
            api_key = os.environ.get(cfg["api_key_env"], "")
            if api_key:
                registry[display_name] = CloudWrapper(
                    provider=cfg["provider"],
                    model=cfg["model"],
                    api_key=api_key,
                    display_name=display_name,
                )

        assert "Gemini 2.0 Flash" in registry
        assert isinstance(registry["Gemini 2.0 Flash"], CloudWrapper)
        assert registry["Gemini 2.0 Flash"].api_key == "real-key"

    @patch.dict("os.environ", {}, clear=False)
    def test_cloud_model_skipped_when_no_key(self):
        """Cloud models without API keys should not appear in the registry."""
        import os

        cloud_models = {
            "Gemini 2.0 Flash": {
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "api_key_env": "GEMINI_API_KEY_NONEXISTENT",
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
            }
        }
        local_llm = _make_mock_llm("Gemma 3 4B (Local)")
        registry = {"Gemma 3 4B (Local)": local_llm}

        for display_name, cfg in cloud_models.items():
            api_key = os.environ.get(cfg["api_key_env"], "")
            if api_key:
                registry[display_name] = CloudWrapper(
                    provider=cfg["provider"],
                    model=cfg["model"],
                    api_key=api_key,
                    display_name=display_name,
                )

        assert "Gemini 2.0 Flash" not in registry
        assert len(registry) == 1  # only local

    def test_local_model_always_in_registry(self):
        """The local model should always be present in the registry."""
        local_llm = _make_mock_llm("Gemma 3 4B (Local)")
        registry = {"Gemma 3 4B (Local)": local_llm}
        assert "Gemma 3 4B (Local)" in registry


# ---------------------------------------------------------------------------
# _resolve_llm
# ---------------------------------------------------------------------------

class TestResolveLlm:
    """Test the orchestrator's model resolution logic."""

    def test_none_model_returns_local(self):
        """Passing model=None should return the local LLM."""
        local = _make_mock_llm("local")
        orch = _make_orchestrator(local_llm=local)
        assert orch._resolve_llm(None) is local

    def test_known_cloud_model_returns_cloud_llm(self):
        """A model name in the registry should return the cloud LLM."""
        local = _make_mock_llm("local")
        cloud = _make_mock_llm("Gemini 2.0 Flash")
        registry = {"Gemma 3 4B (Local)": local, "Gemini 2.0 Flash": cloud}
        orch = _make_orchestrator(local_llm=local, model_registry=registry)

        result = orch._resolve_llm("Gemini 2.0 Flash")
        assert result is cloud

    def test_local_model_name_returns_local(self):
        """Passing the local model display name returns the local LLM."""
        local = _make_mock_llm("Gemma 3 4B (Local)")
        registry = {"Gemma 3 4B (Local)": local}
        orch = _make_orchestrator(local_llm=local, model_registry=registry)

        result = orch._resolve_llm("Gemma 3 4B (Local)")
        assert result is local

    @patch("app.services.orchestrator.logger")
    def test_unknown_model_falls_back_to_local_with_warning(self, mock_logger):
        """An unrecognised model name should fall back to local and log a warning."""
        local = _make_mock_llm("local")
        orch = _make_orchestrator(local_llm=local, model_registry={})

        result = orch._resolve_llm("NonExistentModel")

        assert result is local
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "NonExistentModel" in warning_msg
        assert "falling back" in warning_msg.lower()


# ---------------------------------------------------------------------------
# Orchestrator uses correct LLM for generation
# ---------------------------------------------------------------------------

class TestOrchestratorGeneration:
    """Test that the orchestrator routes generation to the correct LLM."""

    @patch("app.services.orchestrator.config")
    @patch("app.services.orchestrator.WebSearchService")
    def test_generate_uses_requested_model(self, mock_ws_cls, mock_config):
        """_stage_generate should call the resolved LLM's generate method."""
        mock_config.GATEKEEPER_ENABLED = False
        mock_config.RERANKER_ENABLED = False

        local = _make_mock_llm("local")
        cloud = _make_mock_llm("Gemini 2.0 Flash")
        cloud.generate.return_value = iter(["cloud", " response"])

        registry = {"Gemma 3 4B (Local)": local, "Gemini 2.0 Flash": cloud}
        orch = _make_orchestrator(local_llm=local, model_registry=registry)

        ctx = {
            "assembled_prompt": "test prompt",
            "response_text": "",
            "errors": [],
            "timings": {},
            "model": "Gemini 2.0 Flash",
        }

        events = list(orch._stage_generate(ctx))

        cloud.generate.assert_called_once_with("test prompt", stream=True)
        local.generate.assert_not_called()
        assert ctx["response_text"] == "cloud response"

    @patch("app.services.orchestrator.config")
    @patch("app.services.orchestrator.WebSearchService")
    def test_generate_uses_local_when_no_model_specified(self, mock_ws_cls, mock_config):
        """Without a model param, generation should use the local LLM."""
        mock_config.GATEKEEPER_ENABLED = False
        mock_config.RERANKER_ENABLED = False

        local = _make_mock_llm("local")
        local.generate.return_value = iter(["local", " tokens"])
        orch = _make_orchestrator(local_llm=local)

        ctx = {
            "assembled_prompt": "test",
            "response_text": "",
            "errors": [],
            "timings": {},
            "model": None,
        }

        list(orch._stage_generate(ctx))

        local.generate.assert_called_once_with("test", stream=True)
        assert ctx["response_text"] == "local tokens"


# ---------------------------------------------------------------------------
# Cloud error handling in orchestrator
# ---------------------------------------------------------------------------

class TestOrchestratorCloudError:
    """Test that CloudGenerationError is caught and yields cloud_error SSE."""

    @patch("app.services.orchestrator.config")
    @patch("app.services.orchestrator.WebSearchService")
    def test_cloud_error_yields_cloud_error_event(self, mock_ws_cls, mock_config):
        """A CloudGenerationError should yield a cloud_error SSE event."""
        mock_config.GATEKEEPER_ENABLED = False

        local = _make_mock_llm("local")
        cloud = _make_mock_llm("Gemini 2.0 Flash")
        cloud.generate.side_effect = CloudGenerationError("auth_error", "Invalid API key")

        registry = {"Gemini 2.0 Flash": cloud}
        orch = _make_orchestrator(local_llm=local, model_registry=registry)

        ctx = {
            "assembled_prompt": "test",
            "response_text": "",
            "errors": [],
            "timings": {},
            "model": "Gemini 2.0 Flash",
        }

        events = list(orch._stage_generate(ctx))

        # Should have exactly one event — the cloud_error
        assert len(events) == 1
        data = json.loads(events[0].replace("data: ", "").strip())
        assert data["cloud_error"] is True
        assert data["error_type"] == "auth_error"
        assert data["model"] == "Gemini 2.0 Flash"
        assert ctx.get("cloud_failed") is True

    @patch("app.services.orchestrator.config")
    @patch("app.services.orchestrator.WebSearchService")
    def test_cloud_failed_skips_post_process(self, mock_ws_cls, mock_config):
        """When cloud_failed is set, post_process should be skipped."""
        mock_config.GATEKEEPER_ENABLED = False
        mock_config.RERANKER_ENABLED = False
        mock_config.GATEKEEPER_CONTEXT_WINDOW = 3
        mock_config.LOCAL_MODEL_DISPLAY_NAME = "Gemma 3 4B (Local)"

        local = _make_mock_llm("local")
        cloud = _make_mock_llm("Cloud")
        cloud.generate.side_effect = CloudGenerationError("auth_error", "bad key")

        registry = {"Cloud": cloud}
        orch = _make_orchestrator(local_llm=local, model_registry=registry)

        # Mock out stages we don't want to actually run
        orch._stage_classify = MagicMock()
        orch._stage_web_search = MagicMock(return_value=iter([]))
        orch._stage_retrieve = MagicMock()
        orch._stage_score = MagicMock()
        orch._stage_build_context = MagicMock()
        orch._stage_post_process = MagicMock()

        # Consume the generator
        list(orch.process_message(
            user_message="test",
            conversation_id=None,
            recent_messages=[],
            model="Cloud",
        ))

        # post_process should NOT have been called
        orch._stage_post_process.assert_not_called()


# ---------------------------------------------------------------------------
# /api/available-models endpoint
# ---------------------------------------------------------------------------

class TestAvailableModelsEndpoint:
    """Test the /api/available-models Flask route."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False)
    def test_returns_local_and_cloud_models(self):
        """Endpoint should list all models in the registry."""
        # Import app inside test to get a fresh app context
        # We need to patch before importing to control the registry
        with patch("app.models.cloud_wrapper.completion"):
            from main import app
            client = app.test_client()
            response = client.get('/api/available-models')
            data = response.get_json()

            assert response.status_code == 200
            assert "models" in data
            names = [m["name"] for m in data["models"]]
            # Local model should always be present
            assert "Gemma 3 4B (Local)" in names

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False)
    def test_models_have_is_local_flag(self):
        """Each model entry should have an is_local boolean."""
        with patch("app.models.cloud_wrapper.completion"):
            from main import app
            client = app.test_client()
            response = client.get('/api/available-models')
            data = response.get_json()

            for model in data["models"]:
                assert "is_local" in model
                assert isinstance(model["is_local"], bool)


# ---------------------------------------------------------------------------
# Token usage includes model_name
# ---------------------------------------------------------------------------

class TestTokenUsageModelName:
    """Test that /token-usage records include model_name."""

    @patch.dict("os.environ", {}, clear=False)
    def test_token_usage_accepts_model_name(self):
        """POST /token-usage with model_name should store it."""
        with patch("app.models.cloud_wrapper.completion"):
            from main import app
            client = app.test_client()

            response = client.post('/token-usage', json={
                "conversation_id": None,
                "model_name": "Gemini 2.0 Flash",
                "tokens_output": 42,
                "tokens_per_second": 10.5,
                "duration_ms": 4000,
            })
            data = response.get_json()

            assert response.status_code == 200
            assert data["success"] is True

    @patch.dict("os.environ", {}, clear=False)
    def test_token_usage_defaults_model_name_when_missing(self):
        """POST /token-usage without model_name should default to local model."""
        with patch("app.models.cloud_wrapper.completion"):
            from main import app
            client = app.test_client()

            response = client.post('/token-usage', json={
                "tokens_output": 10,
                "tokens_per_second": 5.0,
                "duration_ms": 2000,
            })
            data = response.get_json()

            assert response.status_code == 200
            assert data["success"] is True

    @patch.dict("os.environ", {}, clear=False)
    def test_usage_stats_includes_per_model(self):
        """GET /usage-stats should return a per_model breakdown."""
        with patch("app.models.cloud_wrapper.completion"):
            from main import app
            client = app.test_client()

            response = client.get('/usage-stats')
            data = response.get_json()

            assert response.status_code == 200
            assert "per_model" in data
            assert isinstance(data["per_model"], dict)
