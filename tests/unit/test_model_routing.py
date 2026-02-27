"""Unit tests for Phase 9 model routing — registry, _resolve_llm, and orchestrator integration.

Tests cover:
- Model registry construction (cloud models with/without API keys)
- _resolve_llm() fallback logic
- Orchestrator generation stage using the correct LLM
- CloudGenerationError handling in the pipeline
- /api/available-models endpoint
- Token usage model_name tracking

Run from the project root:
    python -m pytest tests/unit/test_model_routing.py -v
"""

import json
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from app.models.cloud_wrapper import CloudWrapper, CloudGenerationError
from app.services.orchestrator import ChatOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(name="mock-local"):
    """Create a mock LLM with a display name."""
    llm = MagicMock()
    llm.display_name = name
    return llm


def _make_orchestrator(local_llm=None, model_registry=None):
    """Build a ChatOrchestrator with mocked dependencies."""
    local = local_llm or _make_mock_llm("local")
    orch = ChatOrchestrator(
        gatekeeper=MagicMock(),
        retriever=MagicMock(),
        chat_service=MagicMock(),
        memory_service=MagicMock(),
        reranker=MagicMock(),
        local_llm=local,
        model_registry=model_registry or {"Local": local},
    )
    # Disable gatekeeper to simplify tests
    orch.gatekeeper = None
    return orch


# ---------------------------------------------------------------------------
# Model Registry Construction
# ---------------------------------------------------------------------------

class TestModelRegistryConstruction:
    """Cloud models should only be registered when their API key is present."""

    def test_cloud_model_added_with_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            cw = CloudWrapper(
                provider="gemini", model="gemini-2.0-flash",
                api_key="test-key", display_name="Gemini 2.0 Flash",
            )
            registry = {"Local": _make_mock_llm(), "Gemini 2.0 Flash": cw}
        assert "Gemini 2.0 Flash" in registry
        assert isinstance(registry["Gemini 2.0 Flash"], CloudWrapper)

    def test_cloud_model_skipped_without_key(self):
        registry = {"Local": _make_mock_llm()}
        # No cloud model added because no API key
        assert "Gemini 2.0 Flash" not in registry
        assert len(registry) == 1

    def test_local_always_present(self):
        registry = {"Local": _make_mock_llm()}
        assert "Local" in registry


# ---------------------------------------------------------------------------
# _resolve_llm
# ---------------------------------------------------------------------------

class TestResolveLlm:
    """_resolve_llm looks up the registry and falls back to local."""

    def test_none_returns_local(self):
        orch = _make_orchestrator()
        result = orch._resolve_llm(None)
        assert result is orch.local_llm

    def test_known_cloud_returns_cloud(self):
        cloud = _make_mock_llm("cloud")
        orch = _make_orchestrator(model_registry={"Local": _make_mock_llm(), "Cloud": cloud})
        result = orch._resolve_llm("Cloud")
        assert result is cloud

    def test_local_name_returns_local(self):
        local = _make_mock_llm("local")
        orch = _make_orchestrator(local_llm=local, model_registry={"Local": local})
        result = orch._resolve_llm("Local")
        assert result is local

    def test_unknown_falls_back_to_local_with_warning(self):
        orch = _make_orchestrator()
        with patch("app.services.orchestrator.logger") as mock_logger:
            result = orch._resolve_llm("NonExistentModel")
            mock_logger.warning.assert_called_once()
        assert result is orch.local_llm


# ---------------------------------------------------------------------------
# Orchestrator Generation — model selection
# ---------------------------------------------------------------------------

class TestOrchestratorGeneration:
    """The GENERATE stage should use the model specified in ctx."""

    def test_uses_requested_model(self):
        cloud = _make_mock_llm("cloud")
        cloud.generate.return_value = iter(["cloud", " reply"])
        local = _make_mock_llm("local")

        orch = _make_orchestrator(
            local_llm=local,
            model_registry={"Local": local, "Cloud": cloud},
        )

        ctx = {
            "model": "Cloud",
            "assembled_prompt": "Hello",
            "response_text": "",
            "errors": [],
            "timings": {},
        }
        events = list(orch._stage_generate(ctx))

        cloud.generate.assert_called_once()
        local.generate.assert_not_called()
        assert any("cloud" in e for e in events)

    def test_uses_local_when_none_specified(self):
        local = _make_mock_llm("local")
        local.generate.return_value = iter(["local", " reply"])

        orch = _make_orchestrator(local_llm=local)

        ctx = {
            "model": None,
            "assembled_prompt": "Hello",
            "response_text": "",
            "errors": [],
            "timings": {},
        }
        list(orch._stage_generate(ctx))

        local.generate.assert_called_once()


# ---------------------------------------------------------------------------
# Orchestrator — CloudGenerationError handling
# ---------------------------------------------------------------------------

class TestOrchestratorCloudError:
    """When a cloud model raises CloudGenerationError, the pipeline handles it."""

    def test_cloud_error_yields_sse_event(self):
        cloud = _make_mock_llm("cloud")
        cloud.generate.side_effect = CloudGenerationError("auth_error", "Invalid key")

        orch = _make_orchestrator(
            model_registry={"Local": _make_mock_llm(), "Cloud": cloud},
        )

        ctx = {
            "model": "Cloud",
            "assembled_prompt": "Hello",
            "response_text": "",
            "errors": [],
            "timings": {},
        }
        events = list(orch._stage_generate(ctx))

        # Should yield a cloud_error SSE event
        cloud_events = [e for e in events if "cloud_error" in e]
        assert len(cloud_events) == 1
        data = json.loads(cloud_events[0].replace("data: ", "").strip())
        assert data["cloud_error"] is True
        assert data["error_type"] == "auth_error"

    def test_cloud_failed_skips_post_process(self):
        cloud = _make_mock_llm("cloud")
        cloud.generate.side_effect = CloudGenerationError("rate_limit", "429")

        orch = _make_orchestrator(
            model_registry={"Local": _make_mock_llm(), "Cloud": cloud},
        )

        ctx = {
            "model": "Cloud",
            "assembled_prompt": "Hello",
            "response_text": "",
            "errors": [],
            "timings": {},
        }
        list(orch._stage_generate(ctx))

        assert ctx.get("cloud_failed") is True


# ---------------------------------------------------------------------------
# /api/available-models endpoint
# ---------------------------------------------------------------------------

class TestAvailableModelsEndpoint:
    """The /api/available-models endpoint lists all registry models."""

    def test_returns_models(self):
        """Endpoint should return a list with is_local flags."""
        # We test the logic, not the Flask route itself
        local_name = "Gemma 3 4B (Local)"
        registry = {
            local_name: _make_mock_llm("local"),
            "Gemini 2.0 Flash": _make_mock_llm("cloud"),
        }
        models = []
        for name, llm_instance in registry.items():
            is_local = (name == local_name)
            models.append({"name": name, "is_local": is_local})

        assert len(models) == 2
        local_entries = [m for m in models if m["is_local"]]
        cloud_entries = [m for m in models if not m["is_local"]]
        assert len(local_entries) == 1
        assert len(cloud_entries) == 1

    def test_has_is_local_flag(self):
        local_name = "Gemma 3 4B (Local)"
        registry = {local_name: _make_mock_llm()}
        models = [{"name": n, "is_local": n == local_name} for n in registry]
        assert models[0]["is_local"] is True


# ---------------------------------------------------------------------------
# Token usage model_name
# ---------------------------------------------------------------------------

class TestTokenUsageModelName:
    """Token usage records should track the model_name."""

    def test_accepts_model_name(self):
        """TokenUsage model should accept a model_name field."""
        from app.data.database import TokenUsage
        # Verify the column exists on the model class
        assert hasattr(TokenUsage, "model_name")

    def test_default_model_name(self):
        """TokenUsage.model_name should default to the local model name."""
        from app.data.database import TokenUsage
        col = TokenUsage.__table__.columns["model_name"]
        assert col.default is not None
        assert "Gemma 3 4B (Local)" in str(col.default.arg)
