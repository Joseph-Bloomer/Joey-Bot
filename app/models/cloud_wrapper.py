"""Cloud LLM wrapper using LiteLLM for remote provider routing."""

import os
import logging
from typing import Generator, Optional, Union, List

from litellm import completion
from litellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIError,
)

from app.models.base import BaseLLM

logger = logging.getLogger(__name__)


class CloudGenerationError(Exception):
    """Raised when a cloud model fails during generation."""

    def __init__(self, error_type: str, message: str):
        """
        Args:
            error_type: One of "auth_error", "rate_limit", "network_error", "unknown".
            message: Human-readable error description.
        """
        self.error_type = error_type
        self.message = message
        super().__init__(message)


class CloudWrapper(BaseLLM):
    """
    Cloud LLM wrapper using LiteLLM.

    Routes requests to remote providers (Gemini, OpenAI, Anthropic, etc.)
    via LiteLLM's unified interface. Only supports chat generation —
    embeddings and structured JSON extraction must use the local model.
    """

    def __init__(self, provider: str, model: str, api_key: str):
        """
        Initialize cloud wrapper.

        Args:
            provider: LiteLLM provider prefix (e.g., "gemini", "openai")
            model: Model name (e.g., "gemini-2.0-flash")
            api_key: API key for the provider
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.litellm_model = f"{provider}/{model}"

        # LiteLLM reads API keys from env vars per provider
        env_key = self._provider_env_key(provider)
        if env_key:
            os.environ[env_key] = api_key

        logger.info(f"CloudWrapper initialized: {self.litellm_model}")

    @staticmethod
    def _provider_env_key(provider: str) -> Optional[str]:
        """Map provider name to the env var LiteLLM expects."""
        mapping = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        return mapping.get(provider)

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate response from prompt using a cloud LLM.

        Raises CloudGenerationError on provider failures so the orchestrator
        can present a user-facing fallback prompt.
        """
        try:
            response = completion(
                model=self.litellm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=stream,
                temperature=temperature,
            )

            if stream:
                return self._stream_tokens(response)
            return response.choices[0].message.content

        except AuthenticationError as e:
            logger.error(f"[CloudWrapper] Auth error for {self.litellm_model}: {e}")
            raise CloudGenerationError(
                "auth_error",
                "Authentication failed \u2014 check your API key"
            ) from e

        except RateLimitError as e:
            logger.error(f"[CloudWrapper] Rate limit for {self.litellm_model}: {e}")
            raise CloudGenerationError(
                "rate_limit",
                "Rate limit exceeded \u2014 try again later"
            ) from e

        except APIConnectionError as e:
            logger.error(f"[CloudWrapper] Connection error for {self.litellm_model}: {e}")
            raise CloudGenerationError(
                "network_error",
                "Could not reach the cloud provider"
            ) from e

        except APIError as e:
            logger.error(f"[CloudWrapper] API error for {self.litellm_model}: {e}")
            raise CloudGenerationError(
                "unknown",
                f"API error: {e}"
            ) from e

        except CloudGenerationError:
            raise

        except Exception as e:
            logger.error(f"[CloudWrapper] Unexpected error for {self.litellm_model}: {e}")
            raise CloudGenerationError(
                "unknown",
                str(e)
            ) from e

    def _stream_tokens(self, response) -> Generator[str, None, None]:
        """Yield tokens from streaming response."""
        try:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except AuthenticationError as e:
            raise CloudGenerationError("auth_error", "Authentication failed \u2014 check your API key") from e
        except RateLimitError as e:
            raise CloudGenerationError("rate_limit", "Rate limit exceeded \u2014 try again later") from e
        except APIConnectionError as e:
            raise CloudGenerationError("network_error", "Could not reach the cloud provider") from e
        except Exception as e:
            logger.error(f"[CloudWrapper] Streaming error: {e}")
            raise CloudGenerationError("unknown", str(e)) from e

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Not supported — embeddings must use the local model to match existing vectors."""
        raise NotImplementedError(
            "Embeddings must use the local model to match existing vectors"
        )

    def generate_json(self, prompt: str, max_tokens: int = None) -> Optional[str]:
        """Not supported — structured extraction must route to the local model."""
        raise NotImplementedError(
            "Cloud models are not used for structured extraction \u2014 route to local model"
        )
