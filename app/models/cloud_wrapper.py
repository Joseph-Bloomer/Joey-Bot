"""Cloud LLM wrapper for external API providers (Gemini, OpenAI, etc.) via LiteLLM."""

from typing import Generator, Optional, Union, List

from litellm import completion
from app.models.base import BaseLLM
from utils.logger import get_logger

logger = get_logger()


class CloudGenerationError(Exception):
    """Raised when a cloud LLM call fails. Carries a typed error category."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type  # auth_error | rate_limit | network_error | unknown
        self.message = message
        super().__init__(message)


class CloudWrapper(BaseLLM):
    """
    LiteLLM-based wrapper for cloud model providers.

    Only supports ``generate()`` (text + streaming).  Embedding and JSON
    generation are intentionally unsupported — the local model handles those.
    """

    def __init__(self, provider: str, model: str, api_key: str,
                 display_name: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.display_name = display_name or f"{provider}/{model}"
        self.litellm_model = f"{provider}/{model}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> Union[str, Generator[str, None, None]]:
        try:
            if stream:
                return self._stream_tokens(prompt, temperature)
            response = completion(
                model=self.litellm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                api_key=self.api_key,
            )
            return response.choices[0].message.content
        except CloudGenerationError:
            raise
        except Exception as e:
            error_type = self._classify_error(e)
            raise CloudGenerationError(error_type, str(e)) from e

    def get_embedding(self, text: str) -> Optional[List[float]]:
        raise NotImplementedError(
            "Cloud models do not support embeddings. Use local model."
        )

    def generate_json(self, prompt: str, max_tokens: int = None) -> Optional[str]:
        raise NotImplementedError(
            "Cloud models do not support JSON generation. Use local model."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stream_tokens(
        self, prompt: str, temperature: float
    ) -> Generator[str, None, None]:
        try:
            response = completion(
                model=self.litellm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                api_key=self.api_key,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
        except CloudGenerationError:
            raise
        except Exception as e:
            error_type = self._classify_error(e)
            raise CloudGenerationError(error_type, str(e)) from e

    @staticmethod
    def _classify_error(error: Exception) -> str:
        """Map exception text to a typed error category."""
        msg = str(error).lower()
        if any(kw in msg for kw in ("auth", "api_key", "401", "invalid key")):
            return "auth_error"
        if any(kw in msg for kw in ("rate", "429", "quota", "limit")):
            return "rate_limit"
        if any(kw in msg for kw in ("connect", "timeout", "network", "unreachable")):
            return "network_error"
        return "unknown"
