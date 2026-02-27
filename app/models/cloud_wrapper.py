"""Cloud LLM wrapper using LiteLLM for cloud provider abstraction."""

from typing import Generator, Optional, Union, List
from litellm import completion
from app.models.base import BaseLLM


class CloudGenerationError(Exception):
    """Raised when cloud model generation fails."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)


class CloudWrapper(BaseLLM):
    """
    Cloud LLM wrapper using LiteLLM.

    Supports any provider LiteLLM handles (Gemini, OpenAI, Anthropic, etc.)
    by passing the correct model string and API key.
    """

    def __init__(self, provider: str, model: str, api_key: str, display_name: str = ""):
        """
        Initialize cloud wrapper.

        Args:
            provider: LiteLLM provider prefix (e.g., "gemini", "openai")
            model: Model identifier (e.g., "gemini-2.0-flash")
            api_key: API key for the provider
            display_name: Human-readable name for the model
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.display_name = display_name
        self.litellm_model = f"{provider}/{model}"

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.7
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response from cloud model."""
        try:
            if stream:
                return self._stream_tokens(prompt, temperature)
            else:
                response = completion(
                    model=self.litellm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    api_key=self.api_key,
                )
                return response.choices[0].message.content
        except Exception as e:
            error_type = self._classify_error(e)
            raise CloudGenerationError(error_type, str(e))

    def _stream_tokens(self, prompt: str, temperature: float) -> Generator[str, None, None]:
        """Stream tokens from cloud model."""
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
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            error_type = self._classify_error(e)
            raise CloudGenerationError(error_type, str(e))

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Cloud models do not support embeddings — always use local."""
        raise NotImplementedError("Cloud models do not support embeddings. Use local model.")

    def generate_json(self, prompt: str, max_tokens: int = None) -> Optional[str]:
        """Cloud models do not support JSON generation — always use local."""
        raise NotImplementedError("Cloud models do not support JSON generation. Use local model.")

    @staticmethod
    def _classify_error(error: Exception) -> str:
        """Classify an exception into a user-facing error type."""
        error_str = str(error).lower()
        if "auth" in error_str or "api_key" in error_str or "401" in error_str:
            return "auth_error"
        if "rate" in error_str or "429" in error_str or "quota" in error_str:
            return "rate_limit"
        if "connect" in error_str or "timeout" in error_str or "network" in error_str:
            return "network_error"
        return "unknown"
