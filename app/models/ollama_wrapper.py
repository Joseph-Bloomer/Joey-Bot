"""Ollama LLM wrapper using LiteLLM for provider abstraction."""

from typing import Generator, Optional, Union, List, Dict, Any
import requests
from litellm import completion, embedding
from app.models.base import BaseLLM
import config


class OllamaWrapper(BaseLLM):
    """
    Ollama LLM wrapper using LiteLLM.

    LiteLLM provides a unified interface that makes it easy to switch
    between providers (Ollama, OpenAI, Anthropic) by changing model strings.
    """

    def __init__(
        self,
        model: str = config.LITELLM_CHAT_MODEL,
        embedding_model: str = config.LITELLM_EMBEDDING_MODEL,
        api_base: str = config.OLLAMA_BASE_URL,
    ):
        """
        Initialize Ollama wrapper.

        Args:
            model: LiteLLM model string (e.g., "ollama/gemma3:4b")
            embedding_model: LiteLLM embedding model string
            api_base: Ollama API base URL
        """
        self.model = model
        self.embedding_model = embedding_model
        self.api_base = api_base

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.7
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate response from prompt using LiteLLM.

        Args:
            prompt: Input prompt text
            stream: If True, yield tokens as generated
            temperature: Sampling temperature

        Returns:
            Complete response or token generator
        """
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=stream,
                temperature=temperature,
                api_base=self.api_base
            )

            if stream:
                return self._stream_tokens(response)
            return response.choices[0].message.content

        except Exception as e:
            print(f"Generation error: {e}")
            if stream:
                return self._error_generator(str(e))
            return ""

    def _stream_tokens(self, response) -> Generator[str, None, None]:
        """Yield tokens from streaming response."""
        try:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"[Error: {e}]"

    def _error_generator(self, error: str) -> Generator[str, None, None]:
        """Generate error message for streaming mode."""
        yield f"[Error: {error}]"

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding vector using LiteLLM.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None on error
        """
        try:
            response = embedding(
                model=self.embedding_model,
                input=[text],
                api_base=self.api_base
            )
            return response.data[0]["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def generate_json(self, prompt: str, max_tokens: int = None) -> Optional[str]:
        """
        Generate response expecting JSON output (non-streaming).

        Args:
            prompt: Prompt requesting JSON output
            max_tokens: Optional max tokens for response length

        Returns:
            Raw response string or None on error
        """
        try:
            kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=config.JSON_GENERATION_TEMPERATURE,
                api_base=self.api_base
            )
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            response = completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"JSON generation error: {e}")
            return None

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Query Ollama for downloaded models, filter out embedding models.

        Returns:
            List of model info dicts with name, size, etc.
        """
        try:
            response = requests.get(f"{self.api_base}/api/tags")
            response.raise_for_status()
            models = response.json().get('models', [])
            # Filter out embedding models
            return [m for m in models
                    if not any(p in m['name'].lower() for p in config.EMBEDDING_MODEL_PATTERNS)]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def unload_current_model(self) -> bool:
        """
        Unload current model from GPU using keep_alive: 0.

        Returns:
            True if successful, False otherwise
        """
        try:
            model_name = self.model.replace("ollama/", "")
            requests.post(
                f"{self.api_base}/api/generate",
                json={"model": model_name, "keep_alive": 0}
            )
            return True
        except Exception as e:
            print(f"Error unloading model: {e}")
            return False

    def switch_model(self, model_name: str) -> bool:
        """
        Unload current model and switch to new one.

        Args:
            model_name: Name of the model to switch to (without ollama/ prefix)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.unload_current_model()
            self.model = f"ollama/{model_name}"
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False

    def get_current_model(self) -> str:
        """
        Return current model name without ollama/ prefix.

        Returns:
            Model name string
        """
        return self.model.replace("ollama/", "")
