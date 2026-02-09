"""Ollama LLM wrapper using LiteLLM for provider abstraction."""

from typing import Generator, Optional, Union, List
from litellm import completion, embedding
from app.models.base import BaseLLM


class OllamaWrapper(BaseLLM):
    """
    Ollama LLM wrapper using LiteLLM.

    LiteLLM provides a unified interface that makes it easy to switch
    between providers (Ollama, OpenAI, Anthropic) by changing model strings.
    """

    def __init__(
        self,
        model: str = "ollama/gemma3:4b",
        embedding_model: str = "ollama/nomic-embed-text",
        api_base: str = "http://localhost:11434"
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

    def generate_json(self, prompt: str) -> Optional[str]:
        """
        Generate response expecting JSON output (non-streaming).

        Args:
            prompt: Prompt requesting JSON output

        Returns:
            Raw response string or None on error
        """
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.3,  # Lower temperature for structured output
                api_base=self.api_base
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"JSON generation error: {e}")
            return None
