"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Generator, Optional, Union, List


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    Provides a unified interface for chat generation and embeddings,
    enabling easy switching between different LLM backends (Ollama, OpenAI, etc.).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.7
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate response from prompt.

        Args:
            prompt: The input prompt text
            stream: If True, yield tokens as they're generated
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            If stream=False: Complete response string
            If stream=True: Generator yielding tokens
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats, or None on error
        """
        pass

    @abstractmethod
    def generate_json(self, prompt: str) -> Optional[str]:
        """
        Generate response expecting JSON output.

        Args:
            prompt: Prompt requesting JSON output

        Returns:
            Raw response string (caller should parse JSON), or None on error
        """
        pass
