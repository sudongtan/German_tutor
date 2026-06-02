"""
Abstract base class defining the interface for vector store backends.

Subclass VectorStore and implement `index` and `search`.
Swap the backend by changing stores.py — nothing else needs to change.
"""

from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def index(self, text: str, source: str = "manual", title: str = "") -> int:
        """Chunk, embed, and store text. Returns the number of chunks indexed."""

    @abstractmethod
    def search(self, query: str, limit: int = 5, alpha: float = 0.5) -> list[str]:
        """Hybrid search. Returns a list of matching text chunks."""
