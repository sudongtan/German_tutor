"""
Active vector store instance.

To swap databases, change the import and instantiation here — nothing else changes.
"""

from src.german_tutor.vector_store import VectorStore
from src.german_tutor.weaviate_client import WeaviateStore

store: VectorStore = WeaviateStore()
