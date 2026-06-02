"""
Weaviate implementation of VectorStore, plus transport utilities.

fetch_url and chunk_text are not database-specific and stay here as helpers.
WeaviateStore implements the VectorStore protocol.
"""

import logging

import requests
import spacy
import weaviate
import weaviate.classes.config as wvc
from bs4 import BeautifulSoup

from src.german_tutor.config import settings
from src.german_tutor.vector_store import VectorStore

logger = logging.getLogger(__name__)

_COLLECTION = "GermanChunk"
_client: weaviate.WeaviateClient | None = None
_nlp: spacy.Language | None = None

_CONTENT_POS = {"NOUN", "VERB", "ADJ"}


def _get_client() -> weaviate.WeaviateClient:
    global _client
    if _client is None or not _client.is_connected():
        _client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_port,
            headers={"X-OpenAI-Api-Key": settings.openai_api_key},
        )
    return _client


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("de_core_news_sm")
        except OSError:
            raise RuntimeError(
                "German spaCy model not found. "
                "Run: python -m spacy download de_core_news_sm"
            )
    return _nlp


def _ensure_collection() -> None:
    client = _get_client()
    if not client.collections.exists(_COLLECTION):
        client.collections.create(
            name=_COLLECTION,
            vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-small"
            ),
            properties=[
                wvc.Property(name="text", data_type=wvc.DataType.TEXT),
                wvc.Property(name="source", data_type=wvc.DataType.TEXT),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                wvc.Property(name="chunk_index", data_type=wvc.DataType.INT),
                wvc.Property(name="lemmas", data_type=wvc.DataType.TEXT),
            ],
        )
    else:
        # Add lemmas property to existing collections that predate this field
        collection = client.collections.get(_COLLECTION)
        existing = {p.name for p in collection.config.get().properties}
        if "lemmas" not in existing:
            collection.config.add_property(
                wvc.Property(name="lemmas", data_type=wvc.DataType.TEXT)
            )


def fetch_url(url: str) -> tuple[str, str]:
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "GermanTutor/1.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL: {e}") from e
    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text = " ".join(
        p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
    )
    return title, text


def chunk_text(text: str, chunk_words: int = 200, overlap_words: int = 30) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i : i + chunk_words]
        chunks.append(" ".join(chunk))
        if len(chunk) < chunk_words:
            break
        i += chunk_words - overlap_words
    return chunks


def extract_lemmas(text: str) -> str:
    doc = _get_nlp()(text)
    lemmas = sorted(
        {
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and token.pos_ in _CONTENT_POS
            and len(token.lemma_) > 2
        }
    )
    return " ".join(lemmas)


class WeaviateStore(VectorStore):
    def index(self, text: str, source: str = "manual", title: str = "") -> int:
        chunks = chunk_text(text)
        try:
            _ensure_collection()
            collection = _get_client().collections.get(_COLLECTION)
            collection.data.insert_many(
                [
                    {
                        "text": chunk,
                        "source": source,
                        "title": title,
                        "chunk_index": i,
                        "lemmas": extract_lemmas(chunk),
                    }
                    for i, chunk in enumerate(chunks)
                ]
            )
        except Exception as e:
            raise RuntimeError(f"Weaviate indexing failed: {e}") from e
        return len(chunks)

    def search(self, query: str, limit: int = 5, alpha: float = 0.5) -> list[str]:
        try:
            _ensure_collection()
            collection = _get_client().collections.get(_COLLECTION)
            response = collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=limit,
                query_properties=["text", "lemmas"],
            )
        except Exception as e:
            raise RuntimeError(f"Weaviate search failed: {e}") from e
        return [obj.properties["text"] for obj in response.objects]
