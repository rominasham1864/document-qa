"""Multilingual E5 embeddings with distinct query and passage prefixes."""

import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Lazy-load the model only when embeddings are actually requested."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


class E5Embeddings(Embeddings):
    """LangChain-compatible adapter, reusable by ChromaDB in task 7."""

    def __init__(self, model_name: str | None = None, batch_size: int = 16):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL
        )
        self.batch_size = batch_size

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings.")

        vectors = _load_model(self.model_name).encode(
            [f"{prefix}{text}" for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks for indexing."""
        return self._encode(texts, "passage: ")

    def embed_query(self, text: str) -> list[float]:
        """Embed one question using the required retrieval prefix."""
        return self._encode([text], "query: ")[0]
