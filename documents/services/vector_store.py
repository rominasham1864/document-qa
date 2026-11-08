"""Persist and search DOCX chunks using the existing multilingual E5 model.

SQLite stores Document records; Chroma stores chunks, vectors, and metadata.
"""

import os

from chromadb.config import Settings as ChromaSettings
from django.conf import settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from documents.models import Document
from documents.services.chunking import split_document_text
from documents.services.embeddings import E5Embeddings


DEFAULT_COLLECTION_NAME = "document_chunks"


def get_vector_store(
    *,
    persist_directory: str | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Open a local, persistent Chroma collection (or create it if absent).

    Optional parameters allow tests to use temporary storage and fake embeddings.
    """
    directory = persist_directory or os.getenv(
        "CHROMA_PERSIST_DIRECTORY", str(settings.BASE_DIR / "chroma_data")
    )
    return Chroma(
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
        embedding_function=embeddings if embeddings is not None else E5Embeddings(),
        persist_directory=str(directory),
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )


def index_document(
    document: Document, *, vector_store: Chroma | None = None,
    index_key: str | None = None,
) -> int:
    """Upsert current chunks and remove obsolete chunks for just this document.

    Re-running is safe: IDs remain stable; chunks of other documents are preserved.
    """
    if document.pk is None:
        raise ValueError("Save the document in SQLite before indexing.")
    if not document.extracted_text or not document.extracted_text.strip():
        raise ValueError("The document has no extracted text to index.")

    chunks = split_document_text(
        document_id=document.pk,
        title=document.title,
        text=document.extracted_text,
    )
    if not chunks:
        raise ValueError("No chunks were produced for this document.")
    if index_key is not None:
        for chunk in chunks:
            chunk.metadata["index_key"] = index_key

    store = vector_store if vector_store is not None else get_vector_store()
    # Fetch IDs only; vectors are not needed for cleaning up stale chunks.
    previous_ids = set(store.get(where={"document_id": int(document.pk)}, include=[])["ids"])
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    # Chroma upserts explicit IDs; old records with the same IDs get updated.
    # Do not delete older chunks until all new chunks were written successfully.
    store.add_documents(documents=chunks, ids=chunk_ids)
    stale_ids = sorted(previous_ids - set(chunk_ids))
    if stale_ids:
        store.delete(ids=stale_ids)
    return len(chunk_ids)


def search_chunks(
    query: str,
    *,
    k: int = 4,
    document_id: int | None = None,
    vector_store: Chroma | None = None,
):
    """Return the most relevant chunks; no answer generation happens here."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Provide a non-empty query.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if document_id is not None and document_id <= 0:
        raise ValueError("document_id must be positive.")

    # The only search-authorized vectors are those whose generation is the
    # current READY generation in SQLite. This excludes older/partial Chroma
    # data even when an update fails halfway through.
    eligible = Document.objects.filter(index_status=Document.IndexStatus.READY).exclude(
        index_key=""
    )
    if document_id is not None:
        eligible = eligible.filter(pk=document_id)
    keys = list(eligible.values_list("index_key", flat=True))
    if not keys:
        return []

    store = vector_store if vector_store is not None else get_vector_store()
    where = {"index_key": keys[0]} if len(keys) == 1 else {
        "index_key": {"$in": keys}
    }
    return store.similarity_search(query.strip(), k=k, filter=where)


def delete_document_index(document_id: int, *, vector_store: Chroma | None = None) -> int:
    """Remove vectors for one document, including after its SQLite row is deleted."""
    if document_id <= 0:
        raise ValueError("document_id must be positive.")
    store = vector_store if vector_store is not None else get_vector_store()
    ids = store.get(where={"document_id": document_id}, include=[])["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)
