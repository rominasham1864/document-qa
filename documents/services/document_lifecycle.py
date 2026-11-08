"""Explicit, retryable synchronization of Django documents and local Chroma.

Never equate a saved SQLite row with a ready search index. Chroma and SQLite
cannot share one transaction: the retrieval guard excludes non-READY rows and
vectors whose generation (index_key) is not the current SQLite generation.
"""

import hashlib
import uuid

from django.db import transaction

from documents.models import Document
from documents.services.vector_store import (
    delete_document_index,
    get_vector_store,
    index_document,
)


def _schedule_unused_file_cleanup(storage, filename: str) -> None:
    """Remove orphan media only after commit; don't remove shared filenames."""
    if not filename:
        return

    def cleanup():
        if not Document.objects.filter(file=filename).exists():
            storage.delete(filename)

    transaction.on_commit(cleanup, robust=True)


def get_index_snapshot(document: Document) -> dict:
    """Snapshot the *old* committed index before a serializer/Admin saves edits."""
    return {
        "index_status": document.index_status,
        "index_key": document.index_key,
        "indexed_content_hash": document.indexed_content_hash,
        "indexed_title": document.indexed_title,
        "indexed_chunk_count": document.indexed_chunk_count,
    }


def _title_metadata_only(store, *, index_key: str, title: str, expected_count: int) -> int:
    """Change only Chroma metadata; do NOT call embed_documents().

    Chroma's Collection.update(metadatas=...) does not calculate embeddings.
    The LangChain wrapper exposes its underlying Chroma Collection as _collection.
    """
    existing = store.get(where={"index_key": index_key}, include=["metadatas"])
    ids = existing["ids"]
    if not ids or len(ids) != expected_count:
        raise LookupError("The previously indexed Chroma chunks are incomplete.")
    metadatas = [dict(metadata or {}, title=title) for metadata in existing["metadatas"]]
    if any(metadata.get("index_key") != index_key for metadata in metadatas):
        raise LookupError("Indexed chunk metadata does not match the document.")
    # Direct Chroma metadata update, intentionally not LangChain update_documents.
    store._collection.update(ids=ids, metadatas=metadatas)
    return len(ids)


def index_saved_document(
    document: Document,
    *,
    previous_file_name: str | None = None,
    previous_index: dict | None = None,
    vector_store=None,
) -> int:
    """Make a saved document READY or FAILED; never expose stale vectors.

    Retry after a failed update always performs a full upsert. A READY prior
    generation with unchanged text only needs a metadata update, not embeddings.
    The caller must persist content edits with index_status=PENDING and
    index_key='' in the same SQLite save to avoid a stale-search window.
    """
    if document.pk is None:
        raise ValueError("Save the document before indexing.")

    # File cleanup is independent of indexing success: SQLite has already been
    # switched to the replacement filename, so the old file is unreferenced.
    if previous_file_name and previous_file_name != document.file.name:
        _schedule_unused_file_cleanup(document.file.storage, previous_file_name)

    previous = previous_index or get_index_snapshot(document)
    digest = hashlib.sha256(document.extracted_text.encode("utf-8")).hexdigest()
    token = uuid.uuid4().hex
    match = Document.objects.filter(
        pk=document.pk,
        title=document.title,
        extracted_text=document.extracted_text,
        file=document.file.name,
    )
    if not match.update(
        index_status=Document.IndexStatus.PROCESSING,
        index_key="",
        index_attempt=token,
        index_error="",
    ):
        raise RuntimeError("Document changed before indexing began; retry the current version.")

    try:
        if not document.extracted_text.strip():
            raise ValueError("Document contains no extracted text.")

        store = vector_store if vector_store is not None else get_vector_store()
        same_content = (
            previous["index_status"] == Document.IndexStatus.READY
            and bool(previous["index_key"])
            and previous["indexed_content_hash"] == digest
            and previous["indexed_chunk_count"] > 0
        )

        if same_content:
            try:
                count = _title_metadata_only(
                    store,
                    index_key=previous["index_key"],
                    title=document.title,
                    expected_count=previous["indexed_chunk_count"],
                ) if previous["indexed_title"] != document.title else _check_existing(
                    store, previous["index_key"], previous["indexed_chunk_count"]
                )
                key = previous["index_key"]
            except LookupError:
                # Partially missing data: rebuild instead of claiming READY.
                same_content = False

        if not same_content:
            key = f"{document.pk}:{token}"
            count = index_document(document, vector_store=store, index_key=key)
            # Do not publish READY on a partial/no-op Chroma write.
            _check_existing(store, key, count)

        updated = Document.objects.filter(
            pk=document.pk,
            index_attempt=token,
            title=document.title,
            extracted_text=document.extracted_text,
            file=document.file.name,
        ).update(
            index_status=Document.IndexStatus.READY,
            index_key=key,
            indexed_content_hash=digest,
            indexed_title=document.title,
            indexed_chunk_count=count,
            index_error="",
        )
        if not updated:
            raise RuntimeError("Document changed during indexing; result was not published.")
        document.refresh_from_db()
        return count
    except Exception as exc:
        # If another attempt has superseded this one, do not alter its status.
        Document.objects.filter(pk=document.pk, index_attempt=token).update(
            index_status=Document.IndexStatus.FAILED,
            index_key="",
            index_error=f"Indexing failed ({type(exc).__name__})."[:120],
        )
        document.refresh_from_db()
        raise


def _check_existing(store, index_key: str, expected_count: int) -> int:
    ids = store.get(where={"index_key": index_key}, include=[])["ids"]
    if len(ids) != expected_count:
        raise LookupError("Indexed Chroma chunks are missing.")
    return len(ids)


def delete_document_and_index(document: Document, *, vector_store=None) -> None:
    """Hide document from retrieval first, then remove Chroma, SQLite and file."""
    if document.pk is None:
        raise ValueError("Cannot delete an unsaved document.")
    document_id = document.pk
    storage = document.file.storage
    filename = document.file.name

    # A partial Chroma delete must never leave a document advertising READY.
    Document.objects.filter(pk=document_id).update(
        index_status=Document.IndexStatus.PROCESSING, index_key=""
    )
    try:
        delete_document_index(document_id, vector_store=vector_store)
    except Exception as exc:
        Document.objects.filter(pk=document_id).update(
            index_status=Document.IndexStatus.FAILED,
            index_error=f"Deletion failed ({type(exc).__name__})."[:120],
        )
        raise
    document.delete()
    _schedule_unused_file_cleanup(storage, filename)
