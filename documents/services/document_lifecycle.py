"""Keep saved Django documents and their Chroma indexes in sync.

Entry points (Admin/API) call these functions explicitly. Avoid Django signals:
ordinary ORM writes and migrations must not unexpectedly load the E5 model.
"""

from django.db import transaction

from documents.models import Document
from documents.services.vector_store import delete_document_index, index_document


def _schedule_unused_file_cleanup(storage, filename: str) -> None:
    """Delete an unreferenced media file only after its DB transaction commits.

    Django doesn't automatically delete FileField files with their model rows.
    The existence check also avoids deleting a filename used by another row.
    """
    if not filename:
        return

    def clean():
        if not Document.objects.filter(file=filename).exists():
            storage.delete(filename)

    transaction.on_commit(clean, robust=True)


def index_saved_document(
    document: Document, *, previous_file_name: str | None = None, vector_store=None
) -> int:
    """Index the current saved document; clean up a replaced file on success.

    Raises an exception if Chroma is unavailable. The Django row is intentionally
    left intact so ``manage.py index_document --document-id ID`` can retry.
    """
    count = index_document(document, vector_store=vector_store)
    if previous_file_name and previous_file_name != document.file.name:
        _schedule_unused_file_cleanup(document.file.storage, previous_file_name)
    return count


def delete_document_and_index(document: Document, *, vector_store=None) -> None:
    """Remove the Chroma index before removing the DB row and its media file.

    An index failure stops the deletion, avoiding a knowingly orphaned index.
    SQLite and Chroma are independent stores; this is not a distributed transaction.
    """
    document_id = document.pk
    if document_id is None:
        raise ValueError("Cannot delete an unsaved Document.")
    storage = document.file.storage
    filename = document.file.name
    delete_document_index(document_id, vector_store=vector_store)
    document.delete()
    _schedule_unused_file_cleanup(storage, filename)
