"""Task 15: real Chroma integration (fake, local vectors) through Admin and API.

No Hugging Face download, OpenRouter request, or production Chroma data access.
"""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from docx import Document as WordDocument
from langchain_core.embeddings import Embeddings
from rest_framework.test import APITestCase

from documents.models import Document
from documents.services.document_lifecycle import (
    delete_document_and_index, index_saved_document,
)
from documents.services.vector_store import get_vector_store, search_chunks


class TinyEmbeddings(Embeddings):
    """Deterministic embeddings to make the indexing test fast and offline."""

    @staticmethod
    def _vec(text):
        text = text.lower()
        if "machine" in text or "ماشین" in text:
            return [1.0, 0.0, 0.0]
        if "cake" in text or "کیک" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        return [self._vec(x) for x in texts]

    def embed_query(self, text):
        return self._vec(text)


def upload_docx(filename, paragraph):
    word = WordDocument()
    word.add_paragraph(paragraph)
    stream = BytesIO()
    word.save(stream)
    return SimpleUploadedFile(
        filename, stream.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


class AutomaticIndexTests(APITestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        media = override_settings(MEDIA_ROOT=str(root / "media"))
        media.enable()
        self.addCleanup(media.disable)
        self.store = get_vector_store(
            persist_directory=str(root / "chroma"), embeddings=TinyEmbeddings()
        )
        self.user = get_user_model().objects.create_superuser(
            "sync-admin", "sync@example.test", "test-password"
        )
        self.client.force_authenticate(user=self.user)

        self.api_index = patch(
            "documents.views.index_saved_document",
            side_effect=lambda document, **kwargs: index_saved_document(
                document, **kwargs, vector_store=self.store
            ),
        )
        self.api_delete = patch(
            "documents.views.delete_document_and_index",
            side_effect=lambda document: delete_document_and_index(
                document, vector_store=self.store
            ),
        )
        self.api_index.start()
        self.api_delete.start()
        self.addCleanup(self.api_index.stop)
        self.addCleanup(self.api_delete.stop)

    def document_ids(self, doc_id):
        return self.store.get(where={"document_id": doc_id}, include=[])["ids"]

    def upload(self, name="ml.docx", text="Machine learning extracts patterns."):
        return self.client.post(
            reverse("document-list"),
            {"title": "AI", "file": upload_docx(name, text)},
            format="multipart",
        )

    def test_api_upload_is_immediately_searchable(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201)
        doc_id = response.data["id"]
        self.assertEqual(self.document_ids(doc_id), [f"{doc_id}:0"])
        found = search_chunks("machine", k=1, vector_store=self.store)
        self.assertEqual(found[0].metadata["document_id"], doc_id)

    def test_title_change_updates_chroma_metadata_without_duplicate(self):
        response = self.upload()
        doc_id = response.data["id"]
        changed = self.client.patch(
            reverse("document-detail", args=[doc_id]),
            {"title": "Updated title"}, format="json",
        )
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(self.document_ids(doc_id), [f"{doc_id}:0"])
        found = search_chunks("machine", k=1, vector_store=self.store)
        self.assertEqual(found[0].metadata["title"], "Updated title")

    def test_file_replacement_removes_stale_chunks_and_old_file(self):
        response = self.upload(text="Machine learning. " * 130)
        doc_id = response.data["id"]
        old_file = Document.objects.get(pk=doc_id).file.path
        self.assertGreater(len(self.document_ids(doc_id)), 1)
        with self.captureOnCommitCallbacks(execute=True):
            changed = self.client.patch(
                reverse("document-detail", args=[doc_id]),
                {"file": upload_docx("cake.docx", "A cake recipe.")},
                format="multipart",
            )
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(self.document_ids(doc_id), [f"{doc_id}:0"])
        result = search_chunks("cake", k=1, vector_store=self.store)
        self.assertEqual(result[0].page_content, "A cake recipe.")
        self.assertFalse(Path(old_file).exists())

    def test_api_delete_clears_index_file_and_sqlite(self):
        response = self.upload()
        doc_id = response.data["id"]
        filename = Document.objects.get(pk=doc_id).file.path
        with self.captureOnCommitCallbacks(execute=True):
            deleted = self.client.delete(reverse("document-detail", args=[doc_id]))
        self.assertEqual(deleted.status_code, 204)
        self.assertFalse(Document.objects.filter(pk=doc_id).exists())
        self.assertEqual(self.document_ids(doc_id), [])
        self.assertFalse(Path(filename).exists())

    def test_index_failure_is_reported_and_document_can_be_retried(self):
        with patch(
            "documents.views.index_saved_document", side_effect=OSError("Chroma unavailable")
        ):
            response = self.upload()
        self.assertEqual(response.status_code, 503)
        doc_id = response.data["document_id"]
        document = Document.objects.get(pk=doc_id)
        self.assertTrue(document.extracted_text)
        self.assertEqual(self.document_ids(doc_id), [])
        self.assertEqual(index_saved_document(document, vector_store=self.store), 1)
        self.assertEqual(self.document_ids(doc_id), [f"{doc_id}:0"])

    def test_delete_failure_keeps_the_document(self):
        response = self.upload()
        doc_id = response.data["id"]
        with patch(
            "documents.views.delete_document_and_index", side_effect=OSError("store down")
        ):
            failed = self.client.delete(reverse("document-detail", args=[doc_id]))
        self.assertEqual(failed.status_code, 503)
        self.assertTrue(Document.objects.filter(pk=doc_id).exists())
        self.assertEqual(self.document_ids(doc_id), [f"{doc_id}:0"])

    def test_admin_upload_indexes_without_manual_command(self):
        # Use session authentication rather than REST's force_authenticate.
        self.client.force_authenticate(user=None)
        self.client.force_login(self.user)
        with patch(
            "documents.admin.index_saved_document",
            side_effect=lambda document, **kwargs: index_saved_document(
                document, **kwargs, vector_store=self.store
            ),
        ):
            response = self.client.post(
                reverse("admin:documents_document_add"),
                {"title": "Admin AI", "file": upload_docx("admin.docx", "Machine notes")},
            )
        self.assertEqual(response.status_code, 302)
        doc_id = Document.objects.get(title="Admin AI").pk
        self.assertEqual(self.document_ids(doc_id), [f"{doc_id}:0"])

    def test_admin_bulk_delete_removes_all_indexes(self):
        a = self.upload("a.docx", "Machine learning").data["id"]
        b = self.upload("b.docx", "A cake recipe").data["id"]
        self.client.force_authenticate(user=None)
        self.client.force_login(self.user)
        with patch(
            "documents.admin.delete_document_and_index",
            side_effect=lambda document: delete_document_and_index(
                document, vector_store=self.store
            ),
        ):
            response = self.client.post(
                reverse("admin:documents_document_changelist"),
                {"action": "delete_selected", "_selected_action": [a,b], "post": "yes"},
            )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Document.objects.filter(pk__in=[a,b]).exists())
        self.assertEqual(self.document_ids(a), [])
        self.assertEqual(self.document_ids(b), [])
