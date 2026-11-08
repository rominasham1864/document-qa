"""Reliability checks using a local Chroma collection and fake embeddings.

No HF downloads or OpenRouter calls. All tests use temporary Chroma/media.
"""

from unittest.mock import patch

from django.urls import reverse

from documents.models import Document
from documents.services.document_lifecycle import index_saved_document
from documents.services.vector_store import search_chunks
from .test_document_auto_index import AutomaticIndexTests, upload_docx


class IndexReliabilityTests(AutomaticIndexTests):
    """Inherit the isolated store fixture and task-15 integration checks."""

    def test_status_is_ready_after_upload_and_visible_in_api(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201)
        doc_id = response.data["id"]
        document = Document.objects.get(pk=doc_id)
        self.assertEqual(document.index_status, Document.IndexStatus.READY)
        self.assertTrue(document.index_key)
        self.assertEqual(response.data["index_status"], "ready")
        self.assertEqual(len(search_chunks("machine", vector_store=self.store)), 1)

    def test_title_change_updates_metadata_without_new_embeddings(self):
        uploaded = self.upload()
        doc_id = uploaded.data["id"]
        old_key = Document.objects.get(pk=doc_id).index_key
        with patch(
            "documents.services.document_lifecycle.index_document",
            side_effect=AssertionError("Title edit recomputed embeddings"),
        ):
            response = self.client.patch(
                reverse("document-detail", args=[doc_id]),
                {"title": "New metadata title"}, format="json",
            )
        self.assertEqual(response.status_code, 200)
        updated = Document.objects.get(pk=doc_id)
        self.assertEqual(updated.index_key, old_key)
        self.assertEqual(updated.index_status, Document.IndexStatus.READY)
        result = search_chunks("machine", document_id=doc_id, vector_store=self.store)
        self.assertEqual(result[0].metadata["title"], "New metadata title")

    def test_file_replacement_with_unchanged_text_avoids_embedding(self):
        uploaded = self.upload()
        doc_id = uploaded.data["id"]
        text = "Machine learning extracts patterns."
        with patch(
            "documents.services.document_lifecycle.index_document",
            side_effect=AssertionError("Unchanged text was re-embedded"),
        ):
            response = self.client.patch(
                reverse("document-detail", args=[doc_id]),
                {"file": upload_docx("copy.docx", text)}, format="multipart",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Document.objects.get(pk=doc_id).index_status, "ready")

    def test_failed_reindex_is_hidden_and_can_retry(self):
        response = self.upload()
        doc_id = response.data["id"]
        with patch(
            "documents.services.document_lifecycle.index_document",
            side_effect=OSError("Chroma not writable"),
        ):
            failure = self.client.patch(
                reverse("document-detail", args=[doc_id]),
                {"file": upload_docx("new.docx", "A cake recipe.")},
                format="multipart",
            )
        self.assertEqual(failure.status_code, 503)
        self.assertIsInstance(failure.data["document_id"], int)
        failed = Document.objects.get(pk=doc_id)
        self.assertEqual(failed.index_status, "failed")
        self.assertEqual(failed.index_key, "")
        # The old vectors may still be physically in Chroma, but are invisible.
        self.assertEqual(search_chunks("machine", vector_store=self.store), [])
        self.assertEqual(search_chunks("cake", document_id=doc_id, vector_store=self.store), [])

        retried = self.client.post(reverse("document-retry-index", args=[doc_id]))
        self.assertEqual(retried.status_code, 200)
        self.assertEqual(retried.data["index_status"], "ready")
        result = search_chunks("cake", document_id=doc_id, vector_store=self.store)
        self.assertEqual(result[0].page_content, "A cake recipe.")

    def test_retrieval_does_not_read_raw_uncommitted_vectors(self):
        from documents.services.vector_store import index_document

        doc = Document.objects.create(
            title="Not yet ready", file="documents/a.docx", extracted_text="machine"
        )
        index_document(doc, vector_store=self.store, index_key="temporary-write")
        self.assertEqual(self.store.get(where={"document_id": doc.pk}, include=[])["ids"], [f"{doc.pk}:0"])
        self.assertEqual(search_chunks("machine", vector_store=self.store), [])
        self.assertEqual(search_chunks("machine", document_id=doc.pk, vector_store=self.store), [])

    def test_failed_upload_is_saved_but_not_advertised_as_searchable(self):
        with patch(
            "documents.services.document_lifecycle.index_document",
            side_effect=OSError("store down"),
        ):
            response = self.upload()
        self.assertEqual(response.status_code, 503)
        document = Document.objects.get(pk=response.data["document_id"])
        self.assertEqual(document.index_status, "failed")
        self.assertEqual(search_chunks("machine", document_id=document.pk, vector_store=self.store), [])
        self.assertEqual(index_saved_document(document, vector_store=self.store), 1)
        document.refresh_from_db()
        self.assertEqual(document.index_status, "ready")

    def test_retry_endpoint_repairs_missing_chunks_from_ready_index(self):
        uploaded = self.upload()
        doc_id = uploaded.data["id"]
        ids = self.document_ids(doc_id)
        self.store.delete(ids=ids)
        self.assertEqual(search_chunks("machine", document_id=doc_id, vector_store=self.store), [])
        recovered = self.client.post(reverse("document-retry-index", args=[doc_id]))
        self.assertEqual(recovered.status_code, 200)
        self.assertEqual(recovered.data["index_status"], "ready")
        self.assertEqual(len(self.document_ids(doc_id)), 1)

    def test_partial_delete_failure_keeps_row_but_hides_index(self):
        uploaded = self.upload()
        doc_id = uploaded.data["id"]
        with patch(
            "documents.services.document_lifecycle.delete_document_index",
            side_effect=OSError("store down"),
        ):
            response = self.client.delete(reverse("document-detail", args=[doc_id]))
        self.assertEqual(response.status_code, 503)
        document = Document.objects.get(pk=doc_id)
        self.assertEqual(document.index_status, "failed")
        self.assertEqual(search_chunks("machine", vector_store=self.store), [])

    def test_pending_and_interrupted_processing_are_not_searchable(self):
        uploaded = self.upload()
        doc_id = uploaded.data["id"]
        Document.objects.filter(pk=doc_id).update(index_status="processing")
        self.assertEqual(search_chunks("machine", vector_store=self.store), [])
        # The recovery command uses index_saved_document on this row.
        doc = Document.objects.get(pk=doc_id)
        self.assertEqual(index_saved_document(doc, vector_store=self.store), 1)
        self.assertEqual(len(search_chunks("machine", vector_store=self.store)), 1)
