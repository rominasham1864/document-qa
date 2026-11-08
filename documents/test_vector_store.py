"""Local Chroma integration tests with deterministic embeddings (no model download)."""

from tempfile import TemporaryDirectory

from django.test import TestCase
from langchain_core.embeddings import Embeddings

from documents.models import Document
from documents.services.document_lifecycle import index_saved_document
from documents.services.vector_store import (
    delete_document_index,
    get_vector_store,
    index_document,
    search_chunks,
)


class FakeEmbeddings(Embeddings):
    """3-dimensional vectors for predictable, offline retrieval tests."""

    @staticmethod
    def _vector(text):
        if "cat" in text.lower():
            return [1.0, 0.0, 0.0]
        if "dog" in text.lower():
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class VectorStoreTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = get_vector_store(
            persist_directory=self.tmp.name,
            embeddings=FakeEmbeddings(),
        )

    def make_document(self, title, text):
        return Document.objects.create(
            title=title, file="documents/test.docx", extracted_text=text,
        )

    def indexed_ids(self, document):
        return self.store.get(
            where={"document_id": document.pk}, include=[]
        )["ids"]

    def test_index_and_semantic_search(self):
        cat = self.make_document("Pets - cats", "A cat sleeps on a sofa.")
        dog = self.make_document("Pets - dogs", "A dog plays in a park.")
        self.assertEqual(index_saved_document(cat, vector_store=self.store), 1)
        self.assertEqual(index_saved_document(dog, vector_store=self.store), 1)

        found = search_chunks("cat", k=1, vector_store=self.store)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].metadata["document_id"], cat.pk)
        self.assertEqual(found[0].metadata["chunk_id"], f"{cat.pk}:0")

        filtered = search_chunks(
            "cat", k=1, document_id=dog.pk, vector_store=self.store
        )
        self.assertEqual(filtered[0].metadata["document_id"], dog.pk)

    def test_reindex_does_not_duplicate_and_removes_old_chunks(self):
        doc = self.make_document("Old", "cat " * 400)
        first_count = index_saved_document(doc, vector_store=self.store)
        self.assertGreater(first_count, 1)
        self.assertEqual(len(self.indexed_ids(doc)), first_count)

        doc.title = "New"
        doc.extracted_text = "A dog is running."
        doc.index_status = Document.IndexStatus.PENDING
        doc.index_key = ""
        doc.save()
        self.assertEqual(index_saved_document(doc, vector_store=self.store), 1)
        self.assertEqual(self.indexed_ids(doc), [f"{doc.pk}:0"])
        found = search_chunks("dog", k=1, vector_store=self.store)
        self.assertEqual(found[0].metadata["title"], "New")
        self.assertIn("dog", found[0].page_content)

    def test_deleting_one_index_preserves_other_documents(self):
        first = self.make_document("First", "cat")
        second = self.make_document("Second", "dog")
        index_saved_document(first, vector_store=self.store)
        index_saved_document(second, vector_store=self.store)

        self.assertEqual(delete_document_index(first.pk, vector_store=self.store), 1)
        self.assertEqual(self.indexed_ids(first), [])
        self.assertEqual(self.indexed_ids(second), [f"{second.pk}:0"])

    def test_empty_document_and_invalid_search_are_rejected(self):
        doc = self.make_document("Empty", "")
        with self.assertRaises(ValueError):
            index_document(doc, vector_store=self.store)
        with self.assertRaises(ValueError):
            search_chunks("  ", vector_store=self.store)
        with self.assertRaises(ValueError):
            search_chunks("cat", k=0, vector_store=self.store)
