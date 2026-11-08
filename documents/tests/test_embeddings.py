"""Test the embedding adapter without downloading a real Hugging Face model."""

from unittest.mock import patch

from django.test import SimpleTestCase

from documents.services.embeddings import E5Embeddings


class _FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeModel:
    def __init__(self):
        self.inputs = []
        self.options = None

    def encode(self, texts, **kwargs):
        self.inputs = texts
        self.options = kwargs
        return _FakeArray([[0.6, 0.8] for _ in texts])


class E5EmbeddingTests(SimpleTestCase):
    @patch("documents.services.embeddings._load_model")
    def test_documents_use_passage_prefix(self, model_loader):
        fake = _FakeModel()
        model_loader.return_value = fake
        embeddings = E5Embeddings()
        vectors = embeddings.embed_documents(["متن سند", "second passage"])
        self.assertEqual(fake.inputs, ["passage: متن سند", "passage: second passage"])
        self.assertEqual(len(vectors), 2)
        self.assertTrue(fake.options["normalize_embeddings"])

    @patch("documents.services.embeddings._load_model")
    def test_question_uses_query_prefix(self, model_loader):
        fake = _FakeModel()
        model_loader.return_value = fake
        vector = E5Embeddings().embed_query("هوش مصنوعی چیست؟")
        self.assertEqual(fake.inputs, ["query: هوش مصنوعی چیست؟"])
        self.assertEqual(vector, [0.6, 0.8])

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            E5Embeddings().embed_query(" ")

    def test_empty_document_list_returns_empty_result(self):
        self.assertEqual(E5Embeddings().embed_documents([]), [])
