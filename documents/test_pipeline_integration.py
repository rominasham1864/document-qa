"""Task 10: exercise Admin -> DOCX -> SQLite -> Chroma -> RAG -> HTTP API.

Only the embedding model and remote chat model are replaced by deterministic
fakes. Chroma, Django Admin, DOCX parsing, RAG and API code all run normally.
The tests do not touch real Chroma volumes or contact external services.
"""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from docx import Document as WordDocument
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from rest_framework.test import APITestCase

from documents.models import Document
from documents.services.vector_store import (
    get_vector_store,
    index_document,
    search_chunks,
)


class KeywordEmbeddings(Embeddings):
    """Small, predictable vectors for integration tests; NOT a real E5 model."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        lower = text.lower()
        if "ماشین" in lower or "machine" in lower:
            return [1.0, 0.0, 0.0]
        if "کیک" in lower or "cake" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_docx_upload(filename: str, paragraph: str) -> SimpleUploadedFile:
    word = WordDocument()
    word.add_paragraph(paragraph)
    file_obj = BytesIO()
    word.save(file_obj)
    return SimpleUploadedFile(
        filename,
        file_obj.getvalue(),
        content_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    )


class DocumentQAPipelineIntegrationTests(APITestCase):
    """Run a real local pipeline, replacing only expensive/external models."""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        media_dir = Path(self.temporary.name) / "media"
        self.chroma_dir = str(Path(self.temporary.name) / "chroma")
        self.settings_override = override_settings(MEDIA_ROOT=str(media_dir))
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

        self.user = get_user_model().objects.create_superuser(
            username="pipeline-admin",
            email="pipeline@example.test",
            password="test-only-password",
        )
        self.client.force_login(self.user)
        self.store = get_vector_store(
            persist_directory=self.chroma_dir,
            embeddings=KeywordEmbeddings(),
        )

    def upload_document(self, title: str, filename: str, text: str) -> Document:
        response = self.client.post(
            reverse("admin:documents_document_add"),
            {"title": title, "file": make_docx_upload(filename, text)},
        )
        self.assertEqual(response.status_code, 302)
        return Document.objects.get(title=title)

    def ask_via_api(self, payload: dict, answer: str):
        # Only swap the external language model and the default store lookup.
        # The real answer_question(), search_chunks(), Chroma and API still run.
        fake_chat_model = Mock()
        fake_chat_model.invoke.return_value = AIMessage(content=answer)
        with (
            patch(
                "documents.services.rag.search_chunks",
                side_effect=lambda query, *, k, document_id: search_chunks(
                    query, k=k, document_id=document_id, vector_store=self.store
                ),
            ),
            patch(
                "documents.services.rag.get_chat_model",
                return_value=fake_chat_model,
            ),
        ):
            response = self.client.post(
                reverse("ask-question"), payload, format="json"
            )
        return response, fake_chat_model

    def test_word_upload_to_chroma_to_answer_api(self):
        paragraph = "یادگیری ماشین از داده‌ها الگو استخراج می‌کند."
        document = self.upload_document("AI introduction", "ai.docx", paragraph)

        # Verify the real Admin form and DOCX parser saved text to SQLite.
        self.assertEqual(document.extracted_text, paragraph)
        self.assertTrue(Path(document.file.path).is_file())

        # Real chunking + Chroma indexing with deterministic fake embeddings.
        self.assertEqual(index_document(document, vector_store=self.store), 1)
        reopened_store = get_vector_store(
            persist_directory=self.chroma_dir, embeddings=KeywordEmbeddings()
        )
        self.assertEqual(
            reopened_store.get(where={"document_id": document.pk}, include=[])["ids"],
            [f"{document.pk}:0"],
        )

        response, chat = self.ask_via_api(
            {"question": "یادگیری ماشین چیست؟", "k": 2},
            "یادگیری ماشین از داده‌ها الگو استخراج می‌کند [1].",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["question"], "یادگیری ماشین چیست؟")
        self.assertIn("[1]", response.data["answer"])
        self.assertEqual(response.data["sources"][0]["document_id"], document.pk)
        self.assertEqual(response.data["sources"][0]["chunk_id"], f"{document.pk}:0")
        messages = chat.invoke.call_args.args[0]
        self.assertIn(paragraph, messages[1].content)

    def test_document_filter_keeps_other_document_out_of_context(self):
        ai = self.upload_document(
            "AI", "ai.docx", "یادگیری ماشین از داده‌ها الگو می‌گیرد."
        )
        cooking = self.upload_document(
            "Cooking", "cake.docx", "برای پخت کیک به آرد نیاز داریم."
        )
        index_document(ai, vector_store=self.store)
        index_document(cooking, vector_store=self.store)

        response, chat = self.ask_via_api(
            {"question": "یادگیری ماشین چیست؟", "document_id": cooking.pk, "k": 2},
            "این بخش درباره پخت کیک است [1].",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["sources"]), 1)
        self.assertEqual(response.data["sources"][0]["document_id"], cooking.pk)
        context = chat.invoke.call_args.args[0][1].content
        self.assertIn("پخت کیک", context)
        self.assertNotIn("از داده‌ها الگو", context)
