"""Management command tests without loading a real Hugging Face model."""

from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase
from langchain_core.documents import Document as LangChainDocument

from documents.models import Document


class VectorCommandTests(TestCase):
    def test_index_document_command(self):
        document = Document.objects.create(
            title="Example", file="documents/ai.docx", extracted_text="cat"
        )
        output = StringIO()
        with patch(
            "documents.management.commands.index_document.index_document",
            return_value=1,
        ) as indexer:
            call_command("index_document", document_id=document.pk, stdout=output)
        indexer.assert_called_once()
        self.assertIn("1 chunk(s)", output.getvalue())

    def test_index_command_rejects_missing_document(self):
        with self.assertRaises(CommandError):
            call_command("index_document", document_id=99999)

    def test_search_command_shows_chunk_and_metadata(self):
        chunk = LangChainDocument(
            page_content="A cat sleeps on a sofa.",
            metadata={"title": "Cats", "document_id": 2, "chunk_id": "2:0"},
        )
        output = StringIO()
        with patch(
            "documents.management.commands.search_documents.search_chunks",
            return_value=[chunk],
        ):
            call_command("search_documents", query="cat", k=1, stdout=output)
        self.assertIn("Cats", output.getvalue())
        self.assertIn("2:0", output.getvalue())
        self.assertIn("A cat sleeps", output.getvalue())

    def test_delete_index_command(self):
        output = StringIO()
        with patch(
            "documents.management.commands.delete_document_index.delete_document_index",
            return_value=2,
        ):
            call_command("delete_document_index", document_id=2, stdout=output)
        self.assertIn("Removed 2", output.getvalue())
