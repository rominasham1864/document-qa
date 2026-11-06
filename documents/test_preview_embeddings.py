"""Management command smoke tests using a fake model; no downloads."""

from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from .models import Document


class PreviewEmbeddingsTests(TestCase):
    def test_command_shows_chunk_count_and_dimension(self):
        document = Document.objects.create(
            title="Example", file="documents/sample.docx", extracted_text="سلام دنیا"
        )
        output = StringIO()
        with patch(
            "documents.management.commands.preview_embeddings.E5Embeddings"
        ) as mocked_embeddings:
            mocked_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2]]
            call_command("preview_embeddings", document_id=document.pk, stdout=output)
        self.assertIn("Chunks: 1", output.getvalue())
        self.assertIn("Embedding dimension: 2", output.getvalue())
        self.assertIn(f"{document.pk}:0", output.getvalue())

    def test_command_rejects_empty_document(self):
        document = Document.objects.create(
            title="Empty", file="documents/empty.docx", extracted_text=""
        )
        with self.assertRaises(CommandError):
            call_command("preview_embeddings", document_id=document.pk)
