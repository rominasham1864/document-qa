"""Tests for the Document database model."""

from django.core.exceptions import ValidationError
from django.test import TestCase

from documents.models import Document


class DocumentModelTests(TestCase):
    def test_creates_document_with_empty_text(self):
        document = Document.objects.create(
            title="Sample document",
            file="documents/sample.docx",
        )

        self.assertIsNotNone(document.pk)
        self.assertEqual(document.extracted_text, "")
        self.assertEqual(str(document), "Sample document")
        self.assertIsNotNone(document.created_at)

    def test_extracted_text_can_be_updated_later(self):
        document = Document.objects.create(
            title="Sample document",
            file="documents/sample.docx",
        )
        document.extracted_text = "Extracted Word content"
        document.save()
        document.refresh_from_db()

        self.assertEqual(document.extracted_text, "Extracted Word content")

    def test_rejects_non_docx_extension(self):
        document = Document(title="Invalid file", file="documents/sample.pdf")

        with self.assertRaises(ValidationError):
            document.full_clean()
