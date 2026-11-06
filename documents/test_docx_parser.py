"""Unit tests for DOCX text extraction (no database required)."""

from io import BytesIO

from django.test import SimpleTestCase
from docx import Document as WordDocument

from .services.docx_parser import DocumentExtractionError, extract_docx_text


class DocxParserTests(SimpleTestCase):
    @staticmethod
    def as_file(document):
        content = BytesIO()
        document.save(content)
        content.seek(0)
        return content

    def test_preserves_order_of_paragraphs_and_tables(self):
        document = WordDocument()
        document.add_paragraph("مقدمه درباره یادگیری ماشین")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "مدل"
        table.cell(0, 1).text = "دقت"
        document.add_paragraph("Final note")
        stream = self.as_file(document)

        text = extract_docx_text(stream)

        self.assertEqual(
            text,
            "مقدمه درباره یادگیری ماشین\n\nمدل | دقت\n\nFinal note",
        )
        self.assertEqual(stream.tell(), 0)

    def test_rejects_corrupted_docx(self):
        with self.assertRaises(DocumentExtractionError):
            extract_docx_text(BytesIO(b"not a docx archive"))

    def test_rejects_document_without_text(self):
        stream = self.as_file(WordDocument())
        with self.assertRaisesMessage(DocumentExtractionError, "no extractable"):
            extract_docx_text(stream)
