"""Offline tests for document chunking; no model download is needed."""

from django.test import SimpleTestCase

from .services.chunking import split_document_text


class ChunkingTests(SimpleTestCase):
    def test_short_persian_text_keeps_source_metadata(self):
        chunks = split_document_text(
            document_id=7, title="آشنایی با یادگیری ماشین", text="یادگیری ماشین چیست؟"
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "یادگیری ماشین چیست؟")
        self.assertEqual(chunks[0].metadata["document_id"], 7)
        self.assertEqual(chunks[0].metadata["chunk_id"], "7:0")
        self.assertEqual(chunks[0].metadata["title"], "آشنایی با یادگیری ماشین")

    def test_long_text_is_split_into_bounded_chunks(self):
        text = " ".join(["یادگیری عمیق و شبکه عصبی"] * 70)
        chunks = split_document_text(
            document_id=2, title="Sample", text=text, chunk_size=110, chunk_overlap=20
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(c.page_content) <= 110 for c in chunks))
        self.assertEqual(
            [c.metadata["chunk_index"] for c in chunks], list(range(len(chunks)))
        )

    def test_empty_text_creates_no_chunks(self):
        self.assertEqual(
            split_document_text(document_id=2, title="Empty", text="  "), []
        )

    def test_invalid_chunk_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            split_document_text(
                document_id=3, title="Bad", text="Hi", chunk_size=10, chunk_overlap=10
            )
