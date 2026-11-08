"""Tests for document management and DOCX parsing in Django Admin."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from docx import Document as WordDocument

from documents.models import Document


def make_word_upload(filename, paragraph):
    """Build a real DOCX in memory for upload tests."""
    word = WordDocument()
    word.add_paragraph(paragraph)
    output = BytesIO()
    word.save(output)
    return SimpleUploadedFile(
        filename,
        output.getvalue(),
        content_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    )


class DocumentAdminTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin_user = get_user_model().objects.create_superuser(
            username="testadmin",
            email="testadmin@example.test",
            password="test-password-for-admin-tests",
        )

    def setUp(self):
        self.client.force_login(self.admin_user)
        # Old Admin behavior tests stay offline; task-15 tests exercise real Chroma.
        index_mock = patch("documents.admin.index_saved_document", return_value=1)
        delete_mock = patch("documents.admin.delete_document_and_index", side_effect=lambda doc: doc.delete())
        index_mock.start()
        delete_mock.start()
        self.addCleanup(index_mock.stop)
        self.addCleanup(delete_mock.stop)

    def test_document_is_listed_and_searchable(self):
        Document.objects.create(title="Machine Learning", file="documents/ml.docx")
        Document.objects.create(title="Computer Vision", file="documents/cv.docx")

        url = reverse("admin:documents_document_changelist")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Machine Learning")
        self.assertContains(response, "Computer Vision")

        search_response = self.client.get(url, {"q": "MACHINE"})
        self.assertEqual(search_response.status_code, 200)
        self.assertContains(search_response, "Machine Learning")
        self.assertNotContains(search_response, "Computer Vision")

    def test_upload_docx_via_admin_extracts_and_stores_text(self):
        with TemporaryDirectory() as media_dir, override_settings(MEDIA_ROOT=media_dir):
            uploaded_file = make_word_upload("sample.docx", "سلام از سند ورد")
            response = self.client.post(
                reverse("admin:documents_document_add"),
                {"title": "First Word Document", "file": uploaded_file},
            )

            self.assertEqual(response.status_code, 302)
            document = Document.objects.get(title="First Word Document")
            self.assertEqual(document.extracted_text, "سلام از سند ورد")
            self.assertTrue(document.file.name.endswith("sample.docx"))
            self.assertTrue(Path(document.file.path).is_file())

    def test_edit_title_does_not_erase_extracted_text(self):
        document = Document.objects.create(
            title="Old title",
            file="documents/original.docx",
            extracted_text="Original extracted content",
        )
        change_url = reverse("admin:documents_document_change", args=[document.pk])
        response = self.client.post(change_url, {"title": "New title"})
        self.assertEqual(response.status_code, 302)
        document.refresh_from_db()
        self.assertEqual(document.title, "New title")
        self.assertEqual(document.extracted_text, "Original extracted content")

    def test_replace_docx_refreshes_extracted_text(self):
        with TemporaryDirectory() as media_dir, override_settings(MEDIA_ROOT=media_dir):
            document = Document.objects.create(
                title="Example",
                file="documents/original.docx",
                extracted_text="Old content",
            )
            response = self.client.post(
                reverse("admin:documents_document_change", args=[document.pk]),
                {"title": "Example", "file": make_word_upload("new.docx", "New content")},
            )
            self.assertEqual(response.status_code, 302)
            document.refresh_from_db()
            self.assertEqual(document.extracted_text, "New content")
            self.assertTrue(document.file.name.endswith("new.docx"))

    def test_corrupt_docx_is_rejected_without_saving(self):
        with TemporaryDirectory() as media_dir, override_settings(MEDIA_ROOT=media_dir):
            response = self.client.post(
                reverse("admin:documents_document_add"),
                {
                    "title": "Bad file",
                    "file": SimpleUploadedFile("bad.docx", b"not a docx"),
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, "not a readable DOCX")
            self.assertFalse(Document.objects.filter(title="Bad file").exists())

    def test_delete_document_via_admin(self):
        document = Document.objects.create(
            title="To delete", file="documents/original.docx"
        )
        delete_url = reverse("admin:documents_document_delete", args=[document.pk])
        confirmation_response = self.client.get(delete_url)
        self.assertEqual(confirmation_response.status_code, 200)
        delete_response = self.client.post(delete_url, {"post": "yes"})
        self.assertEqual(delete_response.status_code, 302)
        self.assertFalse(Document.objects.filter(pk=document.pk).exists())

    def test_anonymous_user_cannot_access_document_admin(self):
        self.client.logout()
        response = self.client.get(reverse("admin:documents_document_changelist"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response.url)
