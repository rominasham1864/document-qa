"""Tests for document management in Django Admin."""

from pathlib import Path
from tempfile import TemporaryDirectory

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Document


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

    def test_document_is_listed_and_searchable(self):
        Document.objects.create(title="Machine Learning", file="documents/ml.docx")
        Document.objects.create(title="Computer Vision", file="documents/cv.docx")

        url = reverse("admin:documents_document_changelist")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Machine Learning")
        self.assertContains(response, "Computer Vision")

        search_response = self.client.get(url, {"q": "Machine Learning"})
        self.assertEqual(search_response.status_code, 200)
        self.assertContains(search_response, "Machine Learning")
        self.assertNotContains(search_response, "Computer Vision")

    def test_upload_docx_via_admin(self):
        # Upload validation currently checks the extension, not the DOCX contents.
        with TemporaryDirectory() as media_dir, override_settings(MEDIA_ROOT=media_dir):
            uploaded_file = SimpleUploadedFile(
                "sample.docx",
                b"placeholder content for upload test",
                content_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
            )
            response = self.client.post(
                reverse("admin:documents_document_add"),
                {"title": "First Word Document", "file": uploaded_file},
            )

            self.assertEqual(response.status_code, 302)
            document = Document.objects.get(title="First Word Document")
            self.assertEqual(document.extracted_text, "")
            self.assertTrue(document.file.name.endswith("sample.docx"))
            self.assertTrue(Path(document.file.path).is_file())

    def test_edit_title_and_delete_document_via_admin(self):
        document = Document.objects.create(
            title="Old title", file="documents/original.docx"
        )
        change_url = reverse("admin:documents_document_change", args=[document.pk])
        change_response = self.client.post(change_url, {"title": "New title"})
        self.assertEqual(change_response.status_code, 302)
        document.refresh_from_db()
        self.assertEqual(document.title, "New title")

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
