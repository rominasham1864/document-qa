"""Offline tests for document CRUD endpoints, using real in-memory DOCX files."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from docx import Document as WordDocument
from rest_framework import status
from rest_framework.test import APITestCase

from documents.models import Document


def word_upload(name="sample.docx", text="متن تست فارسی"):
    doc = WordDocument()
    doc.add_paragraph(text)
    buffer = BytesIO()
    doc.save(buffer)
    return SimpleUploadedFile(
        name,
        buffer.getvalue(),
        content_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    )


class DocumentAPITests(APITestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        media_override = override_settings(MEDIA_ROOT=self.temp_dir.name)
        media_override.enable()
        self.addCleanup(media_override.disable)

        self.user = get_user_model().objects.create_user(
            username="document-api-tester", password="local-test-password"
        )
        self.client.force_authenticate(user=self.user)
        self.list_url = reverse("document-list")

    def upload(self, title="AI", filename="sample.docx", text="یادگیری ماشین"):
        return self.client.post(
            self.list_url,
            {"title": title, "file": word_upload(filename, text)},
            format="multipart",
        )

    def test_create_extracts_text_and_saves_file(self):
        response = self.upload(title="فارسی", text="یادگیری ماشین چیست؟")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        record = Document.objects.get(pk=response.data["id"])
        self.assertEqual(record.title, "فارسی")
        self.assertEqual(record.extracted_text, "یادگیری ماشین چیست؟")
        self.assertEqual(response.data["extracted_text"], record.extracted_text)
        self.assertTrue(Path(record.file.path).is_file())

    def test_list_omits_long_text_but_detail_includes_it(self):
        document = Document.objects.create(
            title="Document A", file="documents/a.docx", extracted_text="long content"
        )
        list_response = self.client.get(self.list_url)
        self.assertEqual(list_response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(list_response.data), 1)
        self.assertNotIn("extracted_text", list_response.data[0])
        detail = self.client.get(reverse("document-detail", args=[document.pk]))
        self.assertEqual(detail.status_code, status.HTTP_200_OK)
        self.assertEqual(detail.data["extracted_text"], "long content")

    def test_patch_title_does_not_clear_text_or_file(self):
        document = Document.objects.create(
            title="Before", file="documents/before.docx", extracted_text="Original text"
        )
        response = self.client.patch(
            reverse("document-detail", args=[document.pk]),
            {"title": "After"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        document.refresh_from_db()
        self.assertEqual(document.title, "After")
        self.assertEqual(document.extracted_text, "Original text")
        self.assertEqual(document.file.name, "documents/before.docx")

    def test_patch_file_reextracts_text(self):
        document = Document.objects.create(
            title="Example", file="documents/old.docx", extracted_text="Old text"
        )
        response = self.client.patch(
            reverse("document-detail", args=[document.pk]),
            {"file": word_upload("replacement.docx", "New content")},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        document.refresh_from_db()
        self.assertEqual(document.title, "Example")
        self.assertEqual(document.extracted_text, "New content")
        self.assertTrue(document.file.name.endswith("replacement.docx"))

    def test_invalid_files_are_rejected_without_database_changes(self):
        for bad in (
            SimpleUploadedFile("wrong.txt", b"not a docx"),
            SimpleUploadedFile("broken.docx", b"not a zip archive"),
        ):
            with self.subTest(filename=bad.name):
                response = self.client.post(
                    self.list_url, {"title": "Bad", "file": bad}, format="multipart"
                )
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(Document.objects.exists())

    def test_missing_required_data_is_rejected(self):
        response = self.client.post(self.list_url, {"title": "Only title"}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        response = self.client.post(self.list_url, {"file": word_upload()}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(Document.objects.exists())

    def test_cannot_write_extracted_text_manually(self):
        document = Document.objects.create(
            title="Original", file="documents/test.docx", extracted_text="Saved text"
        )
        response = self.client.patch(
            reverse("document-detail", args=[document.pk]),
            {"extracted_text": "INJECTED"}, format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        document.refresh_from_db()
        self.assertEqual(document.extracted_text, "Saved text")

    def test_delete_removes_database_record(self):
        document = Document.objects.create(title="Delete me", file="documents/file.docx")
        response = self.client.delete(reverse("document-detail", args=[document.pk]))
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(Document.objects.filter(pk=document.pk).exists())

    def test_anonymous_requests_are_rejected(self):
        self.client.force_authenticate(user=None)
        self.assertEqual(self.client.get(self.list_url).status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(
            self.client.post(
                self.list_url,
                {"title": "Secret", "file": word_upload()},
                format="multipart",
            ).status_code,
            status.HTTP_401_UNAUTHORIZED,
        )
        self.assertFalse(Document.objects.exists())

    def test_put_method_is_not_enabled(self):
        document = Document.objects.create(title="Example", file="documents/file.docx")
        response = self.client.put(
            reverse("document-detail", args=[document.pk]),
            {"title": "Replaced"}, format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

    def test_ask_route_is_preserved(self):
        self.assertEqual(reverse("ask-question"), "/api/ask/")
