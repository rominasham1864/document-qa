"""Database models for uploaded documents and question-answer history."""

from django.conf import settings
from django.core.validators import FileExtensionValidator
from django.db import models


class Document(models.Model):
    """A Word document and the text extracted from it."""

    title = models.CharField(max_length=255)
    file = models.FileField(
        upload_to="documents/%Y/%m/",
        max_length=255,
        validators=[FileExtensionValidator(allowed_extensions=["docx"])],
    )
    extracted_text = models.TextField(blank=True)
    class IndexStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    index_status = models.CharField(
        max_length=12, choices=IndexStatus.choices, default=IndexStatus.PENDING
    )
    # Only index_key on a READY row is eligible for retrieval.
    index_key = models.CharField(max_length=80, blank=True, default="")
    indexed_content_hash = models.CharField(max_length=64, blank=True, default="")
    indexed_title = models.CharField(max_length=255, blank=True, default="")
    indexed_chunk_count = models.PositiveIntegerField(default=0)
    index_error = models.CharField(max_length=120, blank=True, default="")
    index_attempt = models.CharField(max_length=32, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class QuestionAnswer(models.Model):
    """One user's question, generated answer, and source snapshots."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="question_answers",
    )
    question = models.TextField()
    answer = models.TextField()
    # RAG already returns source dictionaries with rank, document_id,
    # chunk_id and title. Keep snapshots even if documents later change.
    sources = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return self.question[:80]
