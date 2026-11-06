"""Database models for uploaded documents."""

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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title
