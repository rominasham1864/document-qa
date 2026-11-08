from pathlib import Path

from rest_framework import serializers

from .models import Document
from .services.docx_parser import DocumentExtractionError, extract_docx_text


MAX_DOCX_SIZE = 10 * 1024 * 1024  # 10 MiB; API upload limit


class AskQuestionSerializer(serializers.Serializer):
    """A single question with optional retrieval settings."""

    question = serializers.CharField(max_length=2000, trim_whitespace=True)
    k = serializers.IntegerField(default=4, min_value=1, max_value=8)
    document_id = serializers.IntegerField(
        required=False, allow_null=True, min_value=1
    )


class DocumentSerializer(serializers.ModelSerializer):
    """Upload DOCX, extract its text, and expose stored document details."""

    # Return the stored filename, not a public media URL.
    file = serializers.FileField(use_url=False)

    class Meta:
        model = Document
        fields = (
            "id",
            "title",
            "file",
            "extracted_text",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "extracted_text", "created_at", "updated_at")

    def validate_file(self, file):
        """Reject bad uploads before Django saves the file or database row."""
        if Path(file.name).suffix.lower() != ".docx":
            raise serializers.ValidationError("Only .docx files are supported.")
        if file.size > MAX_DOCX_SIZE:
            raise serializers.ValidationError("DOCX files must be at most 10 MiB.")
        try:
            self._new_extracted_text = extract_docx_text(file)
        except DocumentExtractionError as exc:
            raise serializers.ValidationError(str(exc)) from exc
        return file

    def create(self, validated_data):
        validated_data["extracted_text"] = self._new_extracted_text
        return super().create(validated_data)

    def update(self, instance, validated_data):
        # Title-only edits do not touch the old extracted text.
        if "file" in validated_data:
            validated_data["extracted_text"] = self._new_extracted_text
        return super().update(instance, validated_data)


class DocumentListSerializer(DocumentSerializer):
    """Keep large extracted text out of the document listing."""

    class Meta(DocumentSerializer.Meta):
        fields = ("id", "title", "file", "created_at", "updated_at")
