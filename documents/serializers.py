"""Validate the input accepted by the document question-answering API."""

from rest_framework import serializers


class AskQuestionSerializer(serializers.Serializer):
    """A single question with optional retrieval settings."""

    question = serializers.CharField(max_length=2000, trim_whitespace=True)
    k = serializers.IntegerField(default=4, min_value=1, max_value=8)
    document_id = serializers.IntegerField(
        required=False, allow_null=True, min_value=1
    )
