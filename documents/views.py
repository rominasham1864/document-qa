"""HTTP endpoints for document management and question answering."""

import logging

from openai import RateLimitError
from rest_framework import status, viewsets
from rest_framework.exceptions import APIException
from rest_framework.authentication import BasicAuthentication, SessionAuthentication
from rest_framework.generics import GenericAPIView
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Document, QuestionAnswer
from .serializers import (
    AskQuestionSerializer,
    DocumentListSerializer,
    DocumentSerializer,
    QuestionAnswerSerializer,
)
from .services.rag import answer_question
from .services.document_lifecycle import index_saved_document, delete_document_and_index

logger = logging.getLogger(__name__)


class DocumentIndexUnavailable(APIException):
    """An upload/update was saved, but its search index is not ready."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_code = "document_index_unavailable"

    def __init__(self, document_id: int):
        super().__init__({
            "code": "document_index_unavailable",
            "document_id": document_id,
            "message": (
                "Document saved, but indexing failed. "
                "Retry with: python manage.py index_document "
                f"--document-id {document_id}"
            ),
        })
        self.detail["document_id"] = document_id


class DocumentDeletionUnavailable(APIException):
    """Keep the document when removing its Chroma index fails."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "Could not delete document: vector index is unavailable."
    default_code = "document_delete_unavailable"


class DocumentViewSet(viewsets.ModelViewSet):
    """Authenticated CRUD; synchronize indexes on actual document changes."""

    queryset = Document.objects.all()
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]

    def get_serializer_class(self):
        if self.action == "list":
            return DocumentListSerializer
        return DocumentSerializer

    def perform_create(self, serializer):
        document = serializer.save()
        self._index_or_report(document)

    def perform_update(self, serializer):
        before = serializer.instance.file.name
        changed_fields = serializer.validated_data
        document = serializer.save()
        if "title" in changed_fields or "file" in changed_fields:
            self._index_or_report(document, previous_file_name=before)

    @staticmethod
    def _index_or_report(document, *, previous_file_name=None):
        try:
            index_saved_document(document, previous_file_name=previous_file_name)
        except Exception as exc:
            logger.exception("Could not index saved document %s", document.pk)
            raise DocumentIndexUnavailable(document.pk) from exc

    def perform_destroy(self, instance):
        try:
            delete_document_and_index(instance)
        except Exception as exc:
            logger.exception("Could not delete document %s and its index", instance.pk)
            raise DocumentDeletionUnavailable() from exc


class HistoryPagination(PageNumberPagination):
    """Bound history responses so one user cannot fetch unlimited rows at once."""

    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


class QuestionAnswerViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only history; users can see only their own questions and answers."""

    serializer_class = QuestionAnswerSerializer
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = HistoryPagination
    http_method_names = ["get", "head", "options"]

    def get_queryset(self):
        return QuestionAnswer.objects.filter(user=self.request.user)


class AskQuestionView(GenericAPIView):
    """Answer questions using existing RAG services (task 9, unchanged)."""

    serializer_class = AskQuestionSerializer
    # Basic auth is for LOCAL CLI tests; use HTTPS for any remote deployment.
    authentication_classes = [BasicAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            result = answer_question(
                data["question"],
                k=data["k"],
                document_id=data.get("document_id"),
            )
        except RuntimeError as exc:
            # Task 8 wraps OpenAI errors in RuntimeError, preserving the cause.
            # Report a provider limit as a temporary upstream outage, not a
            # bad user request. Never expose internal exception details.
            cause = exc.__cause__
            if (
                isinstance(cause, RateLimitError)
                or getattr(cause, "status_code", None) == 429
                or type(cause).__name__ == "OpenAIRateLimitError"
            ):
                return Response(
                    {
                        "error": {
                            "code": "provider_rate_limited",
                            "message": "The language-model provider is rate-limiting requests. Try again later.",
                        }
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            return Response(
                {
                    "error": {
                        "code": "answer_unavailable",
                        "message": "The answer service is temporarily unavailable. Check the server logs.",
                    }
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        QuestionAnswer.objects.create(
            user=request.user,
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
        )
        return Response(result, status=status.HTTP_200_OK)

