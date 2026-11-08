"""HTTP endpoints for the document application."""

from openai import RateLimitError
from rest_framework import status
from rest_framework.authentication import BasicAuthentication, SessionAuthentication
from rest_framework.generics import GenericAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .serializers import AskQuestionSerializer
from .services.rag import answer_question


class AskQuestionView(GenericAPIView):
    """Answer questions using existing RAG services (no duplicate RAG logic)."""

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

        return Response(result, status=status.HTTP_200_OK)
