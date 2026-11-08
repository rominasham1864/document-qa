"""URLs owned by the document application."""

from django.urls import include, path
from rest_framework.routers import SimpleRouter

from .views import AskQuestionView, DocumentViewSet, QuestionAnswerViewSet

router = SimpleRouter()
router.register("documents", DocumentViewSet, basename="document")
router.register("history", QuestionAnswerViewSet, basename="history")

urlpatterns = [
    path("ask/", AskQuestionView.as_view(), name="ask-question"),
    path("", include(router.urls)),
]
