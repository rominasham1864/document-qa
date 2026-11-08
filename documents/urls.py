"""URLs owned by the document application."""

from django.urls import include, path
from rest_framework.routers import SimpleRouter

from .views import AskQuestionView, DocumentViewSet

router = SimpleRouter()
router.register("documents", DocumentViewSet, basename="document")

urlpatterns = [
    path("ask/", AskQuestionView.as_view(), name="ask-question"),
    path("", include(router.urls)),
]
