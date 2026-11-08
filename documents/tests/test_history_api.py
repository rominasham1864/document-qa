"""Offline tests for automatic history persistence and per-user read access."""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from documents.models import QuestionAnswer


class HistoryAPITests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="history-first-user", password="test-only-password"
        )
        self.other_user = user_model.objects.create_user(
            username="history-second-user", password="test-only-password"
        )
        self.client.force_authenticate(user=self.user)
        self.ask_url = reverse("ask-question")
        self.history_url = reverse("history-list")

    @staticmethod
    def sample_result(question="پرسش", answer="پاسخ [1].", sources=None):
        if sources is None:
            sources = [{
                "rank": 1,
                "document_id": 2,
                "chunk_id": "2:0",
                "title": "AI Notes",
            }]
        return {"question": question, "answer": answer, "sources": sources}

    @patch("documents.views.answer_question")
    def test_successful_ask_is_saved_without_changing_response(self, rag):
        rag.return_value = self.sample_result(
            question="یادگیری ماشین چیست؟", answer="پاسخ بر اساس سند [1]."
        )
        response = self.client.post(
            self.ask_url, {"question": "  یادگیری ماشین چیست؟  ", "k": 2},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, rag.return_value)
        rag.assert_called_once_with("یادگیری ماشین چیست؟", k=2, document_id=None)
        record = QuestionAnswer.objects.get()
        self.assertEqual(record.user_id, self.user.pk)
        self.assertEqual(record.question, rag.return_value["question"])
        self.assertEqual(record.answer, rag.return_value["answer"])
        self.assertEqual(record.sources, rag.return_value["sources"])

    @patch("documents.views.answer_question")
    def test_no_indexed_sources_response_is_also_recorded(self, rag):
        rag.return_value = self.sample_result(
            question="سؤال بی‌پاسخ", answer="سندی یافت نشد.", sources=[]
        )
        response = self.client.post(
            self.ask_url, {"question": "سؤال بی‌پاسخ"}, format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(QuestionAnswer.objects.get().sources, [])

    @patch("documents.views.answer_question")
    def test_bad_input_does_not_create_history(self, rag):
        response = self.client.post(self.ask_url, {"question": ""}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(QuestionAnswer.objects.exists())
        rag.assert_not_called()

    @patch("documents.views.answer_question")
    def test_provider_error_does_not_create_history(self, rag):
        rag.side_effect = RuntimeError("provider unavailable")
        response = self.client.post(
            self.ask_url, {"question": "سؤال"}, format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertFalse(QuestionAnswer.objects.exists())

    def test_list_only_contains_own_history_newest_first(self):
        old = QuestionAnswer.objects.create(
            user=self.user, question="اول", answer="پاسخ ۱"
        )
        new = QuestionAnswer.objects.create(
            user=self.user, question="دوم", answer="پاسخ ۲"
        )
        QuestionAnswer.objects.create(
            user=self.other_user, question="خصوصی", answer="محرمانه"
        )
        response = self.client.get(self.history_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 2)
        self.assertEqual(
            [item["id"] for item in response.data["results"]], [new.pk, old.pk]
        )
        self.assertNotIn("خصوصی", str(response.data))

    def test_retrieve_own_history(self):
        record = QuestionAnswer.objects.create(
            user=self.user, question="پرسش من", answer="پاسخ من", sources=[]
        )
        response = self.client.get(reverse("history-detail", args=[record.pk]))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], record.pk)
        self.assertEqual(response.data["answer"], "پاسخ من")
        self.assertNotIn("user", response.data)

    def test_other_users_history_is_not_accessible(self):
        record = QuestionAnswer.objects.create(
            user=self.other_user, question="private", answer="hidden"
        )
        response = self.client.get(reverse("history-detail", args=[record.pk]))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_anonymous_cannot_list_or_retrieve_history(self):
        record = QuestionAnswer.objects.create(
            user=self.user, question="mine", answer="answer"
        )
        self.client.force_authenticate(user=None)
        self.assertEqual(
            self.client.get(self.history_url).status_code,
            status.HTTP_401_UNAUTHORIZED,
        )
        self.assertEqual(
            self.client.get(reverse("history-detail", args=[record.pk])).status_code,
            status.HTTP_401_UNAUTHORIZED,
        )

    def test_history_is_read_only(self):
        response = self.client.post(
            self.history_url, {"question": "fake", "answer": "fake"}, format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)
        self.assertFalse(QuestionAnswer.objects.exists())

    def test_history_is_paginated(self):
        for index in range(22):
            QuestionAnswer.objects.create(
                user=self.user, question=f"Q{index}", answer="A"
            )
        first = self.client.get(self.history_url)
        self.assertEqual(first.status_code, status.HTTP_200_OK)
        self.assertEqual(first.data["count"], 22)
        self.assertEqual(len(first.data["results"]), 20)
        second = self.client.get(self.history_url, {"page": 2})
        self.assertEqual(len(second.data["results"]), 2)
