"""Offline API tests: no Hugging Face, Chroma or OpenRouter access required."""

from unittest.mock import patch

import httpx
from django.contrib.auth import get_user_model
from django.urls import reverse
from openai import RateLimitError
from rest_framework import status
from rest_framework.test import APITestCase


class AskQuestionAPITests(APITestCase):
    def setUp(self):
        self.url = reverse("ask-question")
        self.user = get_user_model().objects.create_user(
            username="api-tester", password="local-test-password"
        )
        self.client.force_authenticate(user=self.user)

    @patch("documents.views.answer_question")
    def test_valid_question_returns_service_result(self, answer_question):
        answer_question.return_value = {
            "question": "یادگیری ماشین چیست؟",
            "answer": "از داده‌ها الگو یاد می‌گیرد [1].",
            "sources": [
                {"rank": 1, "document_id": 2, "chunk_id": "2:0", "title": "AI"}
            ],
        }
        response = self.client.post(
            self.url,
            {"question": "  یادگیری ماشین چیست؟  ", "k": 3, "document_id": 2},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["sources"][0]["chunk_id"], "2:0")
        answer_question.assert_called_once_with(
            "یادگیری ماشین چیست؟", k=3, document_id=2
        )

    @patch("documents.views.answer_question")
    def test_defaults_and_no_sources(self, answer_question):
        answer_question.return_value = {
            "question": "سؤال",
            "answer": "هیچ بخش ایندکس‌شده‌ای پیدا نشد.",
            "sources": [],
        }
        response = self.client.post(self.url, {"question": "سؤال"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["sources"], [])
        answer_question.assert_called_once_with("سؤال", k=4, document_id=None)

    @patch("documents.views.answer_question")
    def test_invalid_input_is_400_without_invoking_rag(self, answer_question):
        invalid_payloads = [
            {},
            {"question": ""},
            {"question": "   "},
            {"question": "hello", "k": 0},
            {"question": "hello", "k": 9},
            {"question": "hello", "document_id": -1},
            {"question": "hello", "k": "not-a-number"},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post(self.url, payload, format="json")
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        answer_question.assert_not_called()

    @patch("documents.views.answer_question")
    def test_unauthenticated_request_is_rejected(self, answer_question):
        self.client.force_authenticate(user=None)
        response = self.client.post(self.url, {"question": "hello"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        answer_question.assert_not_called()

    def test_get_is_not_allowed(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

    @patch("documents.views.answer_question")
    def test_missing_key_or_model_error_gives_safe_503(self, answer_question):
        answer_question.side_effect = RuntimeError("PRIVATE OPENROUTER ERROR")
        response = self.client.post(self.url, {"question": "hello"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data["error"]["code"], "answer_unavailable")
        self.assertNotIn("PRIVATE", str(response.data))

    @patch("documents.views.answer_question")
    def test_provider_rate_limit_is_reported_without_leaking_details(self, answer_question):
        upstream_response = httpx.Response(
            429,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        )

        def fail_with_rate_limit(*args, **kwargs):
            try:
                raise RateLimitError(
                    "PRIVATE PROVIDER DETAILS", response=upstream_response, body=None
                )
            except RateLimitError as exc:
                raise RuntimeError("OpenRouter request failed") from exc

        answer_question.side_effect = fail_with_rate_limit
        response = self.client.post(self.url, {"question": "hello"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data["error"]["code"], "provider_rate_limited")
        self.assertNotIn("PRIVATE", str(response.data))

    @patch("documents.views.answer_question")
    def test_openai_rate_limit_wrapper_is_also_recognized(self, answer_question):
        # Some model integrations use this name instead of openai.RateLimitError.
        class OpenAIRateLimitError(Exception):
            pass

        def fail(*args, **kwargs):
            try:
                raise OpenAIRateLimitError("not enough quota")
            except OpenAIRateLimitError as exc:
                raise RuntimeError("OpenRouter request failed") from exc

        answer_question.side_effect = fail
        response = self.client.post(self.url, {"question": "hello"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data["error"]["code"], "provider_rate_limited")
