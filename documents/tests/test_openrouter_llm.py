"""Verify the remote LLM client configuration without contacting OpenRouter."""

from unittest.mock import patch

from django.test import SimpleTestCase

from documents.services.llm import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    get_chat_model,
)


class OpenRouterConfigurationTests(SimpleTestCase):
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": ""})
    def test_missing_api_key_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
            get_chat_model()

    @patch.dict("os.environ", {
        "OPENROUTER_API_KEY": "test-key",
        "OPENROUTER_MODEL": DEFAULT_OPENROUTER_MODEL,
    })
    @patch("documents.services.llm.ChatOpenAI")
    def test_client_receives_openrouter_configuration(self, fake_client):
        get_chat_model()
        kwargs = fake_client.call_args.kwargs
        self.assertEqual(kwargs["base_url"], OPENROUTER_BASE_URL)
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["model"], DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(kwargs["max_retries"], 0)
