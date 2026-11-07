"""Management command tests without invoking any external provider."""

from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase


class AskDocumentsCommandTests(SimpleTestCase):
    @patch("documents.management.commands.ask_documents.answer_question")
    def test_command_displays_answer_and_sources(self, answer_question):
        answer_question.return_value = {
            "answer": "Machine learning uses data [1].",
            "sources": [
                {"rank": 1, "document_id": 2, "chunk_id": "2:0", "title": "AI notes"}
            ],
        }
        output = StringIO()
        call_command("ask_documents", query="What is ML?", k=2, stdout=output)
        self.assertIn("Machine learning uses data", output.getvalue())
        self.assertIn("AI notes", output.getvalue())
        answer_question.assert_called_once_with("What is ML?", k=2, document_id=None)

    @patch("documents.management.commands.ask_documents.answer_question")
    def test_command_reports_missing_api_key(self, answer_question):
        answer_question.side_effect = RuntimeError("OPENROUTER_API_KEY is missing")
        with self.assertRaises(CommandError):
            call_command("ask_documents", query="Question")
