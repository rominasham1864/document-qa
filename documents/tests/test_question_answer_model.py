"""Tests for question-and-answer history stored in SQLite."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from documents.models import Document, QuestionAnswer


class QuestionAnswerModelTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="qa-history-user", password="test-only-password"
        )

    def test_save_question_answer_and_sources(self):
        sources = [
            {
                "rank": 1,
                "document_id": 2,
                "chunk_id": "2:0",
                "title": "AI notes",
            }
        ]
        record = QuestionAnswer.objects.create(
            user=self.user,
            question="یادگیری ماشین چیست؟",
            answer="یادگیری ماشین از داده‌ها الگو می‌آموزد [1].",
            sources=sources,
        )
        record.refresh_from_db()

        self.assertEqual(record.user, self.user)
        self.assertEqual(record.question, "یادگیری ماشین چیست؟")
        self.assertIn("داده‌ها", record.answer)
        self.assertEqual(record.sources, sources)
        self.assertIsNotNone(record.created_at)
        self.assertEqual(str(record), record.question)

    def test_sources_default_to_separate_empty_lists(self):
        first = QuestionAnswer.objects.create(
            user=self.user, question="Q1", answer="A1"
        )
        second = QuestionAnswer.objects.create(
            user=self.user, question="Q2", answer="A2"
        )
        first.sources.append({"chunk_id": "1:0"})
        self.assertEqual(second.sources, [])
        second.refresh_from_db()
        self.assertEqual(second.sources, [])

    def test_history_is_related_to_its_user(self):
        other_user = get_user_model().objects.create_user(
            username="other-user", password="test-only-password"
        )
        own = QuestionAnswer.objects.create(
            user=self.user, question="Mine", answer="A"
        )
        QuestionAnswer.objects.create(
            user=other_user, question="Other", answer="B"
        )
        self.assertEqual(list(self.user.question_answers.all()), [own])
        self.assertEqual(QuestionAnswer.objects.count(), 2)

    def test_deleting_user_deletes_their_history(self):
        record = QuestionAnswer.objects.create(
            user=self.user, question="Q", answer="A"
        )
        self.user.delete()
        self.assertFalse(QuestionAnswer.objects.filter(pk=record.pk).exists())

    def test_sources_are_snapshots_not_foreign_keys(self):
        document = Document.objects.create(
            title="Old title", file="documents/old.docx", extracted_text="Old text"
        )
        record = QuestionAnswer.objects.create(
            user=self.user,
            question="Q",
            answer="A [1]",
            sources=[{
                "rank": 1,
                "document_id": document.pk,
                "chunk_id": f"{document.pk}:0",
                "title": document.title,
            }],
        )
        document.delete()
        record.refresh_from_db()
        self.assertEqual(record.sources[0]["title"], "Old title")
        self.assertFalse(Document.objects.exists())

    def test_latest_history_is_first(self):
        first = QuestionAnswer.objects.create(
            user=self.user, question="First", answer="A"
        )
        second = QuestionAnswer.objects.create(
            user=self.user, question="Second", answer="B"
        )
        self.assertEqual(
            list(QuestionAnswer.objects.values_list("pk", flat=True)),
            [second.pk, first.pk],
        )
