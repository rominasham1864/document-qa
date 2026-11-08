"""Offline tests for evaluator's own parsing and safety checks."""

from pathlib import Path
from tempfile import TemporaryDirectory

from django.core.management.base import CommandError
from django.test import SimpleTestCase

from documents.management.commands.evaluate_rag_quality import (
    ANALYSIS_PATTERN,
    CITATION_PATTERN,
    _load_cases,
    normalized,
)


class EvaluationHelpersTests(SimpleTestCase):
    def test_persian_normalization(self):
        self.assertEqual(normalized("كتاب  \n نمونه"), normalized("کتاب نمونه"))

    def test_citations_and_reasoning_leak(self):
        self.assertEqual(CITATION_PATTERN.findall("متن [1] و [3]"), ["1", "3"])
        self.assertIsNotNone(ANALYSIS_PATTERN.search("1. Analyze the Request:"))
        self.assertIsNone(ANALYSIS_PATTERN.search("طبق سند، ۲۱ روز است. [1]"))

    def test_invalid_cases_fail_before_retrieval(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            path.write_text('[{"id":"A","question":""}]', encoding="utf-8")
            with self.assertRaises(CommandError):
                _load_cases(path)
