"""Standalone regression tests: python -m unittest discover -s scripts/tests -v"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_rag_quality import format_report, load_review, percentile, summarize  # noqa: E402
from security_preflight import sensitive_reason  # noqa: E402


def case(ident, *, scope=True, hit=True, evidence=True, answer=None, error=None):
    return {
        "id": ident, "out_of_scope": not scope,
        "expected_document_ids": [1] if scope else [],
        "all_expected_docs_in_top_k": hit if scope else None,
        "evidence_found": evidence if scope else None,
        "answer": answer, "answer_error": error,
        "retrieval_ms": 100, "llm_ms": 200 if answer else None,
        "has_citation": bool(answer and "[1]" in answer),
        "invalid_citations": [], "analysis_leak": False,
    }


class QualitySummaryTests(unittest.TestCase):
    def test_uses_real_denominators_and_no_fake_accuracy(self):
        result = summarize({"mode": "retrieval+llm", "k": 4, "cases": [
            case("a", answer="پاسخ [1]"),
            case("b", hit=False, evidence=False, error="RateLimitError"),
            case("c", scope=False, answer="اطلاعات کافی نیست"),
        ]}, {"a": 2, "b": 2, "c": 2})
        self.assertEqual((result["retrieval_doc_hits"], result["retrieval_doc_evaluable"]), (1, 2))
        self.assertEqual(result["answers_received"], 2)
        self.assertEqual(result["provider_errors"], 1)
        self.assertEqual(result["graded_answered"], 2)
        self.assertFalse(result["human_full_coverage"])
        self.assertIn("Not reportable", format_report(result))

    def test_full_grade_and_latencies(self):
        result = summarize({"mode": "retrieval+llm", "k": 4, "cases": [
            case("a", answer="Correct [1]"), case("b", answer="Incorrect")
        ]}, {"a": 2, "b": 0})
        self.assertEqual(result["human_score"], 2)
        self.assertTrue(result["human_full_coverage"])
        self.assertIn("2/4 (50.0%)", format_report(result))
        self.assertEqual(percentile([1, 2, 3, 4, 5], .95), 5)

    def test_review_grade_validation(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "review.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["id", "human_grade_0_to_2"])
                writer.writeheader()
                writer.writerow({"id": "a", "human_grade_0_to_2": "3"})
            with self.assertRaises(ValueError):
                load_review(path)

    def test_unknown_review_case_rejected(self):
        with self.assertRaises(ValueError):
            summarize({"cases": [case("x")]}, {"y": 2})

    def test_security_path_rules(self):
        self.assertIsNotNone(sensitive_reason(".env"))
        self.assertIsNotNone(sensitive_reason("backup/db.sqlite3"))
        self.assertIsNotNone(sensitive_reason("media/documents/secret.docx"))
        self.assertIsNone(sensitive_reason(".env.example"))
        self.assertIsNone(sensitive_reason("quality_eval/aban.docx"))


if __name__ == "__main__":
    unittest.main()
