"""Generate an auditable aggregate RAG quality report from Task 16 JSON + graded CSV.

Does not connect to OpenRouter, Django, Chroma, or any remote service.
Does not consider keyword overlap a substitute for human correctness grading.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100*numerator/denominator:.1f}%)" if denominator else "N/A"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Same nearest-rank definition used in the task-16 evaluation.
    idx = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[idx], 1)


def ms(value: float | None) -> str:
    return f"{value:.1f} ms" if value is not None else "N/A"


def load_review(path: Path) -> dict[str, int]:
    grades: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"id", "human_grade_0_to_2"}.issubset(reader.fieldnames):
            raise ValueError("Review CSV must contain id and human_grade_0_to_2 columns")
        seen = set()
        for row in reader:
            ident = str(row["id"]).strip()
            if not ident or ident in seen:
                raise ValueError(f"Missing or duplicate case id: {ident!r}")
            seen.add(ident)
            raw = str(row.get("human_grade_0_to_2") or "").strip()
            if raw:
                if raw not in {"0", "1", "2"}:
                    raise ValueError(f"Case {ident}: human grade must be 0, 1, 2, or empty")
                grades[ident] = int(raw)
    return grades


def summarize(report: dict, grades: dict[str, int]) -> dict:
    rows = report.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty Task 16 report with a 'cases' list")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Report case ids must be unique, nonempty strings")
    unknown = set(grades) - set(ids)
    if unknown:
        raise ValueError(f"CSV contains unknown case IDs: {', '.join(sorted(unknown))}")

    in_scope = [r for r in rows if r.get("expected_document_ids")]
    doc_evaluable = [r for r in in_scope if isinstance(r.get("all_expected_docs_in_top_k"), bool)]
    evidence_evaluable = [r for r in in_scope if isinstance(r.get("evidence_found"), bool)]
    answered = [r for r in rows if isinstance(r.get("answer"), str) and r["answer"].strip()]
    error_rows = [r for r in rows if r.get("answer_error")]
    retrieval_lats = [float(r["retrieval_ms"]) for r in rows if isinstance(r.get("retrieval_ms"), (int, float))]
    llm_lats = [float(r["llm_ms"]) for r in answered if isinstance(r.get("llm_ms"), (int, float))]
    answered_ids = {r["id"] for r in answered}
    graded_answered = {ident: value for ident, value in grades.items() if ident in answered_ids}
    ignored_grades = sorted(set(grades) - answered_ids)

    return {
        "mode": report.get("mode", "unknown"),
        "k": report.get("k"),
        "cases": len(rows),
        "in_scope_cases": len(in_scope),
        "retrieval_doc_hits": sum(r.get("all_expected_docs_in_top_k") is True for r in doc_evaluable),
        "retrieval_doc_evaluable": len(doc_evaluable),
        "retrieval_evidence_hits": sum(r.get("evidence_found") is True for r in evidence_evaluable),
        "retrieval_evidence_evaluable": len(evidence_evaluable),
        "answers_received": len(answered),
        "provider_errors": len(error_rows),
        "answered_with_numeric_citations": sum(bool(r.get("has_citation")) for r in answered),
        "invalid_citation_answers": sum(bool(r.get("invalid_citations")) for r in answered),
        "analysis_leak_answers": sum(bool(r.get("analysis_leak")) for r in answered),
        "retrieval_median_ms": round(statistics.median(retrieval_lats), 1) if retrieval_lats else None,
        "retrieval_p95_ms": percentile(retrieval_lats, .95),
        "llm_median_ms": round(statistics.median(llm_lats), 1) if llm_lats else None,
        "llm_p95_ms": percentile(llm_lats, .95),
        "graded_answered": len(graded_answered),
        "human_score": sum(graded_answered.values()),
        "human_score_max_graded": 2 * len(graded_answered),
        "human_score_max_all_cases": 2 * len(rows),
        "human_full_coverage": len(graded_answered) == len(rows),
        "ignored_grades_without_answer": ignored_grades,
        "out_of_scope_cases": sum(bool(r.get("out_of_scope")) for r in rows),
        "graded_out_of_scope": sum(r["id"] in graded_answered for r in rows if r.get("out_of_scope")),
        "out_of_scope_full_credit": sum(graded_answered.get(r["id"]) == 2 for r in rows if r.get("out_of_scope")),
    }


def format_report(s: dict) -> str:
    human_label = (
        f"{s['human_score']}/{s['human_score_max_all_cases']} "
        f"({100*s['human_score']/s['human_score_max_all_cases']:.1f}%)"
        if s["human_full_coverage"] and s["human_score_max_all_cases"]
        else "Not reportable as full-set accuracy (missing grades or LLM answers)"
    )
    lines = [
        "# RAG quality evaluation — measured results",
        "",
        f"- Mode: `{s['mode']}`; Top-k: `{s['k']}`; evaluated questions: **{s['cases']}**.",
        "- Source: actual Task 16 evaluation report + manually scored CSV (not a model judge).",
        "",
        "| Metric | Observed result |",
        "|---|---:|",
        f"| Expected-document hit@k | {ratio(s['retrieval_doc_hits'], s['retrieval_doc_evaluable'])} |",
        f"| Exact evidence phrase present@k | {ratio(s['retrieval_evidence_hits'], s['retrieval_evidence_evaluable'])} |",
        f"| Answers received | {ratio(s['answers_received'], s['cases'])} |",
        f"| Provider errors | {s['provider_errors']} |",
        f"| Answers containing a numeric citation | {ratio(s['answered_with_numeric_citations'], s['answers_received'])} |",
        f"| Answers with invalid numeric citation references | {ratio(s['invalid_citation_answers'], s['answers_received'])} |",
        f"| Answers with heuristic analysis-leak detection | {ratio(s['analysis_leak_answers'], s['answers_received'])} |",
        f"| Retrieval latency median / p95 | {ms(s['retrieval_median_ms'])} / {ms(s['retrieval_p95_ms'])} |",
        f"| LLM latency median / p95 (successful only) | {ms(s['llm_median_ms'])} / {ms(s['llm_p95_ms'])} |",
        f"| Human-scored successful answers | {s['graded_answered']}/{s['answers_received']} |",
        f"| Human score (graded subset only) | {s['human_score']}/{s['human_score_max_graded']} |",
        f"| Human score over complete case set | {human_label} |",
        f"| Out-of-scope questions with full human credit | {s['out_of_scope_full_credit']}/{s['out_of_scope_cases']} (graded: {s['graded_out_of_scope']}) |",
        "",
        "## Interpretation and limitations",
        "",
        "- Retrieval is compared with *labeled documents* and *exact evidence phrases*, not independently judged semantic relevance.",
        "- Valid numeric citation references do **not** prove that the cited passage supports each factual claim; inspect each answer manually.",
        "- A missing answer, provider error, or missing human grade must not be counted as correct.",
        "- Latencies are from this particular run and can include embedding model cold-start; disclose hardware and model for comparisons.",
        "- This small synthetic dataset does not establish generalization to real, diverse private DOCX files.",
    ]
    if s["ignored_grades_without_answer"]:
        lines.extend(["", "**Warning:** Ignored grades for cases without an LLM answer: " + ", ".join(s["ignored_grades_without_answer"])])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("rag_quality_report.json"))
    parser.add_argument("--review", type=Path, default=Path("rag_quality_review.csv"))
    parser.add_argument("--output", type=Path, default=Path("rag_quality_metrics.md"))
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8-sig"))
        grades = load_review(args.review) if args.review.exists() else {}
        summary = summarize(report, grades)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(format_report(summary), encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Summary saved: {args.output}")
    print(f"Retrieval document hit@k: {ratio(summary['retrieval_doc_hits'], summary['retrieval_doc_evaluable'])}")
    if summary["human_full_coverage"]:
        print(f"Full-set human score: {summary['human_score']}/{summary['human_score_max_all_cases']}")
    else:
        print("Full-set answer quality: NOT VERIFIED (complete LLM run + human scores required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
