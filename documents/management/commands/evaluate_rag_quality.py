"""Read-only, human-reviewable Persian RAG quality evaluation.

The evaluation never creates/updates/deletes Document or QuestionAnswer rows.
It uses the actual Chroma retrieval and optionally the real configured LLM.
"""

import csv
import json
import os
import re
import statistics
import time
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from documents.models import Document
from documents.services.rag import answer_question
from documents.services.vector_store import search_chunks


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
ANALYSIS_PATTERN = re.compile(
    r"analyze the request|analysis of the request|thought process|"
    r"chain.of.thought|step.by.step reasoning|internal reasoning|"
    r"\*\*analysis\*\*|^analysis:|^reasoning:|^final answer:",
    re.IGNORECASE | re.MULTILINE,
)
REFUSAL_TERMS = (
    "اطلاعات کافی", "در اسناد", "در متن", "اشاره نشده", "ذکر نشده", 
    "موجود نیست", "مشخص نیست", "نمی‌توان", "نمی توان", "نامشخص", 
    "not enough information", "not mentioned", "cannot determine",
)


def normalized(text):
    """Normalize only for coarse keyword checks; never modify model answers."""
    text = str(text).replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.casefold().strip()


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)], 3)


def _load_cases(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise CommandError(f"Could not parse evaluation JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise CommandError("Cases JSON must be a non-empty list.")
    seen = set()
    for position, case in enumerate(payload, 1):
        if not isinstance(case, dict):
            raise CommandError(f"Case {position} must be an object.")
        ident = case.get("id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise CommandError(f"Case {position}: unique non-empty string id required.")
        seen.add(ident)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise CommandError(f"Case {ident}: non-empty question required.")
        for field in ("expected_docs", "evidence", "answer_keywords"):
            value = case.get(field, [])
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise CommandError(f"Case {ident}: {field} must be a list of strings.")
        if any(name not in ("a", "b") for name in case.get("expected_docs", [])):
            raise CommandError(f"Case {ident}: expected_docs may only contain 'a'/'b'.")
        if not isinstance(case.get("out_of_scope", False), bool):
            raise CommandError(f"Case {ident}: out_of_scope must be boolean.")
    return payload


class Command(BaseCommand):
    help = (
        "Evaluate live retrieval against known Persian DOCX fixtures; optionally "
        "call OpenRouter, then write JSON/CSV for human quality review."
    )

    def add_arguments(self, parser):
        parser.add_argument("--cases", default="quality_eval/cases.json")
        parser.add_argument("--document-a", type=int, required=True)
        parser.add_argument("--document-b", type=int, required=True)
        parser.add_argument("--k", type=int, default=4)
        parser.add_argument("--limit", type=int)
        parser.add_argument("--with-llm", action="store_true")
        parser.add_argument("--output", default="quality_eval/rag_quality_report.json")
        parser.add_argument("--csv-output", default="quality_eval/rag_quality_review.csv")

    def handle(self, *args, **options):
        if not 1 <= options["k"] <= 8:
            raise CommandError("--k must be between 1 and 8.")
        if options["limit"] is not None and options["limit"] < 1:
            raise CommandError("--limit must be positive.")
        ids = {"a": options["document_a"], "b": options["document_b"]}
        if any(value <= 0 for value in ids.values()) or ids["a"] == ids["b"]:
            raise CommandError("Provide two different, positive document IDs.")
        for alias, pk in ids.items():
            try:
                document = Document.objects.get(pk=pk)
            except Document.DoesNotExist as exc:
                raise CommandError(f"Document {alias} (id={pk}) does not exist.") from exc
            if document.index_status != Document.IndexStatus.READY or not document.index_key:
                raise CommandError(
                    f"Document {alias} (id={pk}) is not READY. "
                    f"Current status={document.index_status!r}. Use retry-index first."
                )

        cases = _load_cases(Path(options["cases"]))
        if options["limit"] is not None:
            cases = cases[:options["limit"]]

        results = []
        for pos, case in enumerate(cases, start=1):
            question = case["question"].strip()
            expected = [ids[alias] for alias in case.get("expected_docs", [])]
            row = {
                "id": case["id"],
                "question": question,
                "out_of_scope": case.get("out_of_scope", False),
                "expected_answer": case.get("expected_answer", ""),
                "expected_document_ids": expected,
                "retrieved": [],
                "retrieval_ms": None,
                "all_expected_docs_in_top_k": None,
                "evidence_found": None,
                "missing_evidence": [],
                "answer": None,
                "llm_ms": None,
                "answer_error": None,
                "citations": [],
                "invalid_citations": [],
                "has_citation": None,
                "analysis_leak": None,
                "keyword_coverage": None,
                "possible_refusal": None,
                "human_grade_0_to_2": "",
                "human_notes": "",
            }
            chunks = []
            self.stdout.write(f"[{pos}/{len(cases)}] {case['id']} — {question}")
            started = time.perf_counter()
            try:
                # No document_id filter: this tests whether retrieval distinguishes
                # between two documents rather than being handed the correct one.
                chunks = search_chunks(question, k=options["k"])
            except Exception as exc:
                raise CommandError(
                    f"Retrieval failed on case {case['id']}: {type(exc).__name__}. "
                    "Inspect Chroma/E5 logs."
                ) from exc
            row["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 1)
            row["retrieved"] = [
                {
                    "rank": rank,
                    "document_id": chunk.metadata.get("document_id"),
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "title": chunk.metadata.get("title"),
                    "preview": chunk.page_content[:220].replace("\n", " "),
                }
                for rank, chunk in enumerate(chunks, 1)
            ]
            found_docs = {int(s["document_id"]) for s in row["retrieved"]
                          if s["document_id"] is not None}
            if expected:
                row["all_expected_docs_in_top_k"] = set(expected).issubset(found_docs)
            # Evidence checks test that required source facts made it into the
            # retrieved context, NOT that the generated answer is truthful.
            joined_context = normalized("\n".join(c.page_content for c in chunks))
            evidence = case.get("evidence", [])
            row["missing_evidence"] = [fact for fact in evidence
                                       if normalized(fact) not in joined_context]
            row["evidence_found"] = not row["missing_evidence"] if evidence else None

            if options["with_llm"]:
                started = time.perf_counter()
                try:
                    result = answer_question(
                        question,
                        k=options["k"],
                        search_fn=lambda _query, **_kwargs: chunks,
                    )
                    row["answer"] = result["answer"]
                    row["llm_ms"] = round((time.perf_counter() - started) * 1000, 1)
                    citations = [int(n) for n in CITATION_PATTERN.findall(row["answer"])]
                    row["citations"] = citations
                    valid_ranks = {s["rank"] for s in result["sources"]}
                    row["invalid_citations"] = sorted(set(citations) - valid_ranks)
                    row["has_citation"] = bool(citations)
                    row["analysis_leak"] = bool(ANALYSIS_PATTERN.search(row["answer"]))
                    required = case.get("answer_keywords", [])
                    hits = [word for word in required
                            if normalized(word) in normalized(row["answer"])]
                    row["keyword_coverage"] = (
                        f"{len(hits)}/{len(required)}" if required else None
                    )
                    row["possible_refusal"] = any(
                        normalized(term) in normalized(row["answer"])
                        for term in REFUSAL_TERMS
                    )
                except Exception as exc:
                    row["llm_ms"] = round((time.perf_counter() - started) * 1000, 1)
                    # Avoid serializing providers' raw exception text (may
                    # include sensitive prompts or credentials).
                    row["answer_error"] = type(exc).__name__
                    self.stderr.write(self.style.WARNING(
                        f"  LLM error ({type(exc).__name__}); continuing."
                    ))

            if expected:
                self.stdout.write(
                    f"  retrieval: document-hit={row['all_expected_docs_in_top_k']} "
                    f"evidence={row['evidence_found']} "
                    f"latency={row['retrieval_ms']}ms"
                )
            else:
                self.stdout.write("  out-of-scope: retrieval is not scored as empty")
            results.append(row)

        in_scope = [r for r in results if r["expected_document_ids"]]
        answered = [r for r in results if r["answer"] is not None]
        summary = {
            "total_cases": len(results),
            "in_scope_cases": len(in_scope),
            "retrieval_doc_hit_at_k": (
                round(sum(r["all_expected_docs_in_top_k"] for r in in_scope) / len(in_scope), 3)
                if in_scope else None
            ),
            "retrieval_evidence_hit_at_k": (
                round(sum(r["evidence_found"] is True for r in in_scope) / len(in_scope), 3)
                if in_scope else None
            ),
            "answers_received": len(answered),
            "provider_errors": sum(bool(r["answer_error"]) for r in results),
            "invalid_citation_answers": sum(bool(r["invalid_citations"]) for r in answered),
            "analysis_leak_answers": sum(bool(r["analysis_leak"]) for r in answered),
            "answer_with_citation": sum(bool(r["has_citation"]) for r in answered),
            "retrieval_latency_median_ms": round(statistics.median(
                r["retrieval_ms"] for r in results), 1),
            "retrieval_latency_p95_ms": _p95([r["retrieval_ms"] for r in results]),
            "llm_latency_median_ms": (
                round(statistics.median(r["llm_ms"] for r in answered), 1)
                if answered else None
            ),
        }
        report = {
            "mode": "retrieval+llm" if options["with_llm"] else "retrieval-only",
            "k": options["k"],
            "document_ids": ids,
            "llm_model": os.getenv("OPENROUTER_MODEL", ""),
            "note": (
                "Keyword/citation checks are heuristics, NOT factual correctness. "
                "Read answers and source previews; manually grade each answer "
                "0=incorrect, 1=partial, 2=correct and grounded. "
                "Out-of-scope questions may retrieve unrelated chunks."
            ),
            "summary": summary,
            "cases": results,
        }
        out_json = Path(options["output"])
        out_csv = Path(options["csv_output"])
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        columns = [
            "id", "question", "out_of_scope", "expected_answer", "expected_document_ids",
            "retrieved_document_ids", "retrieved_previews", "all_expected_docs_in_top_k",
            "evidence_found", "missing_evidence", "retrieval_ms", "answer", "llm_ms",
            "answer_error", "citations", "invalid_citations", "has_citation",
            "analysis_leak", "keyword_coverage", "possible_refusal",
            "human_grade_0_to_2", "human_notes",
        ]
        with out_csv.open("w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.DictWriter(fp, fieldnames=columns)
            writer.writeheader()
            for r in results:
                flat = {key: r.get(key) for key in columns}
                flat["expected_document_ids"] = ",".join(map(str, r["expected_document_ids"]))
                flat["retrieved_document_ids"] = ",".join(
                    str(s["document_id"]) for s in r["retrieved"])
                flat["retrieved_previews"] = " | ".join(s["preview"] for s in r["retrieved"])
                flat["missing_evidence"] = " | ".join(r["missing_evidence"])
                flat["citations"] = ",".join(map(str, r["citations"]))
                flat["invalid_citations"] = ",".join(map(str, r["invalid_citations"]))
                writer.writerow(flat)
        self.stdout.write(self.style.SUCCESS(f"Quality report: {out_json}"))
        self.stdout.write(self.style.SUCCESS(f"Manual review CSV: {out_csv}"))
        self.stdout.write("Summary: " + json.dumps(summary, ensure_ascii=False))
        self.stdout.write(self.style.WARNING(
            "Do not treat automated keywords as answer correctness. "
            "Score real answer grounding manually in the CSV."
        ))
