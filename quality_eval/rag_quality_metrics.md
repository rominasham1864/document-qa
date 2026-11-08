# RAG quality evaluation — measured results

- Mode: `retrieval+llm`; Top-k: `4`; evaluated questions: **16**.
- Source: actual Task 16 evaluation report + manually scored CSV (not a model judge).

| Metric | Observed result |
|---|---:|
| Expected-document hit@k | 14/14 (100.0%) |
| Exact evidence phrase present@k | 14/14 (100.0%) |
| Answers received | 15/16 (93.8%) |
| Provider errors | 1 |
| Answers containing a numeric citation | 13/15 (86.7%) |
| Answers with invalid numeric citation references | 0/15 (0.0%) |
| Answers with heuristic analysis-leak detection | 0/15 (0.0%) |
| Retrieval latency median / p95 | 44.5 ms / 6711.4 ms |
| LLM latency median / p95 (successful only) | 1963.0 ms / 7766.7 ms |
| Human-scored successful answers | 15/15 |
| Human score (graded subset only) | 30/30 |
| Human score over complete case set | Not reportable as full-set accuracy (missing grades or LLM answers) |
| Out-of-scope questions with full human credit | 2/2 (graded: 2) |

## Interpretation and limitations

- Retrieval is compared with *labeled documents* and *exact evidence phrases*, not independently judged semantic relevance.
- Valid numeric citation references do **not** prove that the cited passage supports each factual claim; inspect each answer manually.
- A missing answer, provider error, or missing human grade must not be counted as correct.
- Latencies are from this particular run and can include embedding model cold-start; disclose hardware and model for comparisons.
- This small synthetic dataset does not establish generalization to real, diverse private DOCX files.
