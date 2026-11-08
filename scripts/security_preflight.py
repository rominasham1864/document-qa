"""Read-only check for sensitive files accidentally tracked by Git.

This is a *limited* preflight, NOT a secrets scanner or penetration test.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath


IGNORED_EXAMPLE_NAMES = {".env.example", ".env.sample", ".env.template"}
SENSITIVE_DIRS = {"media", "chroma_data", "staticfiles", "data", "hf_models"}
SENSITIVE_SUFFIXES = {".sqlite3", ".db", ".pem", ".p12", ".pfx", ".key"}
SENSITIVE_REPORTS = {"rag_quality_report.json", "rag_quality_review.csv"}


def sensitive_reason(path: str) -> str | None:
    normalized = path.replace("\\", "/").strip("/")
    parts = PurePosixPath(normalized).parts
    if not parts:
        return None
    name = parts[-1].lower()
    if name in IGNORED_EXAMPLE_NAMES:
        return None
    if re.fullmatch(r"\.env(?:\..+)?", name):
        return "environment secrets file"
    if any(part.lower() in SENSITIVE_DIRS for part in parts[:-1]):
        return "runtime data/model storage"
    if name in SENSITIVE_REPORTS:
        return "generated report could contain private documents and answers"
    if any(name.endswith(suffix) for suffix in SENSITIVE_SUFFIXES):
        return "credential, private key, or database file"
    return None


def tracked_paths() -> list[str]:
    process = subprocess.run(
        ["git", "ls-files", "-z"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return [s.decode("utf-8", errors="replace") for s in process.stdout.split(b"\0") if s]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail when sensitive paths are tracked")
    args = parser.parse_args(argv)
    try:
        files = tracked_paths()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: git ls-files failed ({type(exc).__name__}). Run inside a Git repository.", file=sys.stderr)
        return 2
    problems = [(name, sensitive_reason(name)) for name in files]
    problems = [(name, reason) for name, reason in problems if reason]
    if problems:
        print("POTENTIALLY SENSITIVE TRACKED FILES:")
        for name, reason in problems:
            print(f" - {name}: {reason}")
        print("Review these files before pushing. Removing a file now does not erase Git history.")
    else:
        print(f"PASS: inspected {len(files)} Git-tracked paths; no files matched the sensitive-path rules.")
    print("NOTE: This does not inspect file contents or guarantee that secrets are absent.")
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
