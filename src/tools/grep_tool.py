"""Regex/substring search across the ENTIRE repo -- every file type, not just
.py heads. This directly fixes the Phase 2 failure where a real seed loop
living in run_gmixup.sh was invisible to a .py-only context (see
phase_2_iter1_extractor_verifier.md, Failure Case 1). Also the primary
mechanism for absence checks ("does --seed appear ANYWHERE in this repo").
"""

import re
from pathlib import Path

from pydantic import BaseModel, Field

from src.ingest import load_repo

MAX_MATCHES = 40
MAX_FILE_SIZE_BYTES = 2_000_000  # skip anything absurdly large (e.g. vendored data)
CONTEXT_LINES = 2


class ReadFile(BaseModel):
    """Read the full contents of one specific file in the repo (optionally a
    line range). Use this when you need to see a file's actual logic in full,
    not just the lines matching a grep pattern -- e.g. to read an entire
    function body or trace control flow across a file."""

    file_path: str = Field(description="Path to the file, relative to the repo root, e.g. 'src/gmixup.py' or 'run_gmixup.sh'.")
    line_start: int = Field(default=1, description="First line to include (1-indexed).")
    line_end: int = Field(default=400, description="Last line to include (inclusive). Capped at line_start+400.")


def run_read_file(paper_id: str, file_path: str, line_start: int = 1, line_end: int = 400) -> str:
    try:
        repo = load_repo(paper_id)
    except FileNotFoundError as e:
        return f"ERROR: {e}"

    target = (repo["root"] / file_path)
    try:
        target = target.resolve()
        target.relative_to(repo["root"].resolve())
    except (OSError, ValueError):
        return f"ERROR: invalid or out-of-repo path {file_path!r}"

    if not target.exists() or not target.is_file():
        return f"ERROR: {file_path!r} does not exist in this repo."

    try:
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as e:
        return f"ERROR reading {file_path!r}: {e}"

    line_end = min(line_end, line_start + 400, len(lines))
    start_idx = max(0, line_start - 1)
    snippet = "\n".join(f"{i+1}: {lines[i]}" for i in range(start_idx, line_end))
    if not snippet:
        return f"{file_path!r} has {len(lines)} lines; requested range {line_start}-{line_end} returned nothing."
    return f"=== {file_path} (lines {line_start}-{line_end} of {len(lines)}) ===\n{snippet}"


class GrepRepo(BaseModel):
    """Search the ENTIRE repository (all file types -- .py, .sh, .yaml, .md, everything)
    for a regex pattern. Use this to check whether something exists ANYWHERE in the repo
    (e.g. 'is there a --seed flag anywhere', 'is fit_transform called anywhere'), not just
    in the files you've already been shown heads of."""

    pattern: str = Field(description="Python regex pattern to search for (case-insensitive).")
    file_glob: str = Field(
        default="*",
        description="Optional filename glob filter, e.g. '*.py' or '*.sh'. Default '*' searches all files.",
    )


def run_grep_tool(paper_id: str, pattern: str, file_glob: str = "*") -> str:
    try:
        repo = load_repo(paper_id)
    except FileNotFoundError as e:
        return f"ERROR: {e}"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"ERROR: invalid regex pattern {pattern!r}: {e}"

    matches = []
    for f in repo["files"]:
        if not f.match(file_glob):
            continue
        try:
            if f.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel = f.relative_to(repo["root"])
        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - CONTEXT_LINES)
                end = min(len(lines), i + CONTEXT_LINES + 1)
                snippet = "\n".join(f"{j+1}: {lines[j]}" for j in range(start, end))
                matches.append(f"--- {rel}:{i+1} ---\n{snippet}")
                if len(matches) >= MAX_MATCHES:
                    break
        if len(matches) >= MAX_MATCHES:
            break

    if not matches:
        return f"No matches for pattern {pattern!r} (glob {file_glob!r}) in any file in the repo."
    header = f"Found {len(matches)} match(es) for {pattern!r} (glob {file_glob!r}), searched ALL {len(repo['files'])} files in the repo:\n\n"
    return header + "\n\n".join(matches)
