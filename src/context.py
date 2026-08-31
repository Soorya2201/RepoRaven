"""Shared context-formatting helpers, used by baseline.py (Phase 1) and the
extractor/verifier (Phase 2+) so every phase builds prompts from the same
underlying repo/paper representation."""

from src.ingest import load_paper, load_repo

MAX_LINES_PER_FILE = 200
# nemotron-3-super-120b-a12b advertises a 1M-token context window; capping
# well below that is both cheaper/faster and keeps every phase's context
# comparable. Our eval-set repos never come close to this in practice
# (largest is ~40 files x 200 lines).
MAX_CONTEXT_CHARS = 400_000  # ~100k tokens at a ~4 chars/token estimate


def format_paper_text(paper_id: str) -> tuple[str, dict]:
    """Returns (formatted paper text block, raw paper dict)."""
    paper = load_paper(paper_id)
    return f"=== PAPER: {paper['title']} ===\n{paper['raw_text']}", paper


def format_repo_code_context(paper_id: str, max_lines_per_file: int = MAX_LINES_PER_FILE) -> tuple[str, dict]:
    """Returns (formatted README + per-file code-head text block, raw repo dict)."""
    repo = load_repo(paper_id)
    parts = [f"=== README ===\n{repo['readme']}"]
    for py_file in repo["python_files"]:
        try:
            lines = py_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        head = "\n".join(lines[:max_lines_per_file])
        rel_path = py_file.relative_to(repo["root"])
        parts.append(f"\n=== FILE: {rel_path} ===\n{head}")
    return "\n".join(parts), repo


def truncate(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[... truncated to fit context budget ...]"
    return text
