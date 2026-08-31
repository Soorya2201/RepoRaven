"""Phase 0 eval-set curation.

Reads data/eval_set_candidates.json (a manifest of papers hand-selected per the
protocol in instructions.md §4.1 — this repo's own research step, not
automated discovery), downloads each paper's arXiv HTML, shallow-clones each
repo at a pinned commit, and re-checks the five selection filters
programmatically. Candidates that fail a filter are logged, not silently
dropped.

This script does NOT invent findings or ground truth — it only fetches
artifacts and validates the mechanical filters. `data/ground_truth.json` is
hand-annotated separately by reading the fetched code.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import PAPERS_DIR, REPOS_DIR, download_paper, load_repo  # noqa: E402

CANDIDATES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set_candidates.json"
MAX_REPO_MB = 500


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _clone_repo(paper_id: str, repo_url: str, commit: str | None) -> tuple[bool, str]:
    dest = REPOS_DIR / paper_id
    if dest.exists():
        shutil.rmtree(dest)
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True, capture_output=True, text=True, timeout=180,
        )
    except subprocess.CalledProcessError as e:
        return False, f"clone failed: {e.stderr.strip()[:300]}"
    except subprocess.TimeoutExpired:
        return False, "clone timed out"

    if commit:
        try:
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", commit],
                cwd=dest, check=True, capture_output=True, text=True, timeout=60,
            )
            subprocess.run(
                ["git", "checkout", commit],
                cwd=dest, check=True, capture_output=True, text=True, timeout=30,
            )
        except subprocess.CalledProcessError:
            # Commit not reachable via shallow fetch — fall back to whatever
            # the shallow clone gave us (HEAD) and record actual commit below.
            pass
    return True, "ok"


def _current_commit(paper_id: str) -> str:
    dest = REPOS_DIR / paper_id
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def curate() -> dict:
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            f"{CANDIDATES_PATH} not found. Populate it with the hand-researched "
            "candidate list before running curation."
        )
    candidates = json.loads(CANDIDATES_PATH.read_text())

    report = {"accepted": [], "rejected": [], "synthetic": []}

    for c in candidates:
        paper_id = c["paper_id"]

        # Synthetic fixtures are constructed directly under data/repos/ and
        # data/papers/ (no external repo/arXiv source) -- filters don't apply.
        if c.get("synthetic"):
            report["synthetic"].append(c)
            continue

        filters = {}

        # Filter 3: arXiv HTML (or PDF) availability
        source = download_paper(paper_id, c["arxiv_id"])
        if source == "missing":
            pdf_path = PAPERS_DIR / f"{paper_id}.pdf"
            filters["paper_available"] = pdf_path.exists()
        else:
            filters["paper_available"] = True
        c["paper_source"] = source if source != "missing" else ("pdf" if filters["paper_available"] else "missing")

        # Filter 1/2: repo accessible + clone, then size check
        ok, msg = _clone_repo(paper_id, c["repo_url"], c.get("repo_commit"))
        filters["repo_accessible"] = ok
        if not ok:
            report["rejected"].append({**c, "filters": filters, "reason": msg})
            continue

        c["repo_commit"] = _current_commit(paper_id)
        size_mb = _dir_size_mb(REPOS_DIR / paper_id)
        filters["repo_under_500mb"] = size_mb < MAX_REPO_MB
        c["repo_size_mb"] = round(size_mb, 1)

        repo_info = load_repo(paper_id)
        filters["has_python"] = len(repo_info["python_files"]) > 0
        filters["has_readme"] = bool(repo_info["readme"])

        if all(filters.values()):
            report["accepted"].append({**c, "filters": filters})
        else:
            report["rejected"].append({**c, "filters": filters, "reason": "failed filter(s)"})

    return report


if __name__ == "__main__":
    result = curate()
    print(json.dumps(result, indent=2))
    n_a = sum(1 for c in result["accepted"] if c["category"] == "A")
    n_b = sum(1 for c in result["accepted"] if c["category"] == "B")
    print(f"\nAccepted: {len(result['accepted'])} (A={n_a}, B={n_b}). Rejected: {len(result['rejected'])}.")
