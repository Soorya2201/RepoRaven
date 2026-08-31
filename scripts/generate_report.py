"""Generate one polished, human-readable audit report for a phase.

Addresses a real gap: outputs/<phase>/<paper_id>.txt are developer/debug
artifacts (file dumps, JSON blobs, raw error strings). They're useful for
verifying the pipeline works, but nobody's actual intended user -- a PC
chair with 15 minutes -- would want to read them. This script turns the same
underlying data into something meant to be read by that person: findings
sorted by how much they're worth your time, an explicit trust framing per
the README's human-review disclaimer, and the paper/repo cited up front so
you know what you're looking at before you look at anything else.

Usage:
    python scripts/generate_report.py iter1
    python scripts/generate_report.py iter1 --paper-id g_mixup   # single paper

Writes outputs/<phase>/FINAL_REPORT.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _finding_sort_key(f: dict) -> tuple:
    # Worth-your-time-first: severity, then confidence (both descending).
    return (SEVERITY_ORDER.get(f.get("severity", "low"), 2), -float(f.get("confidence", 0) or 0))


def _format_repo_commit(gt_entry: dict) -> str:
    repo_url = gt_entry.get("repo_url", "n/a")
    commit = str(gt_entry.get("repo_commit") or "").strip()
    # Only render as a short `sha` if it actually looks like one -- the
    # synthetic fixture's repo_commit is a descriptive sentence, and blindly
    # slicing the first 10 characters of that mangles it mid-word.
    is_sha = bool(commit) and len(commit) >= 6 and all(c in "0123456789abcdefABCDEF" for c in commit)
    if is_sha:
        return f"{repo_url} @ `{commit[:10]}`"
    return repo_url


def _render_paper(paper_id: str, gt_entry: dict, record: dict) -> str:
    findings = sorted(record.get("findings", []), key=_finding_sort_key)
    score = record.get("score", {})
    lines = []

    lines.append(f"## {gt_entry.get('title', paper_id)}")
    lines.append("")
    arxiv = gt_entry.get("arxiv_id")
    lines.append(
        f"**Paper:** {'arXiv:' + arxiv if arxiv else 'synthetic fixture, not a real paper'}  \n"
        f"**Repo:** {_format_repo_commit(gt_entry)}"
    )
    lines.append("")

    error = record.get("error") or ""
    if error and not error.startswith("Recovered via repair pass"):
        lines.append(f"> ⚠️ **This run hit an error and may be incomplete:** {error[:300]}")
        lines.append("")
    elif error:
        # A successful self-repair, not a real problem -- worth one quiet line, not a warning banner.
        lines.append("<sub>ℹ️ A formatting slip in the model's raw output was automatically repaired before scoring; findings below are unaffected.</sub>")
        lines.append("")

    if not findings:
        lines.append("**No findings.** Nothing in this repo contradicted a claim we could check against the paper.")
        lines.append("")
        lines.append(
            "This is not the same as a guarantee of correctness — it means our specific checks "
            "(hyperparameters, seeds, metrics, data splits, evaluation protocol) turned up nothing. "
            "See the README's human-review note before treating this as a clean bill of health."
        )
    else:
        lines.append(f"**{len(findings)} finding{'s' if len(findings) != 1 else ''}**, ordered by how much they're worth your time:")
        lines.append("")
        lines.append("*Each finding below cites an exact file:line and the paper sentence it's checked against — verify before acting on any of them; see the human-review note in README.md.*")
        lines.append("")
        for i, f in enumerate(findings, 1):
            sev = (f.get("severity") or "?").upper()
            conf = f.get("confidence")
            conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else "?"
            lines.append(f"### {i}. [{sev}] {f.get('failure_mode', 'unknown')} (confidence: {conf_str})")
            lines.append("")
            lines.append(f"> **Paper says:** {f.get('paper_evidence', '(no quote)')}")
            lines.append("")
            lines.append(f"**Code:** `{f.get('code_evidence', '?')}`")
            lines.append("")
            lines.append(f"{f.get('explanation', '')}")
            lines.append("")

    n_expected = len(gt_entry.get("expected_findings", []))
    if n_expected or findings:
        lines.append(
            f"<sub>Scoring (for eval purposes, not shown to end users in a real deployment): "
            f"{score.get('tp', 0)} confirmed against known ground truth, "
            f"{score.get('fp', 0)} unconfirmed, {score.get('fn', 0)} missed. F1 = {score.get('f1', 'n/a')}.</sub>"
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def generate(phase: str, only_paper: str | None = None) -> Path:
    phase_dir = OUTPUTS / phase
    if not phase_dir.exists():
        raise FileNotFoundError(f"No outputs for phase {phase!r}. Run `python -m src.eval {phase}` first.")

    ground_truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())
    summary = json.loads((phase_dir / "_summary.json").read_text())

    paper_ids = [only_paper] if only_paper else list(ground_truth.keys())

    header = [
        f"# RepoRaven Report — {phase}",
        "",
        "**Findings are advisory triage, not adjudication.** This system is measurably "
        f"imperfect (precision {summary['micro_avg']['precision']:.0%} on our own eval set — "
        "roughly 1 in 4-5 findings is wrong). Every finding below cites an exact file:line and "
        "the paper sentence it's checked against, so you can verify it in under a minute. "
        "**Do not forward any finding to an author, editor, or committee without checking the "
        "citation yourself first.** Full context: README.md.",
        "",
        f"Papers audited: {summary['n_papers']}  ·  "
        f"Micro-avg F1: {summary['micro_avg']['f1']:.3f}  ·  "
        f"Precision: {summary['micro_avg']['precision']:.0%}  ·  "
        f"Recall: {summary['micro_avg']['recall']:.0%}",
        "",
        "---",
        "",
    ]

    body = []
    for paper_id in paper_ids:
        record_path = phase_dir / f"{paper_id}.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        gt_entry = ground_truth.get(paper_id, {})
        body.append(_render_paper(paper_id, gt_entry, record))

    out_path = phase_dir / ("FINAL_REPORT.md" if not only_paper else f"REPORT_{only_paper}.md")
    out_path.write_text("\n".join(header) + "\n".join(body), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", help="Phase name, e.g. iter1")
    parser.add_argument("--paper-id", default=None, help="Generate for a single paper only")
    args = parser.parse_args()

    path = generate(args.phase, args.paper_id)
    print(f"wrote {path.relative_to(ROOT)}")
