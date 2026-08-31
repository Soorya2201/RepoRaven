"""One-command reproduction of every RepoRaven phase (micro1 deliverable #2).

Usage:
    python scripts/run_all.py                  # fetch data + run all phases + export trajectories
    python scripts/run_all.py --phases iter2   # run only specific phase(s)
    python scripts/run_all.py --skip-fetch     # reuse already-downloaded papers/repos
    python scripts/run_all.py --rescore-only   # rescore saved predictions, no LLM calls

Phases run in order and each writes outputs/<phase>/. Every phase is scored
against the same data/ground_truth.json, so results are directly comparable.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ALL_PHASES = ["baseline", "iter1", "iter2", "iter3", "final"]
PYTHON = sys.executable


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - start
    ok = result.returncode == 0
    print(f"\n--> {label}: {'OK' if ok else 'FAILED'} ({elapsed:.0f}s)", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce all RepoRaven phases end to end.")
    parser.add_argument("--phases", nargs="+", default=ALL_PHASES,
                        help=f"Which phases to run (default: all of {ALL_PHASES}).")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip downloading papers / cloning repos (reuse what's already in data/).")
    parser.add_argument("--rescore-only", action="store_true",
                        help="Rescore existing saved predictions against current ground truth. No LLM calls, no cost.")
    args = parser.parse_args()

    if args.rescore_only:
        from src.eval import rescore_phase
        for phase in args.phases:
            if (ROOT / "outputs" / phase).exists():
                print(f"\n{'=' * 70}\nRescoring {phase} (no LLM calls)\n{'=' * 70}", flush=True)
                rescore_phase(phase)
        return 0

    if not (ROOT / ".env").exists():
        print("ERROR: .env not found. Copy .env.example to .env and fill in NVIDIA_API_KEY.", file=sys.stderr)
        return 1

    if not args.skip_fetch:
        if not _run([PYTHON, "scripts/curate_eval_set.py"],
                    "STEP 1/3: Fetching papers + cloning repos at pinned commits"):
            print("Curation failed -- check network access and GitHub availability.", file=sys.stderr)
            return 1
    else:
        print("Skipping fetch (--skip-fetch); reusing existing data/papers and data/repos.")

    failures = []
    for i, phase in enumerate(args.phases, start=1):
        label = f"STEP 2/3 ({i}/{len(args.phases)}): Running phase '{phase}'"
        if not _run([PYTHON, "-m", "src.eval", phase], label):
            failures.append(phase)

    _run([PYTHON, "scripts/export_trajectories.py"],
         "STEP 3/4: Exporting agent trajectories")

    for phase in args.phases:
        if (ROOT / "outputs" / phase / "_summary.json").exists():
            _run([PYTHON, "scripts/generate_report.py", phase],
                 f"STEP 4/4: Generating human-readable report for '{phase}'")

    print(f"\n{'=' * 70}\nDONE\n{'=' * 70}")
    print("Per-phase results:    outputs/<phase>/_summary.txt")
    print("Human-readable report: outputs/<phase>/FINAL_REPORT.md")
    print("Per-paper dev logs:   outputs/<phase>/<paper_id>.txt")
    print("Agent trajectories:   outputs/trajectories/")
    if failures:
        print(f"\nWARNING: these phases reported a non-zero exit: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
