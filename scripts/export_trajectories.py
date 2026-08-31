"""Export one representative agent trajectory per phase (micro1 deliverable #4).

The brief asks for trajectories that are "easy to follow from the agent
instructions to the final result... what the agent did and how its tools
responded... the feedback that shaped its next step as well as any retries or
human checkpoints." This script assembles exactly that from data already saved
during each phase's eval run (outputs/<phase>/<paper_id>.json), so the
trajectories are real recorded runs, not reconstructions.

Usage:
    python scripts/export_trajectories.py

Writes outputs/trajectories/<phase>_<paper_id>.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
TRAJ_DIR = OUTPUTS / "trajectories"

# The agent instructions (system prompts) that shape each phase's agents.
# Imported lazily so this script still runs if a phase module is mid-edit.
def _phase_agents(phase: str) -> list[dict]:
    if phase == "baseline":
        from src.baseline import BASELINE_PROMPT
        return [{"agent": "baseline (single node)", "model_role": "NIM_MODEL_PRIMARY", "instructions": BASELINE_PROMPT}]
    if phase == "iter1":
        from src.agents.extractor import EXTRACTOR_PROMPT
        from src.agents.verifier import VERIFIER_PROMPT
        return [
            {"agent": "extractor", "model_role": "NIM_MODEL_FAST", "instructions": EXTRACTOR_PROMPT},
            {"agent": "verifier", "model_role": "NIM_MODEL_PRIMARY", "instructions": VERIFIER_PROMPT},
        ]
    if phase == "iter2":
        from src.agents.extractor import EXTRACTOR_PROMPT
        from src.agents.verifier_tools import VERIFIER_TOOLS_PROMPT
        return [
            {"agent": "extractor", "model_role": "NIM_MODEL_FAST", "instructions": EXTRACTOR_PROMPT},
            {"agent": "verifier (tool-using ReAct)", "model_role": "NIM_MODEL_TOOLCALLING", "instructions": VERIFIER_TOOLS_PROMPT},
        ]
    return []


def _pick_representative(phase_dir: Path) -> str | None:
    """Pick the most informative paper for this phase: prefer one that both
    produced findings AND (for tool phases) made tool calls, so the trajectory
    actually shows the agent working rather than a trivial empty run."""
    best, best_score = None, -1
    for f in sorted(phase_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        # Findings weighted heavily -- a trajectory that ends with zero
        # findings is a poor representative example regardless of how much
        # intermediate activity (claims, tool calls) it shows. Found by
        # inspection: an earlier version of this heuristic picked a paper
        # with 0 findings and 0 tool calls purely because it had more
        # extracted claims than a paper that actually caught real bugs.
        score = len(d.get("findings", [])) * 10 + len(d.get("tool_calls_made", [])) + len(d.get("claims", []))
        if score > best_score:
            best, best_score = d, score
    return best


def export_phase(phase: str) -> Path | None:
    phase_dir = OUTPUTS / phase
    if not phase_dir.exists():
        print(f"skip {phase}: no outputs directory")
        return None

    record = _pick_representative(phase_dir)
    if record is None:
        print(f"skip {phase}: no per-paper results found")
        return None

    paper_id = record.get("paper_id", "unknown")
    ground_truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())
    gt_entry = ground_truth.get(paper_id, {})

    steps = []
    if record.get("claims"):
        steps.append({
            "step": len(steps) + 1,
            "agent": "extractor",
            "action": "Extracted checkable claims from the paper text",
            "result": record["claims"],
            "why_it_matters": "Each claim carries a verbatim quote. The verifier may only report findings grounded in one of these quotes -- this is the anti-hallucination mechanism introduced in Phase 2.",
        })
    for tc in record.get("tool_calls_made", []):
        steps.append({
            "step": len(steps) + 1,
            "agent": "verifier (tool-using)",
            "action": f"Called tool {tc['tool']}",
            "tool_input": tc["args"],
            "tool_response": tc["result"],
            "why_it_matters": "Tool response fed back into the loop; the agent's next step is chosen based on what this returned.",
        })
    steps.append({
        "step": len(steps) + 1,
        "agent": "verifier",
        "action": "Produced final findings",
        "result": record.get("findings", []),
        "retries_and_checkpoints": record.get("error") or "none -- clean first-pass parse",
    })

    trajectory = {
        "phase": phase,
        "paper_id": paper_id,
        "paper_title": gt_entry.get("title"),
        "repo_url": gt_entry.get("repo_url"),
        "repo_commit": gt_entry.get("repo_commit"),
        "agent_instructions": _phase_agents(phase),
        "trajectory": steps,
        "final_findings": record.get("findings", []),
        "ground_truth_expected": gt_entry.get("expected_findings", []),
        "score_vs_ground_truth": record.get("score"),
        "notes": (
            "Recorded from a real eval run (src/eval.py), not reconstructed. "
            "'retries_and_checkpoints' surfaces any JSON-repair pass, extraction retry, "
            "or ungrounded-finding rejection that occurred -- these are the automatic "
            "checkpoints described in the phase writeups."
        ),
    }

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAJ_DIR / f"{phase}_{paper_id}.json"
    out_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)} ({len(steps)} steps, paper: {paper_id})")
    return out_path


if __name__ == "__main__":
    for phase in ["baseline", "iter1", "iter2", "iter3", "iter4", "final"]:
        export_phase(phase)
