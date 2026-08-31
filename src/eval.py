"""Scoring against ground truth + the phase evaluation harness.

CLI: python -m src.eval <phase_name>
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from tabulate import tabulate

from src.llm import get_usage_totals, reset_usage_totals

ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth.json"
OUTPUTS_DIR = ROOT / "outputs"

SEVERITY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}

FAILURE_MODES = [
    "data_leakage_global_preprocessing",
    "hyperparam_drift",
    "single_seed_reporting",
    "metric_misimplementation",
    "test_set_contamination",
    "eval_on_train",
    "other",
]


def _extract_py_basenames(code_evidence: str) -> set:
    """Ground truth code_evidence can cite multiple files (e.g. lbcs's finding
    spans 3 scripts). We match on basename rather than full path so minor
    relative-path differences between predicted and ground-truth citations
    (e.g. 'src/gmixup.py' vs 'gmixup.py') don't cause spurious mismatches --
    within a single repo/paper's scope, same-named .py files are rare."""
    if not code_evidence:
        return set()
    paths = re.findall(r"[\w./-]+\.py", code_evidence)
    return {os.path.basename(p) for p in paths}


def _findings_match(pred: dict, gt: dict) -> bool:
    if pred.get("failure_mode") != gt.get("failure_mode"):
        return False
    pred_files = _extract_py_basenames(pred.get("code_evidence", ""))
    gt_files = _extract_py_basenames(gt.get("code_evidence", ""))
    if not pred_files or not gt_files:
        return False
    return bool(pred_files & gt_files)


def score_findings(predicted: list, ground_truth: list) -> dict:
    """Greedy one-to-one matching per instructions.md §7.2. Returns per-paper
    tp/fp/fn plus which GT findings were matched/missed and which predictions
    were unmatched, for failure-analysis writeups."""
    gt_remaining = list(ground_truth)
    matched_pred_idx = set()
    matched_gt_idx = set()

    for gi, gt_finding in enumerate(gt_remaining):
        for pi, pred_finding in enumerate(predicted):
            if pi in matched_pred_idx:
                continue
            if _findings_match(pred_finding, gt_finding):
                matched_pred_idx.add(pi)
                matched_gt_idx.add(gi)
                break

    tp = len(matched_gt_idx)
    fp = len(predicted) - len(matched_pred_idx)
    fn = len(ground_truth) - len(matched_gt_idx)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    tp_weight = sum(SEVERITY_WEIGHTS.get(predicted[pi].get("severity", "low"), 1) for pi in matched_pred_idx)
    fp_indices = [i for i in range(len(predicted)) if i not in matched_pred_idx]
    fp_weight = sum(SEVERITY_WEIGHTS.get(predicted[i].get("severity", "low"), 1) for i in fp_indices)
    severity_weighted_precision = (
        tp_weight / (tp_weight + fp_weight) if (tp_weight + fp_weight) > 0 else (1.0 if fn == 0 else 0.0)
    )

    unmatched_gt = [g for gi, g in enumerate(ground_truth) if gi not in matched_gt_idx]
    unmatched_pred = [p for pi, p in enumerate(predicted) if pi not in matched_pred_idx]

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "severity_weighted_precision": round(severity_weighted_precision, 4),
        "unmatched_gt": unmatched_gt, "unmatched_pred": unmatched_pred,
    }


def _write_report_txt(path: Path, paper_id: str, gt_entry: dict, predicted: list, score: dict, phase_name: str) -> None:
    lines = []
    lines.append("=" * 65)
    lines.append(f"RepoAudit Report — {paper_id}")
    lines.append(f"Paper: {gt_entry.get('title', 'UNKNOWN')}")
    lines.append(f"arXiv: {gt_entry.get('arxiv_id')}   Repo: {gt_entry.get('repo_url')} @ {gt_entry.get('repo_commit')}")
    lines.append(f"Phase: {phase_name}")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"FINDINGS: {len(predicted)}")
    lines.append("")
    for i, f in enumerate(predicted, 1):
        lines.append(f"[{i}] {f.get('severity', '?')} — {f.get('failure_mode', '?')}")
        lines.append(f"    Paper:  {f.get('paper_evidence', '')}")
        lines.append(f"    Code:   {f.get('code_evidence', '')}")
        lines.append(f"    Why:    {f.get('explanation', '')}")
        lines.append(f"    Confidence: {f.get('confidence', '')}")
        lines.append("")
    lines.append("=" * 65)
    n_expected = len(gt_entry.get("expected_findings", []))
    lines.append(f"Ground truth: {n_expected} findings expected.")
    lines.append(f"Match: {score['tp']} true positive, {score['fp']} false positive, {score['fn']} false negative.")
    lines.append(f"F1: {score['f1']}")
    lines.append("=" * 65)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase(phase_name: str, agent_fn) -> dict:
    """agent_fn(paper_id) -> {'findings': list, 'raw_response': str, 'error': str|None}"""
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    out_dir = OUTPUTS_DIR / phase_name
    out_dir.mkdir(parents=True, exist_ok=True)

    reset_usage_totals()
    start = time.time()

    per_paper = {}
    by_mode = {m: {"tp": 0, "fp": 0, "fn": 0} for m in FAILURE_MODES}
    clean_paper_pred_counts = []
    errors = {}

    for paper_id, gt_entry in ground_truth.items():
        result = agent_fn(paper_id)
        predicted = result.get("findings", [])
        if result.get("error"):
            errors[paper_id] = result["error"]

        expected = gt_entry.get("expected_findings", [])
        score = score_findings(predicted, expected)
        per_paper[paper_id] = {
            "tp": score["tp"], "fp": score["fp"], "fn": score["fn"],
            "precision": score["precision"], "recall": score["recall"], "f1": score["f1"],
        }

        if len(expected) == 0:
            clean_paper_pred_counts.append(len(predicted))

        # by-failure-mode tally: walk matched/unmatched sets per mode
        gt_by_mode = {}
        for g in expected:
            gt_by_mode.setdefault(g["failure_mode"], []).append(g)
        pred_by_mode = {}
        for p in predicted:
            pred_by_mode.setdefault(p.get("failure_mode"), []).append(p)
        for mode in set(list(gt_by_mode.keys()) + list(pred_by_mode.keys())):
            sub_score = score_findings(pred_by_mode.get(mode, []), gt_by_mode.get(mode, []))
            if mode in by_mode:
                by_mode[mode]["tp"] += sub_score["tp"]
                by_mode[mode]["fp"] += sub_score["fp"]
                by_mode[mode]["fn"] += sub_score["fn"]

        (out_dir / f"{paper_id}.json").write_text(
            json.dumps({
                "paper_id": paper_id, "findings": predicted, "score": score, "error": result.get("error"),
                "raw_response": result.get("raw_response", ""),
                "claims": result.get("claims", []),
                "tool_calls_made": result.get("tool_calls_made", []),
                "wall_clock_seconds": result.get("wall_clock_seconds"),
            }, indent=2),
            encoding="utf-8",
        )
        _write_report_txt(out_dir / f"{paper_id}.txt", paper_id, gt_entry, predicted, score, phase_name)

    wall_clock = time.time() - start
    usage = get_usage_totals()

    total_tp = sum(v["tp"] for v in per_paper.values())
    total_fp = sum(v["fp"] for v in per_paper.values())
    total_fn = sum(v["fn"] for v in per_paper.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )
    fpr_on_clean = (
        sum(clean_paper_pred_counts) / len(clean_paper_pred_counts) if clean_paper_pred_counts else 0.0
    )

    by_mode_summary = {}
    for mode, counts in by_mode.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        by_mode_summary[mode] = {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4)}

    summary = {
        "phase": phase_name,
        "model_primary": os.environ.get("NIM_MODEL_PRIMARY"),
        "model_fast": os.environ.get("NIM_MODEL_FAST"),
        "n_papers": len(ground_truth),
        "per_paper": per_paper,
        "micro_avg": {
            "precision": round(micro_precision, 4), "recall": round(micro_recall, 4), "f1": round(micro_f1, 4),
            "fpr_on_clean": round(fpr_on_clean, 4),
        },
        "by_failure_mode": by_mode_summary,
        "wall_clock_seconds": round(wall_clock, 1),
        "token_usage": usage,
        "estimated_nim_credits": None,  # NIM's $/credit rate could not be verified (see phase_0_setup.md known issues) -- token_usage above is the real, verifiable number.
        "errors": errors,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table_rows = [
        [pid, len(ground_truth[pid].get("expected_findings", [])), v["tp"] + v["fp"], v["tp"], v["fp"], v["fn"], v["f1"]]
        for pid, v in per_paper.items()
    ]
    table_rows.append(["MICRO-AVG", "-", "-", total_tp, total_fp, total_fn, round(micro_f1, 4)])
    table_str = tabulate(table_rows, headers=["paper_id", "n_expected", "n_predicted", "TP", "FP", "FN", "F1"])

    summary_txt = [
        f"RepoAudit — Phase: {phase_name}",
        f"Model (primary/fast): {summary['model_primary']} / {summary['model_fast']}",
        f"Papers: {summary['n_papers']}   Wall clock: {summary['wall_clock_seconds']}s",
        f"Token usage: {usage}",
        "",
        table_str,
        "",
        f"Micro-avg precision: {summary['micro_avg']['precision']}  recall: {summary['micro_avg']['recall']}  F1: {summary['micro_avg']['f1']}",
        f"False positive rate on clean papers: {summary['micro_avg']['fpr_on_clean']}",
        "",
        "By failure mode:",
    ]
    for mode, s in by_mode_summary.items():
        summary_txt.append(f"  {mode}: tp={s['tp']} fp={s['fp']} fn={s['fn']} f1={s['f1']}")
    if errors:
        summary_txt.append("")
        summary_txt.append(f"Errors ({len(errors)} papers): {list(errors.keys())}")
    (out_dir / "_summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")

    print("\n".join(summary_txt))
    return summary


def rescore_phase(phase_name: str) -> dict:
    """Re-run scoring for an already-completed phase against the CURRENT
    ground_truth.json, without calling the LLM again. Use this after fixing
    a ground-truth annotation (e.g. Phase 2 surfaced a real bug Phase 0
    missed) so old runs get rescored fairly instead of silently going stale.

    Cost/timing data (wall_clock_seconds, token_usage) is carried over from the
    original run: a replay makes no LLM calls, so measuring it here would
    report zeros and destroy the real cost figures the changelog cites."""
    out_dir = OUTPUTS_DIR / phase_name

    prior_summary_path = out_dir / "_summary.json"
    prior_cost = {}
    if prior_summary_path.exists():
        try:
            prior = json.loads(prior_summary_path.read_text())
            prior_cost = {
                "wall_clock_seconds": prior.get("wall_clock_seconds"),
                "token_usage": prior.get("token_usage"),
            }
        except json.JSONDecodeError:
            pass

    def _replay_from_disk(paper_id: str) -> dict:
        saved = json.loads((out_dir / f"{paper_id}.json").read_text())
        return {
            "findings": saved.get("findings", []),
            "raw_response": saved.get("raw_response", ""),
            "claims": saved.get("claims", []),
            "tool_calls_made": saved.get("tool_calls_made", []),
            "error": saved.get("error"),
        }

    summary = run_phase(phase_name, _replay_from_disk)

    # Restore the original run's real cost figures over the replay's zeros.
    if prior_cost.get("token_usage") and prior_cost["token_usage"].get("total_tokens"):
        summary["wall_clock_seconds"] = prior_cost["wall_clock_seconds"]
        summary["token_usage"] = prior_cost["token_usage"]
        summary["_cost_note"] = (
            "wall_clock_seconds/token_usage are from the original run; this phase was "
            "rescored against updated ground truth without re-calling the LLM."
        )
        prior_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _get_agent_fn(phase_name: str):
    if phase_name == "baseline":
        from src.baseline import run_baseline
        return run_baseline
    if phase_name == "iter1":
        from src.graph import run_iter1
        return run_iter1
    if phase_name == "iter2":
        from src.graph import run_iter2
        return run_iter2
    raise ValueError(f"Unknown phase: {phase_name}. Implemented so far: 'baseline' (Phase 1), 'iter1' (Phase 2), 'iter2' (Phase 3).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.eval <phase_name>")
        sys.exit(1)
    phase = sys.argv[1]
    fn = _get_agent_fn(phase)
    run_phase(phase, fn)
