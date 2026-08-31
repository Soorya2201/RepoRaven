"""Phase 4 checker orchestration: run all 6 specialized checkers against a
paper's extracted claims, combine their findings.

Each checker owns exactly one failure mode and cannot see or be confused by
the other 5 -- a deliberate contrast with Phase 3's single general verifier
juggling all 6 at once with an open-ended toolbox. Every checker follows the
same deterministic-gather-then-single-decision pattern (see base.py);
data_leakage additionally correlates code evidence with vision evidence
(see leakage.py), never accepting a vision-only finding.
"""

from src.agents.checkers.base import run_text_checker
from src.agents.checkers.eval_on_train import CONFIG as EVAL_ON_TRAIN_CONFIG
from src.agents.checkers.hyperparams import CONFIG as HYPERPARAM_CONFIG
from src.agents.checkers.leakage import run_leakage_checker
from src.agents.checkers.metric import CONFIG as METRIC_CONFIG
from src.agents.checkers.seeds import CONFIG as SEED_CONFIG
from src.agents.checkers.test_contamination import CONFIG as TEST_CONTAMINATION_CONFIG

TEXT_CHECKER_CONFIGS = [
    HYPERPARAM_CONFIG,
    SEED_CONFIG,
    METRIC_CONFIG,
    TEST_CONTAMINATION_CONFIG,
    EVAL_ON_TRAIN_CONFIG,
]


def run_all_checkers(paper_id: str, claims: list[dict]) -> dict:
    """Returns {'findings': list, 'per_checker': dict, 'error': str|None}."""
    all_findings = []
    per_checker = {}
    errors = []

    # data_leakage_global_preprocessing first -- it's the dual-modality one.
    leak_result = run_leakage_checker(paper_id, claims)
    all_findings.extend(leak_result["findings"])
    per_checker["data_leakage_global_preprocessing"] = {
        "n_findings": len(leak_result["findings"]),
        "evidence_gathered": leak_result["evidence_gathered"],
    }
    if leak_result["error"]:
        errors.append(leak_result["error"])

    for config in TEXT_CHECKER_CONFIGS:
        result = run_text_checker(paper_id, claims, config)
        all_findings.extend(result["findings"])
        per_checker[config.failure_mode] = {
            "n_findings": len(result["findings"]),
            "evidence_gathered": result["evidence_gathered"],
        }
        if result["error"]:
            errors.append(result["error"])

    return {
        "findings": all_findings,
        "per_checker": per_checker,
        "error": "; ".join(errors) if errors else None,
    }
