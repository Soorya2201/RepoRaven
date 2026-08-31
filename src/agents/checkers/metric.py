"""metric_misimplementation checker: paper says e.g. F1-macro, code computes
something else (wrong averaging, wrong metric function entirely)."""

from src.agents.checkers.base import CheckerConfig
from src.agents.checkers.evidence import gather_function_call_evidence

METRIC_FUNCTIONS = [
    "f1_score", "precision_score", "recall_score", "accuracy_score",
    "roc_auc_score", "average_precision_score", "fbeta_score",
    "precision_recall_fscore_support", "classification_report",
]

INSTRUCTIONS = """The claim states a specific evaluation metric and how it should be computed (e.g. "macro-averaged F1", "top-1 accuracy", "AUROC"). The evidence lists every call to a common sklearn metric function found anywhere in the repo, with its exact arguments. Check whether the metric function used, and its `average=`/parameters, actually match what the claim describes. A metric function call with NO `average=` argument defaults to 'binary' in sklearn -- flag this if the claim specifies macro/micro/weighted averaging."""


def _gather(paper_id: str, claim: dict) -> list:
    return gather_function_call_evidence(paper_id, METRIC_FUNCTIONS)


CONFIG = CheckerConfig(
    failure_mode="metric_misimplementation",
    claim_types=["metric"],
    mode_instructions=INSTRUCTIONS,
    gather_fn=_gather,
)
