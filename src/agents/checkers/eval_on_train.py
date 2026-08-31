"""eval_on_train checker: the "test" evaluation is actually run on training
data, not held-out data."""

from src.agents.checkers.base import CheckerConfig
from src.agents.checkers.evidence import gather_function_call_evidence

EVAL_FUNCTIONS = ["evaluate", "test", "validate", "eval_model", "run_eval"]

INSTRUCTIONS = """The claim states what data the paper's final reported numbers come from (typically the held-out test set, after training completes). The evidence lists every call to a function named evaluate/test/validate/etc. anywhere in the repo, with its exact arguments. Check whether the LAST or FINAL such call in the file (the one that would produce the paper's reported number) is actually passed train-set variables (train_loader, X_train, y_train) instead of test-set ones -- that means the "final" evaluation is silently on training data."""


def _gather(paper_id: str, claim: dict) -> list:
    return gather_function_call_evidence(paper_id, EVAL_FUNCTIONS)


CONFIG = CheckerConfig(
    failure_mode="eval_on_train",
    claim_types=["eval_protocol"],
    mode_instructions=INSTRUCTIONS,
    gather_fn=_gather,
)
