"""test_set_contamination checker: test data used somewhere besides final
evaluation (e.g. inside a training or model-selection loop)."""

from src.agents.checkers.base import CheckerConfig
from src.agents.checkers.evidence import gather_grep_evidence

INSTRUCTIONS = """The claim describes the data split protocol -- typically stating the test set is used only once, for final reporting (e.g. "the test set is touched exactly once"). The evidence lists every line anywhere in the repo referencing test data (test_loader, X_test, y_test, test_set, etc.). Check whether any of these appear INSIDE a loop that also touches model selection, hyperparameter tuning, or training (e.g. a loop over candidate configs that scores each on the test set to decide which to keep) -- that contradicts a "touched once" claim, even if the final reported number is also computed on the test set separately."""


def _gather(paper_id: str, claim: dict) -> list:
    return gather_grep_evidence(paper_id, r"test_loader|test_set|X_test|y_test|testloader", file_glob="*.py")


CONFIG = CheckerConfig(
    failure_mode="test_set_contamination",
    claim_types=["data_split_protocol"],
    mode_instructions=INSTRUCTIONS,
    gather_fn=_gather,
)
