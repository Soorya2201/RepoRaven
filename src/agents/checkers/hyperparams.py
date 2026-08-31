"""hyperparam_drift checker: paper claims one value, code default is another."""

from src.agents.checkers.base import CheckerConfig
from src.agents.checkers.evidence import gather_argparse_evidence

INSTRUCTIONS = """The claim states a specific hyperparameter value (learning rate, batch size, epochs, weight decay, optimizer, etc.). The evidence lists every argparse default actually found in the repo, as `flag default=value`. Check whether any evidence entry's flag corresponds to what the claim describes, and whether its default value matches the claim's stated value. Numeric mismatches of any size count -- a 2x difference is still drift, not just 10x+."""


def _gather(paper_id: str, claim: dict) -> list:
    return gather_argparse_evidence(paper_id)


CONFIG = CheckerConfig(
    failure_mode="hyperparam_drift",
    claim_types=["hyperparameter"],
    mode_instructions=INSTRUCTIONS,
    gather_fn=_gather,
)
