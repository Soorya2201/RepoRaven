"""single_seed_reporting checker: paper claims N-seed/run averaging, code has
no mechanism to actually produce and combine N runs."""

from src.agents.checkers.base import CheckerConfig
from src.agents.checkers.evidence import gather_grep_evidence

INSTRUCTIONS = """The claim states results are averaged/reported over multiple seeds or runs (e.g. "averaged across three random seeds", "results on ten runs"). The evidence is every line anywhere in the repo (including shell scripts, not just .py files) matching "seed". Look for: a loop over multiple seed values (a real multi-seed mechanism -- NOT a bug), vs. a single hardcoded seed with no loop anywhere (a real bug). If a loop exists, check whether the NUMBER of seeds in the loop matches the claim's stated count -- a loop over fewer seeds than claimed is still a real, if milder, mismatch."""


def _gather(paper_id: str, claim: dict) -> list:
    return gather_grep_evidence(paper_id, "seed", file_glob="*")


CONFIG = CheckerConfig(
    failure_mode="single_seed_reporting",
    claim_types=["seed_protocol"],
    mode_instructions=INSTRUCTIONS,
    gather_fn=_gather,
)
