"""Shared engine for Phase 4's specialized text-based checkers.

Design, informed directly by Phase 3's regression: no open-ended ReAct loop.
Each checker (1) deterministically gathers evidence for its one failure mode
(we call the tool in code; the LLM never decides whether/when to investigate),
then (2) makes exactly ONE decision call per relevant claim, forced to ground
its finding in BOTH a real extracted claim quote (Phase 2's mechanism) AND a
real evidence ID from what was actually gathered (new: deterministic
anchoring -- a finding citing a file:line not in the evidence list is
rejected, the same way an ungrounded paper_evidence quote already was).
"""

import json
import os
from dataclasses import dataclass

from src.agents.checkers.evidence import Evidence, evidence_by_id, render_evidence_block
from src.llm import get_llm, invoke_with_retry
from src.parsing import parse_json_list_with_repair

DECISION_PROMPT_TEMPLATE = """You are a specialized checker for exactly ONE reproducibility failure mode: {failure_mode}.

{mode_instructions}

You will see ONE claim extracted from the paper (with its exact quote) and a list of real evidence gathered from the repository, each with an ID like [E1]. You did not choose what evidence to gather -- it was collected deterministically for this claim type. Decide: does this evidence CONTRADICT the claim?

Strict rules:
- If you report a finding, `code_evidence` MUST be exactly one evidence ID from the list below (e.g. "E3"), not a file:line you generate yourself. Citing anything not in the list will be discarded.
- `paper_evidence` MUST be the exact claim quote given below, copied verbatim.
- If the evidence is consistent with the claim, or there isn't enough evidence to tell, return `[]`. Do not guess.

=== CLAIM ===
Quote: "{claim_quote}"
Type: {claim_type}

=== EVIDENCE (cite by ID only) ===
{evidence_block}

Return a JSON list with 0 or 1 findings. Each finding: `failure_mode` (must be "{failure_mode}"), `severity` (high/medium/low), `paper_evidence` (the exact claim quote above), `code_evidence` (an evidence ID from the list, e.g. "E2"), `explanation` (1-2 sentences, referencing what the cited evidence actually shows), `confidence` (0.0-1.0). Return only the JSON, no prose."""


@dataclass
class CheckerConfig:
    failure_mode: str
    claim_types: list[str]
    mode_instructions: str
    gather_fn: callable  # (paper_id, claim) -> list[Evidence]


MAX_DECISION_ATTEMPTS = 2  # the model occasionally emits malformed JSON on a
# single call (observed, nondeterministic even at temperature=0 -- the same
# pattern seen and fixed the same way in src/agents/extractor.py). A plain
# retry with a fresh call succeeded every time this was checked.


def decide_with_retry(llm, prompt: str, repair_model: str) -> tuple[list, str | None]:
    """Returns (findings, error). Retries the whole call once on total parse
    failure before giving up."""
    findings, err = [], None
    for attempt in range(MAX_DECISION_ATTEMPTS):
        try:
            response = invoke_with_retry(llm, [{"role": "user", "content": prompt}])
            raw = response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:  # noqa: BLE001
            # An API-level failure (e.g. a transient overload that outlasted
            # invoke_with_retry's own internal retries) should still be
            # retried by THIS loop, not just parse failures -- a bug in an
            # earlier version of this function returned immediately here,
            # skipping the retry entirely. Caught by testing against a real
            # paper, not by inspection.
            err = f"LLM call failed: {type(e).__name__}: {e}"
            continue

        findings, err = parse_json_list_with_repair(raw, repair_model=repair_model)
        if findings or err is None:
            return findings, err
    if err:
        err = f"{err} (after {MAX_DECISION_ATTEMPTS} attempts)"
    return findings, err


def run_text_checker(paper_id: str, claims: list[dict], config: CheckerConfig) -> dict:
    """Returns {'findings': list, 'evidence_gathered': int, 'error': str|None}."""
    relevant_claims = [c for c in claims if isinstance(c, dict) and c.get("claim_type") in config.claim_types]
    if not relevant_claims:
        return {"findings": [], "evidence_gathered": 0, "error": None}

    primary_model = os.environ["NIM_MODEL_PRIMARY"]
    llm = get_llm(model=primary_model, temperature=0.0, max_tokens=2000)

    all_findings = []
    total_evidence = 0
    errors = []

    for claim in relevant_claims:
        evidence = config.gather_fn(paper_id, claim)
        total_evidence += len(evidence)
        if not evidence:
            continue  # no evidence gathered -- nothing to check this claim against, correctly silent

        prompt = DECISION_PROMPT_TEMPLATE.format(
            failure_mode=config.failure_mode,
            mode_instructions=config.mode_instructions,
            claim_quote=claim.get("quote", ""),
            claim_type=claim.get("claim_type", ""),
            evidence_block=render_evidence_block(evidence),
        )

        findings, parse_err = decide_with_retry(llm, prompt, os.environ["NIM_MODEL_FAST"])
        if parse_err and not findings:
            errors.append(f"{config.failure_mode}: {parse_err}")
            continue

        # Deterministic anchoring enforcement: code_evidence must be a real
        # evidence ID we actually gathered. Resolve it to the real file:line
        # (never trust the model to have copied that part correctly either).
        for f in findings:
            if not isinstance(f, dict):
                continue
            eid = str(f.get("code_evidence", "")).strip()
            ev = evidence_by_id(evidence, eid)
            if ev is None:
                errors.append(f"{config.failure_mode}: dropped finding citing unknown evidence ID {eid!r}")
                continue
            f["code_evidence"] = f"{ev.file}:{ev.line}"
            f["failure_mode"] = config.failure_mode  # enforce, don't trust the model's echo
            all_findings.append(f)

    return {
        "findings": all_findings,
        "evidence_gathered": total_evidence,
        "error": "; ".join(errors) if errors else None,
    }
