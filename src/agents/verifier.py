"""Phase 2 — verifier. Given extracted claims + repo, verify each claim
against the code. Uses NIM_MODEL_PRIMARY.

Grounding rule: every finding must reference which extracted claim it's
checking, via the claim's exact quote. This is enforced by prompt instruction
(not a hard schema constraint -- tool-calling/guided-JSON both return
400/500 errors on this NIM endpoint for this model, tested and confirmed
unavailable) plus a post-hoc check in run_verifier that drops any finding
whose paper_evidence doesn't match a real extracted quote.
"""

import json
import os

from src.context import format_repo_code_context, truncate
from src.llm import get_llm, invoke_with_retry, extract_text
from src.parsing import parse_json_list_with_repair

VERIFIER_PROMPT = """You are verifying whether a repository's code contradicts specific claims extracted from its paper. You will see a list of claims (each with an exact quote from the paper) and the repo's README + code. For EACH claim, decide whether the code contradicts it.

How to check each claim type:
- hyperparameter: compare the claimed value to the actual default/config value in the code.
- seed_protocol: if the claim says results are averaged over N seeds/runs, check whether the code has ANY mechanism (a --seed CLI flag being varied, a wrapper script, an aggregation step) to actually produce and combine N runs. If the code only ever uses one hardcoded seed with no such mechanism visible anywhere shown, that contradicts the claim -> failure_mode "single_seed_reporting".
- metric: check the exact metric function call/parameters in the code (e.g. does it pass the averaging method the paper claims, like average='macro'?).
- data_split_protocol: check whether any shared/global computation (feature scaling, normalization statistics, fitting a preprocessor) happens on data BEFORE the claimed split point, using data from more than one of the claimed splits -> failure_mode "data_leakage_global_preprocessing". Separately, check whether the portion of data the claim describes as test/held-out is used anywhere OTHER than final evaluation (e.g. inside a training or model-selection loop) -> failure_mode "test_set_contamination".
- eval_protocol: check whether the code's final reported evaluation actually uses the data the claim says it uses (e.g. does it call the eval function with a test loader, or accidentally a train loader?) -> failure_mode "eval_on_train" if it doesn't.

Strict grounding rule: only report a finding if (1) you can point to specific, real code visible in what you were given, AND (2) `paper_evidence` in your finding is the EXACT quote text from one of the claims you were given below -- copy it verbatim, do not paraphrase it. If you cannot verify a claim against the code actually shown to you, do not report anything for it -- do not guess or assume code exists that you weren't shown.

Each finding: `failure_mode` (one of: data_leakage_global_preprocessing, hyperparam_drift, single_seed_reporting, metric_misimplementation, test_set_contamination, eval_on_train, other), `severity` (high/medium/low), `paper_evidence` (exact quote from the matching claim), `code_evidence` (file:line or file), `explanation` (1-2 sentences), `confidence` (0.0-1.0).

Return a JSON list. Return `[]` if every claim is consistent with the code shown. Return only the JSON, no prose."""


def _normalize(text: str) -> str:
    """Collapse whitespace and casefold so minor reformatting (extra spaces,
    case, the LaTeX-mangled unicode arXiv HTML sometimes produces) doesn't
    cause a genuinely-grounded quote to be rejected by the check below."""
    return " ".join(text.split()).casefold()


def _claim_quotes(claims: list) -> set:
    return {_normalize(c.get("quote", "")) for c in claims if isinstance(c, dict) and c.get("quote")}


def run_verifier(paper_id: str, claims: list) -> dict:
    """Returns {'findings': list, 'raw_response': str, 'error': str|None, 'dropped_ungrounded': int}."""
    try:
        repo_block, _ = format_repo_code_context(paper_id)
    except FileNotFoundError as e:
        return {"findings": [], "raw_response": "", "error": f"Ingest failed: {e}", "dropped_ungrounded": 0}

    if not claims:
        # No claims extracted -- either the paper made none, or extraction
        # failed. Either way there's nothing to ground a finding in, so we
        # correctly report zero findings rather than free-associating.
        return {"findings": [], "raw_response": "", "error": None, "dropped_ungrounded": 0}

    claims_block = "=== EXTRACTED CLAIMS ===\n" + json.dumps(claims, indent=2)
    context = truncate(f"{claims_block}\n\n{repo_block}")

    primary_model = os.environ["NIM_MODEL_PRIMARY"]
    llm = get_llm(model=primary_model, temperature=0.0, max_tokens=16000)  # see phase_1_baseline.md for why 16000

    try:
        response = invoke_with_retry(llm, [
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": context},
        ])
        raw = extract_text(response)
    except Exception as e:  # noqa: BLE001
        root = e
        last_attempt = getattr(e, "last_attempt", None)
        if last_attempt is not None:
            try:
                root = last_attempt.exception() or e
            except Exception:  # noqa: BLE001
                root = e
        return {"findings": [], "raw_response": "", "error": f"Verifier LLM call failed: {type(root).__name__}: {root}", "dropped_ungrounded": 0}

    findings, err = parse_json_list_with_repair(raw, repair_model=os.environ["NIM_MODEL_FAST"])

    # Post-hoc grounding check: drop any finding whose paper_evidence isn't a
    # real extracted quote (near-exact match, allowing for minor whitespace
    # differences). This is the hard enforcement of the anti-hallucination
    # rule the prompt asks for but can't guarantee on its own.
    valid_quotes = _claim_quotes(claims)
    grounded = []
    dropped = 0
    for f in findings:
        if not isinstance(f, dict):
            dropped += 1
            continue
        evidence = _normalize(f.get("paper_evidence", ""))
        if evidence and any(evidence in q or q in evidence for q in valid_quotes):
            grounded.append(f)
        else:
            dropped += 1

    return {"findings": grounded, "raw_response": raw, "error": err, "dropped_ungrounded": dropped}
