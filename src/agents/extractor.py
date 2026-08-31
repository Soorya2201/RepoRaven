"""Phase 2 — claim extractor. Given paper text, output a list of specific,
checkable claims the paper makes about training setup. Uses NIM_MODEL_FAST.

This is the core anti-hallucination mechanism: the verifier is only allowed
to report a finding grounded in one of these extracted claims (each carrying
an exact quote from the paper). Phase 1's proto2proto failure -- claiming
"the paper reports results from a single random seed" when the paper never
mentions seeds at all -- is structurally impossible here, because there is no
extracted claim to hang that finding on.
"""

import os

from src.context import format_paper_text, truncate
from src.llm import get_llm, invoke_with_retry, extract_text
from src.parsing import parse_json_list_with_repair

EXTRACTOR_PROMPT = """You are extracting specific, checkable claims from an ML research paper about its experimental methodology. Read the paper text below and extract every claim that falls into one of these categories:

- hyperparameter: a specific numeric/named hyperparameter value (learning rate, batch size, epochs, weight decay, optimizer choice, etc.)
- seed_protocol: how many random seeds/runs were used, and whether results are averaged (mean/std) across them
- metric: the exact evaluation metric used and its averaging method (e.g. "F1-macro", "top-1 accuracy")
- data_split_protocol: how train/validation/test data is constructed (ratios, and critically, WHAT ORDER things happen in relative to the split -- e.g. "we normalize features before splitting" vs "we split, then normalize each portion separately")
- eval_protocol: what data the paper's final reported numbers come from (e.g. "all reported numbers are on the held-out test set")

Rules:
- Only extract claims ACTUALLY STATED in the paper text below. Do not infer, assume, or paraphrase-in a claim the paper doesn't make. If the paper never mentions seeds, do not extract a seed_protocol claim.
- Each claim must include an exact short quote (verbatim, copy-pasted from the text) as evidence that this claim was really made.
- Return a JSON list. Each item: `claim_type` (one of the 5 categories above), `quote` (exact verbatim quote), `description` (1 sentence paraphrase).
- Return `[]` if the paper makes no extractable claims in these categories. Return only the JSON, no prose."""


MAX_EXTRACTION_ATTEMPTS = 2  # the fast model occasionally "thinks in content"
# and produces prose instead of JSON on a given call (observed, nondeterministic
# even at temperature=0 -- a plain retry with a fresh call succeeded every time
# this was seen). Cheaper and more reliable than trying to prompt-engineer away
# a flaky-backend issue.


def _attempt_extraction(context: str, fast_model: str) -> tuple[list, str, str | None]:
    llm = get_llm(model=fast_model, temperature=0.0, max_tokens=8000)
    try:
        response = invoke_with_retry(llm, [
            {"role": "system", "content": EXTRACTOR_PROMPT},
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
        return [], "", f"Extractor LLM call failed: {type(root).__name__}: {root}"

    claims, err = parse_json_list_with_repair(raw, repair_model=fast_model)
    return claims, raw, err


def run_extractor(paper_id: str) -> dict:
    """Returns {'claims': list, 'raw_response': str, 'error': str|None}."""
    try:
        paper_block, _ = format_paper_text(paper_id)
    except FileNotFoundError as e:
        return {"claims": [], "raw_response": "", "error": f"Ingest failed: {e}"}

    context = truncate(paper_block)
    fast_model = os.environ["NIM_MODEL_FAST"]

    claims, raw, err = [], "", None
    for attempt in range(MAX_EXTRACTION_ATTEMPTS):
        claims, raw, err = _attempt_extraction(context, fast_model)
        # A non-empty claims list means we got something usable -- stop.
        # An empty list with no error is a legitimate "paper made no
        # extractable claims" result -- also stop, don't retry a real answer.
        if claims or err is None:
            break
        # Otherwise: empty claims AND an error (parse/call failure) -- retry.

    if err and not claims:
        err = f"{err} (after {attempt + 1} attempt(s))"
    return {"claims": claims, "raw_response": raw, "error": err}
