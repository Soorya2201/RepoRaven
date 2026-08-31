"""Phase 1 — deliberately simple baseline agent.

Single LangGraph node, one prompt, paper text + repo README + head of every
.py file dumped in as context, one LLM call, ask for JSON findings. This is
the floor every later phase has to beat (instructions.md §5).
"""

import json
import re
import time

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from src.context import format_paper_text, format_repo_code_context, truncate
from src.llm import get_llm, invoke_with_retry, extract_text
# First attempt used 6000 (padded above instructions.md's 4096 baseline);
# empirically that was still not enough for reasoning_content overhead on
# non-trivial repos -- the model would hit finish_reason='length' mid-thought
# and never emit valid JSON (6/9 papers failed on this in the first real run,
# see phase_1_baseline.md "Failure analysis"). 16000 was enough for every
# paper in this eval set to finish cleanly (finish_reason='stop').
RESPONSE_TOKEN_BUDGET = 16000

BASELINE_PROMPT = """You are auditing an ML research paper and its code repository for reproducibility issues. Read the paper and the code below. Return a JSON list of findings. Each finding must have: `failure_mode` (one of: data_leakage_global_preprocessing, hyperparam_drift, single_seed_reporting, metric_misimplementation, test_set_contamination, eval_on_train, other), `severity` (high/medium/low), `paper_evidence` (short quote or section reference), `code_evidence` (file:line or file), `explanation` (1-2 sentences). Return `[]` if you find nothing. Return only the JSON, no prose."""


class BaselineState(TypedDict):
    paper_id: str
    context: str
    raw_response: str
    findings: list
    error: str | None


def _build_context(paper_id: str) -> str:
    paper_block, _ = format_paper_text(paper_id)
    repo_block, _ = format_repo_code_context(paper_id)
    return truncate(f"{paper_block}\n{repo_block}")


def _parse_findings(raw_response: str) -> tuple[list, str | None]:
    text = raw_response.strip()
    # Strip markdown code fences if the model wrapped the JSON despite instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed, None
        return [], f"Expected a JSON list, got {type(parsed).__name__}"
    except json.JSONDecodeError:
        pass
    # Fallback: find the outermost [ ... ] substring.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list):
                return parsed, "Recovered JSON list from a non-clean response (had surrounding prose)."
        except json.JSONDecodeError:
            pass
    return [], f"Could not parse JSON from model response (first 300 chars): {text[:300]!r}"


def _node_run_baseline(state: BaselineState) -> BaselineState:
    llm = get_llm(temperature=0.0, max_tokens=RESPONSE_TOKEN_BUDGET)
    messages = [
        {"role": "system", "content": BASELINE_PROMPT},
        {"role": "user", "content": state["context"]},
    ]
    try:
        response = invoke_with_retry(llm, messages)
        raw = extract_text(response)
    except Exception as e:  # noqa: BLE001 -- log the real failure, never fabricate a result
        # tenacity's RetryError hides the actual underlying exception behind a
        # repr; unwrap it so the logged error is diagnostic, not just "failed".
        root = e
        last_attempt = getattr(e, "last_attempt", None)
        if last_attempt is not None:
            try:
                root = last_attempt.exception() or e
            except Exception:  # noqa: BLE001
                root = e
        return {**state, "raw_response": "", "findings": [], "error": f"LLM call failed: {type(root).__name__}: {root}"}

    findings, parse_error = _parse_findings(raw)
    return {**state, "raw_response": raw, "findings": findings, "error": parse_error}


def _build_graph():
    graph = StateGraph(BaselineState)
    graph.add_node("run_baseline", _node_run_baseline)
    graph.set_entry_point("run_baseline")
    graph.add_edge("run_baseline", END)
    return graph.compile()


_GRAPH = _build_graph()


def run_baseline(paper_id: str) -> dict:
    """Returns {'findings': list, 'raw_response': str, 'error': str|None, 'wall_clock_seconds': float}."""
    start = time.time()
    try:
        context = _build_context(paper_id)
    except FileNotFoundError as e:
        return {"findings": [], "raw_response": "", "error": f"Ingest failed: {e}", "wall_clock_seconds": 0.0}

    initial_state: BaselineState = {
        "paper_id": paper_id, "context": context, "raw_response": "", "findings": [], "error": None,
    }
    result = _GRAPH.invoke(initial_state)
    result["wall_clock_seconds"] = round(time.time() - start, 1)
    return result
