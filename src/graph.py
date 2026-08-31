"""LangGraph orchestration -- grows each phase. Phase 2: extractor -> verifier."""

import time

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from src.agents.extractor import run_extractor
from src.agents.verifier import run_verifier
from src.agents.verifier_tools import run_verifier_tools


class Iter1State(TypedDict):
    paper_id: str
    claims: list
    extractor_raw: str
    extractor_error: str | None
    findings: list
    verifier_raw: str
    verifier_error: str | None
    dropped_ungrounded: int


def _node_extract(state: Iter1State) -> Iter1State:
    result = run_extractor(state["paper_id"])
    return {
        **state,
        "claims": result["claims"],
        "extractor_raw": result["raw_response"],
        "extractor_error": result["error"],
    }


def _node_verify(state: Iter1State) -> Iter1State:
    result = run_verifier(state["paper_id"], state["claims"])
    return {
        **state,
        "findings": result["findings"],
        "verifier_raw": result["raw_response"],
        "verifier_error": result["error"],
        "dropped_ungrounded": result["dropped_ungrounded"],
    }


def _build_iter1_graph():
    graph = StateGraph(Iter1State)
    graph.add_node("extract", _node_extract)
    graph.add_node("verify", _node_verify)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


_ITER1_GRAPH = _build_iter1_graph()


def run_iter1(paper_id: str) -> dict:
    """Returns {'findings': list, 'raw_response': str, 'error': str|None, 'wall_clock_seconds': float, ...}.
    Matches the same contract as src.baseline.run_baseline for src.eval compatibility,
    plus extra fields (claims, dropped_ungrounded) useful for Phase 6 trajectory logging."""
    start = time.time()
    initial_state: Iter1State = {
        "paper_id": paper_id, "claims": [], "extractor_raw": "", "extractor_error": None,
        "findings": [], "verifier_raw": "", "verifier_error": None, "dropped_ungrounded": 0,
    }
    result = _ITER1_GRAPH.invoke(initial_state)

    # Combine extractor + verifier errors for eval.py's error-tracking column;
    # a verifier error takes priority since it's the one that determines findings.
    combined_error = result["verifier_error"] or result["extractor_error"]
    if result["dropped_ungrounded"] > 0:
        note = f"Dropped {result['dropped_ungrounded']} ungrounded finding(s) (paper_evidence didn't match any extracted quote)."
        combined_error = f"{combined_error}; {note}" if combined_error else note

    return {
        "findings": result["findings"],
        "raw_response": f"--- EXTRACTOR ---\n{result['extractor_raw']}\n\n--- VERIFIER ---\n{result['verifier_raw']}",
        "error": combined_error,
        "claims": result["claims"],
        "wall_clock_seconds": round(time.time() - start, 1),
    }


class Iter2State(TypedDict):
    paper_id: str
    claims: list
    extractor_raw: str
    extractor_error: str | None
    findings: list
    verifier_raw: str
    verifier_error: str | None
    dropped_ungrounded: int
    tool_calls_made: list


def _node_verify_tools(state: Iter2State) -> Iter2State:
    result = run_verifier_tools(state["paper_id"], state["claims"])
    return {
        **state,
        "findings": result["findings"],
        "verifier_raw": result["raw_response"],
        "verifier_error": result["error"],
        "dropped_ungrounded": result["dropped_ungrounded"],
        "tool_calls_made": result["tool_calls_made"],
    }


def _build_iter2_graph():
    graph = StateGraph(Iter2State)
    graph.add_node("extract", _node_extract)
    graph.add_node("verify_tools", _node_verify_tools)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "verify_tools")
    graph.add_edge("verify_tools", END)
    return graph.compile()


_ITER2_GRAPH = _build_iter2_graph()


def run_iter2(paper_id: str) -> dict:
    """Phase 3: same extractor as Phase 2, but a tool-using ReAct verifier
    (real native tool-calling on NIM_MODEL_TOOLCALLING) instead of a fixed
    context dump. Same return contract as run_baseline/run_iter1."""
    start = time.time()
    initial_state: Iter2State = {
        "paper_id": paper_id, "claims": [], "extractor_raw": "", "extractor_error": None,
        "findings": [], "verifier_raw": "", "verifier_error": None, "dropped_ungrounded": 0,
        "tool_calls_made": [],
    }
    result = _ITER2_GRAPH.invoke(initial_state)

    combined_error = result["verifier_error"] or result["extractor_error"]
    if result["dropped_ungrounded"] > 0:
        note = f"Dropped {result['dropped_ungrounded']} ungrounded finding(s) (paper_evidence didn't match any extracted quote)."
        combined_error = f"{combined_error}; {note}" if combined_error else note

    return {
        "findings": result["findings"],
        "raw_response": f"--- EXTRACTOR ---\n{result['extractor_raw']}\n\n--- VERIFIER (tool-using) ---\n{result['verifier_raw']}",
        "error": combined_error,
        "claims": result["claims"],
        "tool_calls_made": result["tool_calls_made"],
        "wall_clock_seconds": round(time.time() - start, 1),
    }
