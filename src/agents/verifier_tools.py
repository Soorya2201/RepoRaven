"""Phase 3 -- tool-using verifier. Same claim-grounding contract as Phase 2's
verifier (src/agents/verifier.py), but instead of reading a fixed dump of
paper.py file heads, it gets the repo's file listing + README and can call
real tools (grep across the WHOLE repo, AST search for exact function calls/
assignments, structured argparse/config parsing) to gather evidence before
deciding.

Uses NIM_MODEL_TOOLCALLING (openai/gpt-oss-120b) -- NIM_MODEL_PRIMARY
(nemotron-3-super) was tested and confirmed to return HTTP 400/500 on both
guided-JSON and tool-calling on this endpoint; gpt-oss-120b was verified with
a real multi-turn tool-call -> result -> answer loop before adopting it here.
"""

import json
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.ingest import load_repo
from src.llm import get_llm, invoke_with_retry
from src.parsing import parse_json_list_with_repair
from src.tools.ast_tool import FindAssignments, FindFunctionCalls, run_find_assignments, run_find_function_calls
from src.tools.config_tool import GetArgparseDefaults, ReadConfigFile, run_get_argparse_defaults, run_read_config_file
from src.tools.grep_tool import GrepRepo, ReadFile, run_grep_tool, run_read_file

MAX_TOOL_ITERATIONS = 10
MAX_TOOL_RESULT_CHARS = 8000  # a single grep/config dump shouldn't blow the whole context budget

TOOLS = [GrepRepo, ReadFile, FindFunctionCalls, FindAssignments, GetArgparseDefaults, ReadConfigFile]

VERIFIER_TOOLS_PROMPT = """You are verifying whether a repository's code contradicts specific claims extracted from its paper. You will see a list of claims (each with an exact quote from the paper) and a listing of every file in the repo plus its README. You do NOT automatically see full file contents -- use the tools available to you to actually inspect the code before deciding anything.

How to check each claim type (use tools to gather real evidence for each):
- hyperparameter: use GetArgparseDefaults or ReadConfigFile to get the EXACT default value, then compare to the claim. Don't guess from a filename.
- seed_protocol: use GrepRepo with pattern like "seed" across ALL files (glob "*", not just "*.py") to check whether ANY mechanism exists anywhere in the repo (a CLI flag, a shell script loop, an aggregation step) to run/combine multiple seeds. Check .sh files too -- a seed loop can live in a shell wrapper script, not just the Python entry point.
- metric: use FindFunctionCalls to get the EXACT arguments of the metric function call (e.g. f1_score) and compare to the claim.
- data_split_protocol: use GrepRepo/FindFunctionCalls to trace where the claimed split happens and whether any shared computation (fit_transform, normalization stats) happens on combined data BEFORE that split -> data_leakage_global_preprocessing. Also check whether the claimed test portion is used anywhere besides final evaluation (e.g. inside a training/selection loop) -> test_set_contamination.
- eval_protocol: use GrepRepo/FindFunctionCalls to check what data the actual evaluation call uses -> eval_on_train if it's actually train data.

Before concluding a repo LACKS something (e.g. "no seed loop exists"), you MUST have actually searched for it with GrepRepo across ALL files (glob "*"), not just the .py files you might have seen referenced. Do not conclude absence from silence -- confirm it with a tool call. Use ReadFile when you need to see a whole function/file's logic, not just grep matches.

IMPORTANT: only use the actual tool-calling mechanism to call a tool -- never write out something like "Tool call: GrepRepo(...)" or a bare JSON object of arguments as your text response. That is not a real tool call and will not be executed. If you are not making a real tool call via the mechanism provided, your response must be the final JSON findings list (starting with `[`) and nothing else.

Strict grounding rule: only report a finding if (1) a tool call actually returned evidence supporting it, AND (2) `paper_evidence` is the EXACT quote text from one of the claims given to you -- copy it verbatim. If you cannot verify a claim with the tools available, do not report anything for it.

When you are done gathering evidence (you may call tools multiple times across several turns), respond with ONLY a JSON list of findings and no further tool calls. Each finding: `failure_mode` (one of: data_leakage_global_preprocessing, hyperparam_drift, single_seed_reporting, metric_misimplementation, test_set_contamination, eval_on_train, other), `severity` (high/medium/low), `paper_evidence` (exact quote from the matching claim), `code_evidence` (file:line), `explanation` (1-2 sentences, cite what the tool call actually showed), `confidence` (0.0-1.0). Return `[]` if every claim is consistent with what you found. Return only the JSON, no prose, once you're done investigating."""


def _dispatch_tool(paper_id: str, name: str, args: dict) -> str:
    try:
        if name == "GrepRepo":
            result = run_grep_tool(paper_id, args.get("pattern", ""), args.get("file_glob", "*"))
        elif name == "ReadFile":
            result = run_read_file(paper_id, args.get("file_path", ""), args.get("line_start", 1), args.get("line_end", 400))
        elif name == "FindFunctionCalls":
            result = run_find_function_calls(paper_id, args.get("function_name", ""))
        elif name == "FindAssignments":
            result = run_find_assignments(paper_id, args.get("variable_name", ""))
        elif name == "GetArgparseDefaults":
            result = run_get_argparse_defaults(paper_id)
        elif name == "ReadConfigFile":
            result = run_read_config_file(paper_id, args.get("file_path", ""))
        else:
            return f"ERROR: unknown tool {name!r}"
    except Exception as e:  # noqa: BLE001 -- a tool crashing shouldn't crash the whole verifier
        return f"ERROR: tool {name!r} raised {type(e).__name__}: {e}"

    if len(result) > MAX_TOOL_RESULT_CHARS:
        result = result[:MAX_TOOL_RESULT_CHARS] + "\n\n[... tool result truncated ...]"
    return result


def _claim_quotes(claims: list) -> set:
    return {_normalize(c.get("quote", "")) for c in claims if isinstance(c, dict) and c.get("quote")}


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def run_verifier_tools(paper_id: str, claims: list) -> dict:
    """Returns {'findings': list, 'raw_response': str, 'error': str|None,
    'dropped_ungrounded': int, 'tool_calls_made': list} -- tool_calls_made is
    kept for Phase 6 trajectory logging (every tool call + its result)."""
    try:
        repo = load_repo(paper_id)
    except FileNotFoundError as e:
        return {"findings": [], "raw_response": "", "error": f"Ingest failed: {e}", "dropped_ungrounded": 0, "tool_calls_made": []}

    if not claims:
        return {"findings": [], "raw_response": "", "error": None, "dropped_ungrounded": 0, "tool_calls_made": []}

    file_listing = "\n".join(str(f.relative_to(repo["root"])) for f in repo["files"])
    intro = (
        f"=== EXTRACTED CLAIMS ===\n{json.dumps(claims, indent=2)}\n\n"
        f"=== REPO FILE LISTING ({len(repo['files'])} files) ===\n{file_listing}\n\n"
        f"=== README ===\n{repo['readme']}"
    )

    toolcalling_model = os.environ["NIM_MODEL_TOOLCALLING"]
    llm = get_llm(model=toolcalling_model, temperature=0.0, max_tokens=8000)
    tooled_llm = llm.bind_tools(TOOLS)

    messages = [SystemMessage(content=VERIFIER_TOOLS_PROMPT), HumanMessage(content=intro)]
    tool_calls_made = []
    final_text = ""

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            ai_msg = invoke_with_retry(tooled_llm, messages)
        except Exception as e:  # noqa: BLE001
            root = e
            last_attempt = getattr(e, "last_attempt", None)
            if last_attempt is not None:
                try:
                    root = last_attempt.exception() or e
                except Exception:  # noqa: BLE001
                    root = e
            return {
                "findings": [], "raw_response": final_text,
                "error": f"Verifier LLM call failed on tool-loop iteration {iteration}: {type(root).__name__}: {root}",
                "dropped_ungrounded": 0, "tool_calls_made": tool_calls_made,
            }

        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            content = (ai_msg.content or "").strip()
            stripped = content
            for fence in ("```json", "```"):
                if stripped.startswith(fence):
                    stripped = stripped[len(fence):].lstrip()
            if stripped.startswith("["):
                final_text = content
                break
            # Not empty, but also not a JSON list -- observed real patterns:
            # (a) truly empty content (a glitch, not deterministic even at
            #     temperature=0 on this backend), (b) the model describing an
            #     imagined tool call as text instead of using the real
            #     tool-calling mechanism (e.g. 'Tool call: OpenFile(...)' or a
            #     bare JSON object of tool-call-shaped arguments). Neither is
            #     a real final answer -- don't accept it as "done".
            nudge = (
                "Your last response was empty." if not content else
                f"Your last response wasn't a JSON findings list (it looked like: {content[:150]!r}). "
                "If you meant to call a tool, use the actual tool-calling mechanism, not text describing one."
            )
            messages.append(HumanMessage(content=f"{nudge} Either call a tool properly to keep investigating, or return your final JSON findings list (starting with '[') now."))
            continue

        for tc in ai_msg.tool_calls:
            tool_result = _dispatch_tool(paper_id, tc["name"], tc["args"])
            tool_calls_made.append({"tool": tc["name"], "args": tc["args"], "result": tool_result})
            messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))
    else:
        # Hit MAX_TOOL_ITERATIONS without a clean final answer. Rather than
        # replaying the whole (possibly confusing, tool-heavy) conversation
        # once more, start a FRESH, compact prompt: claims + a digest of
        # every tool call already made + its result, and ask directly for
        # the JSON. A shorter, cleaner context is more likely to get a real
        # answer than tacking one more message onto an already-long thread.
        digest = "\n\n".join(
            f"Tool call: {tc['tool']}({tc['args']})\nResult: {tc['result'][:1500]}"
            for tc in tool_calls_made
        ) or "(no tool calls were made)"
        fallback_prompt = (
            f"{VERIFIER_TOOLS_PROMPT}\n\n=== EXTRACTED CLAIMS ===\n{json.dumps(claims, indent=2)}\n\n"
            f"=== EVIDENCE ALREADY GATHERED ===\n{digest}\n\n"
            "Based only on the evidence above, return your final JSON findings list now. No more tool calls -- just the JSON."
        )
        try:
            final_msg = invoke_with_retry(llm, [HumanMessage(content=fallback_prompt)])
            final_text = final_msg.content or ""
        except Exception:  # noqa: BLE001
            final_text = ""

    if not final_text.strip():
        return {
            "findings": [], "raw_response": "", "error": "Tool loop ended with no final text response.",
            "dropped_ungrounded": 0, "tool_calls_made": tool_calls_made,
        }

    findings, err = parse_json_list_with_repair(final_text, repair_model=os.environ["NIM_MODEL_FAST"])

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

    return {
        "findings": grounded, "raw_response": final_text, "error": err,
        "dropped_ungrounded": dropped, "tool_calls_made": tool_calls_made,
    }
