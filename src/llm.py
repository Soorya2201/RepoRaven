"""NVIDIA NIM client wrapper. All phases call the LLM through this module — never directly."""

import os
import threading

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Both NIM_MODEL_PRIMARY (nvidia/nemotron-3-super-120b-a12b) and NIM_MODEL_FAST
# (nvidia/nemotron-3-nano-30b-a3b) are reasoning models: they spend part of
# max_tokens on a hidden "reasoning_content" chain-of-thought before the final
# answer lands in "content". Budget max_tokens accordingly (defaults padded vs.
# the instructions.md baseline of 4096) or replies get cut off mid-thought with
# an empty content field.
DEFAULT_MAX_TOKENS = 4096

# Running total of token usage across the process, for credit estimation in eval.py.
_usage_lock = threading.Lock()
_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


# ChatNVIDIA's default HTTP timeout is 60s (langchain_nvidia_ai_endpoints
# _common.py). Large-context calls to a reasoning model (long prompt + long
# hidden reasoning_content + final answer) routinely exceed that -- observed
# empirically as ReadTimeout failures on ~half the Phase 1 baseline eval set
# before this was raised. See phase_1_baseline.md "Failure analysis".
DEFAULT_TIMEOUT_SECONDS = 240


def get_llm(model: str | None = None, temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS,
            timeout: float = DEFAULT_TIMEOUT_SECONDS):
    """Return a ChatNVIDIA instance configured for our NIM endpoint."""
    return ChatNVIDIA(
        model=model or os.environ["NIM_MODEL_PRIMARY"],
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url=os.environ["NIM_BASE_URL"],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        timeout=timeout,
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
def invoke_with_retry(llm, messages):
    response = llm.invoke(messages)
    usage = (response.response_metadata or {}).get("token_usage") or getattr(response, "usage_metadata", None) or {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
    with _usage_lock:
        _usage_totals["prompt_tokens"] += prompt_tokens
        _usage_totals["completion_tokens"] += completion_tokens
        _usage_totals["total_tokens"] += total_tokens
        _usage_totals["calls"] += 1
    return response


def get_usage_totals() -> dict:
    with _usage_lock:
        return dict(_usage_totals)


def reset_usage_totals() -> None:
    with _usage_lock:
        for k in _usage_totals:
            _usage_totals[k] = 0


def extract_text(response) -> str:
    """Return only the final-answer text, stripping any reasoning/thinking content."""
    content = response.content
    if isinstance(content, str):
        return content
    # Some providers return a list of content blocks; keep only text blocks.
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)
