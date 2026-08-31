"""Shared JSON-list parsing with an LLM-repair fallback.

Phase 1 found a systematic failure mode: the model reliably drops the opening
quote on one specific field in ~44% of responses (a real, reproducible
formatting slip, not random noise -- see phase_1_baseline.md). Rather than
regex-patch that one exact bug (fragile, narrow), Phase 2 adds a general
repair pass: on any parse failure, ask a cheap fast model to fix the JSON
syntax without changing content. This is a robustness mechanism, not a
smarter prompt -- it targets "did we lose a correct answer to a formatting
slip," independent of what the slip actually was.
"""

import json
import re

from src.llm import get_llm, invoke_with_retry, extract_text

REPAIR_PROMPT = (
    "The text below was supposed to be valid JSON but has a syntax error "
    "(e.g. a missing quote, trailing comma, or unescaped character). "
    "Return ONLY the corrected, valid JSON with the exact same content -- "
    "do not add, remove, or reword any field values. No prose, no markdown "
    "fences, just the corrected JSON."
)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def _unwrap_dict(parsed: dict) -> list | None:
    """A model instructed to 'return a JSON list' sometimes wraps it in an
    object anyway, e.g. {"findings": [...]} or {"claims": [...]}. If the dict
    has exactly one key whose value is a list, that's almost certainly the
    intended answer -- unwrap it rather than treating it as a hard failure."""
    list_values = [v for v in parsed.values() if isinstance(v, list)]
    if len(list_values) == 1:
        return list_values[0]
    return None


def _try_parse_list(text: str) -> tuple[list | None, str | None]:
    text = _strip_fences(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed, None
        if isinstance(parsed, dict):
            unwrapped = _unwrap_dict(parsed)
            if unwrapped is not None:
                return unwrapped, None
        return None, f"Expected a JSON list, got {type(parsed).__name__}"
    except json.JSONDecodeError as e:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list):
                return parsed, None
        except json.JSONDecodeError:
            pass
    return None, f"Could not parse JSON list (first 300 chars): {text[:300]!r}"


def parse_json_list_with_repair(raw_text: str, repair_model: str | None = None) -> tuple[list, str | None]:
    """Returns (list, error). error is None on a clean first-try parse, a note
    on a repaired parse, or a message if even the repair attempt failed."""
    parsed, err = _try_parse_list(raw_text)
    if parsed is not None:
        return parsed, None

    try:
        repair_llm = get_llm(model=repair_model, temperature=0.0, max_tokens=len(raw_text) // 2 + 2000)
        response = invoke_with_retry(repair_llm, [
            {"role": "system", "content": REPAIR_PROMPT},
            {"role": "user", "content": raw_text},
        ])
        repaired_text = extract_text(response)
        parsed2, err2 = _try_parse_list(repaired_text)
        if parsed2 is not None:
            return parsed2, f"Recovered via repair pass (original parse error: {err})"
    except Exception as e:  # noqa: BLE001 -- repair is best-effort, never let it mask the real failure
        return [], f"Parse failed and repair pass also failed ({type(e).__name__}: {e}). Original error: {err}"

    return [], f"Parse failed even after repair pass. Original error: {err}"
