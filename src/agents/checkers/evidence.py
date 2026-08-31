"""Deterministic evidence gathering for Phase 4's specialized checkers.

The key design change from Phase 3: WE call the tools in code, not the LLM
deciding whether/when/how to. Every checker gets a pre-gathered, ID-tagged
list of evidence and is required to cite an evidence ID for any finding --
it cannot generate a file:line from memory. This eliminates the two failure
classes Phase 3 hit (tool-calling reliability, open-ended investigation
that never concludes) by construction: there is no LLM-driven tool loop
here at all, just one deterministic gather + one grounded decision.
"""

import re
from dataclasses import dataclass, field

from src.ingest import load_repo
from src.tools.ast_tool import _PARSER as _AST_PARSER, _node_text, _walk
from src.tools.config_tool import run_get_argparse_defaults

MAX_EVIDENCE_ITEMS = 25


@dataclass
class Evidence:
    id: str
    file: str
    line: int
    snippet: str
    kind: str  # "grep" | "function_call" | "argparse" | "figure"

    def to_prompt_line(self) -> str:
        return f"[{self.id}] {self.file}:{self.line} | {self.snippet}"


def _next_id(existing: list[Evidence]) -> str:
    return f"E{len(existing) + 1}"


def gather_grep_evidence(paper_id: str, pattern: str, file_glob: str = "*", context_lines: int = 1) -> list[Evidence]:
    try:
        repo = load_repo(paper_id)
    except FileNotFoundError:
        return []
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return []

    out: list[Evidence] = []
    for f in repo["files"]:
        if not f.match(file_glob):
            continue
        try:
            if f.stat().st_size > 2_000_000:
                continue
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel = str(f.relative_to(repo["root"]))
        for i, line in enumerate(lines):
            if regex.search(line):
                out.append(Evidence(id=_next_id(out), file=rel, line=i + 1, snippet=line.strip()[:200], kind="grep"))
                if len(out) >= MAX_EVIDENCE_ITEMS:
                    return out
    return out


def gather_function_call_evidence(paper_id: str, function_names: list[str]) -> list[Evidence]:
    try:
        repo = load_repo(paper_id)
    except FileNotFoundError:
        return []

    out: list[Evidence] = []
    for f in repo["python_files"]:
        try:
            source = f.read_bytes()
        except OSError:
            continue
        try:
            tree = _AST_PARSER.parse(source)
        except Exception:  # noqa: BLE001
            continue
        rel = str(f.relative_to(repo["root"]))
        for node in _walk(tree.root_node):
            if node.type != "call":
                continue
            func_node = node.child_by_field_name("function")
            if func_node is None:
                continue
            func_text = _node_text(func_node, source)
            if any(func_text == fn or func_text.endswith(f".{fn}") for fn in function_names):
                line = node.start_point[0] + 1
                call_text = _node_text(node, source)[:200]
                out.append(Evidence(id=_next_id(out), file=rel, line=line, snippet=call_text, kind="function_call"))
                if len(out) >= MAX_EVIDENCE_ITEMS:
                    return out
    return out


def gather_argparse_evidence(paper_id: str) -> list[Evidence]:
    raw = run_get_argparse_defaults(paper_id)
    out: list[Evidence] = []
    for line in raw.splitlines():
        m = re.match(r"^(.+?):(\d+):\s*(.+)$", line.strip())
        if m:
            file, lineno, rest = m.groups()
            out.append(Evidence(id=_next_id(out), file=file, line=int(lineno), snippet=rest[:200], kind="argparse"))
            if len(out) >= MAX_EVIDENCE_ITEMS:
                break
    return out


def render_evidence_block(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no evidence found -- nothing to cite)"
    return "\n".join(e.to_prompt_line() for e in evidence)


def evidence_by_id(evidence: list[Evidence], eid: str) -> Evidence | None:
    for e in evidence:
        if e.id == eid:
            return e
    return None
