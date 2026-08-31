"""Python AST search via tree-sitter: find function calls (with their exact
arguments), imports, and assignments precisely, instead of pattern-matching
raw text. Built for queries like "find every call to f1_score and its
arguments" -- catches e.g. a missing `average=` kwarg exactly, rather than
relying on the model to notice it in a text dump.
"""

from pydantic import BaseModel, Field
from tree_sitter import Language, Parser
import tree_sitter_python

from src.ingest import load_repo

_LANGUAGE = Language(tree_sitter_python.language())
_PARSER = Parser(_LANGUAGE)

MAX_RESULTS = 30


class FindFunctionCalls(BaseModel):
    """Find every call to a specific function/method name across all .py files
    in the repo, with its exact arguments. Use this for precise checks like
    "does every call to f1_score() pass average=..." instead of eyeballing text."""

    function_name: str = Field(description="The function or method name to search for, e.g. 'f1_score' or 'fit_transform'.")


class FindAssignments(BaseModel):
    """Find every assignment to a specific variable name across all .py files
    in the repo. Use this to trace a value (e.g. 'dataset', 'train_loader')
    across a file to see what it's assigned from."""

    variable_name: str = Field(description="The variable name to search for assignments to, e.g. 'dataset' or 'OPTIMIZER_FN'.")


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def run_find_function_calls(paper_id: str, function_name: str) -> str:
    try:
        repo = load_repo(paper_id)
    except FileNotFoundError as e:
        return f"ERROR: {e}"

    results = []
    for f in repo["python_files"]:
        try:
            source = f.read_bytes()
        except OSError:
            continue
        try:
            tree = _PARSER.parse(source)
        except Exception:  # noqa: BLE001 -- skip unparseable files, don't crash the tool
            continue
        rel = f.relative_to(repo["root"])
        for node in _walk(tree.root_node):
            if node.type != "call":
                continue
            func_node = node.child_by_field_name("function")
            if func_node is None:
                continue
            func_text = _node_text(func_node, source)
            # match bare name or attribute access ending in this name (e.g. sklearn.metrics.f1_score)
            if func_text == function_name or func_text.endswith(f".{function_name}"):
                line = node.start_point[0] + 1
                call_text = _node_text(node, source)
                results.append(f"{rel}:{line}: {call_text}")
                if len(results) >= MAX_RESULTS:
                    break
        if len(results) >= MAX_RESULTS:
            break

    if not results:
        return f"No calls to '{function_name}(...)' found in any .py file in the repo."
    return f"Found {len(results)} call(s) to '{function_name}':\n\n" + "\n".join(results)


def run_find_assignments(paper_id: str, variable_name: str) -> str:
    try:
        repo = load_repo(paper_id)
    except FileNotFoundError as e:
        return f"ERROR: {e}"

    results = []
    for f in repo["python_files"]:
        try:
            source = f.read_bytes()
        except OSError:
            continue
        try:
            tree = _PARSER.parse(source)
        except Exception:  # noqa: BLE001
            continue
        rel = f.relative_to(repo["root"])
        for node in _walk(tree.root_node):
            if node.type != "assignment":
                continue
            left = node.child_by_field_name("left")
            if left is None:
                continue
            left_text = _node_text(left, source)
            if left_text == variable_name:
                line = node.start_point[0] + 1
                assign_text = _node_text(node, source)
                results.append(f"{rel}:{line}: {assign_text}")
                if len(results) >= MAX_RESULTS:
                    break
        if len(results) >= MAX_RESULTS:
            break

    if not results:
        return f"No assignments to '{variable_name}' found in any .py file in the repo."
    return f"Found {len(results)} assignment(s) to '{variable_name}':\n\n" + "\n".join(results)
