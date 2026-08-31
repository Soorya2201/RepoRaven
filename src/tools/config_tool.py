"""Parse YAML/JSON config files and argparse defaults directly as structured
data, instead of the model eyeballing `parser.add_argument(...)` lines in a
text dump. Serves hyperparam_drift checks with exact values rather than
approximate text-matching.
"""

import json

import yaml
from pydantic import BaseModel, Field
from tree_sitter import Language, Parser
import tree_sitter_python

from src.ingest import load_repo

_LANGUAGE = Language(tree_sitter_python.language())
_PARSER = Parser(_LANGUAGE)


class GetArgparseDefaults(BaseModel):
    """Extract every argparse.add_argument(...) call across all .py files in
    the repo as a structured flag -> default value mapping. Use this to get
    EXACT hyperparameter defaults instead of reading raw code text."""
    pass


class ReadConfigFile(BaseModel):
    """Read and parse a specific YAML or JSON config file from the repo,
    returning its contents as structured key -> value data."""

    file_path: str = Field(description="Path to the config file, relative to the repo root, e.g. 'options/model/cifar_birt.yaml'.")


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def run_get_argparse_defaults(paper_id: str) -> str:
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
            if node.type != "call":
                continue
            func_node = node.child_by_field_name("function")
            if func_node is None:
                continue
            func_text = _node_text(func_node, source)
            if not (func_text == "add_argument" or func_text.endswith(".add_argument")):
                continue
            args_node = node.child_by_field_name("arguments")
            if args_node is None:
                continue
            flag_names = []
            default_val = None
            for child in args_node.children:
                if child.type == "string":
                    text = _node_text(child, source).strip("'\"")
                    if text.startswith("-"):
                        flag_names.append(text)
                elif child.type == "keyword_argument":
                    kw_name_node = child.child_by_field_name("name")
                    kw_val_node = child.child_by_field_name("value")
                    if kw_name_node is not None and _node_text(kw_name_node, source) == "default" and kw_val_node is not None:
                        default_val = _node_text(kw_val_node, source)
            if flag_names:
                line = node.start_point[0] + 1
                results.append(f"{rel}:{line}: {'/'.join(flag_names)} default={default_val}")

    if not results:
        return "No argparse.add_argument(...) calls found in any .py file in the repo."
    return f"Found {len(results)} argparse flag(s):\n\n" + "\n".join(results)


def run_read_config_file(paper_id: str, file_path: str) -> str:
    try:
        repo = load_repo(paper_id)
    except FileNotFoundError as e:
        return f"ERROR: {e}"

    target = repo["root"] / file_path
    try:
        target = target.resolve()
        target.relative_to(repo["root"].resolve())  # guard against path traversal outside the repo
    except (OSError, ValueError):
        return f"ERROR: invalid or out-of-repo path {file_path!r}"

    if not target.exists():
        return f"ERROR: {file_path!r} does not exist in this repo."

    try:
        text = target.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return f"ERROR reading {file_path!r}: {e}"

    try:
        if target.suffix in (".yaml", ".yml"):
            parsed = yaml.safe_load(text)
        elif target.suffix == ".json":
            parsed = json.loads(text)
        else:
            return f"ERROR: {file_path!r} is not a .yaml/.yml/.json file (got {target.suffix!r})."
        return f"Parsed contents of {file_path}:\n\n{json.dumps(parsed, indent=2, default=str)}"
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        return f"ERROR parsing {file_path!r}: {e}. Raw contents:\n{text[:2000]}"
