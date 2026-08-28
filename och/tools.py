"""Workspace tools exposed to the model.

Each tool has an Ollama/OpenAI-style JSON schema (for the `tools` field of
/api/chat) and a Python implementation. Implementations always return a
string — errors are returned as text so the model can react to them.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MAX_OUTPUT_CHARS = 40_000
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".och"}


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]
    mutating: bool = False  # mutating tools may require user approval


def _clean_schema(params: dict[str, Any]) -> dict[str, Any]:
    return {k: {kk: vv for kk, vv in v.items() if kk != "_required"} for k, v in params.items()}


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    p = Path(path)
    if not p.is_file():
        return f"Error: file not found: {path}"
    try:
        lines = p.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"Error reading {path}: {exc}"
    offset = max(0, int(offset))
    limit = max(1, int(limit))
    window = lines[offset : offset + limit]
    if not window:
        return f"(file has {len(lines)} lines; offset {offset} is past the end)"
    numbered = "\n".join(f"{i + offset + 1:6}\t{line}" for i, line in enumerate(window))
    suffix = ""
    if offset + limit < len(lines):
        suffix = f"\n... [{len(lines) - offset - limit} more lines, use offset to read further]"
    return _truncate(numbered + suffix)


def write_file(path: str, content: str) -> str:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except OSError as exc:
        return f"Error writing {path}: {exc}"
    return f"Wrote {len(content)} characters to {path}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    p = Path(path)
    if not p.is_file():
        return f"Error: file not found: {path}"
    try:
        text = p.read_text()
    except OSError as exc:
        return f"Error reading {path}: {exc}"
    count = text.count(old_string)
    if count == 0:
        return "Error: old_string not found in file. Read the file and retry with an exact match."
    if count > 1:
        return f"Error: old_string occurs {count} times; provide a larger, unique snippet."
    try:
        p.write_text(text.replace(old_string, new_string, 1))
    except OSError as exc:
        return f"Error writing {path}: {exc}"
    return f"Edited {path} (1 replacement)"


def list_directory(path: str = ".") -> str:
    p = Path(path)
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    entries = []
    try:
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if child.name in SKIP_DIRS:
                continue
            if child.is_dir():
                entries.append(child.name + "/")
            else:
                entries.append(f"{child.name}  ({child.stat().st_size} bytes)")
    except OSError as exc:
        return f"Error listing {path}: {exc}"
    return _truncate("\n".join(entries) or "(empty directory)")


def glob_files(pattern: str, path: str = ".") -> str:
    root = Path(path)
    if not root.is_dir():
        return f"Error: not a directory: {path}"
    matches: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                matches.append(rel)
        if len(matches) > 500:
            break
    matches.sort()
    return _truncate("\n".join(matches[:500]) or "(no matches)")


def grep(pattern: str, path: str = ".", glob: str = "") -> str:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"
    root = Path(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if glob and not fnmatch.fnmatch(name, glob):
                    continue
                files.append(Path(dirpath) / name)
    else:
        return f"Error: path not found: {path}"

    hits: list[str] = []
    for f in files:
        try:
            text = f.read_text(errors="strict")
        except (OSError, UnicodeDecodeError):
            continue  # skip binary/unreadable files
        for lineno, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f}:{lineno}: {line.strip()}")
                if len(hits) >= 500:
                    return _truncate("\n".join(hits) + "\n... [result limit reached]")
    return _truncate("\n".join(hits) or "(no matches)")


def bash(command: str, timeout: int = 300) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    out = proc.stdout or ""
    if proc.stderr:
        out += ("\n" if out else "") + "[stderr]\n" + proc.stderr
    out = out.strip() or "(no output)"
    if proc.returncode != 0:
        out += f"\n[exit code: {proc.returncode}]"
    return _truncate(out)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_tools(bash_timeout: int = 300) -> dict[str, Tool]:
    """Build the tool registry keyed by tool name."""

    def _bash(command: str) -> str:
        return bash(command, timeout=bash_timeout)

    tools = [
        Tool(
            name="read_file",
            description=(
                "Read a text file, returning numbered lines. Use offset/limit "
                "for large files."
            ),
            parameters={
                "path": {"type": "string", "description": "Path to the file", "_required": True},
                "offset": {"type": "integer", "description": "Line offset to start from (0-based)"},
                "limit": {"type": "integer", "description": "Max number of lines to return"},
            },
            fn=read_file,
        ),
        Tool(
            name="write_file",
            description="Create or overwrite a file with the given content.",
            parameters={
                "path": {"type": "string", "description": "Path to the file", "_required": True},
                "content": {"type": "string", "description": "Full file content", "_required": True},
            },
            fn=write_file,
            mutating=True,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact snippet in a file. old_string must appear "
                "exactly once; include surrounding lines to make it unique."
            ),
            parameters={
                "path": {"type": "string", "description": "Path to the file", "_required": True},
                "old_string": {"type": "string", "description": "Exact text to replace", "_required": True},
                "new_string": {"type": "string", "description": "Replacement text", "_required": True},
            },
            fn=edit_file,
            mutating=True,
        ),
        Tool(
            name="list_directory",
            description="List the entries of a directory.",
            parameters={
                "path": {"type": "string", "description": "Directory path (default: cwd)"},
            },
            fn=list_directory,
        ),
        Tool(
            name="glob",
            description="Find files matching a glob pattern (e.g. '*.py', 'src/**/*.ts').",
            parameters={
                "pattern": {"type": "string", "description": "Glob pattern", "_required": True},
                "path": {"type": "string", "description": "Directory to search (default: cwd)"},
            },
            fn=glob_files,
        ),
        Tool(
            name="grep",
            description="Search file contents with a regular expression.",
            parameters={
                "pattern": {"type": "string", "description": "Python regular expression", "_required": True},
                "path": {"type": "string", "description": "File or directory to search (default: cwd)"},
                "glob": {"type": "string", "description": "Only search files matching this glob (e.g. '*.py')"},
            },
            fn=grep,
        ),
        Tool(
            name="bash",
            description=(
                "Run a shell command and return its output. Use for running "
                "tests, git, gh (GitHub CLI), builds, etc."
            ),
            parameters={
                "command": {"type": "string", "description": "Shell command to run", "_required": True},
            },
            fn=_bash,
            mutating=True,
        ),
    ]
    return {t.name: t for t in tools}


def tool_schemas(tools: dict[str, Tool]) -> list[dict[str, Any]]:
    schemas = []
    for t in tools.values():
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": _clean_schema(t.parameters),
                        "required": [k for k, v in t.parameters.items() if v.get("_required")],
                    },
                },
            }
        )
    return schemas
