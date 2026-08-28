"""Configuration and defaults for och."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OCH_MODEL", "qwen2.5-coder:7b")
DEFAULT_CONTEXT_LENGTH = int(os.environ.get("OCH_NUM_CTX", "16384"))
DEFAULT_MAX_ITERATIONS = int(os.environ.get("OCH_MAX_ITERATIONS", "40"))
DEFAULT_BASH_TIMEOUT = int(os.environ.get("OCH_BASH_TIMEOUT", "300"))

SESSIONS_DIR = Path(os.environ.get("OCH_SESSIONS_DIR", Path.home() / ".och" / "sessions"))

SYSTEM_PROMPT = """\
You are och, a coding agent running in a terminal, working in the directory {cwd}.

You help with software engineering tasks: writing new code, fixing bugs,
refactoring, running tests, and preparing git commits and GitHub pull requests.

Rules:
- Use the provided tools to inspect and modify the workspace. Never invent
  file contents — read files before editing them.
- Prefer small, focused edits with the edit_file tool; use write_file only for
  new files or full rewrites.
- After making changes, run the project's tests or a quick sanity check with
  the bash tool when possible.
- For git and GitHub work use the bash tool (git, gh). Never push or open a
  pull request unless the user asked for it.
- When the task is done, reply with a short plain-text summary. Do not include
  tool call syntax in your final answer.
"""


@dataclass
class Config:
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    num_ctx: int = DEFAULT_CONTEXT_LENGTH
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    bash_timeout: int = DEFAULT_BASH_TIMEOUT
    auto_approve: bool = False
    cwd: str = field(default_factory=os.getcwd)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(cwd=self.cwd)
