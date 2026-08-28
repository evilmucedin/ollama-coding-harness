"""The agentic loop: chat with the model, execute tool calls, repeat."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable

from .client import ChatResponse, OllamaClient
from .config import Config
from .tools import Tool, tool_schemas

# Callbacks let the CLI render output / gate approvals without the agent
# knowing anything about terminals.
OnToken = Callable[[str], None]
OnToolCall = Callable[[str, dict[str, Any]], None]
OnToolResult = Callable[[str, str], None]
Approver = Callable[[str, dict[str, Any]], bool]

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_text_tool_calls(content: str, known_names: set[str]) -> list[dict[str, Any]]:
    """Fallback parser for models that emit tool calls as plain text.

    Local models frequently print `{"name": ..., "arguments": {...}}` —
    bare, inside <tool_call> tags, or inside a ```json fence — instead of
    producing structured tool calls that Ollama recognizes. Only JSON objects
    whose "name" matches a known tool are treated as calls, so ordinary
    JSON in an answer is left alone.
    """
    if not content:
        return []
    blocks = _TOOL_CALL_TAG_RE.findall(content)
    if not blocks:
        blocks = _FENCE_RE.findall(content) or [content]
    calls: list[dict[str, Any]] = []
    for block in blocks:
        calls.extend(_scan_json_objects(block, known_names))
    return calls


def _balanced_braces(text: str, start: int) -> tuple[str | None, int]:
    """Return the balanced {...} substring starting at `start`, quote-aware."""
    depth = 0
    quote: str | None = None
    i = start
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
        i += 1
    return None, start + 1


def _parse_objectish(candidate: str) -> dict[str, Any] | None:
    """Parse a dict from JSON or (as a fallback) a Python dict literal.

    Local models often emit tool calls with single-quoted strings; those are
    not JSON, but ast.literal_eval handles them safely (literals only).
    """
    try:
        obj = json.loads(candidate)
    except ValueError:
        try:
            obj = ast.literal_eval(candidate)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    return obj if isinstance(obj, dict) else None


def _scan_json_objects(text: str, known_names: set[str]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        candidate, end = _balanced_braces(text, start)
        obj = _parse_objectish(candidate) if candidate else None
        if obj is None:
            pos = start + 1
            continue
        pos = end
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters"))
        if isinstance(name, str) and name in known_names and (args is None or isinstance(args, dict)):
            calls.append({"function": {"name": name, "arguments": args or {}}})
    return calls


class Agent:
    def __init__(
        self,
        client: OllamaClient,
        config: Config,
        tools: dict[str, Tool],
        approver: Approver | None = None,
        on_token: OnToken | None = None,
        on_tool_call: OnToolCall | None = None,
        on_tool_result: OnToolResult | None = None,
    ):
        self.client = client
        self.config = config
        self.tools = tools
        self.approver = approver
        self.on_token = on_token
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": config.system_prompt()}
        ]

    # -- helpers ---------------------------------------------------------

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self.tools)}"
        if tool.mutating and not self.config.auto_approve and self.approver:
            if not self.approver(name, args):
                return "Error: the user declined this tool call. Ask them how to proceed."
        try:
            return tool.fn(**args)
        except TypeError as exc:
            return f"Error: bad arguments for {name}: {exc}"
        except Exception as exc:  # tool bugs shouldn't kill the session
            return f"Error: {name} raised {type(exc).__name__}: {exc}"

    @staticmethod
    def _parse_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments") or {}
        if isinstance(args, str):  # some models return JSON-encoded strings
            try:
                args = json.loads(args)
            except ValueError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_raw": args}
        return name, args

    # -- main loop --------------------------------------------------------

    def run(self, user_input: str) -> str:
        """Process one user request to completion. Returns the final answer."""
        self.messages.append({"role": "user", "content": user_input})
        schemas = tool_schemas(self.tools)
        options = {"num_ctx": self.config.num_ctx}
        final = ""

        for _ in range(self.config.max_iterations):
            response: ChatResponse = self.client.chat(
                model=self.config.model,
                messages=self.messages,
                tools=schemas,
                options=options,
                on_token=self.on_token,
            )
            self.messages.append(response.message)

            tool_calls = response.tool_calls
            if not tool_calls:
                # Some model/template combinations emit the tool call as
                # plain text instead of a structured tool_calls field.
                tool_calls = extract_text_tool_calls(response.content, set(self.tools))

            if not tool_calls:
                final = response.content
                break

            for call in tool_calls:
                name, args = self._parse_call(call)
                if self.on_tool_call:
                    self.on_tool_call(name, args)
                result = self._execute_tool(name, args)
                if self.on_tool_result:
                    self.on_tool_result(name, result)
                self.messages.append(
                    {"role": "tool", "tool_name": name, "content": result}
                )
        else:
            final = (
                f"[stopped after {self.config.max_iterations} iterations without a "
                "final answer; ask me to continue if needed]"
            )
            self.messages.append({"role": "assistant", "content": final})

        return final
