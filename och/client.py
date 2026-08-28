"""Minimal Ollama HTTP client using only the standard library.

Speaks to the Ollama REST API (default http://localhost:11434):
  - POST /api/chat  (streaming NDJSON, with tool-calling support)
  - GET  /api/tags  (list local models)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class OllamaError(RuntimeError):
    """Raised when the Ollama server returns an error or is unreachable."""


@dataclass
class ChatResponse:
    """Accumulated result of one /api/chat call."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done_reason: str | None = None
    eval_count: int = 0
    prompt_eval_count: int = 0

    @property
    def message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation history."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 600):
        self.host = host.rstrip("/")
        if not self.host.startswith(("http://", "https://")):
            self.host = "http://" + self.host
        self.timeout = timeout

    # -- low level -----------------------------------------------------

    def _request(self, path: str, payload: dict | None = None) -> Any:
        url = self.host + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                detail = json.loads(body).get("error", body)
            except (ValueError, AttributeError):
                detail = body
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host} ({exc.reason}). "
                "Is `ollama serve` running?"
            ) from exc

    def _stream_lines(self, path: str, payload: dict) -> Iterator[dict[str, Any]]:
        with self._request(path, payload) as resp:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                chunk = json.loads(raw)
                if chunk.get("error"):
                    raise OllamaError(str(chunk["error"]))
                yield chunk

    # -- public API ----------------------------------------------------

    def list_models(self) -> list[str]:
        with self._request("/api/tags") as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        """Run one chat completion, streaming tokens via `on_token`.

        Returns the accumulated response, including any tool calls the
        model requested.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options

        result = ChatResponse()
        for chunk in self._stream_lines("/api/chat", payload):
            msg = chunk.get("message") or {}
            token = msg.get("content") or ""
            if token:
                result.content += token
                if on_token:
                    on_token(token)
            for call in msg.get("tool_calls") or []:
                result.tool_calls.append(call)
            if chunk.get("done"):
                result.done_reason = chunk.get("done_reason")
                result.eval_count += chunk.get("eval_count", 0) or 0
                result.prompt_eval_count += chunk.get("prompt_eval_count", 0) or 0
        return result
