"""Session persistence: save/resume conversations as JSON files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import SESSIONS_DIR


def _sessions_dir() -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR


def save(messages: list[dict[str, Any]], model: str, path: Path | None = None) -> Path:
    if path is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = _sessions_dir() / f"session-{stamp}.json"
    payload = {"version": 1, "model": model, "saved_at": time.time(), "messages": messages}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data.get("messages"), list):
        raise ValueError(f"{path} is not a valid och session file")
    return data


def latest() -> Path | None:
    files = sorted(_sessions_dir().glob("session-*.json"))
    return files[-1] if files else None


def list_sessions() -> list[Path]:
    return sorted(_sessions_dir().glob("session-*.json"))
