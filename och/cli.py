"""Interactive REPL and one-shot CLI for och."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, session
from .agent import Agent
from .client import OllamaClient, OllamaError
from .config import Config
from .tools import build_tools

try:  # readline gives history/editing in the REPL; absent on some platforms
    import readline  # noqa: F401
except ImportError:
    pass

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"

HELP = """\
Commands:
  /help              show this help
  /model [name]      show or switch the active model
  /models            list models available in Ollama
  /clear             clear the conversation (keeps the system prompt)
  /save              save the session to disk
  /sessions          list saved sessions
  /yolo              toggle auto-approval of mutating tools (bash, edits)
  /quit or /exit     leave (also Ctrl-D)
Anything else is sent to the model.
"""


def _color(enabled: bool):
    if enabled:
        return BOLD, DIM, CYAN, YELLOW, RESET
    return "", "", "", "", ""


def _format_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 120:
            s = s[:120] + "…"
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


def make_agent(config: Config, interactive: bool, color: bool) -> Agent:
    bold, dim, cyan, yellow, reset = _color(color)
    client = OllamaClient(config.host)
    tools = build_tools(bash_timeout=config.bash_timeout)

    def on_token(token: str) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_tool_call(name: str, args: dict) -> None:
        print(f"\n{cyan}⚒ {name}{reset}{dim}({_format_args(args)}){reset}")

    def on_tool_result(name: str, result: str) -> None:
        preview = result if len(result) <= 400 else result[:400] + "…"
        print(f"{dim}{preview}{reset}")

    def approver(name: str, args: dict) -> bool:
        if not interactive:
            return True  # non-interactive runs behave like --yolo
        prompt = f"{yellow}Allow {name}({_format_args(args)})? [y/N] {reset}"
        try:
            answer = input(prompt)
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes")

    return Agent(
        client,
        config,
        tools,
        approver=approver,
        on_token=on_token,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
    )


def repl(agent: Agent, config: Config, color: bool) -> int:
    bold, dim, cyan, yellow, reset = _color(color)
    print(f"{bold}och{reset} v{__version__} — model {cyan}{config.model}{reset} @ {config.host}")
    print(f"{dim}Type /help for commands, Ctrl-D to exit.{reset}")

    while True:
        try:
            line = input(f"\n{bold}› {reset}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            arg = arg.strip()
            if cmd in ("/quit", "/exit"):
                return 0
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/model":
                if arg:
                    config.model = arg
                print(f"model: {config.model}")
            elif cmd == "/models":
                try:
                    for name in agent.client.list_models():
                        print(f"  {name}")
                except OllamaError as exc:
                    print(f"error: {exc}")
            elif cmd == "/clear":
                agent.messages = agent.messages[:1]
                print("conversation cleared")
            elif cmd == "/save":
                path = session.save(agent.messages, config.model)
                print(f"saved: {path}")
            elif cmd == "/sessions":
                for p in session.list_sessions() or []:
                    print(f"  {p}")
            elif cmd == "/yolo":
                config.auto_approve = not config.auto_approve
                state = "ON (tools run without confirmation)" if config.auto_approve else "OFF"
                print(f"auto-approve: {state}")
            else:
                print(f"unknown command {cmd!r}; try /help")
            continue

        try:
            agent.run(line)
            print()
        except OllamaError as exc:
            print(f"\n{yellow}error:{reset} {exc}")
        except KeyboardInterrupt:
            print(f"\n{dim}[interrupted]{reset}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="och",
        description="och — a coding agent CLI for local Ollama models.",
    )
    parser.add_argument("-p", "--prompt", help="run one prompt non-interactively and exit")
    parser.add_argument("-m", "--model", help="Ollama model to use")
    parser.add_argument("--host", help="Ollama host (default: $OLLAMA_HOST or http://localhost:11434)")
    parser.add_argument("-y", "--yolo", action="store_true", help="auto-approve mutating tools")
    parser.add_argument("--num-ctx", type=int, help="context window size to request")
    parser.add_argument("--max-iterations", type=int, help="max agent loop iterations per request")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="PATH",
                        help="resume a saved session (default: most recent)")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument("--version", action="version", version=f"och {__version__}")
    args = parser.parse_args(argv)

    config = Config()
    if args.model:
        config.model = args.model
    if args.host:
        config.host = args.host
    if args.yolo:
        config.auto_approve = True
    if args.num_ctx:
        config.num_ctx = args.num_ctx
    if args.max_iterations:
        config.max_iterations = args.max_iterations

    interactive = args.prompt is None
    color = (not args.no_color) and sys.stdout.isatty()
    agent = make_agent(config, interactive=interactive, color=color)

    if args.resume:
        path = session.latest() if args.resume == "latest" else Path(args.resume)
        if path is None or not path.exists():
            print("no session to resume", file=sys.stderr)
            return 1
        try:
            data = session.load(path)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"cannot resume {path}: {exc}", file=sys.stderr)
            return 1
        agent.messages = data["messages"]
        print(f"resumed {path} ({len(agent.messages)} messages)")

    if args.prompt is not None:
        try:
            agent.run(args.prompt)
            print()
            return 0
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    return repl(agent, config, color)
