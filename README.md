# ollama-coding-harness

**och** is a small coding-agent CLI harness — in the spirit of the
[pi](https://github.com/badlogic/pi-mono) coding agent — that runs entirely
against **local [Ollama](https://ollama.com) models**. It is written in pure
Python 3 with **only the standard library** (no pip dependencies).

It is focused on coding tasks: writing new code, solving coding problems,
running local tests, and preparing git commits / GitHub pull requests (via
`git` and `gh` through its bash tool).

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`) with at least
  one tool-capable model pulled, e.g.:

  ```sh
  ollama pull qwen2.5-coder:7b
  ```

- Optional: [`gh`](https://cli.github.com) for GitHub PR workflows.

## Usage

No installation needed — run it as a module from the repo root:

```sh
python3 -m och                     # interactive REPL (default model: qwen2.5-coder:7b)
python3 -m och -m llama3.1:8b      # pick a model
python3 -m och -p "add a unit test for utils.py and run it"   # one-shot mode
python3 -m och --resume            # resume the most recent saved session
```

Or install the `och` entry point:

```sh
pip install -e .
och
```

### The agent loop

Each request runs an agentic loop: the model streams its reasoning/answer,
may call workspace tools, sees their results, and iterates until it produces
a final answer (or hits `--max-iterations`).

Available tools:

| tool             | purpose                                            |
|------------------|----------------------------------------------------|
| `read_file`      | read a file with line numbers (offset/limit)       |
| `write_file`     | create or overwrite a file                         |
| `edit_file`      | exact, unique string replacement                   |
| `list_directory` | list a directory                                   |
| `glob`           | find files by pattern                              |
| `grep`           | regex search over file contents                    |
| `bash`           | run shell commands (tests, `git`, `gh`, builds, …) |

Mutating tools (`bash`, `write_file`, `edit_file`) ask for confirmation in
the REPL. Toggle that off with `/yolo` or start with `-y/--yolo`.
One-shot mode (`-p`) auto-approves, so use it deliberately.

### REPL commands

```
/help        show help
/model NAME  switch model         /models    list local Ollama models
/clear       reset conversation   /save      save session to ~/.och/sessions
/sessions    list saved sessions  /yolo      toggle tool auto-approval
/exit        quit (also Ctrl-D)
```

### Configuration

Flags take precedence over environment variables:

| flag               | env var           | default                  |
|--------------------|-------------------|--------------------------|
| `-m/--model`       | `OCH_MODEL`       | `qwen2.5-coder:7b`       |
| `--host`           | `OLLAMA_HOST`     | `http://localhost:11434` |
| `--num-ctx`        | `OCH_NUM_CTX`     | `16384`                  |
| `--max-iterations` | `OCH_MAX_ITERATIONS` | `40`                  |
|                    | `OCH_BASH_TIMEOUT`| `300` (seconds)          |
|                    | `OCH_SESSIONS_DIR`| `~/.och/sessions`        |

### Example: coding task ending in a PR

```sh
python3 -m och
› Create a fibonacci.py with an iterative fib(), add tests, run them,
  then commit on a new branch and open a GitHub PR with gh.
```

The agent will write the files, run `python3 -m unittest`, and use
`git`/`gh` through the bash tool — asking for your approval at each
mutating step unless you `/yolo`.

## Development

Run the test suite (no network or Ollama needed — the client tests spin up a
fake local HTTP server):

```sh
python3 -m unittest discover -s tests -v
```

Layout:

```
och/
  client.py    minimal streaming Ollama HTTP client (urllib)
  tools.py     tool schemas + implementations
  agent.py     the agentic tool loop
  cli.py       argparse CLI + interactive REPL
  session.py   save/resume sessions as JSON
  config.py    defaults, env vars, system prompt
tests/         unittest suite
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
