# logsnatch

A Python CLI for turning raw coding-agent session logs (Claude Code, OpenCode,
Qwen Code) into clean, redacted, quality-scored JSONL that loads directly into
a HuggingFace SFT pipeline.

The pipeline is a chain of JSONL → JSONL stages. Each stage has its own
subcommand and writes a file you can inspect; `run` chains them all together.

```
raw logs ──parse──▶ *_raw.jsonl
                       │
                  redact   (scrub secrets, anonymize paths)
                       ▼
                   *_redacted.jsonl
                       │
                   clean   (drop bad tool calls, empty turns, orphans)
                       ▼
                   *_cleaned.jsonl
                       │
                 evaluate  (attach quality `score`)
                       ▼
                   *_scored.jsonl
                       │
                  filter   (keep score ≥ threshold, format for training)
                       ▼
                  final.jsonl
```

## Install

```bash
pip install -e .                  # core CLI (stdlib only)
pip install -e .[validate]        # adds `transformers` for the `validate` subcommand
pip install -e .[test]            # adds pytest
```

Only `validate` pulls in `transformers` — the rest of the pipeline runs on the
standard library alone, so it works in constrained sandboxes with no network.

## Quick start

End-to-end on Claude Code logs (auto-discovers `~/.claude/projects`):

```bash
python -m logsnatch run --source claude --output training.jsonl
```

This writes the final dataset plus four intermediate files
(`training_raw.jsonl`, `_redacted.jsonl`, `_cleaned.jsonl`,
`_scored.jsonl`) so you can diff stages when debugging.

Higher-quality cut, all supported providers:

```bash
python -m logsnatch run --source all --output data/out.jsonl --min-score 0.7
```

Parse only, then sanity-check records against a real tokenizer:

```bash
python -m logsnatch parse --source claude --output data/raw.jsonl
python -m logsnatch validate --input data/raw.jsonl --model Qwen/Qwen3.5-4B
```

## Commands

| Command    | Purpose                                                          | Required flags                          | Notable optional flags                                                |
| ---------- | ---------------------------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------- |
| `parse`    | Read provider logs and normalize them into OpenAI-style messages | `--source {claude,opencode,qwen,all}`, `--output` | `--input` (defaults to the provider's standard log dir)        |
| `redact`   | Scrub secrets (API keys, tokens, URLs) and anonymize user paths  | `--input`, `--output`                   | —                                                                     |
| `clean`    | Drop malformed tool calls, orphaned tool results, empty turns    | `--input`, `--output`                   | `--no-remove-orphaned`                                                |
| `evaluate` | Attach a `score` (float in [0, 1]) to each conversation          | `--input`, `--output`                   | `--min-turns` (default 2), `--min-token-count` (default 1000)         |
| `filter`   | Keep records with `score ≥ --min-score`, write in training format | `--input`, `--output`                  | `--min-score` (default 0.5)                                           |
| `validate` | Apply a chat template via `transformers.AutoTokenizer` and report tokens, tool calls, and any failures | `--input` *or* `--source` | `--output` (writes only records that passed), `--model` (default `Qwen/Qwen3.5-4B`) |
| `run`      | Chain parse → redact → clean → evaluate → filter end-to-end      | `--source`, `--output`                  | `--input`, `--min-score` (default 0.5)                                |

### Output paths

`--output` paths without a suffix are auto-completed to `.jsonl`. A bare
`data/training` becomes `data/training.jsonl` (with a one-line `Note:` printed
to stderr). Existing suffixes (`.jsonl`, `.json`, anything else) are left
alone. Parent directories are created automatically.

### Default log locations

When you omit `--input`, the parser falls back to each provider's standard
directory:

- **claude** → `~/.claude/projects`
- **opencode** → `~/.local/share/opencode`
- **qwen** → `~/.qwen/projects`

Pass `--input <path>` to point at a specific export instead.

### Choosing flags

- **`--min-score`** — 0.5 is the default and is forgiving. Raise to 0.7+ when
  you have plenty of source logs and want a tighter dataset; lower it when
  you're data-starved or still tuning.
- **`--min-turns` / `--min-token-count`** — what counts as a "real"
  conversation. The evaluator hard-floors anything below these to score 0,
  regardless of other signals. Drop them for sparse logs; raise them when you
  only want substantive sessions.
- **`--source all`** — convenient, but it parses every provider's default log
  dir. Pair `--source <name>` with `--input <path>` to target one export.

## Output schema

A parsed record:

```json
{
  "id": "<session id>",
  "source": "claude",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ]
}
```

After `evaluate`, records gain `score`, `label`, `reasons`, and a `metrics`
block. After `filter`, the first message also carries a `tools` array (the
OpenAI-style tool schemas inferred from observed tool calls), and `messages`
is serialized to `list[str]` (one JSON-encoded turn per element) so PyArrow
can unify the schema across rows.

## Loading the output for training

The final JSONL loads cleanly via vanilla `datasets.load_dataset`:

```python
import json
from datasets import load_dataset

ds = load_dataset("json", data_files="data/training.jsonl", split="train")
for rec in ds:
    msgs = [json.loads(m) for m in rec["messages"]]    # list[str] → list[dict]
    tools = msgs[0].pop("tools", None) if msgs else None
    # tokenizer.apply_chat_template(msgs, tools=tools, ...)
```

## Troubleshooting

- **`validate` fails on a record** — open the failing id in
  `<output>_cleaned.jsonl` and check for missing `tool_call_id` pairings or
  empty assistant content. Most chat-template failures trace back to those.
- **Suspiciously few records survive `filter`** — eyeball the `score`
  distribution in `<output>_scored.jsonl`. If most scores sit below 0.5, your
  logs are short/noisy: lower `--min-score`, or relax `--min-turns` /
  `--min-token-count` in `evaluate`.
- **Secrets leaking through** — add a regex to
  `logsnatch/redaction/secrets.py` and rerun just the `redact` stage on the
  existing `_raw.jsonl` (no need to re-parse).
- **`run` halted mid-pipeline** — every stage is idempotent and reads only the
  previous stage's file, so you can resume by invoking the next stage directly
  with `--input <the partial file>`.

## Extending

Adding a new agent (e.g. Codex) usually means one new file in
`logsnatch/parsers/` plus a one-line entry in `parsers/__init__.py`. The
pipeline stages are agent-agnostic and don't need changes.

See `SKILL.md` for the parser contract (`BaseParser`), step-by-step
instructions, and notes on extending the redaction, scoring, and cleaning
stages.

## Tests

```bash
pytest                                    # full suite
pytest tests/test_parsers.py -k claude    # one parser at a time
```

## Development

Install pre-commit hooks once per clone so lint and formatting run on every commit:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files    # optional: run against the whole repo right now
```

Ruff handles both linting and formatting; config lives in `pyproject.toml` under
`[tool.ruff]`. CI runs `ruff check`, `ruff format --check`, and `pytest` against
Python 3.10, 3.11, and 3.12 on every push to `main` and every pull request
(`.github/workflows/ci.yml`).

## Releasing to PyPI

Publishing uses **Trusted Publishing (OIDC)** — no API tokens stored as
secrets. The workflow at `.github/workflows/publish.yml` builds an sdist + wheel
with `python -m build` and uploads via `pypa/gh-action-pypi-publish`.

### One-time setup

1. **PyPI:** create the project (or register a pending publisher for the first
   release). Under the project's **Publishing** settings, add a Trusted
   Publisher with:
   - Owner: your GitHub username/org
   - Repository: `Logsnatch`
   - Workflow: `publish.yml`
   - Environment: `pypi`
2. **GitHub:** in repo **Settings → Environments**, create an environment named
   `pypi`. Optionally add required reviewers for an extra approval step before
   any release runs.

### Cutting a release

1. Bump `version` in `pyproject.toml` (follow semver).
2. Commit the bump on `main`:
   ```bash
   git commit -am "Release v0.1.1"
   ```
3. Tag and push — this auto-triggers the publish workflow:
   ```bash
   git tag v0.1.1
   git push origin main --tags
   ```

Alternatively, trigger the workflow manually from the **Actions** tab
(`Publish to PyPI` → **Run workflow**). Manual runs publish whatever is on the
selected branch — make sure `version` in `pyproject.toml` is already bumped, or
PyPI will reject the upload as a duplicate.

### Verifying the release

- Watch the run under the **Actions** tab; the publish step prints the uploaded
  files.
- Confirm the new version appears at `https://pypi.org/project/logsnatch/`.
- Smoke test in a clean venv:
  ```bash
  pip install --upgrade logsnatch
  logsnatch --help
  ```
