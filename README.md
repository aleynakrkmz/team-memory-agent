# Team Memory Agent (Second Brain)

IS 773 – Agentic AI · Track A

Keeps a small team's task list correct across meetings: proposes task changes from meeting transcripts, links each change to supporting lines, and commits only coordinator-approved changes.

## Repository layout

| Path | Purpose |
|------|---------|
| `app/ingest` | Transcript loading / speech-to-text, line IDs |
| `app/extract` | One LLM call → candidate task changes (JSON) |
| `app/match` | BM25 matching against open tasks, date resolution |
| `app/route` | Clear vs. difficult case routing |
| `app/agent` | Agent loop and tools (difficult cases only) |
| `app/validate` | Proposal, evidence and injection-rule checks |
| `app/review` | Command-line review list (approval, clarification) |
| `app/commit` | The only write path to the task list |
| `app/store` | Append-only SQLite storage |
| `datagen/` | Team profiles, scenario scripts, synthetic meetings, answer-key check |
| `eval/` | Harness, simulated reviewer, scorer, baselines |
| `prompts/` | Prompt files |
| `config/` | Limits and configuration |
| `tests/` | Tests |
| `runs/` | Run traces and outputs (git-ignored) |
| `data/` | Project SQLite databases (git-ignored) |

## Workstreams

1. Data, audio and storage — Aleyna (lead), Gözde
2. Agent and tools — Bahadır (lead), Aleyna
3. Evaluation, approval and safety — Gözde (lead), Bahadır

## Setup

See [KURULUM.md](KURULUM.md) (setup, Mac and Windows) and [CALISMA_DUZENI.md](CALISMA_DUZENI.md) (branch and pull-request workflow).

```bash
uv sync
cp .env.example .env
uv run pytest
```

## Running the pipeline

Needs Ollama running with `qwen3:8b` (`ollama pull qwen3:8b`).

```bash
# Preview one meeting, task list from a JSON file
uv run python -m app.pipeline samples/demo --meeting 3 --tasks samples/demo/tasks_before_meeting_03.json

# Same, with the project's SQLite database as memory and record
uv run python -m app.pipeline samples/demo --meeting 3 --db data/demo/run1/memory.sqlite --tasks samples/demo/tasks_before_meeting_03.json

# Baseline B1 (fixed rules, no LLM, no Ollama needed), stored the same way
uv run python -m app.pipeline samples/demo --meeting 3 --system b1 --db data/demo/b1/memory.sqlite --tasks samples/demo/tasks_before_meeting_03.json
```

The JSON preview goes to stdout (`--out` also writes it to a file). Progress
goes to stderr. Nothing is written to the task list by the pipeline.

## Reviewing and committing

```bash
# Approve, reject or skip each proposal of meeting 3; an approval is committed at once
uv run python -m app.review data/demo/run1/memory.sqlite --meeting 3
```

See [docs/review_and_commit.md](docs/review_and_commit.md).

## Docs

- [docs/storage.md](docs/storage.md): the append-only SQLite store, its rules, and `--db`
- [docs/review_and_commit.md](docs/review_and_commit.md): validation checks, Rule 1 and Rule 2, the review list and commit
- [docs/baseline_b1.md](docs/baseline_b1.md): baseline B1 (fixed rules, no LLM) and `--system b1`
- [docs/demo_run.md](docs/demo_run.md): results on demo meetings 1–3 and what changed
- [docs/model_selection.md](docs/model_selection.md): how the models were chosen
