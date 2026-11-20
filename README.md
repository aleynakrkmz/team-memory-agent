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
