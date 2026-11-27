# Evaluation harness, scorer and B1

How a configuration is run on a project and scored against the answer key (design §12). Code: `eval/harness.py`,
`eval/scorer.py`, `eval/b1.py`, `eval/compare.py`. Tests: `tests/test_eval.py`.

## Run it

```bash
# The full agent, live (Ollama with qwen3:8b), reset-to-gold and carry-forward
uv run python eval/harness.py samples/demo --config agent --mode reset
uv run python eval/harness.py samples/demo --config agent --mode carry

# Baseline B1 (no model)
uv run python eval/harness.py samples/demo --config b1 --mode reset

# Replay an earlier agent run without calling the model
uv run python eval/harness.py samples/demo --config agent --mode reset --replay runs/eval/<run>/previews

# One table for all runs
uv run python eval/compare.py runs/eval/*
```

Each run writes `runs/eval/<run_id>/` (git-ignored): `report.md`, `scores.json`, `rows.csv` (every expected change
and every proposal), `run_record.json`, `previews/` (one per meeting, replayable) and the databases.

## What one trial does

A trial processes the meetings of one project in order. For each meeting:

1. **The configuration runs.** `agent` is the full pipeline with the database (`app.pipeline --db`). `b1` is the
   rule baseline. If extraction fails twice, the meeting is processed by the B1 rules and the run records an
   extraction fallback (design §11, failure 3). If the model cannot be reached, the meeting stops with
   `tool_failure` and all its changes count as missed.
2. **The simulated reviewer decides.** It is deterministic code, not a model. It approves, as the coordinator,
   every proposal that validation passes, and the approval is committed; invalid and blocked proposals are not
   approved (`validation_blocked`). An approval is not a correctness verdict. Clarification questions get the
   scenario's answer through the pipeline's simulated answers (`simulated_answers.json`).
3. **The scorer compares** the proposals, and in carry-forward mode the task list, with the answer key.

| Mode | Database | Measures |
|---|---|---|
| `reset` (reset-to-gold) | a fresh database per meeting, seeded with the correct list from before the meeting | change F1; each meeting on its own |
| `carry` (carry-forward) | one database for the project; approved changes and mistakes carry on | change F1 and task-state accuracy after every meeting |

## Metrics (`eval/scorer.py`)

- **Change F1 (primary).** A proposal is correct when its type, its task and the required fields are right:
  create: owner and due date (empty when the meeting leaves them open; the coordinator's answer counts for an owner
  left open on purpose); assign: owner; set_due_date: due date; complete and cancel: nothing. Micro-averaged over
  the meetings, and per type. Wrong fields are split into *missing* (left empty) and *wrong* (another value).
- **Task matching.** The system never sees gold task ids. Seeded tasks keep their id. A create proposal is matched
  to a create event of the same meeting when at least 60% of the shorter title's words appear in the other
  (`DESC_THRESHOLD`, fixed); matching is one-to-one. A task the system created is then mapped to that gold task.
- **Task-state accuracy (carry-forward).** After each meeting: owner, due date and status of every gold task,
  compared with the system's task; extra and missing tasks; whether the whole list is right.
- **Unsupported proposals (automatic proxy).** A proposal is supported when it cites a line the key lists for its
  event. The design's manual check of 30 proposals is still to be done.
- **Also recorded:** stop reason per meeting (one of the five), case outcomes (proposed, abstained, escalated,
  stopped, validation_blocked, unresolved), validation results, agent steps, clarification questions, seconds.

## Reproducibility (`run_record.json`)

Every number comes with: the configuration, the mode, the **execution mode** (`live`, `replay`, `mock`, or
`live (no model)` for B1), the model name and Ollama digest, thinking on or off, hashes of `prompts/` and
`config/`, the git commit and whether there were uncommitted changes, package versions, the `uv.lock` hash, a hash
of the project's data and the machine. A replay or mock number is not a live number, and only live runs support a
claim about the model.

## B1 (`eval/b1.py`)

No model. B1 reads a meeting as a conversation:

1. A line that names a task starts an item (a task on the list by its key words, e.g. "the icons" for "Icon set",
   or a new task: "a new task for you - the login page", "we need an icon set", "Ece, can you do X by ..."). The
   item covers that line and up to 3 more, until the next item.
2. Word lists decide the change of an item about a listed task: cancel ("let's drop it", "park this one"),
   complete ("it's all wrapped up", "I finished it yesterday"; not "almost done", not "I'll make sure it's done"),
   assign ("take over", "pass it to Ece", "you've got it now"), new due date ("the deadline moved ... by the 28th").
   An item about a new task is a create.
3. The owner is the person addressed or named after "to", else the next speaker who accepts. "Let's figure out
   who later" keeps the owner open. The date is the first date phrase of the item.
4. Then the same matching and routing as the agent. A hard case becomes a proposal on the top BM25 task only when
   the task was merely unclear or ambiguous; otherwise it stays without a proposal (`unresolved`).

| Version | Written on | F1 on dev set v1 (reset / carry, 7 projects, 134 changes) |
|---|---|---|
| 1: one line at a time | the two sample projects | 0.00-0.14 per project |
| 2: items over several lines | dev set v1 (all 7 projects) | 0.701 / 0.607 |

**Limit:** version 2 was written while reading all seven development projects, so its development score is
optimistic. It is frozen before the test set is generated; the number that counts is the test-set number.

## Limits of these numbers

- The simulated reviewer approves everything validation passes and answers every question at once and correctly.
- The two sample projects are small (3 meetings, 11 changes each) and hand-written; they are not the evaluation set.
- B2 (the single-prompt baseline) and the ablations A1–A3 are not in the harness yet.
