# Baseline B2: one prompt over the whole history

Design §2.2, §6.5 and §6.6: *"B2: one prompt with all transcripts so far and
the current task list; its prompt is developed on the development set with the
same effort as the agent's."* Code: `app/baseline/b2.py`, prompt:
`prompts/b2.md`, tests: `tests/test_b2.py`.

## Why it exists

B2 is the obvious simpler alternative to the agent. The design expects it to
fall short for four reasons (long input, one pass with no search or question,
cost that grows with the project, less precise evidence), but treats it as a
genuine best-effort baseline: if B2 wins, that is the result.

## How it runs

```
task list + meetings 1..n-1 (context) + meeting n  ->  ONE model call  ->  changes of meeting n
                                                     -> proposals -> store -> validate -> review -> commit
```

```bash
uv run python -m app.pipeline samples/demo --meeting 3 --system b2 \
    --db data/demo/b2/memory.sqlite --tasks samples/demo/tasks_before_meeting_03.json
```

- **No match, route or agent.** The model picks the task id from the list
  itself. It can put a statement it cannot resolve under `unclear` (a
  per-case abstention), but it cannot search or ask.
- **History.** With `--db`, the earlier meetings come from the database
  (the pipeline stores any earlier meeting that is missing). Without `--db`,
  they are read from the transcript files.
- **Lines.** Earlier meetings are shown as `M1-L2`, the latest as `L2`.
  Only lines of the latest meeting count as evidence.
- **Same model and settings** as the agent: `qwen3:8b`, temperature 0,
  32k context, thinking off, closed JSON schema.

## B2's ceilings (design §6.6)

| Limit | Value | What happens |
|---|---|---|
| Calls per meeting | 2 | One retry after invalid or cut-off output, then the run fails (`b2_failed`) |
| Tokens per meeting | 30,000 | The oldest meetings are left out first; the run record lists them |
| Time per meeting | 5 minutes | Each call gets the time that is left; no new call after it runs out |

Token counts are estimated from text length (4 characters per token) to decide
what fits. Ollama's exact counts are in its server log.

## From answer to proposals

Each returned change becomes a `ProposedUpdate`, with the task's current
version for updates and the deadline resolved by the shared date resolver
(`datagen.common.find_dates`), the same one match uses. A change that cannot
become a proposal is dropped and recorded with its reason:

- unknown task id, or a task that is no longer open;
- no evidence line from the latest meeting;
- a deadline the resolver cannot read (for `set_due_date`);
- a change that breaks the `ProposedUpdate` contract (for example an assign
  without an owner).

Quotes are not checked here: the validator checks them, as for the agent.

## What is stored

Meetings, proposals and a run record, as for the agent. B2 has no routed
cases, so its proposals have no case id. The run record's metrics hold
`"system": "b2"`, every returned change (`candidate_changes`), and:

| Key | Meaning |
|---|---|
| `b2_calls` | 1 or 2 |
| `history_meetings_sent` / `history_meetings_dropped` | which earlier meetings were in the prompt |
| `prompt_tokens_estimate` | size of the prompt |
| `changes_returned` | how many changes the model returned |
| `rejected_changes` | dropped changes and why |
| `unclear` | the model's abstentions |

A failed run stores `stop_reason = "b2_failed"` and the model's last reply.

## Prompt

`prompts/b2.md` uses the same rules and wording as the extraction prompt where
they overlap (change types, owner, due phrase, verbatim quotes, injection
handling), plus: report only the latest meeting, use earlier meetings to
resolve vague references, never invent a task id, and put unresolvable
statements under `unclear`. The design gives the B2 prompt to workstream 2;
this is a first version to be tuned on the development set with the same
effort as the agent's prompts, then frozen with the system.

## Results on the development meetings

Demo meetings 1–3, each from the reference task list (reset-to-gold), live
`qwen3:8b` on an Apple M5 (27 Nov). All B2 proposals passed the validator.

| Meeting | Expected | Agent | B1 | B2 |
|---|---:|---:|---:|---:|
| 1 | 3 | 3 | 3 | 3 |
| 2 | 4 | 2 | 4 | 3 |
| 3 | 4 | 2 | 3 | 1 |
| Total | 11 | 7 | 10 | 7 |
| Time per meeting | | 68–104 s | < 1 s | 31–41 s |
| Model calls per meeting | | 1–7 | 0 | 1 |

What B2 got wrong:

- Meeting 2: missed "the database schema is finished" (T2 complete).
- Meeting 3: missed "the login page is done" (T1 complete). It returned the
  reassignment of "that API thing" and the cancel of the project report draft,
  but without a task id, so both were dropped (`unknown task id None`).

With three short meetings the whole history is about 1,300 tokens, so the
long-input weakness the design expects does not show yet. B1's lead comes from
rules written on these same meetings (see `baseline_b1.md`). The comparison
that counts is on the held-out projects.
