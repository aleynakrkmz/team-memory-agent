# Baseline B1: fixed rules, no LLM

Design §2.2 and §6.5: *"B1: fixed rules with BM25 matching, no LLM; rules are
fixed before test data are generated."* Code: `app/baseline/b1.py`.
Tests: `tests/test_b1.py`.

## How it runs

B1 replaces only the extraction step. Everything after it is the agent's
path, so the two systems can be compared run for run:

```
transcript -> B1 rules -> match (BM25) -> route -> proposals -> store -> review -> commit
                                                  hard cases: no agent, so no proposal
```

```bash
# B1, results stored in its own database
uv run python -m app.pipeline samples/demo --meeting 3 --system b1 \
    --db data/demo/b1/memory.sqlite --tasks samples/demo/tasks_before_meeting_03.json

# The agent, as before (--system agent is the default)
uv run python -m app.pipeline samples/demo --meeting 3 \
    --db data/demo/agent/memory.sqlite --tasks samples/demo/tasks_before_meeting_03.json
```

No model and no Ollama are needed for `--system b1`. Use a separate database
for each system and run, so trials stay isolated (design §6.3). The review
list and commit work on a B1 database the same way.

## What is stored

The same tables as an agent run: meetings, cases (clear or hard, with the
reason), proposals and a run record. The run record's metrics hold
`"system": "b1"` (or `"agent"`) and `candidate_changes`, every extracted
candidate including the ones that did not become a proposal. A default run id
names the system: `demo-b1-m3-<time>`.

## The rules

Each line is split into sentences. A sentence addressed to an assistant
("Assistant, ...") is skipped.

| Change | Pattern (examples from the development meetings) |
|---|---|
| complete | "*The login page* is done / finished / complete"; "I finished *the X*" |
| cancel | "we're dropping / cancelling / scrapping *the project report draft*" |
| set_due_date | "*The login page* needs a few more days, can I have until the 10th?"; "move *the X* to ..." with a date |
| create, owner named | "Ayşe, can you do *the login page* by the 6th?" with a yes from Ayşe on the next line; "Okan, please take *the customer export*, by the 10th."; "Can, *the database schema* is yours"; "Baran, you'll prepare *the usage report*" |
| create, owner = speaker | "I'll write *the project report draft* myself" |
| create, no owner | "someone has to *run load tests before the release*"; "we need *API documentation*" |
| assign | a named request about a task that is already open under someone else; "X, could you take over *that thing*" |

Then:

- **Merging.** A create about the same task within two lines is merged:
  "We also need API documentation" + "I'll update the API documentation, the
  end of the month is fine" is one create with owner and date.
- **Create or assign.** A request about a task that is already open becomes an
  assign (new owner), a set_due_date (same owner, new date) or nothing (it
  repeats the list). The open task is found by a unique BM25 match, or by a
  unique word overlap on small lists.
- **No agreement, no change.** A question needs a yes from the person asked.
  Offers of help, open questions, pronoun-only tasks ("I'll share them") and
  postponed topics give nothing.
- **Evidence.** Every candidate cites whole lines, so quotes are always
  verbatim.
- **Dates.** Only phrases the shared resolver understands (`by the 18th`,
  `until Monday`, `the end of the month`, ...). "no deadline" gives none.

## Results on the development meetings

Demo meetings 1–3, each from the reference task list (reset-to-gold):

| Meeting | Expected | B1 | Agent (25 Nov) |
|---|---:|---:|---:|
| 1 | 3 | 3 | 3 |
| 2 | 4 | 4 | 2 |
| 3 | 4 | 3 | 2 |
| Total | 11 | **10** | 7 |

B1 misses only the vague reassignment ("that API thing"): the phrase matches
two open tasks, route marks it hard, and B1 has no agent to resolve it.

**These numbers are optimistic for B1.** The rules were written while reading
these same meetings, and the synthetic meetings follow a few fixed phrasings.
The fair comparison is on the held-out projects (model B generator, unseen
names, hand-written and AMI meetings), where wording varies. Per the design,
the rules must not change after the test data are generated.

## Not done yet

- Extraction fallback (design §3.7): when LLM extraction fails twice, the
  agent system should process the meeting with B1 instead of stopping.
  `extract_b1` is ready for that; the pipeline still exits with code 1.
- B2 (one prompt over the whole history) is a separate baseline.
