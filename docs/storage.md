# Storage (`app/store`)

Owner: Aleyna (workstream 1). Design §3.4, §5 and §6.3.

Each project has its own append-only SQLite database. The pipeline stores
meetings, cases, proposals and run records in it, reads the current task list
from it, and gives the agent a read-only view of it. Only the commit step,
after approval, adds a task version.

## Files

| File | What it does |
|------|--------------|
| `app/store/schema.py` | The 11 tables and the triggers that refuse `UPDATE` and `DELETE` |
| `app/store/store.py` | `MemoryStore`: one write method per writer in design §3.4, plus reads |
| `app/store/reader.py` | `SqliteMemory`: the agent's read-only view (implements `MemoryReader`) |
| `app/store/paths.py` | Where database files live |
| `app/store/session.py` | What the pipeline stores before and after a run (`--db`) |
| `tests/test_store.py`, `tests/test_pipeline_store.py` | Tests |

## Tables and who writes them

| Table | Holds | Written by |
|-------|-------|-----------|
| `project` | Project id, schema version | `MemoryStore.create` |
| `team_profiles` | Coordinator and members; the newest row is current | data generator, WS3 lead or coordinator |
| `meetings`, `utterances` | Meetings and their lines (`L1`, `L2`, ...) | ingest |
| `cases` | One row per task-change case (`M3-K0`): clear or hard, reasons | controller |
| `proposals` | `P1`, `P2`, ...: the full `ProposedUpdate` | controller |
| `clarification_questions` | `Q1`, `Q2`, ...: questions the agent asked | controller |
| `clarification_answers` | `C01`, `C02`, ...: answers, who gave them | review list |
| `reviews` | `R1`, `R2`, ...: approve or reject, and who decided | review list |
| `task_versions` | `T1 v1`, `T1 v2`, ...: every version of every task | commit (after approval) or the reset-to-gold setup |
| `run_records` | Stop reason, failure detail, metrics, agent traces | controller |

## Rules the database enforces

- **Append-only.** Every table has triggers that refuse `UPDATE` and `DELETE`.
  Connections turn on `recursive_triggers`, so `INSERT OR REPLACE` cannot
  bypass them. A changed task is a new version row; old rows stay.
- **A commit version needs an approval.** A `task_versions` row with
  `source = 'commit'` must point at a `reviews` row with `decision = 'approve'`,
  and one approval can make only one version.
- **No forked history.** `add_task_version` takes the version the proposal was
  based on and refuses it if the task has moved on.
- **One project per file.** Records for another project are refused, and a
  database is never created over an existing file.
- **The agent cannot write.** `SqliteMemory` opens the file with `mode=ro`.

The decision whether a proposal may be committed (same proposal, same task
version, used once) stays in the commit step (workstream 3). The store only
keeps the history consistent.

## Database files

```
data/<project_id>/<run_id>/memory.sqlite                      carry-forward
data/<project_id>/<run_id>/reset/<meeting_id>/memory.sqlite   reset-to-gold
```

`app/store/paths.py` builds these paths. Ids may use letters, digits, `_`,
`-` and `.` only. `data/` is git-ignored.

## Running the pipeline with the store

```bash
uv run python -m app.pipeline samples/demo --meeting 1 --db data/demo/run1/memory.sqlite
uv run python -m app.pipeline samples/demo --meeting 3 --db data/demo/reset3/memory.sqlite --tasks samples/demo/tasks_before_meeting_03.json
```

With `--db` the pipeline:

1. creates the database if it is missing and stores the team profile;
2. stores the meeting, and any earlier meeting of the project that is not
   stored yet, so the agent can search them;
3. reads the current task list from the database. `--tasks` is only for an
   empty database: it seeds the starting list (rows are marked `seed`);
4. runs extract, match, route and the agent, with `SqliteMemory` as the
   agent's memory;
5. stores the cases, proposals and clarification questions, and a run record
   with the stop reason and metrics. A failed run is recorded too
   (`extraction_failed` or `error`).

Progress lines start with `db:` on stderr. `--run-id` names the run record.
Without `--db` the pipeline works as before (`--tasks` file, transcripts read
from disk).

A meeting that is already stored with different lines is refused: use a new
database for a changed transcript, for example the injected copy of meeting 3.

## For the commit and review steps (workstream 3)

```python
from app.store import MemoryStore

with MemoryStore.open(path, project_id="demo") as store:
    proposal = store.proposal("P3")
    review_id = store.add_review("P3", decision="approve", reviewer="Zeynep")
    record = store.add_task_version(
        approval_id=review_id,
        task_id=proposal.task_id,            # None for create: the store gives T1, T2, ...
        base_version=proposal.task_version,  # None for create
        description=..., owner=..., due_date=..., status=...,
        meeting_no=proposal.meeting_no,
    )
```

`store.current_tasks()` returns the task list, `store.task_history("T4")`
every version with its evidence refs.

Read methods used by the review list and commit:

| Method | Returns |
|--------|---------|
| `meeting_proposals(meeting_no)` | every proposal of a meeting, oldest first, as `(proposal_id, ProposedUpdate)` |
| `pending_proposals(meeting_no)` | the same, only proposals with no review yet |
| `clarification_ids()` | ids of all stored clarification answers (`C01`, ...) |
| `version_for_approval(review_id)` | the task version an approval produced, or `None` if unused |
