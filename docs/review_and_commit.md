# Validate, review and commit

The last three steps of the graph (design §3): a proposal is checked, a person decides, and only then does
the task list change. Code: `app/validate/`, `app/review/`, `app/commit/`. Tests: `tests/test_review_commit.py`.

## Run it

```bash
# 1. Run a meeting with the database (stores the meeting, its cases and proposals)
uv run python -m app.pipeline samples/demo --meeting 3 --db data/demo/r3/memory.sqlite \
    --tasks samples/demo/tasks_before_meeting_03.json

# 2. Review the proposals of that meeting: approve, reject or skip each one
uv run python -m app.review data/demo/r3/memory.sqlite --meeting 3
```

Scripted decisions, without questions: `--decisions decisions.json` with `{"P1": "approve", "*": "skip"}`.

For carry-forward mode, run the next meeting on the same database without `--tasks`: it starts from the task
list that the approvals produced.

## What the review list shows

For each proposal without a review: the change, the task before and after, every evidence line with its text,
and the validation result. Choices: approve, reject, skip, quit.

Only the team's coordinator (from the team profile; aliases accepted) may decide. Every decision is stored as an
approval record: who decided, which proposal, which task version, when. An approval is bound to that one proposal.

| Validation | Meaning | Can it be approved? |
|---|---|---|
| valid | every check passed | yes |
| invalid | a check failed | no; commit refuses it as well |
| blocked | Rule 1 or Rule 2 fired | yes, after typing `yes`: the coordinator has read the lines |

## Checks (`app/validate/validate.py`)

| Code | Check |
|---|---|
| `schema` | the `ProposedUpdate` contract (fields that fit the action, task id and version, evidence) |
| `wrong_project`, `wrong_meeting` | the proposal belongs to this project and meeting |
| `evidence_missing` | every cited line is in this meeting (so no evidence from a later meeting); every clarification id is stored |
| `quote_mismatch`, `quote_not_cited` | every quote is in its line, and every quoted line is cited |
| `unknown_task`, `task_closed` | the task is on the list and still open |
| `stale_version` | the proposal was made on the task's current version |
| `duplicate_task` | a create does not repeat the title of an open task |
| `unknown_owner` | the owner is in the team profile (aliases accepted) |
| `date_in_past` | the due date is not before the meeting |
| `rule1` (block) | one line supports at most one owner change (assign, or create with an owner); otherwise all of them are blocked |
| `rule2` (block) | a line addressed to the assistant is not the only transcript evidence |

## How Rule 2 recognises a line addressed to the assistant

The rule is code, not an instruction to the model. A line is flagged if, ignoring case:

1. it starts with an address word used as a form of address, that is followed by `,` `:` or `!`, optionally
   after a greeting: `Assistant, ...`, `hey bot: ...`, `AI, ...`; or
2. it contains a fixed instruction phrase: `reassign every`, `mark all`, `ignore previous`, `override`, ...

Both lists are in `config/validate.json` and are frozen with the system. "System tests are failing" is not
flagged (no comma after the word); none of the lines of the two demo projects is flagged (tested). How often a
legitimate change is blocked by mistake is counted in the evaluation.

## Commit (`app/commit/commit.py`)

One approved proposal becomes one new task version, in one transaction:

- the review must exist, be an approval, and belong to this proposal;
- the proposal is validated again against the task list of that moment; an invalid one is refused;
- the store refuses a stale base version, a second version for the same approval, and a version whose
  review is a rejection.

So nothing reaches the task list without an approval, and one approval makes at most one version.
