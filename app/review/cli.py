"""The review list: the coordinator approves or rejects each proposal of a meeting (design §3, step 7).

    uv run python -m app.review data/demo/run1/memory.sqlite --meeting 3

For every proposal without a review it shows the change (task before and after), the evidence lines with
their text, and the validation result. Then it asks:

    [a]pprove  [r]eject  [s]kip  [q]uit

- An approval is recorded and committed at once (app.commit): the task list gets one new version.
- An invalid proposal cannot be approved. A blocked one (Rule 1 or Rule 2) can, after typing "yes".
- A rejection is recorded with an optional note. Skipped proposals stay in the list for later.

`--decisions file.json` answers without asking, for scripted runs and tests:
    {"P1": "approve", "P3": "reject", "*": "skip"}      ("*" is used for every other proposal)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from app.commit.commit import CommitError, approve_and_commit, entitled_reviewer, new_record
from app.review.queue import clarification_ids, meeting_proposals, pending
from app.schemas import EvidenceSource, ProposedUpdate, TaskRecord
from app.store.store import MemoryStore
from app.validate.validate import Rule2Settings, ValidationResult, validate_meeting

ACTION_WORDS = {"create": "create", "assign": "reassign", "set_due_date": "new due date",
                "complete": "complete", "cancel": "cancel"}


def _task_line(t: dict | TaskRecord | None) -> str:
    if t is None:
        return "(new task)"
    t = t.model_dump() if isinstance(t, TaskRecord) else t
    due = t.get("due_date") or "no date"
    status = getattr(t.get("status"), "value", t.get("status"))
    return f"owner {t.get('owner') or 'none'}, due {due}, {status}"


def describe(pid: str, p: ProposedUpdate, check: ValidationResult, store: MemoryStore) -> list[str]:
    """The lines shown for one proposal."""
    tasks = {t.task_id: t for t in store.current_tasks()}
    current = tasks.get(p.task_id or "")
    head = f"[{pid}] {ACTION_WORDS[p.action.value]} {p.task_id or 'new task'} “{p.task_description}”  ({p.origin.value})"
    out = [head]
    if check.status != "invalid":
        try:
            after = new_record(p, current, store.team_profile().canonical_member(p.fields.owner))
            out.append(f"    before: {_task_line(current)}" + (f" (v{current.version})" if current else ""))
            out.append(f"    after:  {_task_line(after)}")
        except CommitError:
            pass
    meeting = store.meeting(p.meeting_no)
    for ref in p.evidence_refs:
        if ref.source == EvidenceSource.transcript:
            u = meeting.utterance(ref.id)
            out.append(f"    evidence {ref.id}  {u.speaker}: {u.text}" if u else f"    evidence {ref.id}  (missing)")
        else:
            out.append(f"    evidence {ref.id}  (coordinator's answer)")
    label = {"valid": "valid", "invalid": "INVALID, cannot be approved", "blocked": "BLOCKED, read the evidence"}
    out.append(f"    check: {label[check.status]}" + ("" if check.status == "valid" else f" — {check.summary()}"))
    for n in check.notes:
        out.append(f"    note: {n.message}")
    return out


def review_meeting(
    store: MemoryStore,
    meeting_no: int,
    *,
    reviewer: str | None = None,
    decisions: dict[str, str] | None = None,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    settings: Rule2Settings | None = None,
) -> list[dict]:
    """Go through the meeting's pending proposals. Returns one record per proposal shown."""
    settings = settings or Rule2Settings.load()
    reviewer = entitled_reviewer(store, reviewer or store.team_profile().coordinator)
    todo = pending(store, meeting_no)
    if not todo:
        say(f"Meeting {meeting_no}: nothing to review.")
        return []
    say(f"Meeting {meeting_no}: {len(todo)} proposal(s) to review, reviewer {reviewer}.\n")
    log = []
    for pid, _ in todo:
        # validate against the task list as it is now (earlier approvals in this session included)
        everything = meeting_proposals(store, meeting_no)
        results = validate_meeting([p for _, p in everything], meeting=store.meeting(meeting_no),
                                   team=store.team_profile(), tasks=store.current_tasks(),
                                   clarification_ids=clarification_ids(store), settings=settings)
        check = results[[i for i, _ in everything].index(pid)]
        p = check.proposal
        for line in describe(pid, p, check, store):
            say(line)

        if decisions is not None:
            choice = decisions.get(pid, decisions.get("*", "skip"))
            say(f"    decision (scripted): {choice}")
        else:
            options = "[r]eject [s]kip [q]uit" if check.status == "invalid" else "[a]pprove [r]eject [s]kip [q]uit"
            choice = {"a": "approve", "r": "reject", "s": "skip", "q": "quit"}.get(ask(f"    {options}: ").strip().lower()[:1], "skip")

        record = {"proposal_id": pid, "action": p.action.value, "task_id": p.task_id, "status": check.status,
                  "decision": choice, "review_id": None, "committed": None, "error": None}
        if choice == "quit":
            say("    stopped; the remaining proposals stay in the list.")
            break
        if choice == "approve":
            if check.status == "invalid":
                record["decision"], record["error"] = "skip", "invalid proposals cannot be approved"
                say("    not approved: " + record["error"])
            elif check.status == "blocked" and decisions is None and ask('    Blocked by an injection rule. Type "yes" to approve anyway: ').strip().lower() != "yes":
                record["decision"] = "skip"
                say("    skipped.")
            else:
                try:
                    review_id, task = approve_and_commit(store, pid, reviewer=reviewer, settings=settings)
                    record.update(review_id=review_id, committed=f"{task.task_id} v{task.version}")
                    say(f"    approved ({review_id}) and committed: {task.task_id} version {task.version}")
                except CommitError as exc:
                    record["error"] = str(exc)
                    say(f"    not committed: {exc}")
        elif choice == "reject":
            note = None if decisions is not None else (ask("    reason (optional): ").strip() or None)
            record["review_id"] = store.add_review(pid, decision="reject", reviewer=reviewer, note=note)
            say(f"    rejected ({record['review_id']}).")
        log.append(record)
        say("")

    say("Task list now:")
    for t in store.current_tasks():
        say(f"  {t.task_id} v{t.version}  {t.description} | {_task_line(t)}")
    return log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Review the proposals of one meeting: approve (and commit) or reject.")
    ap.add_argument("db", type=Path, help="the project's SQLite database (the --db of app.pipeline)")
    ap.add_argument("--meeting", type=int, required=True)
    ap.add_argument("--reviewer", help="who decides; default: the team's coordinator")
    ap.add_argument("--decisions", type=Path, help='JSON like {"P1": "approve", "*": "skip"}; no questions are asked')
    a = ap.parse_args(argv)
    decisions = json.loads(a.decisions.read_text(encoding="utf-8")) if a.decisions else None
    with MemoryStore.open(a.db) as store:
        try:
            review_meeting(store, a.meeting, reviewer=a.reviewer, decisions=decisions)
        except CommitError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except (KeyboardInterrupt, EOFError):
            print("\nstopped; nothing else was changed.", file=sys.stderr)
            return 1
    return 0
