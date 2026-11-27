"""The only write path to the task list (design §3.4).

`commit` turns one approved proposal into one new task version:

- the review must exist, be an approval, and belong to this proposal;
- the proposal must still be valid against the current task list (`app.validate`);
- the new version is written in one transaction by `MemoryStore.add_task_version`, which refuses a stale
  base version; the database refuses a second version for the same approval and a version whose approval
  is a rejection.

So nothing reaches the task list without an approval, and one approval makes at most one version.
"""

from __future__ import annotations

from app.schemas import ActionType, ProposedUpdate, TaskRecord, TaskStatus
from app.store.store import MemoryStore, StoreError
from app.validate.validate import Rule2Settings, ValidationResult, validate_meeting


class CommitError(RuntimeError):
    """The proposal could not be committed. Nothing was written to the task list."""


def new_record(proposal: ProposedUpdate, current: TaskRecord | None, canonical_owner: str | None) -> dict:
    """The fields of the next task version: the current task with this proposal's change applied."""
    a = proposal.action
    if a == ActionType.create:
        return {"task_id": None, "base_version": None, "description": proposal.fields.description,
                "owner": canonical_owner, "due_date": proposal.fields.due_date, "status": TaskStatus.open}
    if current is None:
        raise CommitError(f"task {proposal.task_id} is not on the task list")
    row = {"task_id": current.task_id, "base_version": proposal.task_version, "description": current.description,
           "owner": current.owner, "due_date": current.due_date, "status": current.status}
    if a == ActionType.assign:
        row["owner"] = canonical_owner
    elif a == ActionType.set_due_date:
        row["due_date"] = proposal.fields.due_date
    elif a == ActionType.complete:
        row["status"] = TaskStatus.done
    elif a == ActionType.cancel:
        row["status"] = TaskStatus.cancelled
    return row


def revalidate(store: MemoryStore, proposal_id: str, settings: Rule2Settings | None = None) -> ValidationResult:
    """Validate one stored proposal against the task list as it is now (Rule 1 uses the meeting's other proposals)."""
    from app.review.queue import meeting_proposals, clarification_ids  # here to avoid an import cycle

    proposal = store.proposal(proposal_id)
    others = meeting_proposals(store, proposal.meeting_no)
    ids = [pid for pid, _ in others]
    results = validate_meeting([p for _, p in others], meeting=store.meeting(proposal.meeting_no),
                               team=store.team_profile(), tasks=store.current_tasks(),
                               clarification_ids=clarification_ids(store), settings=settings)
    return results[ids.index(proposal_id)]


def commit(store: MemoryStore, proposal_id: str, review_id: str, settings: Rule2Settings | None = None) -> TaskRecord:
    """Write the task version for one approved proposal. Raises CommitError and writes nothing if it cannot."""
    review = store.review(review_id)
    if review is None:
        raise CommitError(f"review {review_id} does not exist")
    if review["proposal_id"] != proposal_id:
        raise CommitError(f"review {review_id} is for proposal {review['proposal_id']}, not {proposal_id}")
    if review["decision"] != "approve":
        raise CommitError(f"review {review_id} is a {review['decision']}, not an approval")
    from app.review.queue import approval_used
    used = approval_used(store, review_id)
    if used:
        raise CommitError(f"approval {review_id} was already used for {used}")

    check = revalidate(store, proposal_id, settings)
    if check.status == "invalid":
        raise CommitError(f"proposal {proposal_id} is invalid now: {check.summary()}")
    # A blocked proposal (Rule 1 or 2) may be committed: the coordinator saw the reason and approved it.

    proposal = check.proposal
    team = store.team_profile()
    current = {t.task_id: t for t in store.current_tasks()}.get(proposal.task_id or "")
    row = new_record(proposal, current, team.canonical_member(proposal.fields.owner))
    try:
        return store.add_task_version(approval_id=review_id, meeting_no=proposal.meeting_no, **row)
    except StoreError as exc:  # stale version, approval used twice, ...
        raise CommitError(str(exc)) from exc
    except Exception as exc:  # sqlite3.IntegrityError from the triggers and UNIQUE constraints
        raise CommitError(f"the store refused the version: {exc}") from exc


def entitled_reviewer(store: MemoryStore, reviewer: str) -> str:
    """Only the team's coordinator may decide on proposals (design: the coordinator approves).
    Returns the coordinator's primary name; raises CommitError for anyone else."""
    team = store.team_profile()
    if team.canonical_member(reviewer) != team.coordinator:
        raise CommitError(f"{reviewer} is not the coordinator of this team ({team.coordinator}); only the coordinator may decide")
    return team.coordinator


def approve_and_commit(store: MemoryStore, proposal_id: str, *, reviewer: str, note: str | None = None,
                       settings: Rule2Settings | None = None) -> tuple[str, TaskRecord]:
    """Record an approval, then commit it. An invalid proposal is refused before the approval is stored."""
    reviewer = entitled_reviewer(store, reviewer)
    check = revalidate(store, proposal_id, settings)
    if check.status == "invalid":
        raise CommitError(f"proposal {proposal_id} cannot be approved: {check.summary()}")
    review_id = store.add_review(proposal_id, decision="approve", reviewer=reviewer, note=note)
    return review_id, commit(store, proposal_id, review_id, settings)
