"""Read-only queries for the review list and commit.

Thin wrappers over `MemoryStore` methods, kept so the review list and commit
call one place. Nothing here writes.
"""

from __future__ import annotations

from app.schemas import ProposedUpdate
from app.store.store import MemoryStore


def meeting_proposals(store: MemoryStore, meeting_no: int) -> list[tuple[str, ProposedUpdate]]:
    """Every stored proposal of one meeting, oldest first, as (proposal_id, proposal)."""
    return store.meeting_proposals(meeting_no)


def pending(store: MemoryStore, meeting_no: int) -> list[tuple[str, ProposedUpdate]]:
    """Proposals of one meeting that have no review yet."""
    return store.pending_proposals(meeting_no)


def clarification_ids(store: MemoryStore) -> set[str]:
    return store.clarification_ids()


def approval_used(store: MemoryStore, review_id: str) -> str | None:
    """The task version an approval already produced ("T4 v2"), or None."""
    record = store.version_for_approval(review_id)
    return f"{record.task_id} v{record.version}" if record is not None else None
