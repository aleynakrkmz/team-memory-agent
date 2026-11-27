"""Read-only queries the review list needs that `MemoryStore` has no public method for yet.

They only SELECT from the store's tables (schema in app/store/schema.py). If the store gets methods for
these, replace the bodies with calls to them.
"""

from __future__ import annotations

from app.schemas import ProposedUpdate
from app.store.store import MemoryStore


def meeting_proposals(store: MemoryStore, meeting_no: int) -> list[tuple[str, ProposedUpdate]]:
    """Every stored proposal of one meeting, oldest first, as (proposal_id, proposal)."""
    rows = store._con.execute(  # noqa: SLF001 - read-only query, see module docstring
        "SELECT proposal_id, payload_json FROM proposals WHERE meeting_no = ? ORDER BY proposal_no", (meeting_no,)
    ).fetchall()
    return [(r["proposal_id"], ProposedUpdate.model_validate_json(r["payload_json"])) for r in rows]


def pending(store: MemoryStore, meeting_no: int) -> list[tuple[str, ProposedUpdate]]:
    """Proposals of one meeting that have no review yet."""
    reviewed = {r["proposal_id"] for r in store._con.execute("SELECT proposal_id FROM reviews").fetchall()}  # noqa: SLF001
    return [(pid, p) for pid, p in meeting_proposals(store, meeting_no) if pid not in reviewed]


def clarification_ids(store: MemoryStore) -> set[str]:
    return {r[0] for r in store._con.execute("SELECT clarification_id FROM clarification_answers").fetchall()}  # noqa: SLF001


def approval_used(store: MemoryStore, review_id: str) -> str | None:
    """The task version an approval already produced ("T4 v2"), or None."""
    row = store._con.execute(  # noqa: SLF001
        "SELECT task_id, version FROM task_versions WHERE approval_id = ?", (review_id,)
    ).fetchone()
    return f"{row['task_id']} v{row['version']}" if row else None
