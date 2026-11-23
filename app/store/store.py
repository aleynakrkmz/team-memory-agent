"""Append-only SQLite memory for one project. Design §3.4 and §5.

`MemoryStore.create` makes a new database file; `MemoryStore.open` opens an
existing one. Each write method stands for one writer in design §3.4. The
store keeps the history honest (append-only rows, one project per file, a
commit version tied to an approve review, no forked versions). Deciding
whether a proposal may be committed (same proposal, same task version,
used once) stays in the commit step.

The agent never gets a `MemoryStore`. It reads through `SqliteMemory`,
which opens the file read-only.
"""

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from app.schemas import (
    EvidenceSource,
    MeetingTranscript,
    ProposedUpdate,
    RouteKind,
    TaskRecord,
    TaskStatus,
    TeamMember,
    TeamProfile,
    Utterance,
)
from app.store.schema import SCHEMA_SQL, SCHEMA_VERSION


class StoreError(RuntimeError):
    """A write that would break the store's rules."""


def _connect(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        con = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    # Without this, INSERT OR REPLACE deletes rows without firing DELETE triggers.
    con.execute("PRAGMA recursive_triggers = ON")
    return con


class MemoryStore:
    """One project's database. Use as a context manager or call `close()`."""

    def __init__(self, con: sqlite3.Connection, project_id: str, *, read_only: bool):
        self._con = con
        self.project_id = project_id
        self.read_only = read_only

    # ----- opening -------------------------------------------------------

    @classmethod
    def create(cls, path: Path, project_id: str) -> "MemoryStore":
        """Make a new, empty database. Refuses to touch an existing file."""
        path = Path(path)
        if path.exists():
            raise StoreError(f"{path} already exists; open it instead of creating it again")
        if not project_id:
            raise StoreError("project_id is required")
        path.parent.mkdir(parents=True, exist_ok=True)
        con = _connect(path, read_only=False)
        try:
            con.executescript(SCHEMA_SQL)
            with con:
                con.execute(
                    "INSERT INTO project (project_id, schema_version) VALUES (?, ?)",
                    (project_id, SCHEMA_VERSION),
                )
        except Exception:
            con.close()
            path.unlink(missing_ok=True)
            raise
        return cls(con, project_id, read_only=False)

    @classmethod
    def open(cls, path: Path, *, project_id: str | None = None, read_only: bool = False) -> "MemoryStore":
        """Open an existing database. Pass `project_id` to check it is the right project."""
        path = Path(path)
        if not path.is_file():
            raise StoreError(f"{path} does not exist")
        con = _connect(path, read_only=read_only)
        row = con.execute("SELECT project_id, schema_version FROM project").fetchone()
        if row is None:
            con.close()
            raise StoreError(f"{path} has no project row")
        if row["schema_version"] != SCHEMA_VERSION:
            con.close()
            raise StoreError(f"{path} has schema version {row['schema_version']}, expected {SCHEMA_VERSION}")
        if project_id is not None and row["project_id"] != project_id:
            con.close()
            raise StoreError(f"{path} belongs to project {row['project_id']!r}, not {project_id!r}")
        return cls(con, row["project_id"], read_only=read_only)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- helpers -------------------------------------------------------

    def _check_project(self, project_id: str) -> None:
        if project_id != self.project_id:
            raise StoreError(f"record is for project {project_id!r}; this store is {self.project_id!r}")

    def _writable(self) -> None:
        if self.read_only:
            raise StoreError("this store was opened read-only")

    def _next_id(self, table: str, prefix: str, width: int = 0) -> str:
        # Rows are never deleted, so the row count gives the next free number.
        count = self._con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return f"{prefix}{count + 1:0{width}d}"

    # ----- team profile (data generator, WS3 lead, or coordinator) ---------

    def add_team_profile(self, team: TeamProfile) -> None:
        """Store a team profile. The newest row is the current profile."""
        self._writable()
        self._check_project(team.project_id)
        members = [member.model_dump(mode="json") for member in team.members]
        with self._con:
            self._con.execute(
                "INSERT INTO team_profiles (coordinator, members_json) VALUES (?, ?)",
                (team.coordinator, json.dumps(members, ensure_ascii=False)),
            )

    def team_profile(self) -> TeamProfile:
        row = self._con.execute(
            "SELECT coordinator, members_json FROM team_profiles ORDER BY profile_no DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise StoreError("no team profile has been stored")
        members = [TeamMember.model_validate(item) for item in json.loads(row["members_json"])]
        return TeamProfile(project_id=self.project_id, coordinator=row["coordinator"], members=members)

    # ----- meetings (ingest) ---------------------------------------------

    def add_meeting(self, meeting: MeetingTranscript) -> None:
        """Store a meeting and all its lines in one transaction."""
        self._writable()
        self._check_project(meeting.project_id)
        with self._con:
            self._con.execute(
                "INSERT INTO meetings (meeting_no, meeting_date) VALUES (?, ?)",
                (meeting.meeting_no, meeting.meeting_date.isoformat()),
            )
            self._con.executemany(
                "INSERT INTO utterances (meeting_no, utterance_id, position, speaker, text, start_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (meeting.meeting_no, u.id, position, u.speaker, u.text, u.start_ms)
                    for position, u in enumerate(meeting.utterances)
                ],
            )

    def meeting_numbers(self) -> list[int]:
        return [row[0] for row in self._con.execute("SELECT meeting_no FROM meetings ORDER BY meeting_no")]

    def meeting(self, meeting_no: int) -> MeetingTranscript:
        row = self._con.execute(
            "SELECT meeting_date FROM meetings WHERE meeting_no = ?", (meeting_no,)
        ).fetchone()
        if row is None:
            raise StoreError(f"meeting {meeting_no} is not stored")
        lines = self._con.execute(
            "SELECT utterance_id, speaker, text, start_ms FROM utterances "
            "WHERE meeting_no = ? ORDER BY position",
            (meeting_no,),
        ).fetchall()
        return MeetingTranscript(
            project_id=self.project_id,
            meeting_no=meeting_no,
            meeting_date=date.fromisoformat(row["meeting_date"]),
            utterances=[
                Utterance(id=line["utterance_id"], speaker=line["speaker"], text=line["text"], start_ms=line["start_ms"])
                for line in lines
            ],
        )

    def utterances_up_to(self, meeting_no: int) -> list[tuple[int, Utterance]]:
        """Lines from this meeting and earlier ones, in meeting and line order."""
        rows = self._con.execute(
            "SELECT meeting_no, utterance_id, speaker, text, start_ms FROM utterances "
            "WHERE meeting_no <= ? ORDER BY meeting_no, position",
            (meeting_no,),
        ).fetchall()
        return [
            (
                row["meeting_no"],
                Utterance(id=row["utterance_id"], speaker=row["speaker"], text=row["text"], start_ms=row["start_ms"]),
            )
            for row in rows
        ]

    # ----- cases, proposals, questions (controller) ----------------------

    def add_case(self, meeting_no: int, candidate_index: int, route: RouteKind, reasons: list[str]) -> str:
        """Record one task-change case. Returns its id, for example M3-K0."""
        self._writable()
        case_id = f"M{meeting_no}-K{candidate_index}"
        with self._con:
            self._con.execute(
                "INSERT INTO cases (case_id, meeting_no, candidate_index, route, reasons_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (case_id, meeting_no, candidate_index, RouteKind(route).value, json.dumps(list(reasons))),
            )
        return case_id

    def add_proposal(self, proposal: ProposedUpdate, *, case_id: str | None = None) -> str:
        """Record a proposal. This is a log entry, not a change to the task list."""
        self._writable()
        self._check_project(proposal.project_id)
        proposal_id = self._next_id("proposals", "P")
        with self._con:
            self._con.execute(
                "INSERT INTO proposals (proposal_id, case_id, meeting_no, candidate_index, action, "
                "task_id, task_version, origin, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    proposal_id,
                    case_id,
                    proposal.meeting_no,
                    proposal.candidate_index,
                    proposal.action.value,
                    proposal.task_id,
                    proposal.task_version,
                    proposal.origin.value,
                    proposal.model_dump_json(),
                ),
            )
        return proposal_id

    def proposal(self, proposal_id: str) -> ProposedUpdate:
        row = self._con.execute(
            "SELECT payload_json FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        if row is None:
            raise StoreError(f"proposal {proposal_id} is not stored")
        return ProposedUpdate.model_validate_json(row["payload_json"])

    def add_clarification_question(self, case_id: str, question: str, options: list[str] | None = None) -> str:
        """Record a question the agent handed to the controller. Returns Q1, Q2, ..."""
        self._writable()
        question_id = self._next_id("clarification_questions", "Q")
        with self._con:
            self._con.execute(
                "INSERT INTO clarification_questions (question_id, case_id, question, options_json) "
                "VALUES (?, ?, ?, ?)",
                (question_id, case_id, question, json.dumps(list(options or []), ensure_ascii=False)),
            )
        return question_id

    # ----- answers and reviews (review list) -----------------------------

    def add_clarification_answer(self, question_id: str, *, answered_by: str, answer: str) -> str:
        """Record the coordinator's answer. Returns C01, C02, ... (an evidence id)."""
        self._writable()
        clarification_id = self._next_id("clarification_answers", "C", width=2)
        with self._con:
            self._con.execute(
                "INSERT INTO clarification_answers (clarification_id, question_id, answered_by, answer) "
                "VALUES (?, ?, ?, ?)",
                (clarification_id, question_id, answered_by, answer),
            )
        return clarification_id

    def clarification_answer(self, clarification_id: str) -> dict[str, Any] | None:
        row = self._con.execute(
            "SELECT a.clarification_id, a.question_id, q.case_id, a.answered_by, a.answer, a.created_at "
            "FROM clarification_answers a JOIN clarification_questions q USING (question_id) "
            "WHERE a.clarification_id = ?",
            (clarification_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def add_review(self, proposal_id: str, *, decision: str, reviewer: str, note: str | None = None) -> str:
        """Record an approval or rejection. Returns R1, R2, ..."""
        self._writable()
        if decision not in ("approve", "reject"):
            raise StoreError("decision must be 'approve' or 'reject'")
        review_id = self._next_id("reviews", "R")
        with self._con:
            self._con.execute(
                "INSERT INTO reviews (review_id, proposal_id, decision, reviewer, note) VALUES (?, ?, ?, ?, ?)",
                (review_id, proposal_id, decision, reviewer, note),
            )
        return review_id

    def review(self, review_id: str) -> dict[str, Any] | None:
        row = self._con.execute(
            "SELECT review_id, proposal_id, decision, reviewer, note, created_at FROM reviews WHERE review_id = ?",
            (review_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    # ----- task versions (commit only, or reset-to-gold setup) -----------

    def add_task_version(
        self,
        *,
        approval_id: str,
        task_id: str | None,
        base_version: int | None,
        description: str,
        owner: str | None,
        due_date: date | None,
        status: TaskStatus,
        meeting_no: int,
    ) -> TaskRecord:
        """Insert the next version of a task. Only the commit step calls this.

        Create: `task_id` and `base_version` are None; the store gives the next
        id (T1, T2, ...) and version 1. Update: `base_version` must be the
        task's current version, so two changes cannot fork the history.
        """
        self._writable()
        with self._con:
            if task_id is None:
                if base_version is not None:
                    raise StoreError("a new task has no base_version")
                task_no = self._con.execute("SELECT COALESCE(MAX(task_no), 0) + 1 FROM task_versions").fetchone()[0]
                task_id, version = f"T{task_no}", 1
            else:
                current = self._current_row(task_id)
                if current is None:
                    raise StoreError(f"task {task_id} does not exist")
                if base_version != current["version"]:
                    raise StoreError(
                        f"task {task_id} is at version {current['version']}, not {base_version}"
                    )
                task_no, version = current["task_no"], current["version"] + 1
            self._insert_version(
                task_no, task_id, version, description, owner, due_date, TaskStatus(status), "commit", approval_id, meeting_no
            )
        return TaskRecord(
            task_id=task_id, description=description, owner=owner, due_date=due_date, status=status, version=version
        )

    def seed_tasks(self, tasks: list[TaskRecord], *, meeting_no: int | None = None) -> None:
        """Load a starting task state into an empty task list.

        This is the reset-to-gold setup step (design §6.3): an evaluation setup
        operation, not an agent change. It only works before any version exists.
        """
        self._writable()
        with self._con:
            if self._con.execute("SELECT COUNT(*) FROM task_versions").fetchone()[0]:
                raise StoreError("seed_tasks only works on an empty task list")
            seen: set[str] = set()
            for task in tasks:
                if task.task_id in seen:
                    raise StoreError(f"task {task.task_id} appears twice in the seed")
                seen.add(task.task_id)
                task_no = _task_number(task.task_id)
                self._insert_version(
                    task_no, task.task_id, task.version, task.description, task.owner,
                    task.due_date, task.status, "seed", None, meeting_no,
                )

    def _insert_version(
        self,
        task_no: int,
        task_id: str,
        version: int,
        description: str,
        owner: str | None,
        due_date: date | None,
        status: TaskStatus,
        source: str,
        approval_id: str | None,
        meeting_no: int | None,
    ) -> None:
        self._con.execute(
            "INSERT INTO task_versions (task_no, task_id, version, description, owner, due_date, status, "
            "source, approval_id, meeting_no) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_no, task_id, version, description, owner,
                due_date.isoformat() if due_date else None,
                status.value, source, approval_id, meeting_no,
            ),
        )

    def _current_row(self, task_id: str) -> sqlite3.Row | None:
        return self._con.execute(
            "SELECT * FROM task_versions WHERE task_id = ? ORDER BY version DESC LIMIT 1", (task_id,)
        ).fetchone()

    def current_tasks(self) -> list[TaskRecord]:
        """The task list: the latest version of every task, in task order."""
        rows = self._con.execute(
            "SELECT tv.* FROM task_versions tv "
            "JOIN (SELECT task_id, MAX(version) AS version FROM task_versions GROUP BY task_id) latest "
            "USING (task_id, version) ORDER BY tv.task_no"
        ).fetchall()
        return [_task_record(row) for row in rows]

    def task_history(self, task_id: str) -> list[dict[str, Any]]:
        """Every stored version of one task, oldest first, with its evidence refs.

        Transcript refs are meeting-qualified (M2-L5); clarification refs keep
        their id (C01). Seeded versions have no evidence.
        """
        rows = self._con.execute(
            "SELECT tv.*, p.payload_json FROM task_versions tv "
            "LEFT JOIN reviews r ON r.review_id = tv.approval_id "
            "LEFT JOIN proposals p ON p.proposal_id = r.proposal_id "
            "WHERE tv.task_id = ? ORDER BY tv.version",
            (task_id,),
        ).fetchall()
        history = []
        for row in rows:
            refs: list[str] = []
            if row["payload_json"]:
                proposal = ProposedUpdate.model_validate_json(row["payload_json"])
                for ref in proposal.evidence_refs:
                    if ref.source == EvidenceSource.transcript:
                        refs.append(f"M{proposal.meeting_no}-{ref.id}")
                    else:
                        refs.append(ref.id)
            history.append(
                {
                    "record": _task_record(row),
                    "source": row["source"],
                    "approval_id": row["approval_id"],
                    "meeting_no": row["meeting_no"],
                    "evidence_refs": refs,
                }
            )
        return history

    # ----- run records (controller) --------------------------------------

    def add_run_record(
        self,
        *,
        run_id: str,
        meeting_no: int,
        stop_reason: str,
        failure_detail: str | None = None,
        metrics: dict[str, Any] | None = None,
        trace: Any = None,
    ) -> None:
        self._writable()
        with self._con:
            self._con.execute(
                "INSERT INTO run_records (run_id, meeting_no, stop_reason, failure_detail, metrics_json, trace_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    meeting_no,
                    stop_reason,
                    failure_detail,
                    json.dumps(metrics or {}, ensure_ascii=False),
                    json.dumps(trace, ensure_ascii=False) if trace is not None else None,
                ),
            )

    def run_records(self, run_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM run_records"
        params: tuple[Any, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params = (run_id,)
        rows = self._con.execute(sql + " ORDER BY record_no", params).fetchall()
        return [dict(row) for row in rows]


def _task_number(task_id: str) -> int:
    if not task_id.startswith("T") or not task_id[1:].isdigit() or int(task_id[1:]) < 1:
        raise StoreError(f"task id {task_id!r} must look like T1, T2, ...")
    return int(task_id[1:])


def _task_record(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        task_id=row["task_id"],
        description=row["description"],
        owner=row["owner"],
        due_date=date.fromisoformat(row["due_date"]) if row["due_date"] else None,
        status=TaskStatus(row["status"]),
        version=row["version"],
    )
