"""The agent's read-only view of the SQLite store.

`SqliteMemory` implements `app.agent.memory.MemoryReader`, so it can replace
`DemoMemory` without changing the agent. It opens the database file with
`mode=ro`: SQLite itself refuses any write through this connection.
"""

from pathlib import Path

from rank_bm25 import BM25Okapi

from app.agent.memory import NO_ANSWER, MeetingHit, SimulatedClarification, TaskVersionView
from app.match.match import tokenize
from app.schemas import TaskRecord, TeamProfile
from app.store.store import MemoryStore


class SqliteMemory:
    """Past meetings, task histories and the team, read from one project database."""

    def __init__(
        self,
        path: Path,
        *,
        project_id: str | None = None,
        clarifications: list[SimulatedClarification] | None = None,
    ):
        self._store = MemoryStore.open(path, project_id=project_id, read_only=True)
        self._clarifications = list(clarifications or [])

    @property
    def project_id(self) -> str:
        return self._store.project_id

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "SqliteMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def current_tasks(self) -> list[TaskRecord]:
        """The task list the pipeline matches against (replaces a --tasks file)."""
        return self._store.current_tasks()

    def search_meetings(self, query: str, *, current_meeting_no: int, max_results: int) -> list[MeetingHit]:
        """BM25 over lines from this meeting and earlier ones. Later meetings are invisible."""
        pool = self._store.utterances_up_to(current_meeting_no)
        tokens = tokenize(query)
        if not pool or not tokens or max_results < 1:
            return []
        documents = [tokenize(f"{u.speaker} {u.text}") for _, u in pool]
        if not any(documents):
            return []
        scores = [float(score) for score in BM25Okapi(documents).get_scores(tokens)]
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], -pool[i][0], pool[i][1].id))
        hits: list[MeetingHit] = []
        for position in order:
            if scores[position] <= 0:
                break
            meeting_no, utterance = pool[position]
            hits.append(
                MeetingHit(
                    meeting_no=meeting_no,
                    utterance_id=utterance.id,
                    ref=f"M{meeting_no}-{utterance.id}",
                    speaker=utterance.speaker,
                    text=utterance.text,
                    score=round(scores[position], 4),
                )
            )
            if len(hits) >= max_results:
                break
        return hits

    def get_task_history(self, task_id: str) -> list[TaskVersionView]:
        views = []
        for item in self._store.task_history(task_id):
            record: TaskRecord = item["record"]
            views.append(
                TaskVersionView(
                    task_id=record.task_id,
                    version=record.version,
                    description=record.description,
                    owner=record.owner,
                    due_date=record.due_date,
                    status=record.status,
                    evidence_refs=item["evidence_refs"],
                )
            )
        return views

    def get_team_profile(self) -> TeamProfile:
        return self._store.team_profile()

    def simulated_answer(self, question: str, *, meeting_no: int) -> str:
        """Same matching rule as DemoMemory: first scripted answer whose text is in the question."""
        folded = question.casefold()
        for item in self._clarifications:
            if item.meeting_no not in (None, meeting_no):
                continue
            needle = item.contains.casefold().strip()
            if needle and needle in folded:
                return item.answer.strip() or NO_ANSWER
        return NO_ANSWER
