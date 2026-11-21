"""Read-only view of past meetings, tasks, and the team.

`MemoryReader` is the seam Aleyna's append-only store can implement later.
`DemoMemory` reads the transcripts and the task rows the caller already has.
It does not open answer keys and it does not write anything.

Simulated clarification answers are supplied by the caller, or by an optional
`clarifications.json` next to `team.json`. A question with no match returns
the exact text "no answer". The agent then abstains or escalates.
"""

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import Field
from rank_bm25 import BM25Okapi

from app.match.match import tokenize
from app.schemas import ApiModel, MeetingTranscript, TaskRecord, TaskStatus, TeamProfile, Utterance
from app.transcript import parse_speaker_transcript

NO_ANSWER = "no answer"
_TRANSCRIPT_FILE = re.compile(r"^meeting_(\d+)\.transcript\.txt$")


@dataclass(frozen=True)
class SimulatedClarification:
    """One scripted coordinator reply. `contains` is matched inside the question."""

    contains: str
    answer: str
    meeting_no: int | None = None


class MeetingHit(ApiModel):
    """One transcript line retrieved by BM25."""

    meeting_no: int = Field(ge=1)
    utterance_id: str
    ref: str = Field(description="Meeting-qualified id, for example M2-L5.")
    speaker: str
    text: str
    score: float


class TaskVersionView(ApiModel):
    """One known version of a task. The demo reader has only the current row."""

    task_id: str
    version: int = Field(ge=1)
    description: str
    owner: str | None = None
    due_date: date | None = None
    status: TaskStatus
    evidence_refs: list[str] = Field(default_factory=list)


@runtime_checkable
class MemoryReader(Protocol):
    """Read-only memory. Implementations must not write the task list."""

    def search_meetings(self, query: str, *, current_meeting_no: int, max_results: int) -> list[MeetingHit]:
        """Lines from this meeting and earlier ones. Later meetings are invisible."""

    def get_task_history(self, task_id: str) -> list[TaskVersionView]:
        """Known versions of one task. Empty when the id is unknown."""

    def get_team_profile(self) -> TeamProfile:
        """Names, aliases, roles, and the coordinator."""

    def simulated_answer(self, question: str, *, meeting_no: int) -> str:
        """A scripted coordinator reply, or the exact text 'no answer'."""


@dataclass(frozen=True)
class _Segment:
    meeting_no: int
    utterance_id: str
    speaker: str
    text: str

    @property
    def ref(self) -> str:
        return f"M{self.meeting_no}-{self.utterance_id}"


class DemoMemory:
    """Transcripts on disk plus the current task rows passed in by the caller.

    Task rows are the list the pipeline already loaded. Until the store keeps
    every version, history is that current row, and evidence is the lines whose
    text contains the task title. Swap this class for the SQLite reader later;
    the agent calls `MemoryReader` only.
    """

    def __init__(
        self,
        *,
        segments: list[_Segment],
        tasks: list[TaskRecord],
        team: TeamProfile,
        clarifications: list[SimulatedClarification],
        as_of: int | None,
    ):
        self._segments = list(segments)
        self._tasks = {task.task_id: task for task in tasks}
        self._team = team
        self._clarifications = list(clarifications)
        self._as_of = as_of

    @classmethod
    def load(
        cls,
        agent_inputs: Path,
        *,
        tasks: list[TaskRecord],
        team: TeamProfile,
        current_meeting: MeetingTranscript | None = None,
        clarifications: list[SimulatedClarification] | None = None,
    ) -> "DemoMemory":
        by_meeting: dict[int, list[Utterance]] = {}
        for path in sorted(agent_inputs.iterdir()):
            match = _TRANSCRIPT_FILE.match(path.name)
            if match is None or not path.is_file():
                continue
            by_meeting[int(match.group(1))] = parse_speaker_transcript(path.read_text(encoding="utf-8"))
        if current_meeting is not None:
            by_meeting[current_meeting.meeting_no] = list(current_meeting.utterances)
        segments: list[_Segment] = []
        for meeting_no in sorted(by_meeting):
            for utterance in by_meeting[meeting_no]:
                segments.append(
                    _Segment(
                        meeting_no=meeting_no,
                        utterance_id=utterance.id,
                        speaker=utterance.speaker,
                        text=utterance.text,
                    )
                )
        if clarifications is None:
            sidecar = agent_inputs / "clarifications.json"
            clarifications = clarifications_from_file(sidecar) if sidecar.is_file() else []
        as_of = current_meeting.meeting_no if current_meeting is not None else None
        return cls(
            segments=segments,
            tasks=list(tasks),
            team=team,
            clarifications=list(clarifications),
            as_of=as_of,
        )

    def search_meetings(self, query: str, *, current_meeting_no: int, max_results: int) -> list[MeetingHit]:
        pool = [segment for segment in self._segments if segment.meeting_no <= current_meeting_no]
        tokens = tokenize(query)
        if not pool or not tokens or max_results < 1:
            return []
        documents = [tokenize(f"{segment.speaker} {segment.text}") for segment in pool]
        if not any(documents):
            return []
        index = BM25Okapi(documents)
        scores = [float(score) for score in index.get_scores(tokens)]
        order = sorted(
            range(len(pool)),
            key=lambda i: (-scores[i], -pool[i].meeting_no, pool[i].utterance_id),
        )
        hits: list[MeetingHit] = []
        for position in order:
            if scores[position] <= 0:
                break
            segment = pool[position]
            hits.append(
                MeetingHit(
                    meeting_no=segment.meeting_no,
                    utterance_id=segment.utterance_id,
                    ref=segment.ref,
                    speaker=segment.speaker,
                    text=segment.text,
                    score=round(scores[position], 4),
                )
            )
            if len(hits) >= max_results:
                break
        return hits

    def get_task_history(self, task_id: str) -> list[TaskVersionView]:
        task = self._tasks.get(task_id)
        if task is None:
            return []
        needle = task.description.casefold()
        refs: list[str] = []
        for segment in self._segments:
            if self._as_of is not None and segment.meeting_no > self._as_of:
                continue
            if needle and needle in segment.text.casefold():
                refs.append(segment.ref)
        return [
            TaskVersionView(
                task_id=task.task_id,
                version=task.version,
                description=task.description,
                owner=task.owner,
                due_date=task.due_date,
                status=task.status,
                evidence_refs=refs,
            )
        ]

    def get_team_profile(self) -> TeamProfile:
        return self._team

    def simulated_answer(self, question: str, *, meeting_no: int) -> str:
        folded = question.casefold()
        for item in self._clarifications:
            if item.meeting_no not in (None, meeting_no):
                continue
            needle = item.contains.casefold().strip()
            if not needle or needle not in folded:
                continue
            answer = item.answer.strip()
            return answer or NO_ANSWER
        return NO_ANSWER


def clarifications_from_file(path: Path) -> list[SimulatedClarification]:
    """Read `{"answers": [{"meeting_no", "contains", "answer"}]}`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("answers") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("clarifications file needs an answers list")
    items: list[SimulatedClarification] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each clarification must be an object")
        meeting_no = row.get("meeting_no")
        items.append(
            SimulatedClarification(
                contains=str(row["contains"]),
                answer=str(row["answer"]),
                meeting_no=int(meeting_no) if meeting_no is not None else None,
            )
        )
    return items
