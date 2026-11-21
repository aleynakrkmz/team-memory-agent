"""Load a ready transcript in the demo text format.

Aleyna's ingest module will own this for audio and for storage. Until
then, extract and the preview CLI read `Speaker: text` files. Line `n`
becomes utterance id `Ln`, which matches the 1-based line numbers in the
demo evidence files. Answer keys are not read here.
"""

import json
import re
from datetime import date
from pathlib import Path

from app.schemas import MeetingTranscript, TeamProfile, Utterance

_LINE = re.compile(r"^([^:]{1,80}):\s*(.+)$")


def parse_speaker_transcript(text: str) -> list[Utterance]:
    """Parse `Name: text` lines. Every non-empty physical line must match."""
    utterances: list[Utterance] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            raise ValueError(f"line {line_no}: empty line")
        match = _LINE.match(raw.strip())
        if not match:
            raise ValueError(f"line {line_no}: expected 'Speaker: text'")
        utterances.append(
            Utterance(id=f"L{line_no}", speaker=match.group(1).strip(), text=match.group(2).strip())
        )
    if not utterances:
        raise ValueError("transcript is empty")
    return utterances


def load_team(path: Path) -> TeamProfile:
    return TeamProfile.model_validate_json(path.read_text(encoding="utf-8"))


def load_meeting(
    agent_inputs: Path,
    meeting_no: int,
    *,
    transcript_file: Path | None = None,
) -> tuple[MeetingTranscript, TeamProfile]:
    """Read team.json, meetings.json and one transcript. Does not read answer keys."""
    if meeting_no < 1:
        raise ValueError("meeting number starts at 1")
    meta = json.loads((agent_inputs / "meetings.json").read_text(encoding="utf-8"))
    dates: list[str] = meta["meeting_dates"]
    if meeting_no > len(dates):
        raise ValueError(f"meeting {meeting_no} is outside this project ({len(dates)} meetings)")
    path = transcript_file or (agent_inputs / f"meeting_{meeting_no:02d}.transcript.txt")
    meeting = MeetingTranscript(
        project_id=meta["project_id"],
        meeting_no=meeting_no,
        meeting_date=date.fromisoformat(dates[meeting_no - 1]),
        utterances=parse_speaker_transcript(path.read_text(encoding="utf-8")),
    )
    return meeting, load_team(agent_inputs / "team.json")


def find_agent_inputs(project: Path) -> Path:
    """Accept either `samples/demo` or the `agent_inputs/<id>` directory itself."""
    project = project.resolve()
    if (project / "team.json").is_file() and (project / "meetings.json").is_file():
        return project
    nested = project / "agent_inputs"
    if nested.is_dir():
        found = sorted(
            path for path in nested.iterdir() if (path / "team.json").is_file() and (path / "meetings.json").is_file()
        )
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise FileNotFoundError(f"{project} has several projects under agent_inputs; pass one of them")
    raise FileNotFoundError(f"no team.json and meetings.json under {project}")
