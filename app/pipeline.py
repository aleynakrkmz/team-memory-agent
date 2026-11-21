"""Preview extract → match → route for one meeting.

This is the first slice of the control loop. It does not validate, it does
not ask for approval, and it does not write the task list. Pass a task-list
JSON until the store can supply the current rows.

    uv run python -m app.pipeline samples/demo --meeting 1

Ollama is used only when no client is injected. Set AGENT_MODEL in .env.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from app.extract import ExtractionError, LLMClient, extract_candidates, ollama_client_from_env
from app.match import match_candidates
from app.route import route_candidates
from app.schemas import MeetingPreview, MeetingTranscript, TaskRecord, TeamProfile
from app.transcript import find_agent_inputs, load_meeting

_TASKS = TypeAdapter(list[TaskRecord])


def run_meeting(
    meeting: MeetingTranscript,
    tasks: list[TaskRecord],
    team: TeamProfile,
    client: LLMClient,
) -> MeetingPreview:
    candidates = extract_candidates(meeting, client)
    matches = match_candidates(candidates, tasks, meeting.meeting_date)
    decisions = route_candidates(meeting, candidates, matches, tasks, team)
    return MeetingPreview(
        project_id=meeting.project_id,
        meeting_no=meeting.meeting_no,
        meeting_date=meeting.meeting_date,
        candidates=candidates,
        matches=matches,
        decisions=decisions,
    )


def load_tasks(path: Path | None) -> list[TaskRecord]:
    if path is None:
        return []
    return _TASKS.validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None, *, client: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract task changes from one meeting and route them.")
    parser.add_argument("project", type=Path, help="samples/demo, or an agent_inputs/<project> directory")
    parser.add_argument("--meeting", type=int, required=True, help="Meeting number, starting at 1")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="JSON list of current TaskRecord rows. Omit only when the list is still empty.",
    )
    parser.add_argument(
        "--transcript-file",
        type=Path,
        default=None,
        help="Replace the meeting file, for example the injected demo transcript.",
    )
    args = parser.parse_args(argv)
    try:
        agent_inputs = find_agent_inputs(args.project)
        meeting, team = load_meeting(agent_inputs, args.meeting, transcript_file=args.transcript_file)
        tasks = load_tasks(args.tasks)
        if client is None:
            client = ollama_client_from_env()
        if args.tasks is None:
            print("warning: no task list was passed; matching against an empty list", file=sys.stderr)
        preview = run_meeting(meeting, tasks, team, client)
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, RuntimeError) else 1
    json.dump(preview.model_dump(mode="json"), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
