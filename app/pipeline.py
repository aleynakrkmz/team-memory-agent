"""Preview extract → match → route → agent for one meeting.

Clear cases keep the route proposal. Hard cases go through the read-only
agent. The printed JSON has `proposals` (both kinds, in candidate order) and
`agent_decisions` (the step trace). Nothing here is approved, and nothing is
written to the task list. Pass a task-list JSON until the store can supply
the current rows.

    uv run python -m app.pipeline samples/demo --meeting 1
    uv run python -m app.pipeline samples/demo --meeting 3 --tasks tasks.json

Ollama is used only when no client is injected. Set AGENT_MODEL in .env
(qwen3:8b). If extraction fails, the process exits 1 so the meeting
controller can fall back to the B1 rules. This module does not run B1.

Stage lines go to stderr (`extract...`, `agent case 1/2 step 1...`).
Stdout stays the preview JSON.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from app.agent import DemoMemory, MemoryReader, resolve_hard_cases
from app.agent.memory import SimulatedClarification, clarifications_from_file
from app.extract import ExtractionError, LLMClient, extract_candidates, ollama_client_from_env
from app.match import match_candidates
from app.route import route_candidates
from app.schemas import (
    AgentDecision,
    MeetingPreview,
    MeetingTranscript,
    ProposedUpdate,
    RouteKind,
    RoutingDecision,
    TaskRecord,
    TeamProfile,
)
from app.transcript import find_agent_inputs, load_meeting

_TASKS = TypeAdapter(list[TaskRecord])


def run_meeting(
    meeting: MeetingTranscript,
    tasks: list[TaskRecord],
    team: TeamProfile,
    client: LLMClient,
    *,
    agent_inputs: Path | None = None,
    reader: MemoryReader | None = None,
    clarifications: list[SimulatedClarification] | None = None,
) -> MeetingPreview:
    _progress("extract...")
    candidates = extract_candidates(meeting, client)
    _progress("match...")
    matches = match_candidates(candidates, tasks, meeting.meeting_date)
    _progress("route...")
    decisions = route_candidates(meeting, candidates, matches, tasks, team)
    agent_decisions: list[AgentDecision] = []
    if any(decision.route == RouteKind.hard for decision in decisions):
        if reader is None:
            if agent_inputs is None:
                raise RuntimeError("hard cases need agent inputs so the agent can read past meetings")
            reader = DemoMemory.load(
                agent_inputs,
                tasks=tasks,
                team=team,
                current_meeting=meeting,
                clarifications=clarifications,
            )
        agent_decisions = resolve_hard_cases(
            meeting=meeting,
            candidates=candidates,
            matches=matches,
            decisions=decisions,
            tasks=tasks,
            team=team,
            client=client,
            reader=reader,
        )
    return MeetingPreview(
        project_id=meeting.project_id,
        meeting_no=meeting.meeting_no,
        meeting_date=meeting.meeting_date,
        candidates=candidates,
        matches=matches,
        decisions=decisions,
        proposals=_proposals(decisions, agent_decisions),
        agent_decisions=agent_decisions,
    )


def load_tasks(path: Path | None) -> list[TaskRecord]:
    if path is None:
        return []
    return _TASKS.validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None, *, client: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract task changes, route them, and run the agent on hard cases."
    )
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
    parser.add_argument(
        "--clarifications",
        type=Path,
        default=None,
        help="Optional simulated coordinator answers. Questions with no match return 'no answer'.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Also write the preview JSON to this path.",
    )
    args = parser.parse_args(argv)
    try:
        agent_inputs = find_agent_inputs(args.project)
        meeting, team = load_meeting(agent_inputs, args.meeting, transcript_file=args.transcript_file)
        tasks = load_tasks(args.tasks)
        clarifications = clarifications_from_file(args.clarifications) if args.clarifications else None
        if client is None:
            client = ollama_client_from_env()
        if args.tasks is None:
            print("warning: no task list was passed; matching against an empty list", file=sys.stderr)
        preview = run_meeting(
            meeting,
            tasks,
            team,
            client,
            agent_inputs=agent_inputs,
            clarifications=clarifications,
        )
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, RuntimeError) else 1
    payload = preview.model_dump(mode="json")
    if args.out is not None:
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _proposals(decisions: list[RoutingDecision], agent_decisions: list[AgentDecision]) -> list[ProposedUpdate]:
    by_index = {item.candidate_index: item for item in agent_decisions}
    proposals: list[ProposedUpdate] = []
    for decision in decisions:
        if decision.proposal is not None:
            proposals.append(decision.proposal)
            continue
        agent = by_index.get(decision.candidate_index)
        if agent is not None and agent.proposal is not None:
            proposals.append(agent.proposal)
    return proposals


if __name__ == "__main__":
    raise SystemExit(main())
