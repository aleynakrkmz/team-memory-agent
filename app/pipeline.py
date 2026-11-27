"""Preview extract → match → route → agent for one meeting.

Clear cases keep the route proposal. Hard cases go through the read-only
agent. The printed JSON has `proposals` (both kinds, in candidate order) and
`agent_decisions` (the step trace). Nothing here is approved, and nothing is
written to the task list.

Without `--db` the task list comes from a `--tasks` JSON file and the agent
reads the transcripts on disk. With `--db` the meeting is stored in the
project's append-only SQLite database, the task list is read from it, the
agent reads it through `SqliteMemory`, and cases, proposals and a run record
are stored. `--tasks` with `--db` seeds an empty task list.

    uv run python -m app.pipeline samples/demo --meeting 1
    uv run python -m app.pipeline samples/demo --meeting 3 --tasks tasks.json
    uv run python -m app.pipeline samples/demo --meeting 3 --db data/demo/run1/memory.sqlite

`--system b1` runs baseline B1 instead: fixed rules find the candidates
(`app/baseline/b1.py`, no LLM), match and route are the same, and hard cases
get no proposal because B1 has no agent. Output and storage are the same, so
the agent and B1 can be compared run for run.

    uv run python -m app.pipeline samples/demo --meeting 3 --system b1 --db data/demo/b1/memory.sqlite

Ollama is used only for `--system agent` and only when no client is injected.
Set AGENT_MODEL in .env (qwen3:8b). If extraction fails, the process exits 1
so the meeting controller can fall back to the B1 rules.

Stage lines go to stderr (`extract...`, `agent case 1/2 step 1...`).
Stdout stays the preview JSON.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import TypeAdapter

from app.agent import DemoMemory, MemoryReader, resolve_hard_cases
from app.agent.memory import SimulatedClarification, clarifications_from_file
from app.baseline import extract_b1
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
from app.store import SqliteMemory, StoreError
from app.store.session import (
    default_run_id,
    open_or_create,
    prepare_meeting,
    record_failure,
    record_preview,
)
from app.transcript import find_agent_inputs, load_meeting

_TASKS = TypeAdapter(list[TaskRecord])
SYSTEMS = ("agent", "b1")


def run_meeting(
    meeting: MeetingTranscript,
    tasks: list[TaskRecord],
    team: TeamProfile,
    client: LLMClient | None,
    *,
    agent_inputs: Path | None = None,
    reader: MemoryReader | None = None,
    clarifications: list[SimulatedClarification] | None = None,
    system: str = "agent",
) -> MeetingPreview:
    """`system` is "agent" (LLM extraction, agent on hard cases) or "b1" (rules, no agent)."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; use one of {', '.join(SYSTEMS)}")
    if system == "b1":
        _progress("extract (B1 rules)...")
        candidates = extract_b1(meeting, tasks, team)
    else:
        if client is None:
            raise RuntimeError("the agent system needs a model client")
        _progress("extract...")
        candidates = extract_candidates(meeting, client, tasks=tasks)
    _progress("match...")
    matches = match_candidates(candidates, tasks, meeting.meeting_date)
    _progress("route...")
    decisions = route_candidates(meeting, candidates, matches, tasks, team)
    agent_decisions: list[AgentDecision] = []
    if system == "agent" and any(decision.route == RouteKind.hard for decision in decisions):
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
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Project SQLite database (created if missing), e.g. data/demo/run1/memory.sqlite.",
    )
    parser.add_argument(
        "--system",
        choices=SYSTEMS,
        default="agent",
        help="agent: LLM extraction and the agent (default). b1: baseline B1, fixed rules, no LLM.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id for the stored run record. Default: <project>-m<meeting>-<UTC time>.",
    )
    args = parser.parse_args(argv)
    if args.db is not None:
        return _main_with_store(args, client)
    try:
        agent_inputs = find_agent_inputs(args.project)
        meeting, team = load_meeting(agent_inputs, args.meeting, transcript_file=args.transcript_file)
        tasks = load_tasks(args.tasks)
        clarifications = clarifications_from_file(args.clarifications) if args.clarifications else None
        if client is None and args.system == "agent":
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
            system=args.system,
        )
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, RuntimeError) else 1
    return _emit(preview, args.out)


def _main_with_store(args: argparse.Namespace, client: LLMClient | None) -> int:
    """The same run, with the project database as memory and record."""
    try:
        agent_inputs = find_agent_inputs(args.project)
        meeting, team = load_meeting(agent_inputs, args.meeting, transcript_file=args.transcript_file)
        seed = load_tasks(args.tasks) if args.tasks is not None else None
        clarifications = clarifications_from_file(args.clarifications) if args.clarifications else None
        store = open_or_create(args.db, team.project_id)
    except (StoreError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    run_id = args.run_id or default_run_id(team.project_id, meeting.meeting_no, args.system)
    started = time.monotonic()
    with store:
        try:
            tasks, notes = prepare_meeting(store, agent_inputs=agent_inputs, meeting=meeting, team=team, seed=seed)
        except (StoreError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for note in notes:
            _progress(f"db: {note}")
        _progress(f"db: {len(tasks)} current tasks, run {run_id} ({args.system})")
        try:
            if client is None and args.system == "agent":
                client = ollama_client_from_env()
            with SqliteMemory(args.db, project_id=team.project_id, clarifications=clarifications) as reader:
                preview = run_meeting(meeting, tasks, team, client, reader=reader, system=args.system)
        except ExtractionError as exc:
            record_failure(
                store, run_id=run_id, meeting_no=meeting.meeting_no, stop_reason="extraction_failed",
                detail=str(exc), seconds=time.monotonic() - started, system=args.system,
                raw_output=exc.last_output,
            )
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except (RuntimeError, ValueError, OSError) as exc:
            record_failure(
                store, run_id=run_id, meeting_no=meeting.meeting_no, stop_reason="error",
                detail=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - started, system=args.system,
            )
            print(f"error: {exc}", file=sys.stderr)
            return 2 if isinstance(exc, RuntimeError) else 1
        metrics = record_preview(
            store, preview, run_id=run_id, seconds=time.monotonic() - started, system=args.system
        )
        _progress(
            f"db: stored {len(preview.decisions)} cases, {metrics['proposals']} proposals, "
            f"run record {run_id}"
        )
    return _emit(preview, args.out)


def _emit(preview: MeetingPreview, out: Path | None) -> int:
    payload = preview.model_dump(mode="json")
    if out is not None:
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
