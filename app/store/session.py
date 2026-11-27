"""Connect one pipeline run to the project's SQLite store.

`prepare_meeting` runs before extraction: it stores the team profile and the
transcripts (earlier meetings too, so the agent can search them), and gives
back the current task list. `record_preview` and `record_failure` run after:
they store cases, proposals, clarification questions and a run record.

None of this changes the task list. Only commit, after approval, adds a task
version (design §3.4). The one exception is `seed_tasks` for an empty list,
the reset-to-gold setup step.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.memory import NO_ANSWER
from app.schemas import MeetingPreview, MeetingTranscript, TaskRecord, TeamProfile
from app.store.store import MemoryStore, StoreError
from app.transcript import load_meeting


def open_or_create(path: Path, project_id: str) -> MemoryStore:
    path = Path(path)
    if path.exists():
        return MemoryStore.open(path, project_id=project_id)
    return MemoryStore.create(path, project_id)


def default_run_id(project_id: str, meeting_no: int, system: str = "agent") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{project_id}-{system}" if system != "agent" else project_id
    return f"{prefix}-m{meeting_no}-{stamp}"


def prepare_meeting(
    store: MemoryStore,
    *,
    agent_inputs: Path,
    meeting: MeetingTranscript,
    team: TeamProfile,
    seed: list[TaskRecord] | None = None,
) -> tuple[list[TaskRecord], list[str]]:
    """Store what this meeting needs and return (current tasks, notes for stderr)."""
    notes: list[str] = []
    _store_team(store, team, notes)
    stored = set(store.meeting_numbers())
    for earlier in range(1, meeting.meeting_no):
        if earlier in stored:
            continue
        try:
            past, _ = load_meeting(agent_inputs, earlier)
        except (OSError, ValueError) as exc:
            notes.append(f"meeting {earlier} is not stored and could not be loaded ({exc})")
            continue
        store.add_meeting(past)
        notes.append(f"stored earlier meeting {earlier}")
    if meeting.meeting_no in stored:
        if store.meeting(meeting.meeting_no) != meeting:
            raise StoreError(
                f"meeting {meeting.meeting_no} is already stored with different lines; "
                "use a new database for a changed transcript"
            )
    else:
        store.add_meeting(meeting)
    later = [no for no in stored if no > meeting.meeting_no]
    if later:
        notes.append(f"this database already holds later meetings {later}; the agent cannot see them")
    current = store.current_tasks()
    if seed is not None:
        if current:
            raise StoreError("--tasks was given but the database already has a task list")
        store.seed_tasks(seed, meeting_no=meeting.meeting_no - 1 or None)
        notes.append(f"seeded {len(seed)} tasks from --tasks")
        current = store.current_tasks()
    return current, notes


def record_preview(
    store: MemoryStore,
    preview: MeetingPreview,
    *,
    run_id: str,
    seconds: float,
    system: str = "agent",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store cases, proposals and clarification questions, then a run record.

    The run record's metrics name the system (agent or b1) and keep every
    extracted candidate, including the ones that did not become a proposal.
    """
    case_ids: dict[int, str] = {}
    for decision in preview.decisions:
        case_ids[decision.candidate_index] = store.add_case(
            preview.meeting_no, decision.candidate_index, decision.route, decision.reasons
        )
    proposal_ids = [
        store.add_proposal(proposal, case_id=case_ids.get(proposal.candidate_index))
        for proposal in preview.proposals
    ]
    questions = 0
    for agent in preview.agent_decisions:
        for step in agent.trace.steps:
            if step.tool != "ask_clarification" or not step.valid:
                continue
            question_id = store.add_clarification_question(
                case_ids[agent.candidate_index],
                str(step.arguments.get("question", "")).strip() or "(empty question)",
                [str(option) for option in step.arguments.get("options", []) or []],
            )
            questions += 1
            answer = _answer(step.observation)
            if answer and answer.casefold() != NO_ANSWER:
                store.add_clarification_answer(question_id, answered_by="simulated coordinator", answer=answer)
    metrics = {
        "system": system,
        "seconds": round(seconds, 2),
        "candidates": len(preview.candidates),
        "hard_cases": len(preview.agent_decisions),
        "proposals": len(preview.proposals),
        "agent_steps": sum(len(item.trace.steps) for item in preview.agent_decisions),
        "agent_outcomes": [item.outcome.value for item in preview.agent_decisions],
        "automatic_escalations": sum(1 for item in preview.agent_decisions if item.automatic),
        "clarification_questions": questions,
        "proposal_ids": proposal_ids,
        "candidate_changes": [candidate.model_dump(mode="json") for candidate in preview.candidates],
    }
    if extra:
        metrics.update(extra)
    store.add_run_record(
        run_id=run_id,
        meeting_no=preview.meeting_no,
        stop_reason="completed",
        metrics=metrics,
        trace=[item.trace.model_dump(mode="json") for item in preview.agent_decisions],
    )
    return metrics


def record_failure(
    store: MemoryStore,
    *,
    run_id: str,
    meeting_no: int,
    stop_reason: str,
    detail: str,
    seconds: float,
    system: str = "agent",
    raw_output: str | None = None,
) -> None:
    """Store a failed run. `raw_output` keeps the model's last rejected reply, if any."""
    metrics: dict[str, Any] = {"system": system, "seconds": round(seconds, 2)}
    if raw_output:
        metrics["last_model_output"] = raw_output
    store.add_run_record(
        run_id=run_id,
        meeting_no=meeting_no,
        stop_reason=stop_reason,
        failure_detail=detail,
        metrics=metrics,
    )


def _store_team(store: MemoryStore, team: TeamProfile, notes: list[str]) -> None:
    try:
        current = store.team_profile()
    except StoreError:
        current = None
    if current != team:
        store.add_team_profile(team)
        notes.append("stored the team profile" if current is None else "stored a changed team profile")


def _answer(observation: str) -> str | None:
    try:
        data = json.loads(observation)
    except (TypeError, ValueError):
        return None
    value = data.get("answer") if isinstance(data, dict) else None
    return str(value).strip() if value is not None else None
