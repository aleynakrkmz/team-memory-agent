"""The only tools the agent can call. None of them write the task list.

propose_update builds a `ProposedUpdate` for validate and review. abstain and
escalate end the case. The other four tools only read `MemoryReader`.
"""

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.memory import NO_ANSWER, MemoryReader
from app.route import POSSIBLE_DUPLICATE, TASK_AMBIGUOUS, TASK_UNCLEAR
from app.schemas import (
    ActionType,
    AgentOutcome,
    AgentStep,
    CandidateChange,
    EvidenceQuote,
    EvidenceRef,
    EvidenceSource,
    MatchResult,
    MeetingTranscript,
    ProposalOrigin,
    ProposedUpdate,
    TaskRecord,
    TeamProfile,
    UpdateFields,
    Utterance,
)

# The router could not name one task. A proposal must follow a lookup.
_IDENTITY_REASONS = frozenset({TASK_UNCLEAR, TASK_AMBIGUOUS, POSSIBLE_DUPLICATE})
_LINE_ID = re.compile(r"^L[1-9][0-9]*$")
_ASSISTANT = re.compile(r"\bassistant\b", re.IGNORECASE)

TOOL_NAMES = (
    "search_meetings",
    "get_task_history",
    "get_team_profile",
    "ask_clarification",
    "propose_update",
    "abstain",
    "escalate",
)

# Ollama turns this schema into a grammar for the reply. `additionalProperties`
# must stay False: with open keys, qwen3 sometimes closes a key with a curly
# quote (`"task_id”: ...`), the grammar accepts it as part of the key, and the
# reply never ends until the timeout or Ollama's repeat limit. With a fixed key
# list the grammar forces a plain closing quote. `arguments` is left out for
# the same reason (an open object); parse_tool_call still accepts it.
TOOL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tool": {"type": "string", "enum": list(TOOL_NAMES)},
        "query": {"type": "string"},
        "max_results": {"type": "integer"},
        "task_id": {"type": ["string", "null"]},
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "action": {
            "type": "string",
            "enum": ["create", "assign", "set_due_date", "complete", "cancel"],
        },
        "description": {"type": ["string", "null"]},
        "owner": {"type": ["string", "null"]},
        "due_date": {"type": ["string", "null"]},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["tool"],
}

_SKIP = {"tool", "name", "arguments"}


class ToolError(Exception):
    """The JSON was a real tool call, but the case rules refused it.

    The loop keeps going and shows this message to the model as data.
    """


@dataclass
class CaseContext:
    meeting: MeetingTranscript
    candidate: CandidateChange
    match: MatchResult
    reasons: list[str]
    tasks: list[TaskRecord]
    team: TeamProfile
    reader: MemoryReader


@dataclass
class ToolOutcome:
    observation: str
    terminal: AgentOutcome | None = None
    proposal: ProposedUpdate | None = None
    reason: str | None = None


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SearchCall(_Model):
    query: str = Field(min_length=1)
    max_results: int | None = Field(default=None, ge=1, le=20)


class HistoryCall(_Model):
    task_id: str = Field(min_length=1)


class TeamCall(_Model):
    pass


class ClarifyCall(_Model):
    question: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)


class ProposeCall(_Model):
    action: ActionType
    task_id: str | None = None
    description: str | None = None
    owner: str | None = None
    due_date: Any = None
    evidence_refs: list[str] = Field(min_length=1)


class ReasonCall(_Model):
    reason: str = Field(min_length=1)


_TOOL_MODELS: dict[str, type[_Model]] = {
    "search_meetings": SearchCall,
    "get_task_history": HistoryCall,
    "get_team_profile": TeamCall,
    "ask_clarification": ClarifyCall,
    "propose_update": ProposeCall,
    "abstain": ReasonCall,
    "escalate": ReasonCall,
}


def parse_tool_call(raw: str) -> tuple[str, _Model, dict[str, Any]]:
    """One JSON object, one tool. Raises ValueError when the reply is unusable."""
    try:
        data = json.loads(_json_text(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if isinstance(data, list):
        if len(data) != 1 or not isinstance(data[0], dict):
            raise ValueError("JSON must be one object, one tool per step")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    tool_name = data.get("tool", data.get("name"))
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("missing tool name")
    tool_name = tool_name.strip()
    model = _TOOL_MODELS.get(tool_name)
    if model is None:
        raise ValueError(f"unknown tool {tool_name}")
    arguments = data.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: arguments: {exc.msg}") from exc
    if arguments is None:
        arguments = {key: value for key, value in data.items() if key not in _SKIP}
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    try:
        parsed = model.model_validate(_prepare(tool_name, arguments))
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(part) for part in first["loc"])
        raise ValueError(f"schema: {first['msg']} ({loc})") from exc
    return tool_name, parsed, parsed.model_dump(mode="json")


def run_tool(
    name: str,
    parsed: _Model,
    ctx: CaseContext,
    prior_steps: list[AgentStep],
    *,
    default_max_results: int,
) -> ToolOutcome:
    if name == "search_meetings":
        assert isinstance(parsed, SearchCall)
        limit = parsed.max_results or default_max_results
        hits = ctx.reader.search_meetings(
            parsed.query,
            current_meeting_no=ctx.meeting.meeting_no,
            max_results=limit,
        )
        payload = {"hits": [hit.model_dump(mode="json") for hit in hits]}
        return ToolOutcome(observation=_dump(payload))
    if name == "get_task_history":
        assert isinstance(parsed, HistoryCall)
        versions = ctx.reader.get_task_history(parsed.task_id)
        payload = {
            "task_id": parsed.task_id,
            "versions": [version.model_dump(mode="json") for version in versions],
        }
        return ToolOutcome(observation=_dump(payload))
    if name == "get_team_profile":
        return ToolOutcome(observation=ctx.reader.get_team_profile().model_dump_json())
    if name == "ask_clarification":
        assert isinstance(parsed, ClarifyCall)
        answer = ctx.reader.simulated_answer(parsed.question, meeting_no=ctx.meeting.meeting_no)
        return ToolOutcome(observation=_dump({"answer": answer}))
    if name == "propose_update":
        assert isinstance(parsed, ProposeCall)
        return _propose(parsed, ctx, prior_steps)
    if name in ("abstain", "escalate"):
        assert isinstance(parsed, ReasonCall)
        outcome = AgentOutcome.abstain if name == "abstain" else AgentOutcome.escalate
        return ToolOutcome(observation=_dump({"status": name}), terminal=outcome, reason=parsed.reason.strip())
    raise ToolError(f"unknown tool {name}")


def _propose(parsed: ProposeCall, ctx: CaseContext, prior_steps: list[AgentStep]) -> ToolOutcome:
    if parsed.action != ctx.candidate.action:
        raise ToolError(
            f"action must stay {ctx.candidate.action.value}; this case was routed for that action"
        )
    lines = _evidence_lines(ctx.meeting, parsed.evidence_refs)
    task: TaskRecord | None = None
    if parsed.action != ActionType.create:
        if not parsed.task_id:
            raise ToolError("updates need a task_id from the task list")
        task = next((item for item in ctx.tasks if item.task_id == parsed.task_id), None)
        if task is None:
            raise ToolError(f"unknown task_id {parsed.task_id}")
        if _IDENTITY_REASONS.intersection(ctx.reasons) and not _identity_supported(task, prior_steps):
            raise ToolError(
                "This case does not identify a single task. "
                "search_meetings or get_task_history must mention that task before propose_update. "
                "Do not guess."
            )
    owner = _resolve_owner(parsed, ctx, lines, prior_steps)
    fields, title, task_id, version = _fields(parsed, ctx, task, owner)
    try:
        proposal = ProposedUpdate(
            project_id=ctx.meeting.project_id,
            meeting_no=ctx.meeting.meeting_no,
            candidate_index=ctx.match.candidate_index,
            action=parsed.action,
            task_id=task_id,
            task_version=version,
            task_description=title,
            fields=fields,
            evidence_refs=[EvidenceRef(source=EvidenceSource.transcript, id=line.id) for line in lines],
            evidence_quotes=[EvidenceQuote(utterance_id=line.id, quote=line.text) for line in lines],
            confidence=None,
            origin=ProposalOrigin.agent,
        )
    except ValidationError as exc:
        raise ToolError(f"proposal rejected: {exc.errors()[0]['msg']}") from exc
    return ToolOutcome(
        observation=_dump(
            {"status": "proposal prepared for review", "task_id": proposal.task_id, "action": proposal.action.value}
        ),
        terminal=AgentOutcome.propose_update,
        proposal=proposal,
    )


def _evidence_lines(meeting: MeetingTranscript, refs: list[str]) -> list[Utterance]:
    lines: list[Utterance] = []
    seen: set[str] = set()
    blocked: list[str] = []
    for ref in refs:
        if not _LINE_ID.fullmatch(ref):
            raise ToolError(f"evidence ref {ref} must be a current meeting line id such as L4")
        utterance = meeting.utterance(ref)
        if utterance is None:
            raise ToolError(f"unknown line {ref}")
        if _ASSISTANT.search(utterance.text):
            blocked.append(ref)
            continue
        if ref not in seen:
            seen.add(ref)
            lines.append(utterance)
    if not lines:
        if blocked:
            raise ToolError(f"lines {', '.join(blocked)} address an assistant and cannot be evidence")
        raise ToolError("cite at least one current meeting line")
    return lines


def _identity_supported(task: TaskRecord, steps: list[AgentStep]) -> bool:
    """True when a prior read tool actually surfaced this task."""
    quoted_id = f'"{task.task_id}"'
    description = task.description.casefold()
    for step in steps:
        if not step.valid:
            continue
        if step.tool == "ask_clarification":
            answer = _answer_text(step.observation)
            if answer.casefold() == NO_ANSWER:
                continue
            blob = answer
        elif step.tool in ("search_meetings", "get_task_history"):
            blob = step.observation
        else:
            continue
        if quoted_id in blob or (description and description in blob.casefold()):
            return True
    return False


def _resolve_owner(
    parsed: ProposeCall,
    ctx: CaseContext,
    lines: list[Utterance],
    prior_steps: list[AgentStep],
) -> str | None:
    if parsed.action in (ActionType.complete, ActionType.cancel):
        if parsed.owner:
            raise ToolError(f"{parsed.action.value} does not take an owner")
        return None
    if parsed.action == ActionType.set_due_date:
        if parsed.owner:
            raise ToolError("set_due_date does not take an owner")
        return None
    if parsed.owner is None:
        if parsed.action == ActionType.assign:
            raise ToolError("assign requires an owner")
        return None
    canonical = ctx.team.canonical_member(parsed.owner)
    if canonical is None:
        raise ToolError(f"owner {parsed.owner} is not on the team")
    if not _owner_supported(canonical, ctx, lines, prior_steps):
        raise ToolError(
            f"owner {canonical} is not stated by the candidate, the cited lines, or a clarification answer"
        )
    return canonical


def _owner_supported(
    canonical: str,
    ctx: CaseContext,
    lines: list[Utterance],
    prior_steps: list[AgentStep],
) -> bool:
    if ctx.team.canonical_member(ctx.candidate.owner) == canonical:
        return True
    names = _member_names(ctx.team, canonical)
    if any(_mentioned(line.text, names) for line in lines):
        return True
    for step in prior_steps:
        if step.tool != "ask_clarification" or not step.valid:
            continue
        answer = _answer_text(step.observation)
        if answer.casefold() == NO_ANSWER:
            continue
        if _mentioned(answer, names):
            return True
    return False


def _fields(
    parsed: ProposeCall,
    ctx: CaseContext,
    task: TaskRecord | None,
    owner: str | None,
) -> tuple[UpdateFields, str, str | None, int | None]:
    if parsed.action == ActionType.create:
        description = (parsed.description or ctx.candidate.description or "").strip()
        if not description:
            raise ToolError("create requires a description")
        if parsed.task_id:
            raise ToolError("create does not take a task_id")
        due = _due_date(parsed.due_date)
        return UpdateFields(description=description, owner=owner, due_date=due), description, None, None
    if task is None:
        raise ToolError("updates need a task_id from the task list")
    if parsed.action == ActionType.assign:
        if not owner:
            raise ToolError("assign requires an owner")
        return UpdateFields(owner=owner), task.description, task.task_id, task.version
    if parsed.action == ActionType.set_due_date:
        if "due_date" not in parsed.model_fields_set:
            raise ToolError("set_due_date requires due_date as YYYY-MM-DD or null")
        return UpdateFields(due_date=_due_date(parsed.due_date)), task.description, task.task_id, task.version
    return UpdateFields(), task.description, task.task_id, task.version


def _due_date(value: Any):
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.casefold() in {"null", "none"}:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ToolError("due_date must be YYYY-MM-DD or null") from exc


def _member_names(team: TeamProfile, canonical: str) -> list[str]:
    for member in team.members:
        if member.name == canonical:
            return [member.name, *member.aliases]
    return [canonical]


def _mentioned(text: str, names: list[str]) -> bool:
    # Case-sensitive on purpose: the member "Can" is also the English verb "can".
    return any(re.search(rf"\b{re.escape(name)}\b", text) for name in names)


def _answer_text(observation: str) -> str:
    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return ""
    if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
        return payload["answer"]
    return ""


def _prepare(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(arguments)
    for key in ("task_id", "description", "owner", "query", "question", "reason"):
        if key in cleaned:
            cleaned[key] = _blank(cleaned[key])
    if tool_name != "propose_update":
        return cleaned
    if "due_date" in cleaned:
        cleaned["due_date"] = _blank(cleaned["due_date"])
    if "evidence_refs" in cleaned:
        cleaned["evidence_refs"] = _coerce_refs(cleaned["evidence_refs"])
    return cleaned


def _coerce_refs(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    refs: list[Any] = []
    for item in value:
        if isinstance(item, str):
            refs.append(item.strip())
        elif isinstance(item, dict):
            ref = item.get("id") or item.get("utterance_id") or ""
            refs.append(str(ref).strip())
        else:
            refs.append(item)
    return refs


def _blank(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.casefold() in {"null", "none"}:
            return None
        return stripped
    return value


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return text
