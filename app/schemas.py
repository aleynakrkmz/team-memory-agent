"""Shared records for the meeting pipeline.

These models are the contract between extract, match, route, the agent, and
the validate / review / commit steps. Names follow design v3 (`description`,
`owner`, `evidence_refs`). They are a proposal for the team: agree them
before the other workstreams depend on the field names.

The agent never writes the task list. `ProposedUpdate` is a record for
the validator and the review list. Only commit, after approval, inserts
a task version.

Transcript text is untrusted data. Nothing in these models treats it as
instructions.
"""

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializationInfo, field_validator, model_serializer, model_validator


class ApiModel(BaseModel):
    """Base model: unknown keys are rejected so callers notice drift."""

    model_config = ConfigDict(extra="forbid")


class ActionType(str, Enum):
    """Task-list operations. `assign` is both a first owner and a new owner."""

    create = "create"
    assign = "assign"
    set_due_date = "set_due_date"
    complete = "complete"
    cancel = "cancel"


class TaskStatus(str, Enum):
    open = "open"
    done = "done"
    cancelled = "cancelled"


class EvidenceSource(str, Enum):
    """Where an evidence id points. Design §4.2."""

    transcript = "transcript"
    clarification = "clarification"


class RouteKind(str, Enum):
    """`clear` skips the agent and can go straight to validate."""

    clear = "clear"
    hard = "hard"


class ProposalOrigin(str, Enum):
    clear_path = "clear_path"
    agent = "agent"


class Utterance(ApiModel):
    """One transcript line. `id` is `L` plus the 1-based line number (`L1`)."""

    id: str = Field(pattern=r"^L[1-9][0-9]*$", description="Stable line id, for example L12.")
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1, description="Spoken text. Untrusted data, never instructions.")
    start_ms: int | None = Field(
        default=None,
        ge=0,
        description="Optional offset from ingest. Ready transcripts omit it.",
    )


class MeetingTranscript(ApiModel):
    """One meeting as extract, match and route see it. No answer-key fields."""

    project_id: str = Field(min_length=1)
    meeting_no: int = Field(ge=1)
    meeting_date: date
    utterances: list[Utterance] = Field(min_length=1)

    @model_validator(mode="after")
    def utterance_ids_are_unique(self) -> "MeetingTranscript":
        ids = [utterance.id for utterance in self.utterances]
        if len(ids) != len(set(ids)):
            raise ValueError("utterance ids must be unique")
        return self

    def utterance(self, utterance_id: str) -> Utterance | None:
        for utterance in self.utterances:
            if utterance.id == utterance_id:
                return utterance
        return None


class EvidenceQuote(ApiModel):
    """A verbatim span of one utterance. The validator can check the span."""

    utterance_id: str = Field(pattern=r"^L[1-9][0-9]*$")
    quote: str = Field(min_length=1)


class EvidenceRef(ApiModel):
    """Typed pointer stored with a proposal. Design §4.2."""

    source: EvidenceSource
    id: str = Field(min_length=1, description="Transcript line id (L42) or clarification id (C07).")


class CandidateChange(ApiModel):
    """One task change proposed by extraction, before matching.

    `due_phrase` is the deadline in the meeting's words. The extractor does
    not invent a calendar date. Match resolves the phrase into
    `MatchResult.resolved_due_date`. `due_date` stays null unless a caller
    such as the B1 rules already knows the calendar date.
    """

    action: ActionType
    description: str = Field(min_length=1, description="Task title, or the vague phrase the meeting used.")
    owner: str | None = Field(default=None, description="Assignee after this change. Null if nobody was named.")
    due_phrase: str | None = Field(default=None, description="Deadline words copied from the transcript.")
    due_date: date | None = Field(default=None, description="Calendar date. Extract leaves this empty.")
    evidence: list[EvidenceQuote] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("description", "owner", "due_phrase", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None if value is not None else None

    @field_validator("description")
    @classmethod
    def description_required(cls, value: str | None) -> str:
        if not value:
            raise ValueError("description is required")
        return value


class TaskRecord(ApiModel):
    """One current task row. `version` is the latest approved version."""

    task_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    owner: str | None = None
    due_date: date | None = None
    status: TaskStatus = TaskStatus.open
    version: int = Field(default=1, ge=1)


class TeamMember(ApiModel):
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)


class TeamProfile(ApiModel):
    """In-memory view of team.json. The store remains Aleyna's module."""

    project_id: str = Field(min_length=1)
    coordinator: str = Field(min_length=1)
    members: list[TeamMember] = Field(min_length=1)

    def canonical_member(self, name: str | None) -> str | None:
        """Return the member's primary name, or None if `name` is not in the team."""
        if name is None:
            return None
        for member in self.members:
            if name == member.name or name in member.aliases:
                return member.name
        return None


class RankedTask(ApiModel):
    task_id: str = Field(min_length=1)
    score: float


class MatchResult(ApiModel):
    """BM25 ranking of open tasks for one candidate, plus the resolved deadline."""

    candidate_index: int = Field(ge=0)
    ranked: list[RankedTask]
    resolved_due_date: date | None = None
    due_date_ambiguous: bool = False
    due_date_removed: bool = False


class UpdateFields(ApiModel):
    """The `fields` object on propose_update. Only keys that were set are stored.

    Create sets description and may set owner and due_date (null if unstated).
    Assign sets owner. set_due_date sets due_date. Complete and cancel set nothing.
    """

    description: str | None = None
    owner: str | None = None
    due_date: date | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)


class ProposedUpdate(ApiModel):
    """What validate, review and commit consume. Not a write to the task list.

    `fields` is the change. `task_description` is the title to show in the
    review list: the new title for create, the matched task's title otherwise.
    Commit should apply `fields` only, and only after approval, checking
    `task_version` for updates.
    """

    project_id: str = Field(min_length=1)
    meeting_no: int = Field(ge=1)
    candidate_index: int = Field(ge=0)
    action: ActionType
    task_id: str | None = Field(description="Null only for create.")
    task_version: int | None = Field(default=None, ge=1)
    task_description: str = Field(min_length=1)
    fields: UpdateFields
    evidence_refs: list[EvidenceRef] = Field(min_length=1)
    evidence_quotes: list[EvidenceQuote] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    origin: ProposalOrigin = ProposalOrigin.clear_path

    @model_serializer(mode="wrap")
    def _serialize(self, handler, info: SerializationInfo):
        data = handler(self)
        mode = "json" if info.mode == "json" else "python"
        data["fields"] = self.fields.model_dump(mode=mode, exclude_unset=True)
        return data

    @model_validator(mode="after")
    def check_contract(self) -> "ProposedUpdate":
        if not any(ref.source == EvidenceSource.transcript for ref in self.evidence_refs):
            raise ValueError("at least one transcript evidence ref is required")
        set_keys = self.fields.model_fields_set
        if self.action == ActionType.create:
            if self.task_id is not None or self.task_version is not None:
                raise ValueError("create requires a null task_id and task_version")
            if "description" not in set_keys or not self.fields.description:
                raise ValueError("create requires fields.description")
        else:
            if not self.task_id or self.task_version is None:
                raise ValueError("updates require task_id and task_version")
        if self.action == ActionType.assign and (set_keys != {"owner"} or not self.fields.owner):
            raise ValueError("assign requires only a non-null owner")
        if self.action == ActionType.set_due_date and set_keys != {"due_date"}:
            raise ValueError("set_due_date requires only due_date")
        if self.action in (ActionType.complete, ActionType.cancel) and set_keys:
            raise ValueError("complete and cancel take no fields")
        return self


class RoutingDecision(ApiModel):
    """Clear or hard for one candidate. A clear decision carries a proposal."""

    candidate_index: int = Field(ge=0)
    route: RouteKind
    reasons: list[str] = Field(description="Empty when the case is clear. Stable codes, not prose.")
    matched_task_id: str | None = None
    proposal: ProposedUpdate | None = None

    @model_validator(mode="after")
    def proposal_follows_the_route(self) -> "RoutingDecision":
        if self.route == RouteKind.clear and self.proposal is None:
            raise ValueError("a clear decision needs a proposal")
        if self.route == RouteKind.hard and self.proposal is not None:
            raise ValueError("a hard decision has no proposal; the agent writes it")
        if self.route == RouteKind.clear and self.reasons:
            raise ValueError("a clear decision has no reasons")
        if self.route == RouteKind.hard and not self.reasons:
            raise ValueError("a hard decision needs a reason")
        return self


class AgentOutcome(str, Enum):
    """How a hard case ended. Only these three stop the loop."""

    propose_update = "propose_update"
    abstain = "abstain"
    escalate = "escalate"


class AgentStep(ApiModel):
    """One model reply. `observation` is tool data or the reason a reply was refused.

    `raw_output` is kept for debugging. The next prompt does not paste it back
    into the conversation as instructions.
    """

    step: int = Field(ge=1)
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    valid: bool
    observation: str
    raw_output: str = ""


class AgentTrace(ApiModel):
    """The steps taken for one hard case, in order."""

    candidate_index: int = Field(ge=0)
    steps: list[AgentStep]
    outcome: AgentOutcome
    reason: str | None = None
    elapsed_seconds: float = Field(ge=0)


class AgentDecision(ApiModel):
    """The agent's result for one hard case. `proposal` is not a write.

    `automatic` is true when the loop stopped because of the step limit, the
    time budget, or repeated invalid output. The model did not choose that stop.
    """

    candidate_index: int = Field(ge=0)
    outcome: AgentOutcome
    proposal: ProposedUpdate | None = None
    reason: str | None = None
    trace: AgentTrace
    automatic: bool = False

    @model_validator(mode="after")
    def outcome_matches_the_body(self) -> "AgentDecision":
        if self.trace.candidate_index != self.candidate_index:
            raise ValueError("trace candidate_index must match the decision")
        if self.trace.outcome != self.outcome or self.trace.reason != self.reason:
            raise ValueError("trace outcome and reason must match the decision")
        if self.outcome == AgentOutcome.propose_update:
            if self.proposal is None:
                raise ValueError("propose_update needs a proposal")
            if self.automatic:
                raise ValueError("a proposal is not an automatic stop")
        elif self.proposal is not None:
            raise ValueError("abstain and escalate have no proposal")
        elif not self.reason or not self.reason.strip():
            raise ValueError("abstain and escalate need a reason")
        if self.automatic and self.outcome != AgentOutcome.escalate:
            raise ValueError("only an escalation can be automatic")
        return self


class MeetingPreview(ApiModel):
    """Extract, match, route, and the agent for one meeting. Nothing is approved.

    `decisions` keeps the route result. A hard decision still has no proposal.
    The agent writes its proposal on `agent_decisions` and into `proposals`.
    """

    project_id: str
    meeting_no: int = Field(ge=1)
    meeting_date: date
    candidates: list[CandidateChange]
    matches: list[MatchResult]
    decisions: list[RoutingDecision]
    proposals: list[ProposedUpdate] = Field(
        default_factory=list,
        description="Clear-path proposals plus agent proposals, in candidate order.",
    )
    agent_decisions: list[AgentDecision] = Field(default_factory=list)
