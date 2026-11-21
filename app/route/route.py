"""Decide whether a candidate skips the agent. Design §3.2.

A candidate is clear only when all three hold:

1. The task is clear. An update needs a top open-task score of at least τ
   and at least `margin` times the second score. A create is clear when no
   open task scores τ or more.
2. The fields that the action requires are resolved. A create may leave
   owner and due date empty when the meeting did not state them. Other
   missing or conflicting fields go to the agent. They are not guessed.
3. No other candidate in this meeting refers to the same task.

τ and the margin are starting values in config/routing.json.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.schemas import (
    ActionType,
    CandidateChange,
    EvidenceRef,
    EvidenceSource,
    MatchResult,
    MeetingTranscript,
    ProposalOrigin,
    ProposedUpdate,
    RouteKind,
    RoutingDecision,
    TaskRecord,
    TeamProfile,
    UpdateFields,
)

TASK_UNCLEAR = "task_unclear"
TASK_AMBIGUOUS = "task_ambiguous"
POSSIBLE_DUPLICATE = "possible_duplicate"
MISSING_OWNER = "missing_owner"
OWNER_NOT_IN_TEAM = "owner_not_in_team"
DUE_DATE_UNRESOLVED = "due_date_unresolved"
DUE_DATE_AMBIGUOUS = "due_date_ambiguous"
EXTRA_OWNER = "extra_owner"
EXTRA_DUE_DATE = "extra_due_date"
SAME_TASK_CONFLICT = "same_task_conflict"

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "routing.json"


class RoutingThresholds(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bm25_tau: float
    bm25_margin: float


def load_thresholds(path: Path | None = None) -> RoutingThresholds:
    config_path = path or CONFIG_PATH
    data = json.loads(config_path.read_text(encoding="utf-8"))
    thresholds = RoutingThresholds.model_validate(data)
    if thresholds.bm25_tau <= 0:
        raise ValueError("bm25_tau must be > 0")
    if thresholds.bm25_margin < 1:
        raise ValueError("bm25_margin must be >= 1")
    return thresholds


def route_candidates(
    meeting: MeetingTranscript,
    candidates: list[CandidateChange],
    matches: list[MatchResult],
    tasks: list[TaskRecord],
    team: TeamProfile,
    *,
    tau: float | None = None,
    margin: float | None = None,
) -> list[RoutingDecision]:
    """Return one decision per candidate. Hard cases have no proposal."""
    if len(candidates) != len(matches):
        raise ValueError("matches must line up with candidates")
    defaults = load_thresholds()
    tau = defaults.bm25_tau if tau is None else tau
    margin = defaults.bm25_margin if margin is None else margin
    if tau <= 0 or margin < 1:
        raise ValueError("tau must be > 0 and margin must be >= 1")

    by_id = {task.task_id: task for task in tasks}
    draft: list[_Draft] = []
    for position, (candidate, match) in enumerate(zip(candidates, matches, strict=True)):
        if match.candidate_index != position:
            raise ValueError("match results must stay in candidate order")
        draft.append(_judge(position, candidate, match, by_id, team, tau=tau, margin=margin))

    referred: dict[str, list[int]] = {}
    for item in draft:
        if item.matched_task_id is not None:
            referred.setdefault(item.matched_task_id, []).append(item.index)
    for indexes in referred.values():
        if len(indexes) > 1:
            for index in indexes:
                draft[index].reasons.append(SAME_TASK_CONFLICT)

    decisions: list[RoutingDecision] = []
    for item in draft:
        proposal = None
        if not item.reasons:
            proposal = _proposal(meeting, item, by_id)
        decisions.append(
            RoutingDecision(
                candidate_index=item.index,
                route=RouteKind.clear if proposal else RouteKind.hard,
                reasons=[] if proposal else item.reasons,
                matched_task_id=item.matched_task_id,
                proposal=proposal,
            )
        )
    return decisions


class _Draft:
    def __init__(self, index: int, candidate: CandidateChange, match: MatchResult):
        self.index = index
        self.candidate = candidate
        self.match = match
        self.matched_task_id: str | None = None
        self.reasons: list[str] = []
        self.owner: str | None = None


def _judge(
    index: int,
    candidate: CandidateChange,
    match: MatchResult,
    tasks: dict[str, TaskRecord],
    team: TeamProfile,
    *,
    tau: float,
    margin: float,
) -> _Draft:
    item = _Draft(index, candidate, match)
    top_id, unique = _unique_task(match, tau=tau, margin=margin)
    if candidate.action == ActionType.create:
        if any(ranked.score >= tau for ranked in match.ranked):
            item.reasons.append(POSSIBLE_DUPLICATE)
            if unique:
                item.matched_task_id = top_id
        # A create that matches nothing is new. Null owner and null date are allowed.
    else:
        if top_id is None or tasks.get(top_id) is None:
            item.reasons.append(TASK_UNCLEAR)
        elif not unique:
            item.reasons.append(TASK_AMBIGUOUS)
        else:
            item.matched_task_id = top_id

    _owner_and_date_reasons(item, team)
    return item


def _unique_task(match: MatchResult, *, tau: float, margin: float) -> tuple[str | None, bool]:
    """Return (top task id or None, whether it is a unique match at or above τ)."""
    if not match.ranked:
        return None, False
    top = match.ranked[0]
    second = match.ranked[1].score if len(match.ranked) > 1 else 0.0
    if top.score < tau:
        return None, False
    unique = second == 0 or top.score >= margin * second
    return top.task_id, unique


def _owner_and_date_reasons(item: _Draft, team: TeamProfile) -> None:
    candidate = item.candidate
    stated_owner = candidate.owner is not None
    item.owner = team.canonical_member(candidate.owner)
    if stated_owner and item.owner is None:
        item.reasons.append(OWNER_NOT_IN_TEAM)

    date_problem = _date_problem(item)
    has_date_info = candidate.due_phrase is not None or candidate.due_date is not None or item.match.due_date_removed

    if candidate.action == ActionType.assign:
        if not stated_owner:
            item.reasons.append(MISSING_OWNER)
        if has_date_info:
            item.reasons.append(EXTRA_DUE_DATE)
    elif candidate.action == ActionType.set_due_date:
        if stated_owner:
            item.reasons.append(EXTRA_OWNER)
        if date_problem:
            item.reasons.append(date_problem)
    elif candidate.action in (ActionType.complete, ActionType.cancel):
        if stated_owner:
            item.reasons.append(EXTRA_OWNER)
        if has_date_info:
            item.reasons.append(EXTRA_DUE_DATE)
    elif candidate.action == ActionType.create:
        if date_problem:
            item.reasons.append(date_problem)


def _date_problem(item: _Draft) -> str | None:
    match = item.match
    if match.due_date_ambiguous:
        return DUE_DATE_AMBIGUOUS
    phrase_or_date = item.candidate.due_phrase is not None or item.candidate.due_date is not None
    if phrase_or_date and match.resolved_due_date is None and not match.due_date_removed:
        return DUE_DATE_UNRESOLVED
    return None


def _proposal(meeting: MeetingTranscript, item: _Draft, tasks: dict[str, TaskRecord]) -> ProposedUpdate:
    candidate = item.candidate
    matched = tasks.get(item.matched_task_id) if item.matched_task_id else None
    if candidate.action == ActionType.create:
        fields = UpdateFields(
            description=candidate.description,
            owner=item.owner,
            due_date=item.match.resolved_due_date,
        )
        task_id = None
        task_version = None
        title = candidate.description
    else:
        if matched is None:
            raise RuntimeError("a clear update has no matched task")
        task_id = matched.task_id
        task_version = matched.version
        title = matched.description
        if candidate.action == ActionType.assign:
            fields = UpdateFields(owner=item.owner)
        elif candidate.action == ActionType.set_due_date:
            fields = UpdateFields(due_date=item.match.resolved_due_date)
        else:
            fields = UpdateFields()
    return ProposedUpdate(
        project_id=meeting.project_id,
        meeting_no=meeting.meeting_no,
        candidate_index=item.index,
        action=candidate.action,
        task_id=task_id,
        task_version=task_version,
        task_description=title,
        fields=fields,
        evidence_refs=_refs(candidate),
        evidence_quotes=list(candidate.evidence),
        confidence=candidate.confidence,
        origin=ProposalOrigin.clear_path,
    )


def _refs(candidate: CandidateChange) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    seen: set[str] = set()
    for quote in candidate.evidence:
        if quote.utterance_id in seen:
            continue
        seen.add(quote.utterance_id)
        refs.append(EvidenceRef(source=EvidenceSource.transcript, id=quote.utterance_id))
    return refs
