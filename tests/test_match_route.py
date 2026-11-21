"""Match and route on the hand-written demo, with no model call."""

from datetime import date

from app.match import match_candidates
from app.route import (
    DUE_DATE_AMBIGUOUS,
    OWNER_NOT_IN_TEAM,
    SAME_TASK_CONFLICT,
    TASK_UNCLEAR,
    load_thresholds,
    route_candidates,
)
from app.schemas import ActionType, CandidateChange, EvidenceQuote, RouteKind
from tests.demo_support import load_demo, open_tasks_before


def _change(action, description, *, owner=None, due_phrase=None, line="L2", quote=None, confidence=0.9):
    return CandidateChange(
        action=action,
        description=description,
        owner=owner,
        due_phrase=due_phrase,
        evidence=[EvidenceQuote(utterance_id=line, quote=quote or description)],
        confidence=confidence,
    )


def _route(meeting_no, candidates):
    meeting, team = load_demo(meeting_no)
    tasks = open_tasks_before(meeting_no)
    matches = match_candidates(candidates, tasks, meeting.meeting_date)
    decisions = route_candidates(meeting, candidates, matches, tasks, team)
    return tasks, matches, decisions


def test_starting_thresholds_match_the_design_margin():
    thresholds = load_thresholds()
    assert thresholds.bm25_tau == 0.5
    assert thresholds.bm25_margin == 1.5


def test_meeting_one_creates_resolve_dates_and_skip_the_agent():
    candidates = [
        _change(ActionType.create, "Login page", owner="Ayşe", due_phrase="by the 6th", line="L2", quote="login page"),
        _change(ActionType.create, "Database schema", owner="Can", due_phrase="by the 4th", line="L4", quote="database schema"),
        _change(ActionType.create, "Project report draft", owner="Zeynep", line="L7", quote="project report draft"),
    ]
    _tasks, matches, decisions = _route(1, candidates)

    assert [item.resolved_due_date for item in matches] == [date(2026, 11, 6), date(2026, 11, 4), None]
    assert [item.route for item in decisions] == [RouteKind.clear, RouteKind.clear, RouteKind.clear]
    login = decisions[0].proposal
    assert login is not None and login.task_id is None
    assert login.fields.owner == "Ayşe"
    assert login.fields.due_date == date(2026, 11, 6)
    assert login.fields.description == "Login page"
    assert login.evidence_refs[0].id == "L2"
    assert decisions[2].proposal.fields.due_date is None
    assert decisions[2].proposal.fields.owner == "Zeynep"


def test_alias_is_stored_as_the_primary_name():
    candidates = [_change(ActionType.create, "Login page", owner="Ayse", line="L2", quote="login page")]
    _tasks, _matches, decisions = _route(1, candidates)
    assert decisions[0].route == RouteKind.clear
    assert decisions[0].proposal.fields.owner == "Ayşe"


def test_meeting_two_clear_updates_hit_the_open_tasks():
    candidates = [
        _change(ActionType.complete, "Database schema", line="L2", quote="database schema"),
        _change(ActionType.set_due_date, "Login page", due_phrase="the 10th", line="L3", quote="login page"),
        _change(
            ActionType.create,
            "Payment API endpoint",
            owner="Ali",
            due_phrase="by the 18th",
            line="L5",
            quote="payment API endpoint",
        ),
        _change(
            ActionType.create,
            "API documentation",
            owner="Ayşe",
            due_phrase="end of the month",
            line="L8",
            quote="API documentation",
        ),
    ]
    _tasks, matches, decisions = _route(2, candidates)

    assert [item.route for item in decisions] == [RouteKind.clear] * 4
    assert [item.proposal.task_id for item in decisions] == ["T2", "T1", None, None]
    assert matches[1].resolved_due_date == date(2026, 11, 10)
    assert decisions[1].proposal.fields.due_date == date(2026, 11, 10)
    assert decisions[1].proposal.fields.model_dump() == {"due_date": date(2026, 11, 10)}
    assert matches[2].resolved_due_date == date(2026, 11, 18)
    assert matches[3].resolved_due_date == date(2026, 11, 30)
    assert decisions[0].proposal.task_description == "Database schema"
    assert decisions[0].proposal.fields.model_dump() == {}
    assert decisions[0].proposal.task_version == 1


def test_vague_api_reassignment_is_not_a_unique_match():
    """Meeting 3: 'that API thing' fits both open API tasks, so the agent must see it."""
    candidates = [
        _change(ActionType.complete, "Login page", line="L2", quote="login page"),
        _change(ActionType.assign, "that API thing from last week", owner="Can", line="L4", quote="that API thing"),
        _change(ActionType.create, "Load tests before the release", line="L6", quote="load tests"),
        _change(ActionType.cancel, "Project report draft", line="L8", quote="project report draft"),
    ]
    tasks, matches, decisions = _route(3, candidates)
    open_ids = {task.task_id for task in tasks if task.status.value == "open"}

    scores = {row.task_id: row.score for row in matches[1].ranked}
    assert scores["T4"] == scores["T5"]
    assert set(scores) == open_ids
    assert "T2" not in scores  # the finished schema is not an open task

    api = decisions[1]
    assert api.route == RouteKind.hard
    assert api.proposal is None
    assert api.matched_task_id is None
    assert decisions[0].route == RouteKind.clear and decisions[0].proposal.task_id == "T1"
    assert decisions[2].route == RouteKind.clear
    assert decisions[2].proposal.task_id is None
    assert decisions[2].proposal.fields.owner is None
    assert decisions[3].route == RouteKind.clear and decisions[3].proposal.task_id == "T3"


def test_two_candidates_for_the_same_task_both_go_to_the_agent():
    candidates = [
        _change(ActionType.complete, "Login page", line="L2", quote="login page"),
        _change(ActionType.set_due_date, "Login page", due_phrase="by Friday", line="L4", quote="that API thing"),
    ]
    meeting, team = load_demo(3)
    tasks = open_tasks_before(3)
    matches = match_candidates(candidates, tasks, meeting.meeting_date)
    decisions = route_candidates(meeting, candidates, matches, tasks, team)
    assert [item.route for item in decisions] == [RouteKind.hard, RouteKind.hard]
    assert all(SAME_TASK_CONFLICT in item.reasons for item in decisions)
    assert all(item.proposal is None for item in decisions)


def test_unknown_owner_and_a_raised_threshold_are_hard():
    candidates = [_change(ActionType.create, "Login page", owner="Bob", line="L2", quote="login page")]
    _tasks, _matches, decisions = _route(1, candidates)
    assert decisions[0].route == RouteKind.hard
    assert OWNER_NOT_IN_TEAM in decisions[0].reasons

    meeting, team = load_demo(2)
    tasks = open_tasks_before(2)
    clear = [_change(ActionType.complete, "Database schema", line="L2", quote="database schema")]
    matches = match_candidates(clear, tasks, meeting.meeting_date)
    forced = route_candidates(meeting, clear, matches, tasks, team, tau=100)
    assert forced[0].route == RouteKind.hard
    assert TASK_UNCLEAR in forced[0].reasons


def test_a_contradictory_deadline_phrase_is_not_guessed():
    meeting, team = load_demo(3)
    tasks = open_tasks_before(3)
    candidates = [
        _change(
            ActionType.set_due_date,
            "Login page",
            due_phrase="no deadline by Friday",
            line="L2",
            quote="login page",
        )
    ]
    matches = match_candidates(candidates, tasks, meeting.meeting_date)
    assert matches[0].due_date_ambiguous is True
    decisions = route_candidates(meeting, candidates, matches, tasks, team)
    assert decisions[0].route == RouteKind.hard
    assert DUE_DATE_AMBIGUOUS in decisions[0].reasons


def test_an_explicit_deadline_removal_stays_null():
    meeting, team = load_demo(3)
    tasks = open_tasks_before(3)
    candidates = [
        _change(ActionType.set_due_date, "Login page", due_phrase="no deadline", line="L2", quote="login page")
    ]
    matches = match_candidates(candidates, tasks, meeting.meeting_date)
    assert matches[0].due_date_removed is True
    assert matches[0].resolved_due_date is None
    decisions = route_candidates(meeting, candidates, matches, tasks, team)
    assert decisions[0].route == RouteKind.clear
    assert "due_date" in decisions[0].proposal.fields.model_fields_set
    assert decisions[0].proposal.fields.due_date is None


def test_weekday_phrase_uses_the_shared_date_resolver():
    meeting, _team = load_demo(3)
    matches = match_candidates(
        [_change(ActionType.set_due_date, "Login page", due_phrase="by Friday", line="L2", quote="login page")],
        open_tasks_before(3),
        meeting.meeting_date,
    )
    assert meeting.meeting_date == date(2026, 11, 12)
    assert matches[0].resolved_due_date == date(2026, 11, 13)
