"""The shared records reject the shapes validate and commit must not see."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas import (
    ActionType,
    EvidenceQuote,
    EvidenceRef,
    EvidenceSource,
    ProposedUpdate,
    RouteKind,
    RoutingDecision,
    TaskRecord,
    TeamProfile,
    UpdateFields,
    Utterance,
)


def test_team_alias_resolves_to_the_primary_name():
    team = TeamProfile.model_validate(
        {
            "project_id": "demo",
            "coordinator": "Zeynep",
            "members": [{"name": "Ayşe", "role": "frontend", "aliases": ["Ayse"]}],
        }
    )
    assert team.canonical_member("Ayse") == "Ayşe"
    assert team.canonical_member("Ayşe") == "Ayşe"
    assert team.canonical_member("Bob") is None
    assert team.canonical_member(None) is None


def test_candidate_confidence_stays_between_zero_and_one():
    from app.schemas import CandidateChange

    with pytest.raises(ValidationError):
        CandidateChange(
            action=ActionType.create,
            description="Login page",
            evidence=[EvidenceQuote(utterance_id="L2", quote="login page")],
            confidence=1.1,
        )


def test_proposal_requires_a_transcript_ref_and_empty_fields_for_complete():
    quote = EvidenceQuote(utterance_id="L2", quote="the login page is done")
    proposal = ProposedUpdate(
        project_id="demo",
        meeting_no=3,
        candidate_index=0,
        action=ActionType.complete,
        task_id="T1",
        task_version=1,
        task_description="Login page",
        fields=UpdateFields(),
        evidence_refs=[EvidenceRef(source=EvidenceSource.transcript, id="L2")],
        evidence_quotes=[quote],
    )
    dumped = proposal.model_dump(mode="json")
    assert dumped["fields"] == {}
    assert dumped["task_id"] == "T1"

    with pytest.raises(ValidationError):
        ProposedUpdate(
            project_id="demo",
            meeting_no=3,
            candidate_index=0,
            action=ActionType.complete,
            task_id="T1",
            task_version=1,
            task_description="Login page",
            fields=UpdateFields(),
            evidence_refs=[EvidenceRef(source=EvidenceSource.clarification, id="C1")],
            evidence_quotes=[quote],
        )


def test_create_proposal_keeps_an_explicit_null_owner():
    proposal = ProposedUpdate(
        project_id="demo",
        meeting_no=3,
        candidate_index=2,
        action=ActionType.create,
        task_id=None,
        task_description="Load tests before the release",
        fields=UpdateFields(description="Load tests before the release", owner=None, due_date=None),
        evidence_refs=[EvidenceRef(source=EvidenceSource.transcript, id="L6")],
        evidence_quotes=[EvidenceQuote(utterance_id="L6", quote="load tests")],
    )
    assert proposal.model_dump(mode="json")["fields"] == {
        "description": "Load tests before the release",
        "owner": None,
        "due_date": None,
    }
    assert proposal.task_id is None


def test_hard_decision_cannot_carry_a_proposal():
    with pytest.raises(ValidationError):
        RoutingDecision(candidate_index=0, route=RouteKind.hard, reasons=[], proposal=None)


def test_utterance_ids_in_one_meeting_are_unique():
    from app.schemas import MeetingTranscript

    with pytest.raises(ValidationError):
        MeetingTranscript(
            project_id="demo",
            meeting_no=1,
            meeting_date=date(2026, 10, 29),
            utterances=[
                Utterance(id="L1", speaker="Zeynep", text="Hello"),
                Utterance(id="L1", speaker="Ali", text="Hello again"),
            ],
        )


def test_task_record_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        TaskRecord.model_validate(
            {
                "task_id": "T1",
                "description": "Login page",
                "status": "open",
                "gold_id": "hidden",
            }
        )
