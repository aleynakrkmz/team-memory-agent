"""B1, the scorer and the harness, on the sample projects (no model; the agent path uses a fake model)."""

import json

from app.schemas import Utterance, ActionType, EvidenceQuote, EvidenceRef, EvidenceSource, ProposedUpdate, TaskRecord, UpdateFields
from eval.b1 import extract
from eval.harness import gold_tasks_before, load_key, run_trial, simulated_answers
from eval.scorer import change_f1, desc_similarity, score_meeting, state_accuracy
from tests.demo_support import ROOT, load_demo
from tests.fakes import FakeLLM
from tests.test_extract import _meeting_one_payload
from tests.test_pipeline import _meeting_three_payload

DEMO = ROOT / "samples" / "demo"


def _p(action, task_id=None, version=None, lines=("L2",), **fields):
    if action == "create":
        fields = {"description": fields.pop("description", "x"), "owner": None, "due_date": None} | fields
    return ProposedUpdate(project_id="demo", meeting_no=3, candidate_index=0, action=ActionType(action), task_id=task_id,
                          task_version=version, task_description=fields.get("description", "t"), fields=UpdateFields(**fields),
                          evidence_refs=[EvidenceRef(source=EvidenceSource.transcript, id=l) for l in lines],
                          evidence_quotes=[EvidenceQuote(utterance_id=lines[0], quote="x")])


# ---------- B1 ----------
def _talk(*lines):
    """A demo meeting with these lines instead of the real ones."""
    meeting, team = load_demo(3)
    utts = [Utterance(id=f"L{i}", speaker=spk, text=t) for i, (spk, t) in enumerate(lines, 1)]
    return meeting.model_copy(update={"utterances": utts}), team


def _found(meeting, team, tasks=()):
    return [(c.action.value, c.description, c.owner, c.due_phrase) for c in extract(meeting, team, list(tasks))]


def test_b1_reads_a_change_spread_over_several_lines():
    meeting, team = _talk(("Zeynep", "Next, we've got a new task - the sales dashboard. Who can take it?"),
                          ("Ali", "I'm full this week."),
                          ("Ayşe", "Sure, I can take it. When do you need it by?"),
                          ("Zeynep", "By Thursday, please."))
    assert _found(meeting, team) == [("create", "sales dashboard", "Ayşe", "By Thursday")]


def test_b1_leaves_an_open_owner_open():
    meeting, team = _talk(("Zeynep", "We need a regression test suite by the 2nd. Any volunteers?"),
                          ("Can", "I'm swamped."), ("Zeynep", "Alright, let's figure out who later."))
    assert _found(meeting, team) == [("create", "regression test suite", None, "by the 2nd")]


def test_b1_tells_done_from_will_be_done():
    tasks = [TaskRecord(task_id="T1", description="Login page", owner="Ayşe"),
             TaskRecord(task_id="T2", description="Database schema", owner="Can")]
    meeting, team = _talk(("Zeynep", "Ayşe, how's the login page?"), ("Ayşe", "Almost done, I'll make sure it's done by Friday."),
                          ("Zeynep", "And the schema, Can?"), ("Can", "Yeah, I wrapped it up yesterday."))
    assert _found(meeting, team, tasks) == [("complete", "Database schema", None, None)]


def test_b1_finds_a_handover_and_skips_the_current_owner():
    tasks = [TaskRecord(task_id="T1", description="Release checklist", owner="Ayşe")]
    meeting, team = _talk(("Zeynep", "Ayşe, could you please pass the release checklist to Can?"), ("Ayşe", "Sure thing."))
    assert _found(meeting, team, tasks) == [("assign", "Release checklist", "Can", None)]


def test_b1_still_reads_the_short_sample_requests():
    meeting, team = load_demo(1)
    found = _found(meeting, team)
    assert ("create", "login page", "Ayşe", "by the 6th") in found
    assert ("create", "database schema", "Can", "by the 4th") in found


# ---------- scorer ----------
def test_description_similarity_and_threshold():
    assert desc_similarity("load tests", "Load tests before the release") == 1.0
    assert desc_similarity("event tracking", "Event tracking setup") == 1.0
    assert desc_similarity("payment API endpoint", "API documentation") < 0.6


def test_scorer_counts_correct_wrong_missed_and_extra():
    scenario, evidence = load_key(DEMO)
    ids = {t.task_id: t.task_id for t in gold_tasks_before(scenario, 3)}
    proposals = [_p("complete", "T1", 2, ("L2",)),                               # correct
                 _p("assign", "T4", 1, ("L4", "L5"), owner="Ali"),             # wrong owner
                 _p("cancel", "T5", 1, ("L8",))]                               # extra
    rows, _ = score_meeting(3, proposals, scenario, evidence[3], ids)
    result = {r.event_id or r.proposed: r.result for r in rows}
    assert result["E8"] == "correct" and result["E9"] == "wrong" and result["E10"] == "missed" and result["E11"] == "missed"
    assert any(r.result == "extra" for r in rows)
    f = change_f1(rows)
    assert (f["all"]["tp"], f["all"]["fp"], f["all"]["fn"]) == (1, 2, 3)
    assert f["field_problems"] == {"owner wrong": 1}


def test_an_owner_left_open_must_stay_empty_unless_the_coordinator_answered():
    scenario, evidence = load_key(DEMO)
    guessed = _p("create", lines=("L6",), description="Load tests before the release", owner="Ali")
    answered = _p("create", lines=("L6",), description="Load tests before the release", owner="Ayşe")
    for p, want in ((guessed, "wrong"), (answered, "correct")):
        rows, _ = score_meeting(3, [p], scenario, evidence[3], {})
        assert next(r for r in rows if r.event_id == "E10").result == want


def test_state_accuracy():
    scenario, _ = load_key(DEMO)
    gold = gold_tasks_before(scenario, 3)
    expected = scenario["expected_state"]["2"]
    ids = {t.task_id: t.task_id for t in gold}
    assert state_accuracy(gold, expected, ids)["fully_correct"]
    wrong = [t.model_copy(update={"owner": "Can"}) if t.task_id == "T4" else t for t in gold]
    s = state_accuracy(wrong + [TaskRecord(task_id="T9", description="extra")], expected, ids)
    assert s["correct_pairs"] == s["pairs"] - 1 and s["extra_tasks"] == ["T9"] and not s["fully_correct"]


def test_simulated_answers_come_from_the_scenario():
    scenario, _ = load_key(DEMO)
    rows = simulated_answers(scenario)["answers"]
    assert {r["contains"] for r in rows} == {"release", "tests", "load"}
    assert {(r["meeting_no"], r["answer"]) for r in rows} == {(3, "Ayşe")}


# ---------- harness ----------
def test_b1_trial_in_both_modes(tmp_path):
    for mode in ("reset", "carry"):
        out = run_trial(DEMO, "b1", mode, out_root=tmp_path, say=lambda _: None)
        scores = json.loads((out / "scores.json").read_text())
        record = json.loads((out / "run_record.json").read_text())
        assert scores["change_f1"]["all"]["tp"] >= 7
        assert record["execution_mode"] == "live (no model)" and record["data_snapshot_sha"]
        assert (out / "report.md").read_text().startswith("# Evaluation run")
        if mode == "carry":
            assert scores["task_state"]["meetings"] == 3


def _meeting_two_payload():
    return json.dumps({"candidates": [
        {"action": "complete", "description": "Database schema", "owner": None, "due_phrase": None,
         "evidence": [{"utterance_id": "L2", "quote": "The database schema is finished"}], "confidence": 0.9}]})


def test_agent_trial_with_a_fake_model_is_marked_mock(tmp_path):
    client = FakeLLM([_meeting_one_payload(), _meeting_two_payload(), _meeting_three_payload(),
                      json.dumps({"tool": "escalate", "reason": "test"})])
    out = run_trial(DEMO, "agent", "reset", out_root=tmp_path, client=client, say=lambda _: None)
    scores = json.loads((out / "scores.json").read_text())
    assert scores["execution_mode"] == "mock"
    assert [m["stop_reason"] for m in scores["meetings"]] == ["completed"] * 3
    assert scores["meetings"][2]["case_outcomes"].get("escalated") == 1
    assert (out / "previews" / "m3.json").exists()
    # the saved previews can be replayed without a model
    again = run_trial(DEMO, "agent", "reset", out_root=tmp_path, replay=out / "previews", say=lambda _: None)
    assert json.loads((again / "scores.json").read_text())["change_f1"] == scores["change_f1"]
