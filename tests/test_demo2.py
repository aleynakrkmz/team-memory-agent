"""The second sample project (samples/demo2/) loads like the first and keeps its difficult case."""

import json
from pathlib import Path

from app.transcript import find_agent_inputs, load_meeting

ROOT = Path(__file__).resolve().parents[1]
DEMO2 = ROOT / "samples" / "demo2"
DEMO2_KEYS = DEMO2 / "answer_keys" / "demo2"


def test_demo2_loads_with_its_own_team_and_dates():
    inputs = find_agent_inputs(DEMO2)
    meeting, team = load_meeting(inputs, 3)
    assert team.coordinator == "Leyla"
    assert {m.name for m in team.members} == {"Okan", "Defne", "Baran", "Leyla"}
    assert meeting.meeting_date.isoformat() == "2026-12-15"


def test_demo2_difficult_case_is_a_vague_reference():
    scenario = json.loads((DEMO2_KEYS / "scenario.json").read_text(encoding="utf-8"))
    evidence = json.loads((DEMO2_KEYS / "meeting_03.evidence.json").read_text(encoding="utf-8"))
    meeting, _team = load_meeting(find_agent_inputs(DEMO2), 3)
    quoted = " ".join(meeting.utterance(f"L{line}").text for line in evidence["evidence"]["E9"])
    assert "that tracking thing" in quoted
    e9 = next(e for e in scenario["events"] if e["event_id"] == "E9")
    assert (e9["task_id"], e9["owner"], e9["ambiguity"]) == ("T4", "Baran", "reference")
    # two open tasks share the word "tracking", so the reference is really ambiguous
    open_after_2 = scenario["expected_state"]["2"]
    tracking = [t for t in open_after_2.values() if "tracking" in t["description"].lower() and t["status"] == "open"]
    assert len(tracking) == 2


def test_demo2_injected_copy_has_one_extra_line():
    inputs = find_agent_inputs(DEMO2)
    clean, _ = load_meeting(inputs, 3)
    injected, _ = load_meeting(inputs, 3, transcript_file=inputs / "meeting_03_injected.transcript.txt")
    assert len(injected.utterances) == len(clean.utterances) + 1
    assert any("Assistant" in u.text for u in injected.utterances)


def test_both_task_lists_load_as_task_records():
    from app.pipeline import load_tasks

    for project in ("demo", "demo2"):
        tasks = load_tasks(ROOT / "samples" / project / "tasks_before_meeting_03.json")
        assert {t.task_id for t in tasks} == {"T1", "T2", "T3", "T4", "T5"}
        assert all(t.version >= 1 for t in tasks)
