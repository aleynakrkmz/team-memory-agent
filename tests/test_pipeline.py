"""The preview command wires extract, match and route without calling Ollama."""

import json

from app.pipeline import main
from app.schemas import RouteKind
from tests.demo_support import ROOT, open_tasks_before
from tests.fakes import FakeLLM
from tests.test_extract import _meeting_one_payload


def _meeting_three_payload() -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "action": "complete",
                    "description": "Login page",
                    "owner": None,
                    "due_phrase": None,
                    "evidence": [{"utterance_id": "L2", "quote": "The login page is done"}],
                    "confidence": 0.95,
                },
                {
                    "action": "assign",
                    "description": "that API thing from last week",
                    "owner": "Can",
                    "due_phrase": None,
                    "evidence": [{"utterance_id": "L4", "quote": "that API thing from last week"}],
                    "confidence": 0.6,
                },
                {
                    "action": "create",
                    "description": "Load tests before the release",
                    "owner": None,
                    "due_phrase": None,
                    "evidence": [{"utterance_id": "L6", "quote": "load tests before the release"}],
                    "confidence": 0.7,
                },
                {
                    "action": "cancel",
                    "description": "Project report draft",
                    "owner": None,
                    "due_phrase": None,
                    "evidence": [{"utterance_id": "L8", "quote": "dropping the project report draft"}],
                    "confidence": 0.9,
                },
            ]
        }
    )


def test_cli_prints_clear_creates_for_the_first_demo_meeting(capsys):
    code = main([str(ROOT / "samples" / "demo"), "--meeting", "1"], client=FakeLLM([_meeting_one_payload()]))
    captured = capsys.readouterr()
    assert code == 0
    preview = json.loads(captured.out)
    assert preview["meeting_date"] == "2026-10-29"
    assert [item["route"] for item in preview["decisions"]] == ["clear", "clear", "clear"]
    assert preview["decisions"][0]["proposal"]["fields"]["due_date"] == "2026-11-06"
    assert preview["agent_decisions"] == []
    assert [item["origin"] for item in preview["proposals"]] == ["clear_path", "clear_path", "clear_path"]
    assert "empty list" in captured.err


def test_cli_routes_the_vague_reassignment_when_given_the_open_tasks(tmp_path, capsys):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps([task.model_dump(mode="json") for task in open_tasks_before(3)]),
        encoding="utf-8",
    )
    code = main(
        [str(ROOT / "samples" / "demo"), "--meeting", "3", "--tasks", str(tasks_path)],
        client=FakeLLM(
            [
                _meeting_three_payload(),
                json.dumps({"tool": "escalate", "reason": "this test only checks the route"}),
            ]
        ),
    )
    captured = capsys.readouterr()
    assert code == 0
    preview = json.loads(captured.out)
    routes = [item["route"] for item in preview["decisions"]]
    assert routes == [RouteKind.clear.value, RouteKind.hard.value, RouteKind.clear.value, RouteKind.clear.value]
    assert preview["decisions"][1]["proposal"] is None
    assert preview["decisions"][1]["matched_task_id"] is None
    assert preview["agent_decisions"][0]["outcome"] == "escalate"
    assert preview["agent_decisions"][0]["proposal"] is None
    assert [item["origin"] for item in preview["proposals"]] == ["clear_path", "clear_path", "clear_path"]
    assert "empty list" not in captured.err


def test_cli_requires_a_model_when_no_client_is_injected(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_MODEL", " ")
    code = main([str(ROOT / "samples" / "demo"), "--meeting", "1"])
    captured = capsys.readouterr()
    assert code == 2
    assert "AGENT_MODEL" in captured.err
    assert captured.out == ""


def test_pipeline_runs_the_agent_only_on_the_hard_case(tmp_path, capsys):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps([task.model_dump(mode="json") for task in open_tasks_before(3)]),
        encoding="utf-8",
    )
    out = tmp_path / "preview.json"
    client = FakeLLM(
        [
            _meeting_three_payload(),
            json.dumps({"tool": "search_meetings", "query": "payment API endpoint"}),
            json.dumps(
                {
                    "tool": "propose_update",
                    "action": "assign",
                    "task_id": "T4",
                    "description": "Payment API endpoint",
                    "owner": "Can",
                    "evidence_refs": ["L4", "L5"],
                }
            ),
        ]
    )
    code = main(
        [
            str(ROOT / "samples" / "demo"),
            "--meeting",
            "3",
            "--tasks",
            str(tasks_path),
            "--out",
            str(out),
        ],
        client=client,
    )
    captured = capsys.readouterr()
    assert code == 0
    preview = json.loads(captured.out)
    assert json.loads(out.read_text(encoding="utf-8")) == preview
    assert [item["route"] for item in preview["decisions"]] == ["clear", "hard", "clear", "clear"]
    assert preview["decisions"][1]["proposal"] is None
    assert [(item["action"], item["origin"], item["task_id"]) for item in preview["proposals"]] == [
        ("complete", "clear_path", "T1"),
        ("assign", "agent", "T4"),
        ("create", "clear_path", None),
        ("cancel", "clear_path", "T3"),
    ]
    assert preview["proposals"][2]["fields"]["owner"] is None
    agent = preview["agent_decisions"][0]
    assert agent["outcome"] == "propose_update"
    assert [step["tool"] for step in agent["trace"]["steps"]] == ["search_meetings", "propose_update"]
    assert agent["proposal"]["fields"] == {"owner": "Can"}
    assert agent["proposal"]["task_description"] == "Payment API endpoint"
    assert [ref["id"] for ref in agent["proposal"]["evidence_refs"]] == ["L4", "L5"]
    assert len(client.calls) == 3
    assert "empty list" not in captured.err


def test_extraction_failure_stops_before_the_agent(capsys):
    code = main(
        [str(ROOT / "samples" / "demo"), "--meeting", "3"],
        client=FakeLLM(["not json", "still not json"]),
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "extraction failed" in captured.err
    assert captured.out == ""


def test_cli_prints_stage_progress_on_stderr_and_keeps_stdout_json(tmp_path, capsys):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps([task.model_dump(mode="json") for task in open_tasks_before(3)]),
        encoding="utf-8",
    )
    code = main(
        [str(ROOT / "samples" / "demo"), "--meeting", "3", "--tasks", str(tasks_path)],
        client=FakeLLM(
            [
                _meeting_three_payload(),
                json.dumps({"tool": "escalate", "reason": "progress check"}),
            ]
        ),
    )
    captured = capsys.readouterr()
    assert code == 0
    preview = json.loads(captured.out)
    assert preview["meeting_no"] == 3
    assert "extract..." in captured.err
    assert "agent case 1/1 step 1..." in captured.err
    assert "extract..." not in captured.out
    assert "agent case" not in captured.out


def test_application_code_does_not_open_answer_keys():
    app = ROOT / "app"
    for path in app.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "answer_keys" not in text
        assert "scenario.json" not in text
