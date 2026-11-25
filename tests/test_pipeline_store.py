"""The pipeline with --db, and model calls that Ollama aborts."""

import json
import sqlite3

import pytest

from app.agent.config import AgentConfig
from app.agent.loop import run_case
from app.extract import ExtractionError, extract_candidates
from app.extract.client import OllamaClient, OllamaGenerationError
from app.match import match_candidates
from app.pipeline import main
from app.route import route_candidates
from app.schemas import AgentOutcome, RouteKind
from app.store import MemoryStore, SqliteMemory
from tests.demo_support import ROOT, load_demo, open_tasks_before
from tests.fakes import FakeLLM
from tests.test_extract import _meeting_one_payload
from tests.test_pipeline import _meeting_three_payload

DEMO = str(ROOT / "samples" / "demo")


def _tasks_file(tmp_path, meeting_no):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([t.model_dump(mode="json") for t in open_tasks_before(meeting_no)]), encoding="utf-8")
    return path


def _agent_reply_assign_t4():
    return [
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


# ----- --db --------------------------------------------------------------------


def test_db_run_stores_the_meeting_cases_proposals_and_run_record(tmp_path, capsys):
    db = tmp_path / "memory.sqlite"
    code = main([DEMO, "--meeting", "1", "--db", str(db), "--run-id", "r1"], client=FakeLLM([_meeting_one_payload()]))
    captured = capsys.readouterr()
    assert code == 0
    preview = json.loads(captured.out)
    with MemoryStore.open(db, project_id="demo", read_only=True) as store:
        assert store.meeting_numbers() == [1]
        assert store.meeting(1) == load_demo(1)[0]
        assert store.team_profile() == load_demo(1)[1]
        assert store.current_tasks() == []  # proposals are not commits
        (record,) = store.run_records("r1")
        assert record["stop_reason"] == "completed"
        metrics = json.loads(record["metrics_json"])
        assert metrics["proposals"] == len(preview["proposals"]) == 3
        assert [store.proposal(pid).action.value for pid in metrics["proposal_ids"]] == ["create"] * 3
    assert "db: stored 3 cases, 3 proposals" in captured.err


def test_db_seeds_tasks_then_reads_them_and_stores_earlier_meetings(tmp_path, capsys):
    db = tmp_path / "memory.sqlite"
    client = FakeLLM([_meeting_three_payload(), *_agent_reply_assign_t4()])
    code = main(
        [DEMO, "--meeting", "3", "--db", str(db), "--tasks", str(_tasks_file(tmp_path, 3)), "--run-id", "r3"],
        client=client,
    )
    captured = capsys.readouterr()
    assert code == 0
    preview = json.loads(captured.out)
    assert [d["route"] for d in preview["decisions"]] == ["clear", "hard", "clear", "clear"]
    assert preview["agent_decisions"][0]["outcome"] == "propose_update"
    with MemoryStore.open(db, read_only=True) as store:
        assert store.meeting_numbers() == [1, 2, 3]
        assert [t.task_id for t in store.current_tasks()] == ["T1", "T2", "T3", "T4", "T5"]
        assert {item["source"] for item in store.task_history("T4")} == {"seed"}
    # The agent searched the database, which now holds meetings 1-3.
    search = json.loads(preview["agent_decisions"][0]["trace"]["steps"][0]["observation"])
    assert search
    assert "seeded 5 tasks" in captured.err


def test_db_second_run_reads_the_stored_task_list(tmp_path, capsys):
    db = tmp_path / "memory.sqlite"
    with MemoryStore.create(db, "demo") as store:
        store.seed_tasks(open_tasks_before(3))
    code = main(
        [DEMO, "--meeting", "3", "--db", str(db)],
        client=FakeLLM([_meeting_three_payload(), *_agent_reply_assign_t4()]),
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "db: 5 current tasks" in captured.err
    assert "empty list" not in captured.err


def test_db_refuses_tasks_when_the_list_is_not_empty(tmp_path, capsys):
    db = tmp_path / "memory.sqlite"
    with MemoryStore.create(db, "demo") as store:
        store.seed_tasks(open_tasks_before(3))
    code = main(
        [DEMO, "--meeting", "3", "--db", str(db), "--tasks", str(_tasks_file(tmp_path, 3))],
        client=FakeLLM([]),
    )
    assert code == 1
    assert "already has a task list" in capsys.readouterr().err


def test_db_refuses_a_changed_transcript_for_a_stored_meeting(tmp_path, capsys):
    db = tmp_path / "memory.sqlite"
    assert main([DEMO, "--meeting", "1", "--db", str(db)], client=FakeLLM([_meeting_one_payload()])) == 0
    injected = ROOT / "samples" / "demo" / "agent_inputs" / "demo" / "meeting_03_injected.transcript.txt"
    with MemoryStore.open(db) as store:
        store.add_meeting(load_demo(3)[0])
    capsys.readouterr()
    code = main([DEMO, "--meeting", "3", "--db", str(db), "--transcript-file", str(injected)], client=FakeLLM([]))
    assert code == 1
    assert "already stored with different lines" in capsys.readouterr().err


def test_db_records_an_extraction_failure(tmp_path, capsys):
    db = tmp_path / "memory.sqlite"
    code = main([DEMO, "--meeting", "1", "--db", str(db), "--run-id", "bad"], client=FakeLLM(["nope", "nope"]))
    assert code == 1
    with MemoryStore.open(db, read_only=True) as store:
        (record,) = store.run_records("bad")
        assert record["stop_reason"] == "extraction_failed"
        assert "invalid JSON" in record["failure_detail"]


def test_db_run_never_writes_through_the_agent_reader(tmp_path):
    db = tmp_path / "memory.sqlite"
    main([DEMO, "--meeting", "1", "--db", str(db)], client=FakeLLM([_meeting_one_payload()]))
    with SqliteMemory(db) as reader:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader._store._con.execute("INSERT INTO meetings (meeting_no, meeting_date) VALUES (9, '2026-01-01')")


# ----- Ollama aborts a reply ----------------------------------------------------


class _ResponseError(Exception):
    """Same shape as ollama.ResponseError (name, .error, .status_code)."""

    def __init__(self, error, status_code):
        super().__init__(error)
        self.error = error
        self.status_code = status_code


_ResponseError.__name__ = "ResponseError"


def test_client_turns_a_5xx_reply_into_a_generation_error():
    def chat(**kwargs):
        raise _ResponseError("prediction aborted, token repeat limit reached", 500)

    with pytest.raises(OllamaGenerationError, match="token repeat limit"):
        OllamaClient(model="m", chat=chat).complete_json(system="s", user="u", json_schema={})


def test_client_keeps_a_missing_model_fatal():
    def chat(**kwargs):
        raise _ResponseError("model 'x' not found", 404)

    with pytest.raises(_ResponseError):
        OllamaClient(model="m", chat=chat).complete_json(system="s", user="u", json_schema={})


class _ScriptedLLM(FakeLLM):
    """FakeLLM that raises when a scripted response is an exception."""

    def complete_json(self, **kwargs):
        value = super().complete_json(**kwargs)
        if isinstance(value, Exception):
            raise value
        return value


def test_extraction_retries_once_after_an_aborted_reply():
    meeting, _ = load_demo(1)
    client = _ScriptedLLM([OllamaGenerationError("token repeat limit reached", 500), _meeting_one_payload()])
    assert len(extract_candidates(meeting, client)) == 3
    assert "repeated itself" in client.calls[1]["user"]


def test_extraction_gives_up_after_two_aborted_replies():
    meeting, _ = load_demo(1)
    error = OllamaGenerationError("token repeat limit reached", 500)
    with pytest.raises(ExtractionError, match="token repeat limit"):
        extract_candidates(meeting, _ScriptedLLM([error, error]))


def _hard_case():
    meeting, team = load_demo(3)
    tasks = open_tasks_before(3)
    from app.extract.extract import _parse_output

    candidates = _parse_output(_meeting_three_payload(), meeting)
    matches = match_candidates(candidates, tasks, meeting.meeting_date)
    decisions = route_candidates(meeting, candidates, matches, tasks, team)
    index = next(i for i, d in enumerate(decisions) if d.route == RouteKind.hard)
    return meeting, team, tasks, candidates[index], matches[index], decisions[index]


def test_an_aborted_agent_call_escalates_only_that_case(tmp_path):
    meeting, team, tasks, candidate, match, decision = _hard_case()
    from app.agent.memory import DemoMemory
    from tests.demo_support import DEMO_INPUTS

    reader = DemoMemory.load(DEMO_INPUTS, tasks=tasks, team=team, current_meeting=meeting)
    error = OllamaGenerationError("prediction aborted, token repeat limit reached", 500)
    result = run_case(
        meeting=meeting, candidate=candidate, match=match, decision=decision, tasks=tasks, team=team,
        client=_ScriptedLLM([error, error]), reader=reader,
        config=AgentConfig(max_steps=6, time_budget_seconds=300, max_invalid_outputs=2, model="m", search_max_results=5),
    )
    assert result.outcome == AgentOutcome.escalate
    assert result.automatic
    assert "token repeat limit" in result.reason
    assert [step.valid for step in result.trace.steps] == [False, False]


# ----- reply grammars stay closed ---------------------------------------------


def _objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _objects(item)


def test_agent_reply_schema_allows_only_listed_keys():
    from app.agent.tools import TOOL_JSON_SCHEMA

    assert TOOL_JSON_SCHEMA["additionalProperties"] is False
    assert "arguments" not in TOOL_JSON_SCHEMA["properties"]


def test_extraction_reply_schema_allows_only_listed_keys():
    meeting, _ = load_demo(1)
    client = FakeLLM([_meeting_one_payload()])
    extract_candidates(meeting, client)
    objects = list(_objects(client.calls[0]["json_schema"]))
    assert objects
    assert all(obj["additionalProperties"] is False for obj in objects)


# ----- extraction message ------------------------------------------------------


def test_quote_rejection_shows_the_exact_line():
    meeting, _ = load_demo(1)
    bad = json.loads(_meeting_one_payload())
    bad["candidates"][1]["evidence"] = [{"utterance_id": "L4", "quote": "Can, please finish it by the 4th."}]
    client = FakeLLM([json.dumps(bad), _meeting_one_payload()])
    extract_candidates(meeting, client)
    retry = client.calls[1]["user"]
    assert "quote is not in that line" in retry
    assert '"Can, please finish it by the 4th."' in retry
    assert '"Can, the database schema is yours, please finish it by the 4th."' in retry


def test_extraction_sees_the_current_task_list():
    from app.extract.extract import FENCE_START, TASKS_END, TASKS_START

    meeting, _ = load_demo(3)
    client = FakeLLM([_meeting_three_payload()])
    extract_candidates(meeting, client, tasks=open_tasks_before(3))
    user = client.calls[0]["user"]
    assert user.index(TASKS_START) < user.index(TASKS_END) < user.index(FENCE_START)
    assert "T4 | Payment API endpoint | Ali" in user


def test_an_empty_task_list_is_shown_as_empty_and_none_leaves_it_out():
    from app.extract.extract import TASKS_START

    meeting, _ = load_demo(1)
    client = FakeLLM([_meeting_one_payload(), _meeting_one_payload()])
    extract_candidates(meeting, client, tasks=[])
    extract_candidates(meeting, client)
    assert "no tasks exist yet" in client.calls[0]["user"]
    assert TASKS_START not in client.calls[1]["user"]
