"""SQLite store: append-only rules, project isolation, and the read-only agent view."""

import sqlite3
from datetime import date

import pytest

from app.agent.memory import MemoryReader, SimulatedClarification
from app.schemas import (
    ActionType,
    EvidenceQuote,
    EvidenceRef,
    EvidenceSource,
    ProposedUpdate,
    RouteKind,
    TaskRecord,
    TaskStatus,
    UpdateFields,
)
from app.store import MemoryStore, SqliteMemory, StoreError, project_db_path, reset_db_path
from app.store.schema import TABLES
from tests.demo_support import load_demo


@pytest.fixture
def db(tmp_path):
    return tmp_path / "memory.sqlite"


def _demo_store(path, meetings=(1,)):
    store = MemoryStore.create(path, "demo")
    team = None
    for meeting_no in meetings:
        meeting, team = load_demo(meeting_no)
        store.add_meeting(meeting)
    store.add_team_profile(team)
    return store


def _create_proposal(meeting_no=1, description="Login page", owner="Ayşe", line="L2"):
    return ProposedUpdate(
        project_id="demo",
        meeting_no=meeting_no,
        candidate_index=0,
        action=ActionType.create,
        task_id=None,
        task_description=description,
        fields=UpdateFields(description=description, owner=owner, due_date=date(2026, 11, 6)),
        evidence_refs=[EvidenceRef(source=EvidenceSource.transcript, id=line)],
        evidence_quotes=[EvidenceQuote(utterance_id=line, quote="login page")],
    )


def _commit_create(store, proposal, reviewer="Zeynep"):
    """What the commit step will do: record the review, then insert the version."""
    case_id = store.add_case(proposal.meeting_no, proposal.candidate_index, RouteKind.clear, [])
    proposal_id = store.add_proposal(proposal, case_id=case_id)
    review_id = store.add_review(proposal_id, decision="approve", reviewer=reviewer)
    return store.add_task_version(
        approval_id=review_id,
        task_id=None,
        base_version=None,
        description=proposal.fields.description,
        owner=proposal.fields.owner,
        due_date=proposal.fields.due_date,
        status=TaskStatus.open,
        meeting_no=proposal.meeting_no,
    )


# ----- append-only ---------------------------------------------------------


def test_every_table_has_append_only_triggers(db):
    with MemoryStore.create(db, "demo"):
        pass
    con = sqlite3.connect(db)
    names = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    for table in TABLES:
        assert f"{table}_no_update" in names
        assert f"{table}_no_delete" in names


@pytest.mark.parametrize("table", TABLES)
def test_update_and_delete_are_refused_on_every_table(db, table):
    with _demo_store(db) as store:
        _commit_create(store, _create_proposal())
        store.add_clarification_answer(
            store.add_clarification_question("M1-K0", "Who owns it?"), answered_by="Zeynep", answer="Ayşe"
        )
        store.add_run_record(run_id="r1", meeting_no=1, stop_reason="completed")
    con = sqlite3.connect(db)
    con.execute("PRAGMA recursive_triggers = ON")
    assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0
    column = con.execute(f"PRAGMA table_info({table})").fetchone()[1]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute(f"UPDATE {table} SET {column} = {column}")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute(f"DELETE FROM {table}")


def test_insert_or_replace_cannot_overwrite_a_task_version(db):
    with _demo_store(db) as store:
        _commit_create(store, _create_proposal())
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store._con.execute(
                "INSERT OR REPLACE INTO task_versions (task_no, task_id, version, description, status, source) "
                "VALUES (1, 'T1', 1, 'rewritten', 'open', 'seed')"
            )
        assert store.current_tasks()[0].description == "Login page"


def test_a_new_version_keeps_the_old_one(db):
    with _demo_store(db, meetings=(1, 2)) as store:
        created = _commit_create(store, _create_proposal())
        update = ProposedUpdate(
            project_id="demo",
            meeting_no=2,
            candidate_index=0,
            action=ActionType.complete,
            task_id=created.task_id,
            task_version=created.version,
            task_description="Login page",
            fields=UpdateFields(),
            evidence_refs=[EvidenceRef(source=EvidenceSource.transcript, id="L2")],
            evidence_quotes=[EvidenceQuote(utterance_id="L2", quote="login")],
        )
        proposal_id = store.add_proposal(update)
        review_id = store.add_review(proposal_id, decision="approve", reviewer="Zeynep")
        done = store.add_task_version(
            approval_id=review_id,
            task_id="T1",
            base_version=1,
            description="Login page",
            owner="Ayşe",
            due_date=date(2026, 11, 6),
            status=TaskStatus.done,
            meeting_no=2,
        )
        assert done.version == 2
        assert [task.status for task in store.current_tasks()] == [TaskStatus.done]
        history = store.task_history("T1")
        assert [item["record"].version for item in history] == [1, 2]
        assert history[0]["record"].status == TaskStatus.open
        assert history[1]["evidence_refs"] == ["M2-L2"]


# ----- commit rules the store keeps -----------------------------------------


def test_a_commit_version_needs_an_approve_review(db):
    with _demo_store(db) as store:
        proposal_id = store.add_proposal(_create_proposal())
        review_id = store.add_review(proposal_id, decision="reject", reviewer="Zeynep")
        kwargs = dict(
            task_id=None, base_version=None, description="Login page", owner=None,
            due_date=None, status=TaskStatus.open, meeting_no=1,
        )
        with pytest.raises(sqlite3.DatabaseError, match="approve review"):
            store.add_task_version(approval_id=review_id, **kwargs)
        with pytest.raises(sqlite3.DatabaseError, match="approve review"):
            store.add_task_version(approval_id="R99", **kwargs)
        assert store.current_tasks() == []


def test_one_approval_makes_one_version(db):
    with _demo_store(db) as store:
        created = _commit_create(store, _create_proposal())
        approval = store.task_history(created.task_id)[0]["approval_id"]
        with pytest.raises(sqlite3.IntegrityError):
            store.add_task_version(
                approval_id=approval, task_id=None, base_version=None, description="Again",
                owner=None, due_date=None, status=TaskStatus.open, meeting_no=1,
            )


def test_a_stale_base_version_is_refused(db):
    with _demo_store(db) as store:
        store.seed_tasks([TaskRecord(task_id="T1", description="Login page", version=2)])
        proposal_id = store.add_proposal(_create_proposal())
        review_id = store.add_review(proposal_id, decision="approve", reviewer="Zeynep")
        with pytest.raises(StoreError, match="version 2, not 1"):
            store.add_task_version(
                approval_id=review_id, task_id="T1", base_version=1, description="Login page",
                owner=None, due_date=None, status=TaskStatus.done, meeting_no=1,
            )


def test_new_task_ids_follow_the_highest_existing_number(db):
    with _demo_store(db) as store:
        store.seed_tasks([TaskRecord(task_id="T3", description="Old task")])
        created = _commit_create(store, _create_proposal())
        assert created.task_id == "T4"
        assert created.version == 1


def test_seed_only_fills_an_empty_task_list(db):
    with _demo_store(db) as store:
        store.seed_tasks([TaskRecord(task_id="T1", description="Login page")])
        with pytest.raises(StoreError, match="empty task list"):
            store.seed_tasks([TaskRecord(task_id="T2", description="Other")])
        assert store.task_history("T1")[0]["source"] == "seed"


def test_a_review_is_given_once_per_proposal(db):
    with _demo_store(db) as store:
        proposal_id = store.add_proposal(_create_proposal())
        store.add_review(proposal_id, decision="reject", reviewer="Zeynep")
        with pytest.raises(sqlite3.IntegrityError):
            store.add_review(proposal_id, decision="approve", reviewer="Zeynep")


# ----- project and run isolation -------------------------------------------


def test_records_from_another_project_are_refused(db):
    with _demo_store(db) as store:
        other = _create_proposal().model_copy(update={"project_id": "demo2"})
        with pytest.raises(StoreError, match="demo2"):
            store.add_proposal(other)


def test_opening_with_the_wrong_project_is_refused(db):
    MemoryStore.create(db, "demo").close()
    with pytest.raises(StoreError, match="belongs to project"):
        MemoryStore.open(db, project_id="demo2")


def test_create_never_reuses_an_existing_file(db):
    MemoryStore.create(db, "demo").close()
    with pytest.raises(StoreError, match="already exists"):
        MemoryStore.create(db, "demo")


def test_paths_keep_projects_runs_and_meetings_apart(tmp_path):
    a = project_db_path("demo", "run1", data_root=tmp_path)
    b = project_db_path("demo", "run2", data_root=tmp_path)
    r = reset_db_path("demo", "run1", "3", data_root=tmp_path)
    assert a == tmp_path / "demo" / "run1" / "memory.sqlite"
    assert r == tmp_path / "demo" / "run1" / "reset" / "3" / "memory.sqlite"
    assert len({a, b, r}) == 3
    for bad in ("..", "a/b", "", "../x"):
        with pytest.raises(ValueError):
            project_db_path(bad, "run1", data_root=tmp_path)


def test_two_runs_do_not_see_each_other(tmp_path):
    with _demo_store(project_db_path("demo", "run1", data_root=tmp_path)) as one:
        _commit_create(one, _create_proposal())
    with _demo_store(project_db_path("demo", "run2", data_root=tmp_path)) as two:
        assert two.current_tasks() == []


# ----- the agent's read-only view ------------------------------------------


def test_sqlite_memory_is_a_memory_reader_and_cannot_write(db):
    _demo_store(db).close()
    with SqliteMemory(db, project_id="demo") as memory:
        assert isinstance(memory, MemoryReader)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            memory._store._con.execute("INSERT INTO meetings (meeting_no, meeting_date) VALUES (9, '2026-01-01')")
        with pytest.raises(StoreError, match="read-only"):
            memory._store.add_run_record(run_id="r", meeting_no=1, stop_reason="x")


def test_round_trip_of_the_demo_meetings(db):
    with _demo_store(db, meetings=(1, 2, 3)) as store:
        for meeting_no in (1, 2, 3):
            assert store.meeting(meeting_no) == load_demo(meeting_no)[0]
        assert store.team_profile() == load_demo(1)[1]


def test_search_never_shows_later_meetings(db):
    _demo_store(db, meetings=(1, 2, 3)).close()
    with SqliteMemory(db) as memory:
        hits = memory.search_meetings("login page", current_meeting_no=1, max_results=5)
        assert hits
        assert all(hit.meeting_no == 1 for hit in hits)
        assert hits[0].ref.startswith("M1-L")
        later = memory.search_meetings("login page", current_meeting_no=3, max_results=20)
        assert {hit.meeting_no for hit in later} > {1}


def test_task_history_and_current_tasks_through_the_reader(db):
    with _demo_store(db) as store:
        _commit_create(store, _create_proposal())
    with SqliteMemory(db) as memory:
        assert [task.task_id for task in memory.current_tasks()] == ["T1"]
        (view,) = memory.get_task_history("T1")
        assert view.owner == "Ayşe"
        assert view.evidence_refs == ["M1-L2"]
        assert memory.get_task_history("T9") == []
        assert memory.get_team_profile().coordinator == "Zeynep"


def test_simulated_answers_match_the_demo_rule(db):
    _demo_store(db).close()
    answers = [SimulatedClarification(contains="api", answer="T4", meeting_no=3)]
    with SqliteMemory(db, clarifications=answers) as memory:
        assert memory.simulated_answer("Which API task?", meeting_no=3) == "T4"
        assert memory.simulated_answer("Which API task?", meeting_no=2) == "no answer"
