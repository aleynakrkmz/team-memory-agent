"""Demo-project fixtures. Answer keys are read only as the test oracle."""

import json
from pathlib import Path

from app.schemas import TaskRecord, TaskStatus
from app.transcript import load_meeting

ROOT = Path(__file__).resolve().parents[1]
DEMO_INPUTS = ROOT / "samples" / "demo" / "agent_inputs" / "demo"
DEMO_KEYS = ROOT / "samples" / "demo" / "answer_keys" / "demo"


def load_demo(meeting_no: int, *, transcript_name: str | None = None):
    transcript = None if transcript_name is None else DEMO_INPUTS / transcript_name
    return load_meeting(DEMO_INPUTS, meeting_no, transcript_file=transcript)


def open_tasks_before(meeting_no: int) -> list[TaskRecord]:
    """Current list the pipeline would see at the start of `meeting_no`."""
    if meeting_no == 1:
        return []
    scenario = json.loads((DEMO_KEYS / "scenario.json").read_text(encoding="utf-8"))
    state = scenario["expected_state"][str(meeting_no - 1)]
    tasks = []
    for task_id, row in state.items():
        tasks.append(
            TaskRecord(
                task_id=task_id,
                description=row["description"],
                owner=row["owner"],
                due_date=row["due_date"],
                status=TaskStatus(row["status"]),
            )
        )
    return tasks
