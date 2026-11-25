# Demo run: meetings 1–3

Date: 25 Nov 2026. Model `qwen3:8b` through Ollama, thinking off,
temperature 0, 32k context, 120 s per call. Apple M5, 24 GB.

Each meeting ran in its own fresh database (reset-to-gold): the database held
the earlier meetings and the reference task list from before the meeting.
Expected changes come from `samples/demo/answer_keys/demo/scenario.json`.

## Results

| Meeting | Expected | First run (24 Nov) | This run (25 Nov) | Time |
|---------|---------:|--------------------|-------------------|-----:|
| 1 | 3 | 0, extraction rejected | **3**, all correct | 75 s |
| 2 | 4 | 0, stopped by a stuck model call | **2** | 67 s |
| 3 | 4 | 2 | **2** | 90 s |
| Total | 11 | 2 | **7** | |

No wrong proposals in either run. No model call timed out in this run.

### Meeting 1
All three new tasks proposed: Login page (Ayşe, 6 Nov), Database schema
(Can, 4 Nov), Project report draft (Zeynep, no date).

### Meeting 2
- Correct: create Payment API endpoint (Ali, 18 Nov) and API documentation
  (Ayşe, 30 Nov).
- Missed: T2 Database schema done (L2) was not extracted.
- Missed: T1 Login page moved to 10 Nov. Extraction labelled it `assign`; the
  agent may not change a case's action, so it abstained.

### Meeting 3
- Correct: complete T1 Login page, cancel T3 Project report draft.
- Escalated: reassign T4 to Can. Extraction left the owner empty (the owner is
  in L5, "Sure, I can take it"). The agent used all 6 steps and the case went
  to the coordinator.
- Missed: create "Load tests before the release" (found in the first run).

## What changed between the runs

1. **Stuck model calls.** qwen3 sometimes closed a JSON key with a curly
   quote (`"task_id”: ...`). The reply schemas allowed any key name, so the
   grammar accepted the broken key and the reply never ended. The agent and
   extraction schemas now allow only their listed keys. The same captured
   call that timed out at 120 s now answers in about 9 s.
2. **One bad call no longer stops a meeting.** An Ollama 5xx reply
   (`token repeat limit reached`) becomes `OllamaGenerationError`. The agent
   escalates only that case; extraction retries once.
3. **Exact quotes.** A rejected quote's retry message shows the wrong quote
   and the exact line, so the model copies it correctly.
4. **Create or assign.** Extraction sees the current task list and the prompt
   says create is for tasks that are not on it.
5. **Store connected.** The pipeline runs with `--db` (see `storage.md`).

## Open problems

- Extraction misses some lines (a finished task, a task with no owner).
- A deadline change can come out as `assign`, and the agent cannot fix the
  action.
- When the new owner is named in a later line, extraction leaves it empty.
- Validate, review and commit are not built yet, so approved changes do not
  update the task list.

## Reproduce

```bash
uv run python -m app.pipeline samples/demo --meeting 1 --db data/demo/r1/memory.sqlite
uv run python -m app.pipeline samples/demo --meeting 3 --db data/demo/r3/memory.sqlite --tasks samples/demo/tasks_before_meeting_03.json
```

Meeting 2 needs the reference task list after meeting 1 as `--tasks`
(T1–T3 from `expected_state["1"]` in the demo scenario).
