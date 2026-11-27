"""The evaluation harness: run one configuration on one project and score it (design §12.4, §12.7).

    uv run python eval/harness.py samples/demo --config agent --mode reset
    uv run python eval/harness.py samples/demo --config b1 --mode carry
    uv run python eval/harness.py samples/demo --config agent --mode reset --replay runs/eval/<earlier run>/previews

A trial processes every meeting of the project in order. Each meeting:

1. **Runs the configuration.** `agent`: the full pipeline (app.pipeline, with the database). `b1`: the rule
   baseline (eval/b1.py), no model. `--replay DIR` uses the previews an earlier agent run saved instead of
   calling the model (recorded replay). If extraction fails twice the meeting is processed by the B1 rules
   (design §11, failure 3) and this is recorded as an extraction fallback.
2. **Simulated reviewer.** Deterministic code, not a model: every proposal that validation passes is approved
   by the team's coordinator and committed; invalid and blocked proposals are not (`validation_blocked`).
   Approval is not a correctness verdict: a wrong proposal that passes validation is committed too.
   Clarification questions get the scenario's answer (or "not decided") through the pipeline's simulated answers.
3. **Scoring** (eval/scorer.py): change F1 in both modes, and task-state accuracy after each meeting in carry mode.

Modes: `reset` (reset-to-gold) starts every meeting in a fresh database seeded with the correct task list from
before the meeting; `carry` (carry-forward) uses one database, so approved changes and mistakes carry on.

Everything goes to runs/eval/<run_id>/ (git-ignored): run_record.json (what is needed to reproduce the number:
model and digest, prompt and config hashes, package versions, data snapshot, git commit, execution mode),
scores.json, rows.csv, report.md, previews/ and the databases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.commit.commit import CommitError, approve_and_commit  # noqa: E402
from app.match.match import tokenize  # noqa: E402
from app.review.queue import clarification_ids, meeting_proposals  # noqa: E402
from app.schemas import MeetingPreview, TaskRecord  # noqa: E402
from app.store.session import open_or_create, prepare_meeting, record_preview  # noqa: E402
from app.store.store import MemoryStore  # noqa: E402
from app.transcript import find_agent_inputs, load_meeting  # noqa: E402
from app.validate.validate import Rule2Settings, validate_meeting  # noqa: E402
from eval.b1 import run_b1  # noqa: E402
from eval.scorer import change_f1, score_meeting, state_accuracy  # noqa: E402

CONFIGS = ("agent", "b1")
MODES = ("reset", "carry")
STOP_REASONS = ("completed", "budget_exhausted", "invalid_action", "tool_failure", "approval_denied")


# ---------- the answer key and the reproducibility record ----------
def load_key(project: Path) -> tuple[dict, dict[int, dict]]:
    pid = find_agent_inputs(project).name
    keys = project / "answer_keys" / pid
    scenario = json.loads((keys / "scenario.json").read_text(encoding="utf-8"))
    evidence = {}
    for m in range(1, len(scenario["meeting_dates"]) + 1):
        evidence[m] = json.loads((keys / f"meeting_{m:02d}.evidence.json").read_text(encoding="utf-8"))["evidence"]
    return scenario, evidence


def gold_tasks_before(scenario: dict, meeting_no: int) -> list[TaskRecord]:
    """The correct task list at the start of a meeting (reset-to-gold seed)."""
    state = scenario["expected_state"].get(str(meeting_no - 1), {})
    return [TaskRecord(task_id=tid, description=t["description"], owner=t["owner"], due_date=t["due_date"],
                       status=t["status"],
                       version=sum(1 for e in scenario["events"] if e["task_id"] == tid and e["meeting"] < meeting_no))
            for tid, t in state.items()]


def simulated_answers(scenario: dict) -> dict:
    """Answers for clarification questions: for every event whose owner is left open on purpose, the scenario's
    answer. The pipeline gives the first answer whose `contains` word appears in the question, so every word of
    the task title (4 letters or more, longest first) gets a row."""
    rows = []
    for e in scenario["events"]:
        if e.get("ambiguity") == "owner":
            words = sorted({w for w in tokenize(scenario["tasks"][e["task_id"]]["description"]) if len(w) >= 4},
                           key=lambda w: (-len(w), w))
            for w in words:
                rows.append({"meeting_no": e["meeting"], "contains": w,
                             "answer": e.get("coordinator_answer") or "not decided"})
    return {"answers": rows}


def _sha(paths: list[Path], base: Path = ROOT) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(base)).encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def model_identity(config: str, execution_mode: str) -> dict:
    if config == "b1":
        return {"model": None, "note": "B1 uses no model"}
    from app.agent.config import load_agent_config, resolve_agent_model
    cfg = load_agent_config()
    info = {"model": resolve_agent_model(), "think": cfg.think, "num_ctx": 32768, "temperature": 0}
    if execution_mode == "live":
        try:
            import os
            from ollama import Client
            for m in Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434")).list()["models"]:
                if m["model"] == info["model"]:
                    info["digest"] = m["digest"]
        except Exception as exc:  # noqa: BLE001
            info["digest"] = f"not recorded ({type(exc).__name__})"
    return info


def run_record(project: Path, config: str, mode: str, execution_mode: str, run_id: str, replay: Path | None) -> dict:
    pkgs = {}
    for name in ("ollama", "pydantic", "rank-bm25", "python-dotenv"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    return {
        "run_id": run_id, "project": project.name, "configuration": config, "mode": mode,
        "execution_mode": execution_mode, "replayed_from": str(replay) if replay else None,
        "model": model_identity(config, execution_mode),
        "prompts_sha": _sha(list((ROOT / "prompts").glob("*.md"))),
        "config_sha": _sha(list((ROOT / "config").glob("*.json"))),
        "code": {"git_commit": _git("rev-parse", "HEAD"), "uncommitted_changes": bool(_git("status", "--porcelain"))},
        "packages": pkgs, "python": platform.python_version(), "uv_lock_sha": _sha([ROOT / "uv.lock"]),
        "data_snapshot_sha": _sha([p for p in project.rglob("*") if p.is_file()], base=project),
        "machine": platform.platform(), "seed": None,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------- one meeting ----------
def make_preview(config: str, project: Path, meeting_no: int, db: Path, seed: list[TaskRecord] | None,
                 out: Path, answers_file: Path, run_id: str, replay: Path | None, client=None) -> tuple[MeetingPreview | None, dict]:
    """Run the configuration for one meeting and store its proposals. Returns (preview, events)."""
    inputs = find_agent_inputs(project)
    meeting, team = load_meeting(inputs, meeting_no)
    events = {"stop_reason": "completed", "failure_detail": None, "extraction_fallback": False}
    if config == "agent" and replay is None:
        from app.pipeline import main as pipeline_main
        argv = [str(project), "--meeting", str(meeting_no), "--db", str(db), "--out", str(out),
                "--clarifications", str(answers_file), "--run-id", f"{run_id}-m{meeting_no}"]
        if seed is not None:
            seed_file = out.with_suffix(".seed.json")
            seed_file.write_text(json.dumps([t.model_dump(mode="json") for t in seed]), encoding="utf-8")
            argv += ["--tasks", str(seed_file)]
        log = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(log):
            code = pipeline_main(argv, client=client)
        events["log"] = log.getvalue()[-2000:]
        if code == 0:
            return MeetingPreview.model_validate_json(out.read_text(encoding="utf-8")), events
        with MemoryStore.open(db) as store:
            last = (store.run_records() or [{}])[-1]
        if last.get("stop_reason") == "extraction_failed":
            events.update(extraction_fallback=True, failure_detail="invalid_output")
            with MemoryStore.open(db) as store:
                preview = run_b1(meeting, store.current_tasks(), team)
                record_preview(store, preview, run_id=f"{run_id}-m{meeting_no}-b1", seconds=0)
            out.write_text(preview.model_dump_json(indent=2), encoding="utf-8")
            return preview, events
        events.update(stop_reason="tool_failure", failure_detail=(last.get("failure_detail") or events["log"])[-300:])
        return None, events

    with open_or_create(db, meeting.project_id) as store:
        tasks, _ = prepare_meeting(store, agent_inputs=inputs, meeting=meeting, team=team, seed=seed)
        if replay is not None:
            preview = MeetingPreview.model_validate_json((replay / f"m{meeting_no}.json").read_text(encoding="utf-8"))
        else:
            preview = run_b1(meeting, tasks, team)
        record_preview(store, preview, run_id=f"{run_id}-m{meeting_no}", seconds=0)
    out.write_text(preview.model_dump_json(indent=2), encoding="utf-8")
    return preview, events


def simulated_review(db: Path, meeting_no: int, settings: Rule2Settings) -> list[dict]:
    """Approve every proposal of the meeting that validation passes; record what happened to each."""
    out = []
    with MemoryStore.open(db) as store:
        coordinator = store.team_profile().coordinator
        stored = meeting_proposals(store, meeting_no)
        results = validate_meeting([p for _, p in stored], meeting=store.meeting(meeting_no), team=store.team_profile(),
                                   tasks=store.current_tasks(), clarification_ids=clarification_ids(store), settings=settings)
        for (pid, _), check in zip(stored, results):
            row = {"proposal_id": pid, "validation": check.status, "issues": check.summary(), "committed_as": None}
            if check.status == "valid":
                try:
                    _, task = approve_and_commit(store, pid, reviewer=coordinator, settings=settings)
                    row["committed_as"] = task.task_id
                except CommitError as exc:   # e.g. stale after an earlier approval in the same meeting
                    row["validation"], row["issues"] = "stale", str(exc)
            out.append(row)
        tasks_after = store.current_tasks()
    return out, tasks_after


def case_outcomes(preview: MeetingPreview | None, review: list[dict], config: str) -> dict:
    """Counts of case outcomes for one meeting (design §11)."""
    c = {}
    if preview is None:
        return c
    agent = {a.candidate_index: a for a in preview.agent_decisions}
    proposal_of = {p.candidate_index: i for i, p in enumerate(preview.proposals)}
    for d in preview.decisions:
        i = d.candidate_index
        if i in proposal_of:
            r = review[proposal_of[i]] if proposal_of[i] < len(review) else {}
            key = {"valid": "proposed", "stale": "stopped"}.get(r.get("validation"), "validation_blocked")
        elif i in agent:
            a = agent[i]
            key = "stopped" if a.automatic else {"abstain": "abstained", "escalate": "escalated"}.get(a.outcome.value, "proposed")
        else:
            key = "unresolved"
        c[key] = c.get(key, 0) + 1
    return c


# ---------- a trial ----------
def run_trial(project: Path, config: str, mode: str, *, replay: Path | None = None, out_root: Path | None = None,
              client=None, say=print) -> Path:
    project = project.resolve()
    if config not in CONFIGS or mode not in MODES:
        raise SystemExit(f"config must be one of {CONFIGS}, mode one of {MODES}")
    execution_mode = "replay" if replay else ("live" if config == "agent" else "live (no model)")
    if config == "agent" and client is not None and replay is None:
        execution_mode = "mock"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "replay" if replay else execution_mode.split()[0]
    run_id = f"{project.name}-{config}-{mode}-{tag}-{stamp}"
    base = out_root or ROOT / "runs" / "eval"
    n = 1
    while (base / run_id).exists():                  # two runs in the same second
        n += 1
        run_id = f"{project.name}-{config}-{mode}-{tag}-{stamp}-{n}"
    out = base / run_id
    (out / "previews").mkdir(parents=True)
    scenario, evidence = load_key(project)
    answers_file = out / "simulated_answers.json"
    answers_file.write_text(json.dumps(simulated_answers(scenario), ensure_ascii=False, indent=2), encoding="utf-8")
    record = run_record(project, config, mode, execution_mode, run_id, replay)
    settings = Rule2Settings.load()

    meetings = len(scenario["meeting_dates"])
    rows, per_meeting, id_map = [], [], {}
    for m in range(1, meetings + 1):
        say(f"meeting {m}/{meetings} ...")
        started = time.monotonic()
        db = out / ("memory.sqlite" if mode == "carry" else f"m{m}.sqlite")
        seed = gold_tasks_before(scenario, m) if mode == "reset" else None
        if mode == "reset":
            id_map = {t.task_id: t.task_id for t in seed}
        preview, events = make_preview(config, project, m, db, seed, out / "previews" / f"m{m}.json",
                                       answers_file, run_id, replay, client)
        review, tasks_after = simulated_review(db, m, settings) if preview is not None else ([], [])
        proposals = preview.proposals if preview is not None else []
        committed = {i for i, r in enumerate(review) if r["committed_as"]}
        m_rows, created = score_meeting(m, proposals, scenario, evidence[m], id_map, committed)
        for i, gold in created.items():                 # tasks the system created now map to gold tasks
            if i < len(review) and review[i]["committed_as"]:
                id_map[review[i]["committed_as"]] = gold
        rows += m_rows
        entry = {"meeting": m, "seconds": round(time.monotonic() - started, 1), "stop_reason": events["stop_reason"],
                 "failure_detail": events["failure_detail"], "extraction_fallback": events["extraction_fallback"],
                 "proposals": len(proposals), "committed": len(committed),
                 "validation": {s: sum(1 for r in review if r["validation"] == s) for s in ("valid", "blocked", "invalid", "stale")},
                 "case_outcomes": case_outcomes(preview, review, config),
                 "agent_steps": sum(len(a.trace.steps) for a in preview.agent_decisions) if preview else 0,
                 "clarification_questions": sum(1 for a in (preview.agent_decisions if preview else [])
                                                for s in a.trace.steps if s.tool == "ask_clarification" and s.valid),
                 "change_f1": change_f1(m_rows)["all"]}
        if mode == "carry":
            entry["task_state"] = state_accuracy(tasks_after, scenario["expected_state"][str(m)], id_map)
        per_meeting.append(entry)

    record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    scores = {"run_id": run_id, "configuration": config, "mode": mode, "execution_mode": execution_mode,
              "change_f1": change_f1(rows), "meetings": per_meeting}
    if mode == "carry":
        states = [p["task_state"] for p in per_meeting]
        scores["task_state"] = {"mean_accuracy": round(sum(s["accuracy"] or 0 for s in states) / len(states), 3),
                                "fully_correct_meetings": sum(s["fully_correct"] for s in states), "meetings": len(states)}
    (out / "run_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with (out / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else ["meeting"])
        w.writeheader()
        for r in rows:
            d = asdict(r); d["problems"] = "; ".join(d["problems"]); w.writerow(d)
    (out / "report.md").write_text(report(record, scores, rows), encoding="utf-8")
    say(f"done: {out}")
    return out


def report(record: dict, scores: dict, rows) -> str:
    f = scores["change_f1"]
    a = f["all"]
    lines = [f"# Evaluation run {record['run_id']}", "",
             f"- Configuration **{record['configuration']}**, mode **{record['mode']}**, execution **{record['execution_mode']}**",
             f"- Model: {record['model'].get('model') or 'none'}" + (f" (digest {record['model'].get('digest')})" if record['model'].get('digest') else ""),
             f"- Data: {record['project']} (snapshot {record['data_snapshot_sha']}); prompts {record['prompts_sha']}; config {record['config_sha']}; "
             f"code {str(record['code']['git_commit'])[:10]}{' + uncommitted changes' if record['code']['uncommitted_changes'] else ''}",
             "", "## Change F1 (micro, all meetings)", "",
             f"Precision {a['precision']}, recall {a['recall']}, **F1 {a['f1']}** (TP {a['tp']}, FP {a['fp']}, FN {a['fn']}). "
             f"Unsupported proposals (automatic proxy): {f['unsupported_rate_proxy']}.", "",
             "| Type | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    for t, v in f["by_type"].items():
        lines.append(f"| {t} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']} | {v['recall']} | {v['f1']} |")
    if f["field_problems"]:
        lines += ["", "Field problems: " + ", ".join(f"{k} {v}" for k, v in f["field_problems"].items())]
    lines += ["", "## Meetings", "", "| Meeting | Stop reason | Proposals | Committed | Case outcomes | F1 | Task state | Seconds |",
              "|---|---|---|---|---|---|---|---|"]
    for m in scores["meetings"]:
        ts = m.get("task_state")
        lines.append(f"| {m['meeting']} | {m['stop_reason']}{' (B1 fallback)' if m['extraction_fallback'] else ''} | {m['proposals']} | "
                     f"{m['committed']} | {', '.join(f'{k} {v}' for k, v in m['case_outcomes'].items())} | {m['change_f1']['f1']} | "
                     f"{'' if not ts else str(ts['accuracy']) + (' ✓' if ts['fully_correct'] else '')} | {m['seconds']} |")
    lines += ["", "## Every expected change and proposal", "", "| M | Event | Expected | Proposed | Result | Problems |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.meeting} | {r.event_id or ''} | {r.expected or ''} | {r.proposed or ''} | {r.result} | {'; '.join(r.problems)} |")
    if record["configuration"] == "b1":
        lines += ["", "Note: B1's rules were written while reading the two sample projects (samples/demo, samples/demo2), "
                  "so its score on them is optimistic. Compare configurations on the development and test sets."]
    lines += ["", "Limits: simulated reviewer approves every proposal that validation passes, answers every clarification at once "
              "and correctly; unsupported rate is an automatic proxy until the manual check of 30 proposals."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one configuration on one project and score it against the answer key.")
    ap.add_argument("project", type=Path, help="a project folder with agent_inputs/ and answer_keys/, e.g. samples/demo")
    ap.add_argument("--config", choices=CONFIGS, default="agent")
    ap.add_argument("--mode", choices=MODES, default="reset")
    ap.add_argument("--replay", type=Path, help="previews/ folder of an earlier agent run: replay it instead of calling the model")
    ap.add_argument("--out", type=Path, help="where runs go (default runs/eval)")
    a = ap.parse_args(argv)
    run_trial(a.project, a.config, a.mode, replay=a.replay, out_root=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
