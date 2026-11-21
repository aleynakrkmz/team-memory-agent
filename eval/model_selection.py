"""Model selection for the Team Memory Agent.

What it answers (instructor feedback on the design document, item 1, and the Week 9 decision table):
  "Name your models. Give the agent's model and both generation models, with versions.
   Say whether the generation models come from different families."

Roles (design document):
  agent model  - extraction, the agent loop and baseline B2 ("all LLM setups use the same model");
                 needs a 32,768-token context and tool calling
  generator A  - writes the development meetings (datagen); not the agent's model
  generator B  - writes the test meetings; not the agent's model; a different family from A
  checker      - check (iv) of the answer-key check (suggested: not a generator, not the agent)

Every candidate runs the same four tests on development material only (never the test set):
  1. extraction   - 8 meetings with known answers: precision, recall, F1, guessed owners, time
  2. agent case   - the "that API thing" case as a tool-calling loop with scripted tool results
  3. long input   - one ~20k-token prompt like B2's: read in full? within 5 minutes? correct list?
  4. generation   - one development project through the real datagen pipeline: answer-key check pass rate.
                   Check (iv) is done by ONE fixed checker model for every generator (GEN_CHECKER), so the
                   generators are compared on equal terms.

Run history:
  run 1 (21 Nov 2026) - check (iv) used a different checker per generator and matched change types exactly;
                        most meetings failed on false "unscripted change" flags, so generator B was undecided.
  run 2 (21 Nov 2026) - one fixed checker; create/assign/set_due_date count as one kind of change; recaps of
                        the existing task list are not flagged; flagged changes are saved for manual review;
                        5 generated meetings; a generator must pass at least one meeting to be suggested.

Run on your own computer (Ollama installed and running with OLLAMA_CONTEXT_LENGTH=32768):
  uv run python eval/model_selection.py
  uv run python eval/model_selection.py --models qwen3:8b mistral-nemo:12b --gen-meetings 3   # a smaller run

Run on a free Colab T4 GPU (Runtime -> Change runtime type -> T4 GPU), in one code cell:
  !curl -fsSL https://ollama.com/install.sh | sh
  !git clone <repository URL> repo && cd repo && pip -q install ollama && python eval/model_selection.py --start-ollama
  (a private repository needs a GitHub token in the URL; or upload a zip of the repository and unzip it)
  (--start-ollama starts Ollama with a 32k context and restarts it if it stops.) The run takes about 2 hours
  for the 5 default models. It can be resumed: run the same command again and finished tests are skipped.

Outputs:
  eval/model_selection_results/   raw results (*.jsonl), model versions, generated meetings, summary CSV,
                                  model_selection_flags.csv (check (iv) flags to review by hand); not committed
  docs/model_selection.md         the report; docs/model_selection_table.csv: the Week 9 decision table
  checker_<model>.csv             made by --measure-checker MODEL: check (iv) run again with MODEL on the saved
                                  meetings and compared with the run's (hand-reviewed) flags
  docs/model_selection_review.md  written by hand after the run (manual review of the flags, team decision);
                                  copied into the report when it exists
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------- Where things are ----------
# In the repository this file is eval/model_selection.py, so the root is one level up.
# If the files are copied next to each other (e.g. into /content on Colab), that works too.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "eval" else HERE
for p in (ROOT / "datagen", ROOT, HERE):
    if (p / "common.py").exists():
        sys.path.insert(0, str(p))
        break
import common as C   # shared datagen code: same checks, prompts and scenario generator as the pipeline

CONTEXT_LENGTH = 32768          # design document 3.5
TIME_LIMIT_S = 300              # design document 3.5: 5 minutes per meeting
GEN_CHECKER = "qwen3:8b"        # the same checker for every generator in test 4; it is not a generator candidate
THINKING_PREFIXES = ("qwen3", "deepseek-r1")   # thinking mode is turned off so timings are comparable

DEFAULT_CANDIDATES = [          # examples; the team decides the list. Check tags on ollama.com/library.
    {"tag": "qwen3:8b",         "family": "Qwen (Alibaba)"},
    {"tag": "llama3.1:8b",      "family": "Llama (Meta)"},
    {"tag": "gemma3:12b",       "family": "Gemma (Google)"},
    {"tag": "mistral-nemo:12b", "family": "Mistral (Mistral AI)"},
    {"tag": "granite3.3:8b",    "family": "Granite (IBM)"},
]

client = None   # set by connect()


def ollama_up(host="http://127.0.0.1:11434"):
    import urllib.request
    try:
        urllib.request.urlopen(host + "/api/tags", timeout=3)
        return True
    except OSError:
        return False


def start_ollama(log_path, host="http://127.0.0.1:11434"):
    """Starts (or restarts) `ollama serve` with a 32k context (for Colab; on a laptop start Ollama yourself).
    Ollama's default context is 4k tokens on GPUs with less than 24 GB, so the design's 32,768 is set here."""
    subprocess.run(["pkill", "-9", "-f", "ollama serve"], check=False)
    time.sleep(2)
    env = dict(__import__("os").environ, OLLAMA_CONTEXT_LENGTH=str(CONTEXT_LENGTH),
               OLLAMA_HOST=host.split("//")[-1], OLLAMA_LOAD_TIMEOUT="15m")  # a 12B model can load slowly on a T4
    # stdin is closed so that Colab's "!python ..." cell ends when the script ends, while Ollama keeps running
    subprocess.Popen(["ollama", "serve"], env=env, stdin=subprocess.DEVNULL, stdout=open(log_path, "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        if ollama_up(host):
            print("Ollama is running")
            return
        time.sleep(1)
    raise RuntimeError(f"Ollama did not start; see {log_path}")


def connect(host="http://127.0.0.1:11434"):
    """Connects to a running Ollama server."""
    global client
    client = C.make_client(host)
    return client


def chat(model, messages, **kw):
    """One chat call (temperature 0, 32k context, thinking off where possible). Returns (response, seconds)."""
    opts = {"temperature": 0, "num_ctx": CONTEXT_LENGTH}
    opts.update(kw.pop("options", {}))
    t0 = time.time()
    if model.startswith(THINKING_PREFIXES):
        try:
            r = client.chat(model=model, messages=messages, options=opts, think=False, **kw)
        except TypeError:
            r = client.chat(model=model, messages=messages, options=opts, **kw)
    else:
        r = client.chat(model=model, messages=messages, options=opts, **kw)
    return r, time.time() - t0


# ---------- Candidates: download and record the exact version ----------
def model_info(tag):
    """Exact version (digest), size, context length and capabilities as reported by Ollama."""
    info = {"tag": tag, "digest": None, "size_gb": None, "context_length": None, "capabilities": None}
    for m in client.list()["models"]:
        if m["model"] == tag:
            info["digest"] = m["digest"]
            info["size_gb"] = round(m["size"] / 1e9, 2)
    try:
        s = client.show(tag)
        mi = s.get("modelinfo") or {}
        info["context_length"] = next((v for k, v in mi.items() if k.endswith(".context_length")), None)
        info["capabilities"] = s.get("capabilities")
    except Exception as e:  # noqa: BLE001 - recorded, not fatal
        info["show_error"] = str(e)[:200]
    return info


def prepare_models(candidates, out, pull=True):
    """Pulls every candidate; a model that cannot be pulled is recorded and skipped."""
    info = {}
    for c in candidates:
        tag = c["tag"]
        print("model:", tag)
        try:
            if pull:
                client.pull(tag)
            info[tag] = {**model_info(tag), "family": c["family"]}
        except Exception as e:  # noqa: BLE001
            info[tag] = {"tag": tag, "family": c["family"], "pull_error": str(e)[:200]}
    (out / "model_info.json").write_text(json.dumps(info, indent=2, default=str))
    return info


# ---------- Test data (development material only) ----------
def build_test_set(samples_dir):
    """3 demo meetings + 5 meetings of a mock development project, all with known answers."""
    test_set = []
    team, scen, tr, _ = C.load_project(samples_dir, "demo")
    for m in tr:
        test_set.append(("demo", m, team, scen, tr))
    dteam, dscen = C.generate_project("sel_dev", 4242, "dev", n_meetings=5)
    dtr = {}
    for m in range(1, 6):
        lines, _ = C.split_tags(C.render_mock(dteam, dscen, m, random.Random(m)))
        dtr[m] = [f"{s}: {t}" for s, t in lines]
        test_set.append(("dev_mock", m, dteam, dscen, dtr))
    return test_set, team


def meeting_inputs(scenario, transcripts, m):
    """What extraction sees for meeting m: open tasks before the meeting and numbered lines."""
    before = scenario["expected_state"].get(str(m - 1), {})
    open_tasks = [{"task_id": t, **{k: v[k] for k in ("description", "owner", "due_date")}}
                  for t, v in before.items() if v["status"] == "open"]
    lines = [f"M{m}-L{i:02d} | {line}" for i, line in enumerate(transcripts[m], 1)]
    return open_tasks, lines


# ---------- Test 1: extraction ----------
SYSTEM_EXTRACT = "You extract task changes from a meeting transcript for a team task list. Answer with JSON only."


def extract_prompt(team, scenario, m, open_tasks, lines):
    """Measurement prompt only; the system's real prompt lives in prompts/ (workstream 2)."""
    return (f"Meeting date: {scenario['meeting_dates'][m-1]}\nTeam: {json.dumps(team, ensure_ascii=False)}\n"
            f"Open tasks: {json.dumps(open_tasks, ensure_ascii=False)}\n\nTranscript:\n" + "\n".join(lines) +
            "\n\nReturn {\"changes\": [...]} where each change has: type (create|assign|set_due_date|complete|cancel), "
            "task_description, task_id (an open task ID, or null for a new task), owner (team member, or null if not stated), "
            "due_date (YYYY-MM-DD, or null if not stated), evidence_line_ids (list of line IDs). "
            "Never guess an owner or a date that the meeting does not state.")


def score_meeting(changes, scenario, m):
    """Design 6.2, simplified: a change is correct only if type, task and required fields match.
    Returns (tp, fp, fn, guessed_owner)."""
    events = [e for e in scenario["events"] if e["meeting"] == m]
    changes = changes if isinstance(changes, list) else []
    used, tp, guessed = set(), 0, 0
    for ch in changes:
        if not isinstance(ch, dict):
            continue
        for e in events:
            if e["event_id"] in used or ch.get("type") != e["type"]:
                continue
            same_task = (ch.get("task_id") == e["task_id"]) or (
                e["type"] == "create" and C._words(ch.get("task_description"))
                & C._words(scenario["tasks"][e["task_id"]]["description"]))
            if not same_task:
                continue
            ok = True
            if e["type"] in ("create", "assign"):
                ok &= (ch.get("owner") or None) == e.get("owner")
                if e.get("ambiguity") == "owner" and ch.get("owner"):
                    guessed += 1
            if e["type"] in ("create", "set_due_date"):
                ok &= (ch.get("due_date") or None) == e.get("due_date")
            if ok:
                used.add(e["event_id"]); tp += 1
                break
    return tp, len(changes) - tp, len(events) - tp, guessed


def test_extraction(model, test_set, runs):
    rows = []
    for src, m, team, scen, tr in test_set:
        open_tasks, lines = meeting_inputs(scen, tr, m)
        for run in range(runs):
            r, secs = chat(model, [{"role": "system", "content": SYSTEM_EXTRACT},
                                   {"role": "user", "content": extract_prompt(team, scen, m, open_tasks, lines)}],
                           format="json")
            try:
                changes = json.loads(r["message"]["content"]).get("changes", []); valid = True
            except (json.JSONDecodeError, AttributeError):
                changes, valid = [], False
            tp, fp, fn, guessed = score_meeting(changes, scen, m)
            rows.append({"model": model, "source": src, "meeting": m, "run": run + 1, "seconds": round(secs, 1),
                         "valid_json": valid, "tp": tp, "fp": fp, "fn": fn, "guessed_owner": guessed,
                         "prompt_tokens": r.get("prompt_eval_count"), "output_tokens": r.get("eval_count")})
    return rows


# ---------- Test 2: the agent case ("that API thing", design 3.8) ----------
def _tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


TOOLS = [
    _tool("search_meetings", "Search lines of the current and earlier meetings of this project.",
          {"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]),
    _tool("get_task_history", "All versions of a task with their evidence.", {"task_id": {"type": "string"}}, ["task_id"]),
    _tool("get_team_profile", "Names, aliases and roles of the team.", {}),
    _tool("ask_clarification", "Ask the coordinator one question about this case.",
          {"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}, ["question"]),
    _tool("propose_update", "Propose one task change with evidence line IDs.",
          {"action": {"type": "string", "enum": ["create", "assign", "set_due_date", "complete", "cancel"]},
           "task_id": {"type": "string"}, "fields": {"type": "object"},
           "evidence_line_ids": {"type": "array", "items": {"type": "string"}}},
          ["action", "fields", "evidence_line_ids"]),
    _tool("abstain", "End the case with no change.", {"reason": {"type": "string"}}, ["reason"]),
    _tool("escalate", "Hand the case to the coordinator.", {"reason": {"type": "string"}}, ["reason"]),
]

AGENT_SYSTEM = ("You resolve one difficult case for a team task list. Use the tools; take one step at a time. "
                "Rules: search before asking; ask before guessing; abstain rather than invent. "
                "Propose a change only with evidence lines that state the owner or the date.")
AGENT_USER = ("Meeting 3 (2026-11-12). Candidate change from extraction: reassign a task to Can.\n"
              "Lines: M3-L03 'Ali: I'm completely swamped with the mobile release this week.' "
              "M3-L04 'Zeynep: Ali is swamped. Can, could you take over that API thing from last week?' "
              "M3-L05 'Can: Sure, I can take it.'\n"
              "Open tasks: T4 Payment API endpoint (owner Ali), T5 API documentation (owner Ayşe), "
              "T3 Project report draft (owner Zeynep). The reference 'that API thing' fits more than one task.")


def scripted_tool(name, args, team):
    """The same tool answers every time (from the demo data), so only the model's decisions are tested."""
    if name == "search_meetings":
        return {"lines": [
            {"id": "M2-L05", "text": "Zeynep: Next, payments. Ali, can you build the payment API endpoint by the 18th?"},
            {"id": "M2-L06", "text": "Ali: Sure, the payment API endpoint by the 18th."},
            {"id": "M2-L08", "text": "Ayşe: I'll update the API documentation, the end of the month is fine."}]}
    if name == "get_task_history":
        hist = {"T4": [{"version": 1, "owner": "Ali", "due_date": "2026-11-18", "evidence": ["M2-L05", "M2-L06"]}],
                "T5": [{"version": 1, "owner": "Ayşe", "due_date": "2026-11-30", "evidence": ["M2-L07", "M2-L08"]}]}
        return {"task_id": args.get("task_id"), "versions": hist.get(args.get("task_id"), "unknown task")}
    if name == "get_team_profile":
        return team
    if name == "ask_clarification":
        return {"status": "question stored; case paused"}
    return {"status": "ok"}


def run_agent_case(model, team, max_steps=6):
    """Solved = propose_update for T4 with owner Can and at least one evidence line, within 6 steps."""
    msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": AGENT_USER}]
    steps, trace, t0 = 0, [], time.time()
    while steps < max_steps:
        r, _ = chat(model, msgs, tools=TOOLS)
        calls = r["message"].get("tool_calls") or []
        if not calls:
            return {"solved": False, "end": "no_tool_call", "steps": steps, "trace": trace,
                    "seconds": round(time.time() - t0, 1)}
        msgs.append(r["message"])
        call = calls[0]   # one step at a time
        name, args = call["function"]["name"], call["function"]["arguments"] or {}
        steps += 1; trace.append(name)
        if name == "propose_update":
            f = args.get("fields") or {}
            solved = args.get("task_id") == "T4" and f.get("owner") == "Can" and bool(args.get("evidence_line_ids"))
            return {"solved": solved, "end": "propose_update", "steps": steps, "trace": trace,
                    "proposal": json.dumps(args, ensure_ascii=False), "seconds": round(time.time() - t0, 1)}
        if name in ("abstain", "escalate", "ask_clarification"):
            return {"solved": False, "end": name, "steps": steps, "trace": trace, "seconds": round(time.time() - t0, 1)}
        msgs.append({"role": "tool", "content": json.dumps(scripted_tool(name, args, team), ensure_ascii=False),
                     "tool_name": name})
    return {"solved": False, "end": "step_limit", "steps": steps, "trace": trace, "seconds": round(time.time() - t0, 1)}


def test_agent(model, team, runs):
    rows = []
    for run in range(runs):
        try:
            res = run_agent_case(model, team); res["tools_supported"] = True
        except Exception as e:  # noqa: BLE001
            res = {"solved": False, "end": "error", "error": str(e)[:200],
                   "tools_supported": "does not support tools" not in str(e)}
        res["trace"] = ">".join(res.get("trace", []))
        rows.append({"model": model, "run": run + 1, **res})
    return rows


# ---------- Test 3: long input, as in baseline B2 ----------
def build_long_prompt(target_tokens=20000):
    """Five development meetings padded with small talk to about target_tokens (about 4 characters per token)."""
    lteam, lscen = C.generate_project("sel_long", 777, "dev", n_meetings=5)
    rng = random.Random(1)
    names = [p["name"] for p in lteam["members"]]
    pad = ["Sorry, can you repeat that? The connection dropped for a second.",
           "I think we discussed the office move, but that is not for this list.",
           "Let me share my screen so everyone can see the board."]
    meetings = []
    for m in range(1, 6):
        lines, _ = C.split_tags(C.render_mock(lteam, lscen, m, random.Random(100 + m)))
        meetings.append([f"{s}: {t}" for s, t in lines])
    text = ""
    while len(text) / 4 < target_tokens:
        meetings = [ml + [f"{rng.choice(names)}: {rng.choice(pad)}" for _ in range(10)] for ml in meetings]
        text = "".join(f"\n=== Meeting {m} ({lscen['meeting_dates'][m-1]}) ===\n" + "\n".join(ml) + "\n"
                       for m, ml in enumerate(meetings, 1))
    prompt = (f"Team: {json.dumps(lteam, ensure_ascii=False)}\nAll meetings so far:{text}\n"
              "Return {\"tasks\": [...]} with the current state of every task: description, owner (or null), "
              "due_date (YYYY-MM-DD or null), status (open, done or cancelled).")
    return prompt, lscen["expected_state"]["5"], int(len(prompt) / 4)


def test_long(model, long_prompt, gold, est_tokens):
    r, secs = chat(model, [{"role": "system", "content": "Answer with JSON only."},
                           {"role": "user", "content": long_prompt}], format="json")
    try:
        tasks = json.loads(r["message"]["content"]).get("tasks", [])
    except (json.JSONDecodeError, AttributeError):
        tasks = None
    correct = 0
    for g in gold.values():
        for t in tasks or []:
            if isinstance(t, dict) and C._words(t.get("description")) & C._words(g["description"]):
                correct += int((t.get("owner") or None) == g["owner"] and t.get("status") == g["status"])
                break
    read = r.get("prompt_eval_count")
    return {"model": model, "seconds": round(secs, 1), "prompt_tokens_read": read, "estimated_input_tokens": est_tokens,
            "input_read_in_full": bool(read and read >= 0.8 * est_tokens), "valid_json": tasks is not None,
            "tasks_correct": correct, "tasks_total": len(gold), "within_5_min": secs <= TIME_LIMIT_S}


# ---------- Test 4: generation through the real datagen pipeline ----------
def test_generation(model, checker_model, out, n_meetings=5):
    """Writes one development project with this model and runs the answer-key check (up to 3 tries per meeting)."""
    gteam, gscen = C.generate_project("sel_gen", 2026, "dev", n_meetings=n_meetings)
    outdir = out / "generated" / model.replace(":", "_")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in range(1, n_meetings + 1):
        for attempt in range(1, 4):
            seed = 1000 * m + attempt
            t0 = time.time()
            lines = C.render(gteam, gscen, m, seed, backend="ollama", client=client, model=model)
            gen_secs = time.time() - t0
            probs = C.auto_check(gteam, gscen, m, lines, C.make_checker(client, checker_model, seed))
            rows.append({"model": model, "meeting": m, "attempt": attempt, "passed": not probs,
                         "reasons": ";".join(sorted({p[0] for p in probs})), "gen_seconds": round(gen_secs, 1),
                         "n_lines": len(lines), "checker": checker_model,
                         # every problem with its detail, so the check (iv) flags can be reviewed by hand
                         "problems": [{"reason": r, "event": e, "detail": d} for r, e, d in probs]})
            (outdir / f"meeting_{m:02d}_try{attempt}.txt").write_text(
                "\n".join(f"{s}: {t}" for s, t in lines), encoding="utf-8")
            if not probs:
                break
    return rows


def review_flags(out, models=None):
    """The check (iv) flags of the generation test, one row per flagged change, for review by hand.
    For each row decide: is this really a change the scenario does not have (true flag), or not (false flag)?"""
    rows = []
    for r in load(out, "generation"):
        if models and r.get("model") not in models:
            continue
        for p in r.get("problems", []):
            if p["reason"] in ("unscripted_change", "checker_unparsable"):
                rows.append({"model": r["model"], "meeting": r["meeting"], "attempt": r["attempt"],
                             "reason": p["reason"], "change": p["detail"],
                             "file": f"generated/{r['model'].replace(':', '_')}/meeting_{r['meeting']:02d}_try{r['attempt']}.txt"})
    return rows


def measure_checker(model, out):
    """Measures a checker model on the meetings saved by the generation test, without generating anything new.
    Check (iv) is run again with `model` on every saved meeting and compared with the flags already in
    generation.jsonl (made by the run's checker and reviewed by hand). Returns one row per meeting; also
    writes checker_<model>.csv. Rows where only the new checker flags something must be read by hand."""
    gteam, gscen = None, None
    rows = []
    for r in load(out, "generation"):
        if "passed" not in r:
            continue
        if gscen is None or len(gscen["meeting_dates"]) < r["meeting"]:
            n = max(x.get("meeting", 0) for x in load(out, "generation"))
            gteam, gscen = C.generate_project("sel_gen", 2026, "dev", n_meetings=n)
        f = out / "generated" / r["model"].replace(":", "_") / f"meeting_{r['meeting']:02d}_try{r['attempt']}.txt"
        if not f.exists():
            continue
        lines = C.parse_transcript(f.read_text(encoding="utf-8"), gteam)
        seed = 1000 * r["meeting"] + r["attempt"]
        probs = C.auto_check(gteam, gscen, r["meeting"], lines, C.make_checker(client, model, seed))
        new = [d for x, _, d in probs if x in ("unscripted_change", "checker_unparsable")]
        old = [p["detail"] for p in r.get("problems", []) if p["reason"] in ("unscripted_change", "checker_unparsable")]
        rows.append({"generator": r["model"], "meeting": r["meeting"], "attempt": r["attempt"],
                     "flags_run_checker": len(old), "flags_new_checker": len(new),
                     "agree": bool(old) == bool(new), "new_checker_flags": " | ".join(new),
                     "file": str(f.relative_to(out))})
    if rows:
        with (out / f"checker_{model.replace(':', '_')}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    flagged_before = [x for x in rows if x["flags_run_checker"]]
    caught = sum(1 for x in flagged_before if x["flags_new_checker"])
    only_new = [x for x in rows if x["flags_new_checker"] and not x["flags_run_checker"]]
    print(f"{model} as checker on {len(rows)} saved meetings:")
    print(f"  meetings with a true flag (run checker, reviewed by hand): {len(flagged_before)}; also flagged by {model}: {caught}")
    print(f"  meetings flagged only by {model}: {len(only_new)} (read these by hand)")
    for x in only_new:
        print("   ", x["file"], "->", x["new_checker_flags"][:200])
    return rows


# ---------- Running, resuming and saving ----------
def _jsonl(out, name):
    return out / f"{name}.jsonl"


def done(out, name, model):
    p = _jsonl(out, name)
    return p.exists() and any(json.loads(line).get("model") == model for line in p.read_text().splitlines())


def save(out, name, rows):
    with _jsonl(out, name).open("a") as f:
        for r in rows if isinstance(rows, list) else [rows]:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load(out, name):
    p = _jsonl(out, name)
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def run_all(models, samples_dir, out, runs=2, gen_meetings=5, long_tokens=20000, checker=GEN_CHECKER, before_each=None):
    """Runs the four tests for every model. Already finished (model, test) pairs are skipped,
    so a disconnect does not lose work: just run again.
    checker: the one model that does check (iv) for every generator (default GEN_CHECKER);
             if it is not among the models, the next other model is used and recorded.
    before_each: optional function called before every test (--start-ollama uses it to restart Ollama if it stopped)."""
    test_set, team = build_test_set(samples_dir)
    long_prompt, long_gold, long_est = build_long_prompt(long_tokens)
    print(f"{len(test_set)} extraction meetings; long prompt about {long_est} tokens")
    for model in models:
        print(f"\n===== {model} =====")
        gen_checker = checker if checker in models else next((m for m in models if m != model), model)
        tests = [("extraction", lambda: test_extraction(model, test_set, runs)),
                 ("agent_case", lambda: test_agent(model, team, runs)),
                 ("long_input", lambda: test_long(model, long_prompt, long_gold, long_est)),
                 ("generation", lambda: test_generation(model, gen_checker, out, gen_meetings))]
        for name, fn in tests:
            if done(out, name, model):
                print(f"  {name}: already done"); continue
            if before_each:
                before_each()
            t0 = time.time()
            try:
                save(out, name, fn())
            except Exception as e:  # noqa: BLE001 - recorded so the run can continue
                save(out, name, {"model": model, "error": str(e)[:300]})
            print(f"  {name}: {time.time() - t0:.0f} s")


# ---------- Summary, suggestion and decision table ----------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def summarize(out, info):
    """One summary row per model, from the raw results."""
    ext, agt, lng, gen = (load(out, n) for n in ("extraction", "agent_case", "long_input", "generation"))
    summary = {}
    for tag in [t for t, i in info.items() if "pull_error" not in i]:
        e = [r for r in ext if r.get("model") == tag and "tp" in r]
        tp, fp, fn = (sum(r[k] for r in e) for k in ("tp", "fp", "fn"))
        prec = round(tp / (tp + fp), 3) if tp + fp else 0.0
        rec = round(tp / (tp + fn), 3) if tp + fn else 0.0
        a = [r for r in agt if r.get("model") == tag]
        lg = next((r for r in lng if r.get("model") == tag and "seconds" in r), {})
        g = [r for r in gen if r.get("model") == tag and "passed" in r]
        meetings = sorted({r["meeting"] for r in g})
        first = [next(r["passed"] for r in g if r["meeting"] == m and r["attempt"] == 1) for m in meetings]
        anyp = [any(r["passed"] for r in g if r["meeting"] == m) for m in meetings]
        reasons = {}
        for r in g:
            for x in filter(None, r["reasons"].split(";")):
                reasons[x] = reasons.get(x, 0) + 1
        summary[tag] = {
            "model": tag, "family": info[tag]["family"], "digest": info[tag].get("digest"),
            "size_gb": info[tag].get("size_gb"), "context_length": info[tag].get("context_length"),
            "extract_precision": prec, "extract_recall": rec,
            "extract_f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "extract_valid_json": _mean([float(r["valid_json"]) for r in e]),
            "extract_guessed_owner": sum(r["guessed_owner"] for r in e),
            "extract_median_s": _median([r["seconds"] for r in e]),
            "tools_supported": all(r.get("tools_supported", False) for r in a) if a else False,
            "agent_solved_rate": _mean([float(r.get("solved", False)) for r in a]),
            "agent_ends": ", ".join(sorted({r.get("end", "?") for r in a})),
            "long_seconds": lg.get("seconds"), "long_read_in_full": lg.get("input_read_in_full", False),
            "long_within_5_min": lg.get("within_5_min", False),
            "long_tasks_correct": f"{lg.get('tasks_correct', 0)}/{lg.get('tasks_total', '?')}",
            "gen_first_try_pass": _mean([float(x) for x in first]),
            "gen_pass_after_3": _mean([float(x) for x in anyp]),
            "gen_seconds_per_meeting": _median([r["gen_seconds"] for r in g]),
            "gen_failure_reasons": "; ".join(f"{k} {v}" for k, v in sorted(reasons.items())),
            "gen_meetings": len(meetings),
            "gen_checker": ", ".join(sorted({r.get("checker", "?") for r in g})),
        }
    return summary


def suggest(summary):
    """Suggestion rules, fixed before looking at the results. The team decides; this only orders the evidence."""
    rows = list(summary.values())
    fam = {r["model"]: r["family"] for r in rows}
    eligible = [r for r in rows if r["tools_supported"] and (r["context_length"] or 0) >= CONTEXT_LENGTH
                and r["long_read_in_full"] and r["long_within_5_min"]]
    agent_rank = [r["model"] for r in sorted(eligible, key=lambda r: (-(r["agent_solved_rate"] or 0),
                                                                        -r["extract_f1"], r["extract_median_s"] or 1e9))]
    agent = agent_rank[0] if agent_rank else None
    pool = [r for r in rows if r["model"] != agent and fam[r["model"]] != fam.get(agent)]
    gen_rank = [r["model"] for r in sorted(pool, key=lambda r: (-(r["gen_pass_after_3"] or 0),
                                                                 -(r["gen_first_try_pass"] or 0),
                                                                 r["gen_seconds_per_meeting"] or 1e9))]
    # A generator must pass at least one meeting; otherwise no generator is suggested for that role.
    gen_rank = [m for m in gen_rank if (summary[m]["gen_pass_after_3"] or 0) > 0]
    gen_a = gen_rank[0] if gen_rank else None
    gen_b = next((m for m in gen_rank[1:] if fam[m] not in (fam.get(gen_a), fam.get(agent))), None)
    chk_rank = [r["model"] for r in sorted(rows, key=lambda r: -r["extract_f1"]) if r["model"] not in (agent, gen_a, gen_b)]
    return {"agent": agent, "agent_rank": agent_rank, "generator_a": gen_a, "generator_b": gen_b,
            "generator_rank": gen_rank, "checker": chk_rank[0] if chk_rank else None, "checker_rank": chk_rank}


def decision_table(summary, pick):
    """The Week 9 model-selection decision table: task, reasoning depth, privacy, latency, cost,
    local/free feasibility, fallback (plus the exact version)."""
    def fallback(model, rank):
        return next((m for m in rank if m != model), "none tested")

    def row(role, task, model, rank, latency, depth):
        s = summary.get(model, {})
        return {"role": role, "model": model, "family": s.get("family"), "version (digest)": s.get("digest"),
                "task": task, "reasoning depth": depth, "privacy": "local (Ollama); no data leaves the machine",
                "latency": latency, "cost": "0 (local)", "local/free": "yes: free Colab T4 or laptop",
                "fallback": fallback(model, rank) if model else None}

    a, ga, gb, ck = pick["agent"], pick["generator_a"], pick["generator_b"], pick["checker"]
    rows = [
        row("agent (extraction, agent loop, B2)", "find task changes; choose tools step by step", a, pick["agent_rank"],
            f"{summary[a]['extract_median_s']} s per extraction; {summary[a]['long_seconds']} s for a ~20k-token B2 prompt" if a else None,
            "medium: several tool steps"),
        row("generator A (development data)", "write development meetings from a scenario", ga, pick["generator_rank"],
            f"{summary[ga]['gen_seconds_per_meeting']} s per meeting" if ga else None, "low: follows a script"),
        row("generator B (test data)", "write test meetings from a scenario", gb,
            [m for m in pick["generator_rank"] if m != ga], f"{summary[gb]['gen_seconds_per_meeting']} s per meeting" if gb else None,
            "low: follows a script"),
        row("checker (answer-key check iv)", "list the changes in a generated meeting", ck, pick["checker_rank"],
            f"{summary[ck]['extract_median_s']} s per meeting" if ck else None, "low"),
    ]
    fixed = [
        {"role": "speech-to-text", "model": "faster-whisper (size to be chosen)", "task": "audio to transcript",
         "privacy": "local", "cost": "0", "local/free": "yes (CPU)", "fallback": "Parakeet TDT 0.6B v2 on a free Colab GPU (stretch 1)"},
        {"role": "retrieval comparison", "model": "nomic-embed-text v1.5", "task": "compared with BM25 on the development set only",
         "privacy": "local", "cost": "0", "local/free": "yes", "fallback": "BM25 only"},
        {"role": "text-to-speech (held-out audio)", "model": "Kokoro-82M", "task": "audio of the held-out synthetic projects",
         "privacy": "local", "cost": "0", "local/free": "yes", "fallback": "transcript condition only"},
    ]
    return rows + fixed


def _md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |\n" for r in rows)


def write_outputs(out, summary, pick, table, docs_dir, gpu=None, runs=None):
    """Writes CSV files and the Markdown report docs/model_selection.md (for the Week 10 first evaluation pass).
    If docs/model_selection_review.md exists (written by hand after the run), it is copied into the report."""
    def to_csv(path, rows):
        if not rows:
            return
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    to_csv(out / "model_selection_summary.csv", list(summary.values()))
    to_csv(out / "model_selection_table.csv", table)
    (out / "suggestion.json").write_text(json.dumps(pick, indent=2))

    fam = {m: s["family"] for m, s in summary.items()}
    a, ga, gb = pick["agent"], pick["generator_a"], pick["generator_b"]
    diff = "yes" if ga and gb and fam[ga] != fam[gb] else "no / not decided"
    md = f"""# Model selection

*Generated by `eval/model_selection.py` on {dt.date.today().isoformat()}. Hardware: {gpu or 'not recorded'}. Execution mode: live (Ollama). Repeats per test: {runs}.*

![How model selection works](model_selection_flow.png)

This answers item 1 of the instructor's feedback on the design document ("Name your models ... with versions. Say whether the generation models come from different families") and fills in the Week 9 model-selection decision table.

## Suggested choice (the team decides)

| Role | Model | Family | Version (digest) |
|---|---|---|---|
| Agent (extraction, agent loop, B2) | {a} | {fam.get(a)} | {summary.get(a, {}).get('digest')} |
| Generator A (development data) | {ga} | {fam.get(ga)} | {summary.get(ga, {}).get('digest')} |
| Generator B (test data) | {gb} | {fam.get(gb)} | {summary.get(gb, {}).get('digest')} |
| Checker (answer-key check iv) | {pick['checker']} | {fam.get(pick['checker'])} | {summary.get(pick['checker'], {}).get('digest')} |

- Generators A and B come from different families: **{diff}**.
- Neither generator is the agent's model or from the agent's family: **{'yes' if a and ga and gb and fam[a] not in (fam[ga], fam[gb]) else 'check'}**.
- **Team decision:** {"see *Reading of the results (by hand)* at the end." if (docs_dir / "model_selection_review.md").exists() else "_to be filled in: write docs/model_selection_review.md and run with --report-only._"}

**Suggestion rules** (fixed before the results were seen):
- **Agent:** must support tools, have a context of at least {CONTEXT_LENGTH} tokens, read the ~20k-token B2 prompt in full and finish it within 5 minutes. Then the highest agent-case solve rate, then extraction F1, then speed.
- **Generator A:** the highest answer-key-check pass rate after three tries, then first-try pass rate, then speed. It must not be the agent model or from its family.
- **Generator B:** the best remaining model from a family different from A and from the agent.
- **Both generators:** must pass the answer-key check in at least one meeting (added after run 1, where a model with a 0% pass rate was suggested only because it was faster).
- **Checker:** the best extraction F1 among the remaining models.

## Decision table (Week 9)

{_md_table(table, ["role", "model", "version (digest)", "task", "reasoning depth", "privacy", "latency", "cost", "local/free", "fallback"])}
## Results per model

{_md_table(list(summary.values()), ["model", "family", "context_length", "extract_precision", "extract_recall", "extract_f1", "extract_guessed_owner", "extract_valid_json", "extract_median_s", "tools_supported", "agent_solved_rate", "agent_ends", "long_read_in_full", "long_seconds", "long_tasks_correct", "gen_first_try_pass", "gen_pass_after_3", "gen_seconds_per_meeting"])}
Failure reasons in the generation test: {'; '.join(f"{m}: {s['gen_failure_reasons'] or 'none'}" for m, s in summary.items())}

## What these results do and do not show

- Only development material was used: the hand-written demo project and generated development projects. The test set was not touched.
- Extraction was measured on 8 short meetings; 5 of them were written by fixed templates, so their language is simpler than real or LLM-written meetings.
- The agent case is one case with scripted tool results; it tests tool choice, not the full system.
- The generation test wrote {max([s.get('gen_meetings', 0) for s in summary.values()] or [0])} meetings of one development project per model; check (iv) was done by: {', '.join(sorted({s.get('gen_checker') for s in summary.values() if s.get('gen_checker')}))}. Flagged changes are saved in `generation.jsonl` ("problems") for manual review.
- The long-input test is one synthetic prompt of about 20,000 tokens.
- The per-meeting time limit (5 minutes) will be measured again with the real agent loop before the system freeze.
- All numbers are from live runs on the hardware above, with the model versions (digests) listed.

## Run history

- **Run 1 (21 Nov 2026):** each generator had a different checker for check (iv) and change types had to match exactly. Most generated meetings failed on false "unscripted change" flags, so generator B could not be chosen.
- **Run 2 (21 Nov 2026):** one fixed checker for every generator; `create`, `assign` and `set_due_date` count as one kind of change; repeating the existing task list is not flagged; every flag is saved for review by hand; 5 generated meetings; a generator must pass at least one meeting. The numbers in this report are from this run.
"""
    review = docs_dir / "model_selection_review.md"
    if review.exists():
        md += "\n" + review.read_text(encoding="utf-8").strip() + "\n"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "model_selection.md").write_text(md, encoding="utf-8")
    shutil.copy(out / "model_selection_table.csv", docs_dir / "model_selection_table.csv")
    return md


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Team Memory Agent model selection (Ollama must be running).")
    ap.add_argument("--models", nargs="*", help="Ollama tags; default: the example list in this file")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--runs", type=int, default=1, help="repeats of each extraction meeting and of the agent case "
                    "(temperature is 0, so a repeat rarely changes the result)")
    ap.add_argument("--gen-meetings", type=int, default=5)
    ap.add_argument("--checker", default=GEN_CHECKER, help="the one model that does check (iv) for every generator")
    ap.add_argument("--long-tokens", type=int, default=20000)
    ap.add_argument("--no-pull", action="store_true", help="do not download models (they must already exist)")
    ap.add_argument("--samples", default=str(ROOT / "samples" / "demo"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "model_selection_results"))
    ap.add_argument("--docs", default=str(ROOT / "docs"))
    ap.add_argument("--start-ollama", action="store_true",
                    help="start Ollama with a 32k context and restart it if it stops (for Colab)")
    ap.add_argument("--measure-checker", metavar="MODEL",
                    help="no new meetings: run check (iv) with MODEL on the saved meetings and compare it with the run's checker")
    ap.add_argument("--report-only", action="store_true",
                    help="no new tests: rebuild the report from the saved results (e.g. after writing the review)")
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.measure_checker:
        if a.start_ollama:
            start_ollama(out / "ollama.log", a.host)
        connect(a.host)
        if not a.no_pull:
            client.pull(a.measure_checker)
        measure_checker(a.measure_checker, out)
        return

    if a.report_only:
        info = json.loads((out / "model_info.json").read_text())
        run = json.loads((out / "run_info.json").read_text()) if (out / "run_info.json").exists() else {}
        finish(out, info, Path(a.docs), run.get("gpu"), run.get("runs", a.runs))
        return

    fam = {c["tag"]: c["family"] for c in DEFAULT_CANDIDATES}
    cands = [{"tag": t, "family": fam.get(t, t.split(":")[0])} for t in a.models] if a.models else DEFAULT_CANDIDATES
    out.mkdir(parents=True, exist_ok=True)
    before_each = None
    if a.start_ollama:
        log = out / "ollama.log"
        start_ollama(log, a.host)
        before_each = lambda: ollama_up(a.host) or start_ollama(log, a.host)  # noqa: E731
    connect(a.host)
    info = prepare_models(cands, out, pull=not a.no_pull)
    models = [t for t, i in info.items() if "pull_error" not in i]
    gpu = gpu_name()
    (out / "run_info.json").write_text(json.dumps({"gpu": gpu, "runs": a.runs, "gen_meetings": a.gen_meetings,
                                                   "checker": a.checker}, indent=2))
    run_all(models, a.samples, out, runs=a.runs, gen_meetings=a.gen_meetings, long_tokens=a.long_tokens,
            checker=a.checker, before_each=before_each)
    finish(out, info, Path(a.docs), gpu, a.runs)


def finish(out, info, docs, gpu, runs):
    """Summary, suggestion, decision table, report and the list of check (iv) flags to review by hand."""
    summary = summarize(out, info)
    pick = suggest(summary)
    table = decision_table(summary, pick)
    write_outputs(out, summary, pick, table, docs, gpu=gpu, runs=runs)
    flags = review_flags(out)
    if flags:
        with (out / "model_selection_flags.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flags[0])); w.writeheader(); w.writerows(flags)
    print("\nSuggestion:", json.dumps({k: v for k, v in pick.items() if not k.endswith("rank")}))
    print(f"Check (iv) flags to review by hand: {len(flags)} -> {out / 'model_selection_flags.csv'}")
    print("Report:", docs / "model_selection.md")


if __name__ == "__main__":
    main()
