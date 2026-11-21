"""Shared code for the Team Memory Agent data notebooks.

Used by:
  datagen/demo_data.ipynb        - writes the hand-written demo project (samples/demo/)
  datagen/datagen_pipeline.ipynb - generates the development and test data for the evaluation
  model_selection.ipynb          - tests candidate models (Colab)

Everything here follows Section 6.1 of the system design document. Keeping it in one file means
the three notebooks always use the same file format, the same labels and the same checks.
"""
import csv
import datetime as dt
import hashlib
import json
import random
import re
from pathlib import Path

# ---------- Date resolver ----------
# People say dates in words, not as calendar dates. This function turns the words into a date,
# counting from the day of the meeting. Examples, for a meeting on Thursday 12 November 2026:
#     "by the 18th"             -> 2026-11-18
#     "by Friday"               -> 2026-11-13  (the first Friday after the meeting)
#     "by the end of the month" -> 2026-11-30
#     "by the 4th"              -> 2026-12-04  (the 4th has passed this month, so next month)
# Why we need it: check (ii) in Step C asks "does the meeting really say the date that the answer key
# expects?" We answer that by resolving every date phrase in the line and comparing.
# Note: the system's Match step (app/match) needs the same logic; a shared module is recommended.
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

def _last_day(year, month):
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return nxt - dt.timedelta(days=1)

def find_dates(text: str, meeting: dt.date) -> list:
    """Finds every date phrase in the text and returns the calendar dates they mean.
    Example: find_dates("Ali, the API by the 18th?", date(2026, 11, 12)) -> [date(2026, 11, 18)]"""
    t = text.lower()
    found = []
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        found.append(dt.date(int(m[1]), int(m[2]), int(m[3])))
    month_re = "|".join(MONTHS)
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_re})\b|\b({month_re})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", t):
        day = int(m[1] or m[4]); mon = MONTHS.index(m[2] or m[3]) + 1
        y = meeting.year + (1 if mon < meeting.month else 0)
        found.append(dt.date(y, mon, day))
    for m in re.finditer(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", t):
        day = int(m[1])
        if day > meeting.day:
            found.append(dt.date(meeting.year, meeting.month, day))
        else:
            y, mo = (meeting.year + 1, 1) if meeting.month == 12 else (meeting.year, meeting.month + 1)
            found.append(dt.date(y, mo, day))
    if re.search(r"\bend of (the )?month\b", t):
        found.append(_last_day(meeting.year, meeting.month))
    if re.search(r"\btomorrow\b", t):
        found.append(meeting + dt.timedelta(days=1))
    if re.search(r"\bin (two|2) weeks\b", t):
        found.append(meeting + dt.timedelta(days=14))
    for m in re.finditer(r"\b(next\s+)?(" + "|".join(WEEKDAYS) + r")\b", t):
        delta = (WEEKDAYS.index(m[2]) - meeting.weekday()) % 7 or 7   # first such weekday after the meeting
        found.append(meeting + dt.timedelta(days=delta))
    return found

# 1 -> "1st", 2 -> "2nd", 18 -> "18th" (used to write phrases such as "by the 18th")
def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

# ---------- Tagged lines ----------
# To build the answer key, we need to know which line talks about which task change.
# So the line gets a label at the end, for example:
#     "Can: Sure, I can take it. [E9]"   ->  this line is about event E9
# The LLM adds these labels when it writes the meeting (we add them by hand for the demo data).
#
# The labels are then split off:
#   - The system sees the line WITHOUT the label ("Can: Sure, I can take it."),
#     otherwise it would see the answer.
#   - The answer key keeps where each label was ("E9 -> lines 4 and 5"),
#     so we can check later whether the system cited the right lines.
# (Design document, Section 6.1-B)
TAG_RE = re.compile(r"\s*\[(E\d+(?:\s*,\s*E\d+)*)\]\s*$")

def split_tags(lines):
    """Splits the labels off the lines.
    Input:  [("Can", "Sure, I can take it. [E9]"), ...]
    Output: transcript = [("Can", "Sure, I can take it."), ...]     <- what the system will see
            evidence   = {"E9": [5], ...}                           <- goes to the answer key"""
    transcript, evidence = [], {}
    for i, (spk, text) in enumerate(lines, start=1):
        m = TAG_RE.search(text)
        clean = TAG_RE.sub("", text).strip()
        if m:
            for eid in re.split(r"\s*,\s*", m[1]):
                evidence.setdefault(eid, []).append(i)
        transcript.append((spk, clean))
    return transcript, evidence

def apply_events(state: dict, events: list) -> dict:
    """Updates the task list with one meeting's events and returns the new list.
    Example: the list has {"T4": owner "Ali"}; the event "assign T4 to Can" -> {"T4": owner "Can"}.
    This is how we know what the correct task list looks like after every meeting."""
    s = json.loads(json.dumps(state))
    for e in events:
        tid = e["task_id"]
        if e["type"] == "create":
            s[tid] = {"description": e["description"], "owner": e.get("owner"),
                      "due_date": e.get("due_date"), "status": "open"}
        elif e["type"] == "assign":
            s[tid]["owner"] = e["owner"]
        elif e["type"] == "set_due_date":
            s[tid]["due_date"] = e["due_date"]
        elif e["type"] == "complete":
            s[tid]["status"] = "done"
        elif e["type"] == "cancel":
            s[tid]["status"] = "cancelled"
    return s

# The correct task list after meeting 1, after meeting 2, ... (stored in scenario.json as "expected_state").
# In carry-forward mode the system's list is compared with these after every meeting.
def expected_states(events, n_meetings):
    states, s = {}, {}
    for m in range(1, n_meetings + 1):
        s = apply_events(s, [e for e in events if e["meeting"] == m])
        states[str(m)] = s
    return states

def write_project(base: Path, project_id: str, team: dict, scenario: dict, meetings: dict):
    """meetings: {meeting_no: [(speaker, text_with_tag), ...]}.
    Writes one project to two separate folders:
      agent_inputs/<project>/  team.json, meetings.json, meeting_01.transcript.txt ...  <- the system reads only this
      answer_keys/<project>/   scenario.json, meeting_01.evidence.json ...               <- only for scoring, never shown to the system"""
    inp = base / "agent_inputs" / project_id
    key = base / "answer_keys" / project_id
    inp.mkdir(parents=True, exist_ok=True); key.mkdir(parents=True, exist_ok=True)
    (inp / "team.json").write_text(json.dumps(team, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {"project_id": project_id, "meeting_dates": scenario["meeting_dates"]}
    (inp / "meetings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (key / "scenario.json").write_text(json.dumps(scenario, indent=2, ensure_ascii=False), encoding="utf-8")
    for m, lines in meetings.items():
        transcript, evidence = split_tags(lines)
        (inp / f"meeting_{m:02d}.transcript.txt").write_text(
            "\n".join(f"{s}: {t}" for s, t in transcript) + "\n", encoding="utf-8")
        (key / f"meeting_{m:02d}.evidence.json").write_text(
            json.dumps({"meeting": m, "evidence": evidence}, indent=2), encoding="utf-8")

# ---------- Automatic check (6.1-C) ----------
# An LLM does not always write what we asked for. It may forget an event, name the wrong person,
# or say a different date. These checks catch that before a meeting enters the data set.
# The answer key comes from the scenario, so a meeting that does not match it would make the key wrong.
# {"Ayşe": ["Ayşe", "Ayse"], ...}: every name a person can be called in the meeting
def names_of(team):
    out = {}
    for p in team["members"]:
        out[p["name"]] = [p["name"]] + p.get("aliases", [])
    return out

# True if one of the names appears as a whole word ("Can" matches "Can," but not "scan")
def mentions(text, names):
    return any(re.search(rf"\b{re.escape(n)}\b", text) for n in names)

def auto_check(team, scenario, meeting_no, lines, checker=None):
    """Checks one meeting against the scenario.
    Returns a list of problems as (reason, event_id, detail). An empty list means the meeting passed.
    Example problem: ("missing_event", "E9", "no tagged line") -> the meeting never mentions event E9."""
    problems = []
    members = names_of(team)
    meeting_date = dt.date.fromisoformat(scenario["meeting_dates"][meeting_no - 1])
    transcript, evidence = split_tags(lines)
    events = [e for e in scenario["events"] if e["meeting"] == meeting_no]
    # Every speaker must be a team member, and no label may be left in the clean text.
    for i, (spk, txt) in enumerate(transcript, 1):
        if spk not in members:
            problems.append(("unknown_speaker", None, f"line {i}: {spk}"))
        if re.search(r"\[E\d+", txt):
            problems.append(("tag_leak", None, f"line {i}"))
    for e in events:
        eid = e["event_id"]
        tagged = [transcript[i - 1][1] for i in evidence.get(eid, [])]
        # (i) Every planned event must appear in the meeting (at least one line has its label).
        if not tagged:
            problems.append(("missing_event", eid, "no tagged line")); continue
        joined = " ".join(tagged)
        speakers = {transcript[i - 1][0] for i in evidence.get(eid, [])}
        # (ii) If the event gives a task to someone, that person must be named in the line
        #      ("Can, could you take over...") or must be the one speaking ("I'll do it").
        #      If the event sets a date, the line must contain a phrase that means exactly that date.
        if e.get("owner") and e.get("ambiguity") != "owner" and e["type"] in ("create", "assign"):
            if not (mentions(joined, members.get(e["owner"], [e["owner"]])) or e["owner"] in speakers):
                problems.append(("missing_owner_or_date_cue", eid, f"owner {e['owner']} not named"))
        if e.get("due_date") and e["type"] in ("create", "set_due_date"):
            if dt.date.fromisoformat(e["due_date"]) not in find_dates(joined, meeting_date):
                problems.append(("missing_owner_or_date_cue", eid, f"no phrase resolving to {e['due_date']}"))
        # (iii) Some events are planned as unclear on purpose: "someone has to run load tests" (no owner said).
        #       Then the owner's name must NOT appear; otherwise the case is not really unclear and the
        #       clarification test would be meaningless.
        if e.get("ambiguity") == "owner":
            ans = e.get("coordinator_answer")
            if ans and ans != "not decided" and mentions(joined, members.get(ans, [ans])):
                problems.append(("ambiguity_not_real", eid, f"owner {ans} is named"))
    # (iv) A second model reads the meeting and lists every change it finds.
    #      A change that is not in the scenario means the generator invented something extra,
    #      which would make the answer key incomplete.
    if checker is not None:
        for ch in checker("\n".join(f"{s}: {t}" for s, t in transcript)):
            if not _matches_scripted(ch, events, scenario):
                problems.append(("unscripted_change", None, json.dumps(ch, ensure_ascii=False)))
    return problems

# Small helper for (iv): compares task names by their important words ("payment API endpoint" ~ "API endpoint").
STOP = {"the", "a", "an", "for", "of", "to", "and", "on", "in", "with", "from", "last", "week", "that", "thing"}
def _words(s):
    return {w for w in re.findall(r"[a-zçğıöşü]+", (s or "").lower()) if w not in STOP}

def _matches_scripted(change, events, scenario):
    for e in events:
        if e["type"] != change.get("type"):
            continue
        desc = scenario["tasks"][e["task_id"]]["description"] + " " + scenario["tasks"][e["task_id"]].get("ref", "")
        if _words(desc) & _words(change.get("task")):
            return True
    return False


# Name lists. The test projects use names never used in development (design 6.1),
# so the system cannot do well on the test set just because it got used to certain names.
DEV_NAMES = ["Ali", "Ayşe", "Kaan", "Zeynep", "Deniz", "Elif", "Mert", "Selin", "Burak", "Ece", "Kerem", "Derya"]
TEST_NAMES = ["Omar", "Lina", "Jonas", "Mira", "Tarik", "Nora", "Felix", "Yara", "Emre", "Sena", "Hugo", "Ines"]
ROLES = ["backend", "frontend", "data", "design", "qa"]
# Note: names that are also English words (such as "Can") are not used in the generator lists, because they confuse the name check.
# "Can" in the demo data is deliberate (it is the example in the design document).

# Task lists: (task name, which role usually does it, a vague way people refer to it).
# The vague reference is used for the difficult case, e.g. "that API thing" instead of "Payment API endpoint".
# Test projects again use their own task list.
DEV_TASKS = [
    ("Payment API endpoint", "backend", "that API thing"),
    ("Login page", "frontend", "the login stuff"),
    ("Database schema", "backend", "the schema thing"),
    ("User survey analysis", "data", "the survey work"),
    ("Onboarding screens", "design", "those onboarding screens"),
    ("Regression test suite", "qa", "the testing thing"),
    ("Search filters", "frontend", "the filter stuff"),
    ("Sales dashboard", "data", "that dashboard"),
    ("Email notifications", "backend", "the email thing"),
    ("Icon set", "design", "the icons"),
    ("Release checklist", "qa", "that checklist"),
    ("Caching layer", "backend", "the caching work"),
]
TEST_TASKS = [
    ("Invoice export", "backend", "that export thing"),
    ("Profile settings page", "frontend", "the settings stuff"),
    ("Data retention script", "data", "the retention work"),
    ("Accessibility review", "qa", "the accessibility thing"),
    ("Landing page mockups", "design", "those mockups"),
    ("Rate limiter", "backend", "the limiter thing"),
    ("Churn report", "data", "that report"),
    ("Mobile navigation bar", "frontend", "the nav stuff"),
    ("Load test plan", "qa", "the load testing thing"),
    ("Brand colour guide", "design", "the colour thing"),
    ("Audit log viewer", "backend", "the audit thing"),
    ("Session timeout handling", "frontend", "the timeout stuff"),
]

def date_phrase(rng, meeting: dt.date):
    """Picks a deadline and the words used to say it, e.g. ("by the 18th", "2026-11-18").
    The date is computed with the same resolver that check (ii) uses, so phrase and key always agree."""
    kind = rng.choice(["weekday", "dom", "dom", "eom"])
    if kind == "weekday":
        target = meeting + dt.timedelta(days=rng.randint(2, 7))
        phrase = f"by {WEEKDAYS[target.weekday()].capitalize()}"
    elif kind == "dom":
        target = meeting + dt.timedelta(days=rng.randint(5, 20))
        phrase = f"by the {ordinal(target.day)}"
    else:
        target = _last_day(meeting.year, meeting.month)
        if (target - meeting).days < 4:
            target = meeting + dt.timedelta(days=10); phrase = f"by the {ordinal(target.day)}"
        else:
            phrase = "by the end of the month"
    assert target in find_dates(phrase, meeting), (phrase, target)
    return phrase, target.isoformat()

def generate_project(project_id, seed, split, n_meetings=5, team_size=4):
    """Writes the plan (scenario) of one project: who is in the team and what happens in each meeting.
    No LLM here; plain code with a fixed seed, so the same seed always gives the same project.
    Meeting 1 opens three tasks. Each later meeting changes two or three open tasks
    (new owner, new date, done, cancelled) and opens one or two new ones.
    Every project contains at least one vague reassignment (meeting 3) and one task with no stated owner."""
    rng = random.Random(seed)   # all random choices come from this seed -> reproducible
    names = rng.sample(DEV_NAMES if split == "dev" else TEST_NAMES, team_size)
    bank = list(DEV_TASKS if split == "dev" else TEST_TASKS); rng.shuffle(bank)
    roles = rng.sample(ROLES, team_size - 1)
    members = [{"name": names[0], "role": "coordinator", "aliases": []}] + [
        {"name": n, "role": r, "aliases": []} for n, r in zip(names[1:], roles)]
    team = {"project_id": project_id, "coordinator": names[0], "members": members}
    workers = members[1:]
    # Weekly meetings, always on the same weekday, starting on a seeded date.
    start = dt.date(2026, 9, 3) + dt.timedelta(days=7 * rng.randint(0, 8))
    dates = [(start + dt.timedelta(days=7 * i)) for i in range(n_meetings)]

    tasks, events, state = {}, [], {}
    ev = iter(range(1, 1000)); tk = iter(range(1, 1000))
    def new_event(**kw):
        e = {"event_id": f"E{next(ev)}", "ambiguity": None, "coordinator_answer": None}
        e.update(kw); events.append(e); return e
    def create(m, ambiguous_owner=False):
        # Opens a new task in meeting m. The owner has a matching role when possible.
        # 80% of new tasks get a deadline. With ambiguous_owner=True the meeting will not say who does it.
        if not bank: return
        desc, role, ref = bank.pop()
        tid = f"T{next(tk)}"; tasks[tid] = {"description": desc, "ref": ref, "role": role}
        cands = [w for w in workers if w["role"] == role] or workers
        owner = rng.choice(cands)["name"]
        phrase, due = date_phrase(rng, dates[m - 1]) if rng.random() < 0.8 else (None, None)
        if ambiguous_owner:
            answer = owner if rng.random() < 0.75 else "not decided"
            new_event(meeting=m, type="create", task_id=tid, description=desc, owner=None,
                      due_date=due, due_phrase=phrase, ambiguity="owner", coordinator_answer=answer)
        else:
            new_event(meeting=m, type="create", task_id=tid, description=desc, owner=owner,
                      due_date=due, due_phrase=phrase)

    for _ in range(3):
        create(1)
    for m in range(2, n_meetings + 1):
        state = expected_states(events, m - 1)[str(m - 1)]
        open_ids = [t for t, v in state.items() if v["status"] == "open"]
        rng.shuffle(open_ids)
        n_updates = min(len(open_ids), rng.randint(2, 3))
        for i, tid in enumerate(open_ids[:n_updates]):
            cur = state[tid]
            if m == 3 and i == 0:   # every project gets at least one reassignment with a vague reference (a difficult case)
                op = "assign_ref"
            else:
                op = rng.choice(["assign", "set_due_date", "complete", "complete", "cancel"])
            if op in ("assign", "assign_ref"):
                others = [w["name"] for w in workers if w["name"] != cur["owner"]]
                new_event(meeting=m, type="assign", task_id=tid, owner=rng.choice(others),
                          previous_owner=cur["owner"],
                          ambiguity="reference" if op == "assign_ref" else None)
            elif op == "set_due_date":
                phrase, due = date_phrase(rng, dates[m - 1])
                new_event(meeting=m, type="set_due_date", task_id=tid, due_date=due, due_phrase=phrase)
            else:
                new_event(meeting=m, type=op, task_id=tid)
        create(m, ambiguous_owner=(m == 2 or (m >= 4 and rng.random() < 0.3)))
        if rng.random() < 0.5:
            create(m)

    scenario = {"project_id": project_id, "split": split, "seed": seed,
                "meeting_dates": [d.isoformat() for d in dates],
                "tasks": tasks, "events": events,
                "expected_state": expected_states(events, n_meetings)}
    return team, scenario


# Small-talk lines used by the mock backend, so mock meetings are not only task talk.
FILLER = ["Morning everyone.", "Can you all hear me?", "Let's start with updates.",
          "Sorry, I was on mute.", "Anything else before we close?", "Okay, sounds good.",
          "Let me check my notes.", "Thanks, see you next week."]

def render_mock(team, scenario, m, rng):
    """Writes meeting m from fixed sentence templates (no model). Only to test the code.
    Example output: "Zeynep: Ali, can you take the login page by the 18th? [E3]" """
    coord = team["coordinator"]
    workers = [p["name"] for p in team["members"] if p["name"] != coord]
    lines = [(coord, FILLER[0]), (rng.choice(workers), rng.choice(FILLER[1:4]))]
    for e in [e for e in scenario["events"] if e["meeting"] == m]:
        tag = f" [{e['event_id']}]"; task = scenario["tasks"][e["task_id"]]
        d = task["description"]; low = d[0].lower() + d[1:]
        if e["type"] == "create" and e["ambiguity"] == "owner":
            when = f" {e['due_phrase']}" if e.get("due_phrase") else ""
            lines += [(coord, f"Someone has to do the {low}{when}." + tag),
                      (rng.choice(workers), "Let's decide who takes it later.")]
        elif e["type"] == "create":
            when = f" {e['due_phrase']}" if e.get("due_phrase") else ""
            lines += [(coord, f"{e['owner']}, can you take the {low}{when}?" + tag),
                      (e["owner"], "Yes, I'll do it." + tag)]
        elif e["type"] == "assign" and e["ambiguity"] == "reference":
            lines += [(coord, f"{e['previous_owner']} is swamped. {e['owner']}, could you take over {task['ref']} from last week?" + tag),
                      (e["owner"], "Sure, I can take it." + tag)]
        elif e["type"] == "assign":
            lines += [(coord, f"{e['owner']}, please take over the {low} from {e['previous_owner']}." + tag),
                      (e["owner"], "Okay, it's mine now." + tag)]
        elif e["type"] == "set_due_date":
            lines += [(coord, f"The {low} will slip, let's move it {e['due_phrase']}." + tag)]
        elif e["type"] == "complete":
            lines += [(coord, f"The {low} is done, right?"), (rng.choice(workers), "Yes, it's finished." + tag)]
        elif e["type"] == "cancel":
            lines += [(coord, f"We're dropping the {low}, we don't need it anymore." + tag)]
        if rng.random() < 0.4:
            lines.append((rng.choice(workers), rng.choice(FILLER[4:7])))
    lines.append((coord, FILLER[-1]))
    return lines

def build_prompt(team, scenario, m):
    """Builds the instructions for the generator LLM (MIMIC style).
    The model gets the participants and an agenda (one line per planned event) and plays everyone.
    Example agenda line: "E9: The task 'Payment API endpoint' moves from Ali to Can.
                          Refer to the task ONLY as 'that API thing' ..."
    The rules at the end ask for labels such as [E9] and forbid extra task changes."""
    people = "\n".join(f"- {p['name']} ({p['role']})" for p in team["members"])
    agenda = []
    for e in [e for e in scenario["events"] if e["meeting"] == m]:
        t = scenario["tasks"][e["task_id"]]; d = t["description"]
        if e["type"] == "create" and e["ambiguity"] == "owner":
            s = f"Someone says the task '{d}' must be done but NOBODY says who will do it."
        elif e["type"] == "create":
            s = f"{e['owner']} agrees to do a new task '{d}'."
        elif e["type"] == "assign" and e["ambiguity"] == "reference":
            s = (f"The task '{d}' moves from {e['previous_owner']} to {e['owner']}. "
                 f"Refer to the task ONLY as '{t['ref']}' (from an earlier meeting), never by its full name.")
        elif e["type"] == "assign":
            s = f"The task '{d}' moves from {e['previous_owner']} to {e['owner']}; say {e['owner']}'s name."
        elif e["type"] == "set_due_date":
            s = f"The deadline of '{d}' changes."
        elif e["type"] == "complete":
            s = f"The task '{d}' is reported as finished."
        else:
            s = f"The team decides to drop the task '{d}'."
        if e.get("due_phrase"):
            s += f" The deadline must be said exactly as '{e['due_phrase']}'."
        agenda.append(f"{e['event_id']}: {s}")
    system = ("You write realistic, slightly messy transcripts of weekly software project meetings. "
              "You role-play every participant with their own way of speaking: interruptions, small talk, "
              "hesitations and short replies are welcome.")
    user = (f"Meeting date: {scenario['meeting_dates'][m-1]}. Coordinator: {team['coordinator']}.\n"
            f"Participants:\n{people}\n\nAgenda (each item must happen in the conversation):\n" + "\n".join(agenda) +
            "\n\nRules:\n- Write 20 to 35 lines, each as 'Name: text'. Use only the participants above.\n"
            "- At the END of every line that states an agenda item, add its tag, e.g. [E3]. Do not tag other lines.\n"
            "- Do not agree any other task, owner change, deadline, completion or cancellation.\n"
            "- Output only the transcript.")
    return system, user

# Turns the model's answer into [(speaker, text), ...] and drops lines that are not "Name: text"
# for a real team member (for example a title or an explanation the model added).
def parse_transcript(text, team):
    names = [p["name"] for p in team["members"]]
    out = []
    for raw in text.splitlines():
        m = re.match(r"^\s*\**([^:*]{1,30})\**\s*:\s*(.+)$", raw)
        if m and m[1].strip() in names:
            out.append((m[1].strip(), m[2].strip()))
    return out

# ---------- Talking to a model through Ollama ----------
def make_client(host="http://localhost:11434"):
    """Creates an Ollama client. Imported here so the mock path works without the ollama package."""
    from ollama import Client
    return Client(host=host)


def ollama_chat(client, model, system, user, seed, temperature=0.8, num_ctx=32768):
    """Sends one request and returns the model's text answer.
    The seed makes a retry produce a different meeting (attempt 2 and 3 use different seeds)."""
    r = client.chat(model=model, messages=[{"role": "system", "content": system},
                                          {"role": "user", "content": user}],
                    options={"temperature": temperature, "seed": seed, "num_ctx": num_ctx})
    return r["message"]["content"]


def render(team, scenario, m, seed, backend="mock", client=None, model=None):
    """Writes meeting m: with fixed templates (backend="mock") or with an LLM (backend="ollama")."""
    if backend == "mock":
        return render_mock(team, scenario, m, random.Random(seed))
    system, user = build_prompt(team, scenario, m)
    return parse_transcript(ollama_chat(client, model, system, user, seed), team)


def make_checker(client, model, seed):
    """Check (iv): a model reads the finished meeting (without labels) and lists every task change it finds.
    Returns a function that takes the transcript text and returns a list of changes."""
    def checker(transcript_text):
        system = "You extract task changes from meeting transcripts. Answer with JSON only."
        user = ("List every task change agreed in this transcript as a JSON list of objects "
                "{\"type\": \"create|assign|set_due_date|complete|cancel\", \"task\": \"short description\", "
                "\"owner\": \"name or null\"}.\n\n" + transcript_text)
        raw = ollama_chat(client, model, system, user, seed, temperature=0)
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            return json.loads(m[0]) if m else []
        except json.JSONDecodeError:
            return [{"type": "unparsable_checker_output", "task": raw[:200]}]
    return checker


def load_project(base, project_id):
    """Reads a project written by write_project. Returns (team, scenario, transcripts, evidence)."""
    base = Path(base)
    inp, key = base / "agent_inputs" / project_id, base / "answer_keys" / project_id
    team = json.loads((inp / "team.json").read_text(encoding="utf-8"))
    scenario = json.loads((key / "scenario.json").read_text(encoding="utf-8"))
    transcripts, evidence = {}, {}
    for m in range(1, len(scenario["meeting_dates"]) + 1):
        transcripts[m] = (inp / f"meeting_{m:02d}.transcript.txt").read_text(encoding="utf-8").splitlines()
        evidence[m] = json.loads((key / f"meeting_{m:02d}.evidence.json").read_text())["evidence"]
    return team, scenario, transcripts, evidence
