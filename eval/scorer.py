"""Scoring a run against the answer key (design §12.3). No model is used.

Inputs per meeting: the proposals the configuration made, which of them were committed, and (in carry-forward
mode) the task list after the meeting. The answer key is the project's scenario.json and evidence files.

**Task matching.** The system never sees gold task ids. An update proposal names a system task id; it is mapped
to a gold task through `id_map` (seeded tasks keep their id; a task the system created is mapped when its create
proposal is matched). A create proposal is matched to a create event of the same meeting by its description:
the share of the shorter description's words that also appear in the other one (words as app.match tokenises
them), at least DESC_THRESHOLD. Matching is one-to-one and greedy by that share; ties are broken by order.

**Change F1 (primary).** A proposal is correct when its type, its gold task and the required fields are right:
create: owner and due date (null when the meeting leaves them open; the coordinator's answer is accepted for a
task whose owner was left open on purpose); assign: owner; set_due_date: due date; complete and cancel: nothing.
Counts are summed over meetings before precision, recall and F1 are computed (micro-averaging), also per type.
A wrong field is reported as "missing" (left empty although the key has a value) or "wrong" (another value).

**Unsupported proposals (automatic proxy).** A proposal is supported when it cites at least one line that the
key lists for its event. A proposal that matches no event is unsupported. The design's manual check of 30
proposals is still to be done; this number only screens for it.

**Accumulated task-state accuracy (carry-forward).** After each meeting, for every gold task: owner, due date and
status are compared with the system's task; the share of correct (task, field) pairs, plus extra and missing
tasks, and whether the whole list is right.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from app.match.match import tokenize
from app.schemas import ActionType, EvidenceSource, ProposedUpdate, TaskRecord

DESC_THRESHOLD = 0.6     # fixed on the development material; see the module docstring
TYPES = [a.value for a in ActionType]
NOT_DECIDED = "not decided"


def desc_similarity(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _date(v) -> date | None:
    if v is None or isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


@dataclass
class Row:
    """One expected change or one proposal, after matching."""
    meeting: int
    event_id: str | None
    gold_task: str | None
    expected: str | None
    proposal_index: int | None
    proposed: str | None
    action: str
    result: str                     # correct | wrong | missed | extra
    problems: list[str] = field(default_factory=list)
    supported: bool | None = None
    committed: bool | None = None


def _describe_event(e: dict, tasks: dict) -> str:
    s = f"{e['type']} {e['task_id']} '{tasks[e['task_id']]['description']}'"
    if e["type"] in ("create", "assign"):
        s += f" owner={e.get('owner')}"
    if e["type"] in ("create", "set_due_date"):
        s += f" due={e.get('due_date')}"
    return s


def _describe_proposal(p: ProposedUpdate) -> str:
    f = p.fields.model_dump()
    s = f"{p.action.value} {p.task_id or 'new'} '{p.task_description}'"
    if "owner" in f:
        s += f" owner={f['owner']}"
    if "due_date" in f:
        s += f" due={f['due_date']}"
    return s


def check_fields(p: ProposedUpdate, e: dict) -> list[str]:
    """The required fields that are wrong, as 'owner missing', 'due_date wrong', ..."""
    out = []
    f = p.fields
    if e["type"] in ("create", "assign"):
        want = e.get("owner")
        ok_values = {want}
        if e.get("ambiguity") == "owner" and e.get("coordinator_answer") not in (None, NOT_DECIDED):
            ok_values.add(e["coordinator_answer"])          # the coordinator's answer to a clarification
        if f.owner not in ok_values:
            out.append("owner missing" if f.owner is None else "owner wrong")
    if e["type"] in ("create", "set_due_date"):
        want = _date(e.get("due_date"))
        if _date(f.due_date) != want:
            out.append("due_date missing" if f.due_date is None and want is not None else "due_date wrong")
    return out


def score_meeting(meeting_no: int, proposals: list[ProposedUpdate], scenario: dict, evidence: dict,
                  id_map: dict[str, str], committed: set[int] | None = None) -> tuple[list[Row], dict[int, str]]:
    """Match one meeting's proposals with its expected events.
    Returns the rows and, for matched create proposals, {proposal index: gold task id}."""
    events = [e for e in scenario["events"] if e["meeting"] == meeting_no]
    tasks = scenario["tasks"]
    pairs: list[tuple[float, int, int]] = []          # (similarity, proposal index, event index)
    for pi, p in enumerate(proposals):
        for ei, e in enumerate(events):
            if p.action == ActionType.create:
                if e["type"] != "create":
                    continue
                sim = desc_similarity(p.fields.description or p.task_description, tasks[e["task_id"]]["description"])
                if sim >= DESC_THRESHOLD:
                    pairs.append((sim, pi, ei))
            else:
                gold = id_map.get(p.task_id or "")
                if gold == e["task_id"] and e["type"] != "create":
                    pairs.append((2.0 if p.action.value == e["type"] else 1.0, pi, ei))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_p, used_e, link = set(), set(), {}
    for _, pi, ei in pairs:
        if pi in used_p or ei in used_e:
            continue
        used_p.add(pi); used_e.add(ei); link[pi] = ei

    rows, created = [], {}
    for pi, p in enumerate(proposals):
        cited = {r.id for r in p.evidence_refs if r.source == EvidenceSource.transcript}
        if pi in link:
            e = events[link[pi]]
            lines = {f"L{n}" for n in evidence.get(e["event_id"], [])}
            problems = [] if p.action.value == e["type"] else [f"type {p.action.value}, expected {e['type']}"]
            problems += check_fields(p, e) if not problems else []
            rows.append(Row(meeting_no, e["event_id"], e["task_id"], _describe_event(e, tasks), pi, _describe_proposal(p),
                            e["type"], "wrong" if problems else "correct", problems, bool(cited & lines),
                            None if committed is None else pi in committed))
            if p.action == ActionType.create:
                created[pi] = e["task_id"]
        else:
            rows.append(Row(meeting_no, None, id_map.get(p.task_id or ""), None, pi, _describe_proposal(p), p.action.value,
                            "extra", [], False, None if committed is None else pi in committed))
    for ei, e in enumerate(events):
        if ei not in used_e:
            rows.append(Row(meeting_no, e["event_id"], e["task_id"], _describe_event(e, tasks), None, None, e["type"], "missed"))
    return rows, created


def change_f1(rows: list[Row]) -> dict:
    """Micro precision, recall and F1, overall and per type, plus field problems."""
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f = 2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": _r(p), "recall": _r(r), "f1": _r(f)}
    out = {"all": prf(*_counts(rows, None))}
    out["by_type"] = {t: prf(*_counts(rows, t)) for t in TYPES}
    out["field_problems"] = dict(Counter(x for r in rows for x in r.problems))
    proposals = [r for r in rows if r.proposal_index is not None]
    out["unsupported_rate_proxy"] = _r(sum(1 for r in proposals if not r.supported) / len(proposals)) if proposals else None
    return out


def _counts(rows, t):
    sel = [r for r in rows if t is None or r.action == t]
    tp = sum(1 for r in sel if r.result == "correct")
    fp = sum(1 for r in sel if r.result in ("wrong", "extra"))
    fn = sum(1 for r in sel if r.result in ("wrong", "missed"))
    return tp, fp, fn


def _r(x):
    return None if x is None else round(x, 3)


def state_accuracy(system: list[TaskRecord], expected: dict, id_map: dict[str, str]) -> dict:
    """Compare the system's task list after a meeting with the key's expected list (carry-forward)."""
    by_gold = {id_map[t.task_id]: t for t in system if t.task_id in id_map}
    extra = [t.task_id for t in system if t.task_id not in id_map]
    pairs = correct = 0
    missing = []
    for gid, g in expected.items():
        s = by_gold.get(gid)
        pairs += 3
        if s is None:
            missing.append(gid)
            continue
        correct += (s.owner == g["owner"]) + (s.due_date == _date(g["due_date"])) + (s.status.value == g["status"])
    return {"accuracy": _r(correct / pairs) if pairs else None, "correct_pairs": correct, "pairs": pairs,
            "missing_tasks": missing, "extra_tasks": extra,
            "fully_correct": not missing and not extra and correct == pairs}
