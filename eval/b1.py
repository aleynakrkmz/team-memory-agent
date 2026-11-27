"""Baseline B1: fixed rules and BM25 matching, no language model (design §2, §12).

B1 reads a meeting as a conversation, not line by line. Version 2 (written on the generated development set,
dev set v1; never on the test set):

1. **Items.** A line that names a task starts an item: a task already on the list (its key words, e.g. "the
   icons" for "Icon set"), a task created earlier in the meeting, or a new task ("a new task for you - the
   login page", "we need an icon set", "could you take on the 'Release checklist'"). The item runs over that
   line and the next lines, up to 3, until the next item starts.
2. **Change.** Fixed word lists decide the change of an item about a task on the list, in this order:
   cancel ("let's drop it", "park this one"), complete ("it's all wrapped up", "I finished it yesterday"; not
   "almost done" or "I'll make sure it's done"), assign ("take over", "pass it to Ece", "over to you, Burak",
   "you've got it now"), set_due_date ("the deadline has changed ... by the 28th", "pushed back to Sunday").
   An item about a new task is a create.
3. **Owner and date.** The owner is the person a sentence addresses ("Ece, could you ...") or names after
   "to", else the next speaker who accepts ("Sure, I can take it"). "Let's figure out who later", "any
   volunteers?" and similar leave the owner open. The date is the first date phrase in the item.
4. **Matching and routing** are the agent's (app.match, app.route). Clear cases become proposals. A hard case
   becomes a proposal on the top BM25 task only when the task was merely unclear or ambiguous; every other
   hard case is left without a proposal (B1 has no agent).

Version 1 matched one line at a time with patterns written while reading the two sample projects; on the
development set it reached F1 0.00-0.14, because a generated meeting spreads one change over several lines.
The rules are frozen before the test set is generated and are not tuned on it.
"""
from __future__ import annotations

import re

from datagen.common import find_dates

from app.match import match_candidates
from app.route import route_candidates
from app.schemas import (
    ActionType,
    CandidateChange,
    EvidenceQuote,
    EvidenceRef,
    EvidenceSource,
    MeetingPreview,
    MeetingTranscript,
    ProposalOrigin,
    ProposedUpdate,
    RouteKind,
    TaskRecord,
    TeamProfile,
    UpdateFields,
)

STOP = {"the", "a", "an", "that", "this", "those", "these", "our", "my", "your", "task", "thing", "stuff", "work",
        "new", "one", "of", "for", "on", "to", "and", "with", "it", "them"}
GENERIC = {"set", "page", "layer", "suite", "test", "tests", "analysis", "endpoint", "screens", "screen", "task"}
DATE = re.compile(r"(?:(?:by|until|before|to|on|for)\s+)?(?:the\s+)?(?:end of (?:the )?month|\d{1,2}(?:st|nd|rd|th)\b"
                  r"|(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|tomorrow|in (?:two|2) weeks)", re.I)
NEW_TASK = [  # a new task is named after one of these; the first group that matches gives the name
    re.compile(r"\b(?:new|another)\s+(?:task|one)\b[^.?!]*?(?:\s[-–:]\s*|\.\s+|,\s+)(?:it's\s+|it is\s+)?(?P<task>[^.?!,]+)", re.I),
    re.compile(r"\b(?:we|i)\s+(?:also\s+)?need\s+(?:you\s+to\s+(?:work on|handle|do|build|tackle)\s+|someone to (?:work on|tackle|handle|do)\s+)?"
               r"(?P<task>(?:an?|the|some|those)\s+[^.?!,]+)", re.I),
    re.compile(r"\b(?:take on|handle|tackle|work on|creating|create|build|get started on|start on|start working on)\s+"
               r"(?P<task>(?:the|an?|that|those)?\s*'?[^.?!,]+)", re.I),
    # short requests, as in the hand-written sample projects (B1 version 1)
    re.compile(r"\b(?:can|could) you\s+(?:please\s+)?(?:do|build|take|write|prepare|set up|run|update|draft|make|own)\s+"
               r"(?P<task>[^.?!,]+)", re.I),
    re.compile(r"\b(?:please|you'll|i'll|i will)\s+(?:write|prepare|set up|run|update|draft|build|take|own|do)\s+"
               r"(?P<task>(?:the|an?)\s+[^.?!,]+)", re.I),
    re.compile(r"^(?:[^\W\d_]+,\s+)?(?P<task>(?:the|a|an)\s+[\w\s-]+?)\s+is yours\b", re.I),
]
QUOTED = re.compile(r"['‘“\"]([A-Z][^'’”\"]{2,40})['’”\"]")
TASK_END = re.compile(r"\s+(?:by|before|until|for|from|too|myself|yourself|no|as well|this|next|today|tomorrow|task|that|if|so|since|and|but|"
                      r"which|we|you|i|it's|is|are|can|could|would|will|please|right|now|first|then|though)\b.*$", re.I)
OPEN_OWNER = re.compile(r"figure out who|no one|nobody|not assigned|keep (?:this|it) open|decide who|revisit", re.I)
ASK_OWNER = re.compile(r"any volunteers|who(?:'s| is| will| can)? (?:taking|take|tackle|going to)|who can take|"
                       r"someone (?:needs|has) to|we need someone", re.I)       # open unless someone then accepts
CANCEL = re.compile(r"\b(?:drop(?:ping)?|scrap(?:ping)?|cancel(?:l?ing|led)?|shelve|park (?:this|it|that) one|"
                    r"put (?:it|this|that) on hold|no longer need)\b", re.I)
DONE = re.compile(r"\b(?:(?:is|are|it's|that's|they're|'s|been|was)\s+(?:all\s+|now\s+)?(?:done|finished|complete|completed|"
                  r"set|wrapped up|ready to go|live|merged|shipped)|(?:finished|wrapped up|completed|nailed|wrapped)\s+(?:it|that|them|"
                  r"those|this|the\b|up\b)|all wrapped up|all done|done and dusted|is done)\b", re.I)
NOT_DONE = re.compile(r"\b(?:almost|mostly|nearly|not yet|will|'ll|make sure|should|need to|needs to|have to|by then|"
                      r"once|until|going to|gonna|how's|have you|did you|is it)\b", re.I)
ASSIGN = re.compile(r"\b(?:take (?:it |this |that )?over|take it from here|hand (?:it |this |that )?(?:off|over)|"
                    r"pass (?:it|this|that|the [^.?!]+?) (?:on )?to|over to you|you've got (?:it|the [^.?!]+?) now|shift(?:ing|ed)? .*?to|"
                    r"shuffle .*?to|move (?:it|this|that) to|assign(?:ed)? (?:this|it|that|you)|reassign|"
                    r"want you to take|take (?:it|this|that) on|pick (?:it|this|that) up)\b", re.I)
DUE_CHANGE = re.compile(r"\b(?:deadline|due|moved|move|pushed|push|shift(?:ed|ing)?|changed?|adjust|extend(?:ed)?|"
                        r"now|bring (?:it )?forward|slip|until|more (?:days|time)|extension)\b", re.I)
ACCEPT = re.compile(r"^\s*(?:sure|yes|yeah|yep|of course|absolutely|okay|ok|alright|no problem|count me in|happy to)\b|"
                    r"\b(?:i can (?:take|do|handle|manage|work)|i'll (?:take|do|handle|get|make|dive|start|pick)|i'm on it|"
                    r"i can get started|i can have)\b", re.I)


def _words(text: str) -> set[str]:
    return {w.rstrip("s") if len(w) > 4 else w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def _key_words(description: str) -> set[str]:
    return _words(description) - {w.rstrip("s") for w in GENERIC} or _words(description)


def _task_in(text: str, known: dict[str, str]) -> str | None:
    """The known task (id -> description) a text names by its key words, the best one with half of them or more."""
    words, best, score = _words(text), None, 0.0
    for tid, desc in known.items():
        key = _key_words(desc)
        hit = len(key & words) / len(key) if key else 0
        if hit >= 0.5 and hit > score:
            best, score = tid, hit
    return best


def _new_task(text: str) -> str | None:
    """The name of a new task the line introduces, or None."""
    m = QUOTED.search(text)
    if m:
        return m.group(1).strip()
    for rule in NEW_TASK:
        m = rule.search(text)
        if m:
            name = re.sub(r"^(?:the|an?|that|those|some)\s+", "", m.group("task").strip(" '’"), flags=re.I)
            name = TASK_END.sub("", name).strip(" '-")
            if len(name) >= 3 and len(name.split()) <= 5 and _words(name):
                return name
    return None


def addressed(text: str, team: TeamProfile, speaker: str | None = None) -> str | None:
    """The team member a sentence of the line names with a comma: "Ece, could you ..." / "for you, Burak"."""
    for m in re.finditer(r"(?:^|[.!?,]\s+)([^\W\d_][\w'-]*)\s*[,?!.]|,\s*([^\W\d_][\w'-]*)\s*[.?!]", text):
        name = team.canonical_member(m.group(1) or m.group(2))
        if name and name != speaker:
            return name
    return None


def _named_after_to(text: str, team: TeamProfile) -> str | None:
    for m in re.finditer(r"\b(?:to|for)\s+([^\W\d_][\w'-]*)", text):
        name = team.canonical_member(m.group(1))
        if name:
            return name
    return None


def _date_phrase(lines: list[str], meeting_date) -> str | None:
    for text in lines:
        for m in DATE.finditer(text):
            if find_dates(m.group(0), meeting_date):
                return m.group(0)
    return None


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def extract(meeting: MeetingTranscript, team: TeamProfile, tasks: list[TaskRecord] | None = None) -> list[CandidateChange]:
    """All B1 candidates of one meeting."""
    utts = meeting.utterances
    known = {t.task_id: t.description for t in (tasks or []) if t.status.value == "open"}
    owners = {t.task_id: t.owner for t in (tasks or [])}
    starts: list[tuple[int, str, str | None]] = []           # (line index, existing task id or "", new task name)
    created: dict[str, str] = {}
    for i, u in enumerate(utts):
        tid = _task_in(u.text, known)
        new = None if tid else _new_task(u.text)
        if new and _task_in(new, {**created}):               # named again later in the meeting
            new, tid = None, ""
        if tid or new:
            if starts and starts[-1][1] == (tid or "") and starts[-1][2] is None and tid:
                continue                                     # the same task again: same item
            if new:
                created[f"new{len(created)}"] = new
            starts.append((i, tid or "", new))
    out: list[CandidateChange] = []
    for k, (i, tid, new) in enumerate(starts):
        end = min(starts[k + 1][0] if k + 1 < len(starts) else len(utts), i + 4)
        items = utts[i:end]
        texts = [u.text for u in items]
        joined = " ".join(texts)
        ev = [EvidenceQuote(utterance_id=items[0].id, quote=items[0].text)]
        date_ = _date_phrase(texts, meeting.meeting_date)

        def accepter():
            for u in items[1:]:
                if ACCEPT.search(u.text) and team.canonical_member(u.speaker):
                    return u
            return None

        if new is not None:                                  # create
            owner = None
            acc = accepter()
            if not OPEN_OWNER.search(joined) and not (ASK_OWNER.search(joined) and acc is None):
                owner = addressed(items[0].text, team, items[0].speaker)
                if owner is None and acc is not None:
                    owner = acc.speaker
                if acc is not None:
                    ev.append(EvidenceQuote(utterance_id=acc.id, quote=acc.text))
            out.append(CandidateChange(action=ActionType.create, description=new, owner=owner, due_phrase=date_,
                                       evidence=ev, confidence=1.0))
            continue
        if not tid:
            continue
        desc = known[tid]
        sents = [(u, s) for u in items for s in _sentences(u.text)]
        if any(CANCEL.search(s) for _, s in sents):
            u = next(u for u, s in sents if CANCEL.search(s))
            action, owner, date_ = ActionType.cancel, None, None
        elif any(DONE.search(s) and not NOT_DONE.search(s) for _, s in sents):
            u = next(u for u, s in sents if DONE.search(s) and not NOT_DONE.search(s))
            action, owner, date_ = ActionType.complete, None, None
        elif any(ASSIGN.search(s) for _, s in sents):
            u, s = next((u, s) for u, s in sents if ASSIGN.search(s))
            owner = _named_after_to(s, team) or addressed(s, team, u.speaker) or addressed(items[0].text, team, items[0].speaker)
            acc = accepter()
            if owner is None and acc is not None:
                owner = acc.speaker
            if owner is None or owner == owners.get(tid):
                continue
            action, date_ = ActionType.assign, None
        elif date_ and any(DUE_CHANGE.search(s) and DATE.search(s) for _, s in sents):
            u = next(u for u, s in sents if DUE_CHANGE.search(s) and DATE.search(s))
            action, owner = ActionType.set_due_date, None
        else:
            continue
        if u.id != items[0].id:
            ev.append(EvidenceQuote(utterance_id=u.id, quote=u.text))
        out.append(CandidateChange(action=action, description=desc, owner=owner, due_phrase=date_,
                                   evidence=ev, confidence=1.0))
    return out


def run_b1(meeting: MeetingTranscript, tasks: list[TaskRecord], team: TeamProfile) -> MeetingPreview:
    """B1 for one meeting, in the same shape as the agent pipeline's preview (no agent decisions)."""
    candidates = extract(meeting, team, tasks)
    matches = match_candidates(candidates, tasks, meeting.meeting_date)
    decisions = route_candidates(meeting, candidates, matches, tasks, team)
    by_id = {t.task_id: t for t in tasks}
    proposals: list[ProposedUpdate] = []
    for d in decisions:
        if d.route == RouteKind.clear and d.proposal is not None:
            proposals.append(d.proposal)
            continue
        c, m = candidates[d.candidate_index], matches[d.candidate_index]
        only_task_reasons = set(d.reasons) <= {"task_unclear", "task_ambiguous"}
        if c.action == ActionType.create or not only_task_reasons or not m.ranked or m.ranked[0].score <= 0:
            continue
        top = by_id[m.ranked[0].task_id]
        fields = {"assign": UpdateFields(owner=team.canonical_member(c.owner)),
                  "set_due_date": UpdateFields(due_date=m.resolved_due_date)}.get(c.action.value, UpdateFields())
        try:
            proposals.append(ProposedUpdate(
                project_id=meeting.project_id, meeting_no=meeting.meeting_no, candidate_index=d.candidate_index,
                action=c.action, task_id=top.task_id, task_version=top.version, task_description=top.description,
                fields=fields,
                evidence_refs=[EvidenceRef(source=EvidenceSource.transcript, id=i) for i in dict.fromkeys(q.utterance_id for q in c.evidence)],
                evidence_quotes=list(c.evidence), confidence=c.confidence, origin=ProposalOrigin.clear_path))
        except ValueError:
            continue
    return MeetingPreview(project_id=meeting.project_id, meeting_no=meeting.meeting_no, meeting_date=meeting.meeting_date,
                          candidates=candidates, matches=matches, decisions=decisions, proposals=proposals)
