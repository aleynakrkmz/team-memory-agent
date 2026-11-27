"""Baseline B1: fixed rules with BM25 matching, no LLM (design §2.2, §6.5).

`extract_b1` reads a transcript line by line and turns explicit statements
into the same `CandidateChange` objects that LLM extraction returns:

    complete      "The login page is done"
    cancel        "we're dropping the project report draft"
    set_due_date  "The login page needs a few more days, can I have until the 10th?"
    create        "Ayşe, can you do the login page by the 6th?" + "Yes ..."  (owner Ayşe)
                  "Can, the database schema is yours, please finish it by the 4th."
                  "I'll write the project report draft myself"            (owner = speaker)
                  "someone has to run load tests before the release"      (no owner)
    assign        a request to a named person about a task that is already open
                  under someone else, or "X, could you take over ..."

A request phrased as a question needs a yes from the person asked on the next
line. Questions without an agreement, offers of help, pronouns ("I'll share
them") and lines addressed to an assistant give nothing. Each candidate cites
its whole lines as evidence, so every quote is verbatim.

The pipeline then runs the usual match and route. B1 has no agent: hard
cases get no proposal. The rules are fixed before test data are generated;
they were written on the development meetings only (samples/demo, demo2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from datagen.common import find_dates

from app.match.match import match_candidates, tokenize
from app.schemas import (
    ActionType,
    CandidateChange,
    EvidenceQuote,
    MeetingTranscript,
    TaskRecord,
    TaskStatus,
    TeamProfile,
    Utterance,
)

CONFIDENCE = 0.9
# Same thresholds as route (config/routing.json defaults); used only to decide create vs assign.
TAU = 0.5
MARGIN = 1.5

_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_DAY = r"(?:the\s+\d{1,2}(?:st|nd|rd|th))"
_DUE = re.compile(
    rf"\b(?:(?:by|until|before|on)\s+(?:{_DAY}|(?:next\s+)?{_WEEKDAY}|tomorrow|the\s+end\s+of\s+(?:the\s+)?month)"
    rf"|the\s+end\s+of\s+(?:the\s+)?month|in\s+(?:two|2)\s+weeks)\b",
    re.IGNORECASE,
)
_NO_DEADLINE = re.compile(r"\bno deadline\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.?!]+[.?!]?")
_YES = re.compile(r"^(?:yes|yeah|yep|sure|ok|okay|fine|of course|no problem|will do|i will|i'll|i can)\b", re.IGNORECASE)
_ASSISTANT = re.compile(r"^(?:hey\s+|hi\s+|ok\s+|okay\s+)?(?:assistant|ai|bot|agent|system|computer)\s*[,:!]", re.IGNORECASE)
_PRONOUNS = {"it", "them", "that", "this", "those", "these", "one"}
_ARTICLE = re.compile(r"^(?:the|a|an|our|my|your)\s+", re.IGNORECASE)
_TRAILING = re.compile(r"\s+(?:myself|yourself|first|too|as well|for now)$", re.IGNORECASE)

_TASK = r"(?P<task>(?:the|our|a|an)\s+[^,.;?!]+?)"
_COMPLETE = [
    re.compile(rf"^{_TASK}\s+(?:is|are|was|were)\s+(?:done|finished|complete|completed|ready)\b", re.IGNORECASE),
    re.compile(rf"^I(?:'ve| have)?\s+(?:finished|completed)\s+{_TASK}(?=[,.;!]|$)", re.IGNORECASE),
]
_CANCEL = [
    re.compile(rf"\b(?:dropping|drop|cancell?ing|cancel|scrapping|scrap)\s+{_TASK}(?=[,.;!]|$)", re.IGNORECASE),
]
_MORE_TIME = re.compile(rf"^{_TASK}\s+needs?\s+(?:a\s+few\s+|a\s+bit\s+|some\s+)?more\s+(?:days|time)\b", re.IGNORECASE)
_MOVE = re.compile(rf"\b(?:move|push|postpone)\s+{_TASK}\s+to\b", re.IGNORECASE)
_VERBS = r"(?:do|build|take|write|prepare|handle|own|finish|update|run|set\s+up|create|make|fix|start|review|test|draft)"
_REQUEST_Q = re.compile(rf"^(?:can|could|would|will)\s+you\s+(?:please\s+)?(?P<verb>take\s+over|{_VERBS})\s+(?P<task>[^?]+)", re.IGNORECASE)
_REQUEST_DO = re.compile(rf"^please\s+(?P<verb>take\s+over|{_VERBS})\s+(?P<task>[^?]+)", re.IGNORECASE)
_YOU_WILL = re.compile(rf"^you(?:'ll|\s+will)\s+(?P<verb>{_VERBS})\s+(?P<task>.+)", re.IGNORECASE)
_YOURS = re.compile(rf"^{_TASK}\s+is\s+yours\b", re.IGNORECASE)
_SELF = re.compile(rf"^(?:\w+,\s+)?I(?:'ll|\s+will)\s+(?P<verb>{_VERBS})\s+(?P<task>.+)", re.IGNORECASE)
_NEED = re.compile(
    r"\b(?:we\s+(?:also\s+)?need|someone\s+(?:has|needs)\s+to|somebody\s+(?:has|needs)\s+to)\s+(?P<task>[^.?!]+)",
    re.IGNORECASE,
)


@dataclass
class _Draft:
    action: ActionType
    task: str
    owner: str | None
    due: str | None
    lines: list[str] = field(default_factory=list)
    take_over: bool = False


def extract_b1(meeting: MeetingTranscript, tasks: list[TaskRecord], team: TeamProfile) -> list[CandidateChange]:
    """Rule-based candidates for one meeting. Deterministic; no model call."""
    names = _names(team)
    lines = meeting.utterances
    drafts: list[_Draft] = []
    for position, line in enumerate(lines):
        speaker = team.canonical_member(line.speaker) or line.speaker
        following = lines[position + 1] if position + 1 < len(lines) else None
        for sentence in _sentences(line.text):
            if _ASSISTANT.match(sentence):
                continue
            for draft in _rules(sentence, line, speaker, following, names, team):
                _add(drafts, draft, position, lines)
    candidates = [_candidate(draft, meeting) for draft in drafts]
    return _classify(candidates, drafts, tasks, meeting.meeting_date)


# ----- rules --------------------------------------------------------------


def _rules(
    sentence: str,
    line: Utterance,
    speaker: str,
    following: Utterance | None,
    names: dict[str, str],
    team: TeamProfile,
) -> list[_Draft]:
    found: list[_Draft] = []
    due = _due(sentence)
    for pattern in _COMPLETE:
        if (m := pattern.search(sentence)) and (task := _task(m["task"])):
            return [_Draft(ActionType.complete, task, None, None, [line.id])]
    for pattern in _CANCEL:
        if (m := pattern.search(sentence)) and (task := _task(m["task"])):
            return [_Draft(ActionType.cancel, task, None, None, [line.id])]
    if (m := _MORE_TIME.search(sentence) or _MOVE.search(sentence)) and due and (task := _task(m["task"])):
        return [_Draft(ActionType.set_due_date, task, None, due, [line.id])]

    addressee, rest = _addressed(sentence, names)
    if addressee is not None:
        if (m := _REQUEST_Q.search(rest)) and (task := _task(m["task"], due)):
            if following is not None and _agrees(following, addressee, team):
                found.append(_Draft(ActionType.create, task, addressee, due, [line.id, following.id], _take_over(m)))
            return found
        for pattern in (_REQUEST_DO, _YOU_WILL):
            if (m := pattern.search(rest)) and (task := _task(m["task"], due)):
                lines = [line.id]
                if following is not None and _agrees(following, addressee, team):
                    lines.append(following.id)
                return [_Draft(ActionType.create, task, addressee, None if _NO_DEADLINE.search(rest) else due, lines, _take_over(m))]
        if (m := _YOURS.search(rest)) and (task := _task(m["task"])):
            return [_Draft(ActionType.create, task, addressee, due, [line.id])]

    if not sentence.rstrip().endswith("?"):
        if (m := _SELF.search(sentence)) and (task := _task(m["task"], due)):
            return [_Draft(ActionType.create, task, speaker, None if _NO_DEADLINE.search(sentence) else due, [line.id])]
        if (m := _NEED.search(sentence)) and (task := _task(m["task"], due)):
            return [_Draft(ActionType.create, task, None, due, [line.id])]
    return found


def _take_over(match: re.Match) -> bool:
    return bool(re.match(r"take\s+over", match["verb"], re.IGNORECASE))


def _agrees(line: Utterance, person: str, team: TeamProfile) -> bool:
    return team.canonical_member(line.speaker) == person and bool(_YES.match(line.text.strip()))


def _addressed(sentence: str, names: dict[str, str]) -> tuple[str | None, str]:
    """`Name, rest` -> (canonical name, rest). Also after a leading `Okay,` / `New item:`."""
    text = sentence.strip()
    for _ in range(2):
        head, sep, rest = text.partition(",")
        if sep and head.strip().casefold() in names:
            return names[head.strip().casefold()], rest.strip()
        if sep and head.strip().casefold() in {"okay", "ok", "so", "and", "also"}:
            text = rest.strip()
            continue
        break
    return None, text


def _due(text: str) -> str | None:
    if _NO_DEADLINE.search(text):
        return None
    match = _DUE.search(text)
    if match is None:
        return None
    phrase = match.group(0)
    return phrase if find_dates(phrase, date(2026, 1, 1)) else None


def _task(raw: str, due: str | None = None) -> str | None:
    """Clean a task phrase. None for pronouns and empty phrases."""
    text = raw
    if due:
        text = text.split(due)[0]
    text = re.split(r"[,;:?!]|\s+no deadline\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = _DUE.split(text)[0] if _DUE.search(text) else text
    text = _TRAILING.sub("", text.strip().rstrip("."))
    text = _ARTICLE.sub("", text).strip()
    words = [w for w in tokenize(text) if w not in _PRONOUNS]
    if not words or text.casefold() in _PRONOUNS:
        return None
    return text


# ----- merging and classification -----------------------------------------


def _add(drafts: list[_Draft], new: _Draft, position: int, lines: list[Utterance]) -> None:
    """Merge with a create about the same task from the last two lines; else append."""
    recent = {line.id for line in lines[max(0, position - 2) : position + 1]}
    for old in reversed(drafts):
        if old.action != new.action or old.action != ActionType.create:
            continue
        if not (set(old.lines) & recent):
            continue
        if _same_task(old.task, new.task):
            old.owner = old.owner or new.owner
            old.due = old.due or new.due
            old.take_over = old.take_over or new.take_over
            old.lines.extend(i for i in new.lines if i not in old.lines)
            return
    for old in drafts:
        if old.action == new.action and _same_task(old.task, new.task) and set(old.lines) & set(new.lines):
            return
    drafts.append(new)


def _same_task(a: str, b: str) -> bool:
    left, right = set(tokenize(a)), set(tokenize(b))
    if not left or not right:
        return False
    return len(left & right) / min(len(left), len(right)) >= 0.5


def _candidate(draft: _Draft, meeting: MeetingTranscript) -> CandidateChange:
    evidence = []
    for line_id in draft.lines:
        line = meeting.utterance(line_id)
        if line is not None:
            evidence.append(EvidenceQuote(utterance_id=line.id, quote=line.text))
    return CandidateChange(
        action=draft.action,
        description=draft.task,
        owner=draft.owner,
        due_phrase=draft.due,
        evidence=evidence,
        confidence=CONFIDENCE,
    )


def _classify(
    candidates: list[CandidateChange],
    drafts: list[_Draft],
    tasks: list[TaskRecord],
    meeting_date: date,
) -> list[CandidateChange]:
    """A request about a task that is already open is an assign or a date change, not a create."""
    if not candidates:
        return []
    by_id = {task.task_id: task for task in tasks if task.status == TaskStatus.open}
    matches = match_candidates(candidates, tasks, meeting_date)
    result: list[CandidateChange] = []
    for candidate, draft, match in zip(candidates, drafts, matches, strict=True):
        if candidate.action != ActionType.create:
            result.append(candidate)
            continue
        existing = _open_task(candidate.description, match, by_id)
        if existing is None:
            if draft.take_over:
                # "take over that thing": an assign whose task route cannot pin down.
                result.append(candidate.model_copy(update={"action": ActionType.assign, "due_phrase": None}))
            else:
                result.append(candidate)
            continue
        if candidate.owner and candidate.owner != existing.owner:
            result.append(candidate.model_copy(update={"action": ActionType.assign, "due_phrase": None}))
        elif candidate.due_phrase and match.resolved_due_date and match.resolved_due_date != existing.due_date:
            result.append(candidate.model_copy(update={"action": ActionType.set_due_date, "owner": None}))
        # Otherwise the open task already says this: no change.
    return result


def _open_task(description: str, match, by_id: dict[str, TaskRecord]) -> TaskRecord | None:
    """The one open task this phrase names: a unique BM25 match, else a unique word overlap.

    BM25 gives no usable score on a list of one or two tasks (every word is in
    half the documents), so the overlap check covers small lists.
    """
    top = match.ranked[0] if match.ranked else None
    second = match.ranked[1].score if len(match.ranked) > 1 else 0.0
    if top and top.score >= TAU and top.score >= MARGIN * second:
        return by_id.get(top.task_id)
    overlapping = [task for task in by_id.values() if _same_task(task.description, description)]
    return overlapping[0] if len(overlapping) == 1 else None


def _names(team: TeamProfile) -> dict[str, str]:
    names: dict[str, str] = {}
    for member in team.members:
        names[member.name.casefold()] = member.name
        for alias in member.aliases:
            names[alias.casefold()] = member.name
    return names


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]
