"""BM25 match of a candidate title against open tasks, and deadline resolution.

Design §3.1: rank open tasks with BM25, and turn phrases such as "by the
18th" into a calendar date from the meeting date. The date logic is
`datagen.common.find_dates`, the same resolver the answer-key check uses.

Task titles are only a few words, so length normalization is off (`b=0`).
Otherwise a shorter title wins when both titles share the same words.
A word that appears in half of the open tasks has an IDF of zero in
rank-bm25, so it does not separate those tasks. "that API thing" then
scores 0 against both API tasks and route sends the case to the agent.
"""

import re
from datetime import date

from rank_bm25 import BM25Okapi

from datagen.common import find_dates

from app.schemas import CandidateChange, MatchResult, RankedTask, TaskRecord, TaskStatus

# Function words, plus "thing", which the demo uses only as a vague pointer.
# Content words (api, login, report, tests, ...) stay in the query.
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "before",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "last",
        "of",
        "on",
        "over",
        "that",
        "the",
        "thing",
        "this",
        "to",
        "week",
        "with",
    }
)
_WORD = re.compile(r"[a-z0-9çğıöşü]+")
_REMOVAL = re.compile(
    r"\b(?:no deadline|without a deadline|drop the deadline|remove the deadline|deadline (?:is )?removed)\b",
    re.IGNORECASE,
)

# rank-bm25 defaults, except b. See the module docstring.
BM25_K1 = 1.5
BM25_B = 0.0


def tokenize(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOP]


def match_candidates(
    candidates: list[CandidateChange],
    tasks: list[TaskRecord],
    meeting_date: date,
) -> list[MatchResult]:
    """Rank open tasks for every candidate. Closed tasks are not in the index."""
    open_tasks = [task for task in tasks if task.status == TaskStatus.open]
    tokenized = [tokenize(task.description) for task in open_tasks]
    index = _Index(tokenized)
    results: list[MatchResult] = []
    for index_no, candidate in enumerate(candidates):
        scores = index.scores(tokenize(candidate.description), len(open_tasks))
        order = sorted(range(len(open_tasks)), key=lambda i: (-scores[i], open_tasks[i].task_id))
        resolved, ambiguous, removed = _resolve_due(candidate, meeting_date)
        results.append(
            MatchResult(
                candidate_index=index_no,
                ranked=[RankedTask(task_id=open_tasks[i].task_id, score=scores[i]) for i in order],
                resolved_due_date=resolved,
                due_date_ambiguous=ambiguous,
                due_date_removed=removed,
            )
        )
    return results


class _Index:
    def __init__(self, tokenized: list[list[str]]):
        self._usable = bool(tokenized) and any(tokenized)
        self._bm25 = BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B) if self._usable else None

    def scores(self, query: list[str], n_docs: int) -> list[float]:
        if not self._usable or not query or self._bm25 is None:
            return [0.0] * n_docs
        return [float(score) for score in self._bm25.get_scores(query)]


def _resolve_due(candidate: CandidateChange, meeting_date: date) -> tuple[date | None, bool, bool]:
    """Return (calendar date, ambiguous, explicitly removed)."""
    phrase = candidate.due_phrase
    removed = bool(phrase and _REMOVAL.search(phrase))
    found = list(dict.fromkeys(find_dates(phrase, meeting_date))) if phrase else []
    if candidate.due_date is not None and found and candidate.due_date not in found:
        return None, True, False
    if len(found) > 1:
        return None, True, False
    if len(found) == 1 and removed:
        return None, True, False
    if len(found) == 1:
        return found[0], False, False
    if candidate.due_date is not None:
        return candidate.due_date, False, False
    if removed:
        return None, False, True
    return None, False, False
