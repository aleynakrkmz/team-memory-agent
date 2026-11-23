"""SQLite tables for one project's memory. Design §3.4 and §5.

Every table is append-only. Triggers refuse UPDATE and DELETE on all of
them, so a new task version is a new row and old rows stay. Connections
also turn on `recursive_triggers`, because SQLite skips DELETE triggers for
`INSERT OR REPLACE` unless that pragma is on.

Who writes which table (design §3.4):
- meetings, utterances: ingest
- team_profiles: data generator, WS3 lead, or the coordinator
- cases, proposals, clarification_questions, run_records: the controller
- clarification_answers, reviews: the review list
- task_versions: commit only, after approval (`source = 'commit'`), or the
  reset-to-gold setup step (`source = 'seed'`)
"""

SCHEMA_VERSION = 1

TABLES = (
    "project",
    "team_profiles",
    "meetings",
    "utterances",
    "cases",
    "proposals",
    "clarification_questions",
    "clarification_answers",
    "reviews",
    "task_versions",
    "run_records",
)

_CREATE_TABLES = """
CREATE TABLE project (
    project_id      TEXT PRIMARY KEY CHECK (length(project_id) > 0),
    schema_version  INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE team_profiles (
    profile_no      INTEGER PRIMARY KEY AUTOINCREMENT,
    coordinator     TEXT NOT NULL,
    members_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE meetings (
    meeting_no      INTEGER PRIMARY KEY CHECK (meeting_no >= 1),
    meeting_date    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE utterances (
    meeting_no      INTEGER NOT NULL REFERENCES meetings(meeting_no),
    utterance_id    TEXT NOT NULL CHECK (utterance_id GLOB 'L[1-9]*'),
    position        INTEGER NOT NULL,
    speaker         TEXT NOT NULL,
    text            TEXT NOT NULL,
    start_ms        INTEGER CHECK (start_ms IS NULL OR start_ms >= 0),
    PRIMARY KEY (meeting_no, utterance_id)
);

CREATE TABLE cases (
    case_id         TEXT PRIMARY KEY,
    meeting_no      INTEGER NOT NULL REFERENCES meetings(meeting_no),
    candidate_index INTEGER NOT NULL CHECK (candidate_index >= 0),
    route           TEXT NOT NULL CHECK (route IN ('clear', 'hard')),
    reasons_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (meeting_no, candidate_index)
);

CREATE TABLE proposals (
    proposal_no     INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id     TEXT NOT NULL UNIQUE,
    case_id         TEXT REFERENCES cases(case_id),
    meeting_no      INTEGER NOT NULL REFERENCES meetings(meeting_no),
    candidate_index INTEGER NOT NULL,
    action          TEXT NOT NULL,
    task_id         TEXT,
    task_version    INTEGER,
    origin          TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE clarification_questions (
    question_no     INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id     TEXT NOT NULL UNIQUE,
    case_id         TEXT NOT NULL REFERENCES cases(case_id),
    question        TEXT NOT NULL CHECK (length(question) > 0),
    options_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE clarification_answers (
    answer_no       INTEGER PRIMARY KEY AUTOINCREMENT,
    clarification_id TEXT NOT NULL UNIQUE,
    question_id     TEXT NOT NULL UNIQUE REFERENCES clarification_questions(question_id),
    answered_by     TEXT NOT NULL CHECK (length(answered_by) > 0),
    answer          TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE reviews (
    review_no       INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id       TEXT NOT NULL UNIQUE,
    proposal_id     TEXT NOT NULL UNIQUE REFERENCES proposals(proposal_id),
    decision        TEXT NOT NULL CHECK (decision IN ('approve', 'reject')),
    reviewer        TEXT NOT NULL CHECK (length(reviewer) > 0),
    note            TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE task_versions (
    task_no         INTEGER NOT NULL CHECK (task_no >= 1),
    task_id         TEXT NOT NULL,
    version         INTEGER NOT NULL CHECK (version >= 1),
    description     TEXT NOT NULL CHECK (length(description) > 0),
    owner           TEXT,
    due_date        TEXT,
    status          TEXT NOT NULL CHECK (status IN ('open', 'done', 'cancelled')),
    source          TEXT NOT NULL CHECK (source IN ('commit', 'seed')),
    approval_id     TEXT UNIQUE REFERENCES reviews(review_id),
    meeting_no      INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (task_id, version),
    CHECK ((source = 'commit' AND approval_id IS NOT NULL)
        OR (source = 'seed' AND approval_id IS NULL))
);

CREATE TABLE run_records (
    record_no       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    meeting_no      INTEGER NOT NULL,
    stop_reason     TEXT NOT NULL,
    failure_detail  TEXT,
    metrics_json    TEXT NOT NULL,
    trace_json      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- A commit version must point at an approval, not at a rejection.
CREATE TRIGGER task_versions_need_approval
BEFORE INSERT ON task_versions
WHEN NEW.source = 'commit'
 AND NOT EXISTS (SELECT 1 FROM reviews WHERE review_id = NEW.approval_id AND decision = 'approve')
BEGIN
    SELECT RAISE(ABORT, 'a commit version needs an approve review');
END;
"""


def _append_only_triggers() -> str:
    parts = []
    for table in TABLES:
        parts.append(
            f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, 'append-only: {table} rows cannot be updated'); END;\n"
            f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, 'append-only: {table} rows cannot be deleted'); END;\n"
        )
    return "".join(parts)


SCHEMA_SQL = _CREATE_TABLES + _append_only_triggers()
