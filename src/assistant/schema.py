"""SQLite schema + initialisation.

Design:
  - One `db.sqlite3` file, created on first run, seeded from the bundled CBSE
    JSON and (optionally) the EdNet sample.
  - All domain state (students, topics, attempts, materials, tests,
    conversation sessions) lives in the DB.
  - Attempts are the grain for student modelling: the `performance_history`
    table holds one row per answered question (EdNet style). Static "weak /
    strong" labels live in a separate `student_topic_label` table so they
    degrade gracefully when no attempt history exists.
  - Single-file schema init via `init_db()` — no Alembic; the schema is small
    enough that versioned migrations are overkill for this stage.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    id                        TEXT PRIMARY KEY,
    name                      TEXT NOT NULL,
    grade                     INTEGER,
    board                     TEXT,
    target_exam               TEXT,
    daily_study_time_minutes  INTEGER NOT NULL DEFAULT 60,
    source                    TEXT NOT NULL DEFAULT 'cbse'  -- 'cbse' | 'ednet' | etc.
);

CREATE TABLE IF NOT EXISTS topics (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL UNIQUE,
    subject  TEXT
);

-- Declared labels from the student's profile (if any). These are the
-- fallback when we don't have enough attempt data to model weakness.
CREATE TABLE IF NOT EXISTS student_topic_label (
    student_id  TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    topic_id    INTEGER NOT NULL REFERENCES topics(id),
    label       TEXT NOT NULL CHECK(label IN ('weak','strong')),
    PRIMARY KEY (student_id, topic_id)
);

-- One row per answered question. EdNet's grain. Used for student modelling.
CREATE TABLE IF NOT EXISTS attempts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    topic_id      INTEGER NOT NULL REFERENCES topics(id),
    question_id   TEXT,                     -- external id, e.g. EdNet 'q7333'
    correct       INTEGER NOT NULL,         -- 0 | 1
    elapsed_ms    INTEGER,
    answered_at   TEXT NOT NULL             -- ISO8601
);

CREATE INDEX IF NOT EXISTS idx_attempts_student_topic_time
    ON attempts(student_id, topic_id, answered_at DESC);

CREATE TABLE IF NOT EXISTS materials (
    id                 TEXT PRIMARY KEY,       -- e.g. M101
    topic_id           INTEGER NOT NULL REFERENCES topics(id),
    title              TEXT NOT NULL,
    content_type       TEXT,                   -- notes | video | practice
    difficulty         TEXT,                   -- beginner | intermediate | advanced
    estimated_minutes  INTEGER,
    description        TEXT
);

CREATE TABLE IF NOT EXISTS tests (
    id             TEXT PRIMARY KEY,           -- e.g. T201
    student_id     TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    subject        TEXT,
    name           TEXT NOT NULL,
    scheduled_for  TEXT NOT NULL               -- ISO date
);

CREATE TABLE IF NOT EXISTS test_topics (
    test_id   TEXT NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
    topic_id  INTEGER NOT NULL REFERENCES topics(id),
    PRIMARY KEY (test_id, topic_id)
);

-- Conversation persistence (scaffolding for multi-turn memory). Not used
-- for retrieval yet; we persist messages so a future "last week we worked on
-- Algebra" recall path has data to read from.
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    student_id  TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    started_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,                 -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- Prerequisite graph. A row (topic_id, prereq_topic_id) means the prereq
-- should be fluent before attempting `topic_id`. Used by plan_study_week
-- to refuse to recommend a topic when its foundations are still weak.
CREATE TABLE IF NOT EXISTS topic_prerequisites (
    topic_id          INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    prereq_topic_id   INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    rationale         TEXT,
    PRIMARY KEY (topic_id, prereq_topic_id),
    CHECK (topic_id != prereq_topic_id)
);

CREATE INDEX IF NOT EXISTS idx_prereq_topic ON topic_prerequisites(topic_id);
"""


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "db.sqlite3"


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create the DB file (if missing) and apply the schema idempotently."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def is_empty(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 0
