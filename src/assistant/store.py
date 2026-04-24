"""Store — SQLite-backed replacement for the JSON DataStore.

The rest of the codebase (tools, agent) depends on a narrow interface:
`profile`, `performance` snapshot, `materials`, `tests`, plus a few helpers.
We keep that interface and back it with SQLite so we can hold many students,
persist conversations, and ingest real attempt data.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .data import Material
from .schema import connect, init_db, is_empty, DEFAULT_DB_PATH


# ---------------------------------------------------------------- Store

@dataclass
class Store:
    """Per-student read view backed by SQLite. One instance per student/session.

    Construction is cheap (a few indexed queries on an in-memory-cached SQLite
    connection) so the agent can build one at the start of every chat.
    """

    conn: sqlite3.Connection
    student_id: str

    # Cached read state (populated by .load())
    profile: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    materials: list[Material] = field(default_factory=list)
    tests: list[dict] = field(default_factory=list)

    @classmethod
    def open(cls, student_id: str, db_path: Path | str = DEFAULT_DB_PATH) -> "Store":
        conn = init_db(db_path)
        store = cls(conn=conn, student_id=student_id)
        store.load()
        return store

    # ------------------------------------------------------------ load

    def load(self) -> None:
        self.profile = self._load_profile()
        self.performance = self._load_performance()
        self.materials = self._load_materials()
        self.tests = self._load_tests()

    def _load_profile(self) -> dict:
        row = self.conn.execute(
            "SELECT * FROM students WHERE id = ?", (self.student_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown student_id: {self.student_id}")
        labels = self.conn.execute(
            """
            SELECT t.name, stl.label
            FROM student_topic_label stl
            JOIN topics t ON t.id = stl.topic_id
            WHERE stl.student_id = ?
            """,
            (self.student_id,),
        ).fetchall()
        weak = [r["name"] for r in labels if r["label"] == "weak"]
        strong = [r["name"] for r in labels if r["label"] == "strong"]
        return {
            "student_id": row["id"],
            "name": row["name"],
            "grade": row["grade"],
            "board": row["board"],
            "target_exam": row["target_exam"],
            "daily_study_time_minutes": row["daily_study_time_minutes"],
            "source": row["source"],
            "strong_topics": strong,
            "weak_topics": weak,
        }

    def _load_performance(self) -> dict:
        """Aggregate attempt rows into subject + topic performance snapshots.

        Snapshot shape stays the same as the JSON version so existing tools
        keep working; the *numbers* now come from the attempts table.
        """
        rows = self.conn.execute(
            """
            SELECT t.name AS topic, t.subject,
                   AVG(a.correct) * 100.0 AS score_pct,
                   COUNT(*)               AS n,
                   MAX(a.answered_at)     AS last_assessed
            FROM attempts a
            JOIN topics t ON t.id = a.topic_id
            WHERE a.student_id = ?
            GROUP BY t.id
            """,
            (self.student_id,),
        ).fetchall()

        topic_performance = [
            {
                "topic": r["topic"],
                "subject": r["subject"],
                "score_percentage": round(r["score_pct"], 1),
                "attempts": r["n"],
                "last_assessed": (r["last_assessed"] or "")[:10],
            }
            for r in rows
        ]

        subj: dict[str, list[float]] = {}
        for r in rows:
            subj.setdefault(r["subject"] or "General", []).append(r["score_pct"])
        subject_performance = [
            {
                "subject": name,
                "overall_score_percentage": round(sum(vals) / len(vals), 1),
            }
            for name, vals in subj.items()
        ]

        return {
            "student_id": self.student_id,
            "subject_performance": subject_performance,
            "topic_performance": topic_performance,
        }

    def _load_materials(self) -> list[Material]:
        rows = self.conn.execute(
            """
            SELECT m.id AS material_id, t.name AS topic, m.title, m.content_type,
                   m.difficulty, m.estimated_minutes, m.description
            FROM materials m
            JOIN topics t ON t.id = m.topic_id
            ORDER BY m.id
            """
        ).fetchall()
        return [Material(**dict(r)) for r in rows]

    def _load_tests(self) -> list[dict]:
        tests = self.conn.execute(
            """
            SELECT id AS test_id, subject, name AS test_name, scheduled_for AS date
            FROM tests
            WHERE student_id = ?
            ORDER BY scheduled_for
            """,
            (self.student_id,),
        ).fetchall()
        result = []
        for t in tests:
            topics = self.conn.execute(
                """
                SELECT topics.name FROM test_topics
                JOIN topics ON topics.id = test_topics.topic_id
                WHERE test_topics.test_id = ?
                """,
                (t["test_id"],),
            ).fetchall()
            result.append({
                "test_id": t["test_id"],
                "subject": t["subject"],
                "test_name": t["test_name"],
                "date": t["date"],
                "topics": [r["name"] for r in topics],
            })
        return result

    # ------------------------------------------------------- convenience

    def topic_score(self, topic: str) -> float | None:
        for row in self.performance.get("topic_performance", []):
            if row["topic"] == topic:
                return float(row["score_percentage"])
        return None

    def subject_score(self, subject: str) -> float | None:
        for row in self.performance.get("subject_performance", []):
            if row["subject"] == subject:
                return float(row["overall_score_percentage"])
        return None

    # ------------------------------------------------------- sessions

    def start_session(self, session_id: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions(id, student_id, started_at) VALUES(?,?,?)",
            (session_id, self.student_id, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def log_message(self, session_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (session_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    # ------------------------------------------------------- listing

    @staticmethod
    def list_students(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
        conn = init_db(db_path)
        rows = conn.execute(
            "SELECT id, name, grade, board, source FROM students ORDER BY source, id"
        ).fetchall()
        return [dict(r) for r in rows]


def ensure_seeded(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the DB and seed it from bundled JSON on first run. Idempotent."""
    conn = init_db(db_path)
    if is_empty(conn):
        from .seed import seed_cbse, seed_ednet_sample, seed_extra_roster, seed_prereqs
        seed_cbse(conn)          # Arjun (grade 10 CBSE, from JSON)
        seed_ednet_sample(conn)  # Priya / Kenji / Mei (TOEIC prep, EdNet-schema)
        seed_extra_roster(conn)  # Ananya / Vikram / Rhea / Aarav / Neha
        seed_prereqs(conn)       # topic prerequisite graph
    return conn
