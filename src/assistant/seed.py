"""Seeders for the SQLite DB.

Two sources, no intermediate storage:

  1. **CBSE JSON** (data/*.json) — one student (Arjun, grade 10). We synthesise
     ~15 attempts per declared weak/strong topic so the modelling path has
     real attempt data to aggregate over, not just static labels. Accuracies
     (~40% weak, ~85% strong) stay consistent with the declared scores.

  2. **EdNet-schema sample** — two TOEIC-prep students (Priya, Kenji) whose
     attempts are generated directly into SQLite; no CSV intermediary. The
     tag taxonomy and accuracy profiles match EdNet KT1's shape.

If you drop real EdNet KT1 CSVs at `data/ednet_raw/` (layout documented in
`load_ednet_raw` below), they're ingested instead of the synthetic sample.
Real EdNet files are the only place the CSV loader runs — our own sample
goes straight to INSERT.
"""

from __future__ import annotations

import csv
import json
import random
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
EDNET_RAW_DIR = DATA_DIR / "ednet_raw"


# ------------------------------------------------------------- helpers

def _upsert_topic(conn: sqlite3.Connection, name: str, subject: str | None) -> int:
    row = conn.execute("SELECT id FROM topics WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO topics(name, subject) VALUES(?, ?)", (name, subject)
    )
    return cur.lastrowid


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- CBSE

def seed_cbse(conn: sqlite3.Connection) -> None:
    profile = json.loads((DATA_DIR / "student_profile.json").read_text())
    performance = json.loads((DATA_DIR / "performance_history.json").read_text())
    materials = json.loads((DATA_DIR / "study_materials.json").read_text())["materials"]
    tests = json.loads((DATA_DIR / "upcoming_tests.json").read_text())["upcoming_tests"]

    # topic -> subject map derived from performance rows, materials, and tests.
    topic_subject: dict[str, str | None] = {
        r["topic"]: r.get("subject") for r in performance.get("topic_performance", [])
    }
    for m in materials:
        topic_subject.setdefault(m["topic"], None)
    for t in tests:
        for topic in t["topics"]:
            topic_subject.setdefault(topic, t.get("subject"))

    conn.execute(
        """
        INSERT INTO students(id, name, grade, board, target_exam,
            daily_study_time_minutes, source)
        VALUES(?,?,?,?,?,?,?)
        """,
        (profile["student_id"], profile["name"], profile["grade"],
         profile["board"], profile["target_exam"],
         profile["daily_study_time_minutes"], "cbse"),
    )

    topic_ids = {
        name: _upsert_topic(conn, name, subj) for name, subj in topic_subject.items()
    }

    for t in profile.get("weak_topics", []):
        tid = topic_ids.get(t) or _upsert_topic(conn, t, None)
        conn.execute(
            "INSERT OR IGNORE INTO student_topic_label(student_id, topic_id, label) VALUES(?,?,?)",
            (profile["student_id"], tid, "weak"),
        )
    for t in profile.get("strong_topics", []):
        tid = topic_ids.get(t) or _upsert_topic(conn, t, None)
        conn.execute(
            "INSERT OR IGNORE INTO student_topic_label(student_id, topic_id, label) VALUES(?,?,?)",
            (profile["student_id"], tid, "strong"),
        )

    for m in materials:
        tid = topic_ids[m["topic"]]
        conn.execute(
            """
            INSERT OR REPLACE INTO materials(id, topic_id, title, content_type,
                difficulty, estimated_minutes, description)
            VALUES(?,?,?,?,?,?,?)
            """,
            (m["material_id"], tid, m["title"], m["content_type"],
             m["difficulty"], m["estimated_minutes"], m["description"]),
        )

    for t in tests:
        conn.execute(
            "INSERT OR REPLACE INTO tests(id, student_id, subject, name, scheduled_for) VALUES(?,?,?,?,?)",
            (t["test_id"], profile["student_id"], t["subject"], t["test_name"], t["date"]),
        )
        for topic in t["topics"]:
            tid = topic_ids.get(topic) or _upsert_topic(conn, topic, t.get("subject"))
            conn.execute(
                "INSERT OR IGNORE INTO test_topics(test_id, topic_id) VALUES(?,?)",
                (t["test_id"], tid),
            )

    # Synthesise attempts so the modelling path has data to aggregate. Fixed
    # seed ⇒ reproducible across runs.
    rng = random.Random(42)
    now = datetime.now(timezone.utc)
    topic_targets = {
        r["topic"]: r.get("score_percentage", 55) / 100.0
        for r in performance.get("topic_performance", [])
    }
    all_topics = (
        set(profile.get("weak_topics", []))
        | set(profile.get("strong_topics", []))
        | set(topic_targets.keys())
    )
    for topic_name in all_topics:
        target = topic_targets.get(
            topic_name,
            0.42 if topic_name in profile.get("weak_topics", []) else 0.85,
        )
        tid = topic_ids[topic_name]
        for i in range(15):
            correct = 1 if rng.random() < target else 0
            elapsed = rng.randint(15_000, 60_000)
            when = now - timedelta(days=rng.randint(1, 30), minutes=rng.randint(0, 600))
            conn.execute(
                """
                INSERT INTO attempts(student_id, topic_id, question_id,
                    correct, elapsed_ms, answered_at)
                VALUES(?,?,?,?,?,?)
                """,
                (profile["student_id"], tid, f"cbse-{topic_name[:4]}-{i}".lower(),
                 correct, elapsed, _iso(when)),
            )

    conn.commit()


# ------------------------------------------------------------- EdNet

def seed_ednet_sample(conn: sqlite3.Connection) -> None:
    """Load an EdNet-style sample. Reads real CSVs from `data/ednet_raw/` if
    present, otherwise generates a synthetic sample directly into SQLite.
    """
    if EDNET_RAW_DIR.exists() and any(EDNET_RAW_DIR.rglob("*.csv")):
        load_ednet_raw(conn, EDNET_RAW_DIR)
    else:
        _seed_ednet_synthetic(conn)


def _seed_ednet_synthetic(conn: sqlite3.Connection) -> None:
    """Generate two TOEIC-prep students + attempts + tests + materials, all
    inserted directly. No CSV intermediary.
    """
    # Tag taxonomy: (name, subject). Tag ids are only relevant for the real-
    # EdNet path; here we map tags → topics by index.
    tags = [
        ("Grammar - Subject-Verb Agreement",   "TOEIC Grammar"),
        ("Grammar - Tense Consistency",         "TOEIC Grammar"),
        ("Grammar - Conditionals",              "TOEIC Grammar"),
        ("Vocabulary - Business Idioms",        "TOEIC Vocabulary"),
        ("Vocabulary - Academic Words",         "TOEIC Vocabulary"),
        ("Reading - Inference Questions",       "TOEIC Reading"),
        ("Reading - Detail Questions",          "TOEIC Reading"),
        ("Listening - Part 3 Conversations",    "TOEIC Listening"),
    ]
    topic_ids = [_upsert_topic(conn, name, subject) for name, subject in tags]

    # Synthetic "questions" that carry topic ids. Each question hits 1-2 topics.
    qrng = random.Random(7)
    questions: list[tuple[str, list[int]]] = []
    for qi in range(1, 81):
        n_tags = qrng.choice([1, 1, 1, 2])
        tag_idxs = qrng.sample(range(len(tags)), n_tags)
        questions.append((f"q{qi:04d}", [topic_ids[i] for i in tag_idxs]))

    # Students.
    students = [
        {
            "id": "EN-u001", "name": "Priya", "target_exam": "TOEIC",
            "daily_minutes": 60,
            "strong_idxs": {6, 7},     # Reading-Detail, Listening
            "weak_idxs":   {2, 3},     # Conditionals, Business Idioms
            "test_id": "EN-T1", "test_name": "Reading Mock",
            "test_days_away": 6,
            "test_topic_idxs": [2, 3],
        },
        {
            "id": "EN-u002", "name": "Kenji", "target_exam": "TOEIC",
            "daily_minutes": 45,
            "strong_idxs": {0, 4},     # Subject-Verb, Academic Words
            "weak_idxs":   {5, 1},     # Inference, Tense
            "test_id": "EN-T2", "test_name": "Full TOEIC Mock",
            "test_days_away": 12,
            "test_topic_idxs": [5, 1],
        },
        {
            "id": "EN-u003", "name": "Mei", "target_exam": "TOEIC",
            "daily_minutes": 75,
            "strong_idxs": {0, 1, 4, 6}, # Grammar fundamentals + Reading-Detail
            "weak_idxs":   {7},          # Listening Part 3
            "test_id": "EN-T3", "test_name": "Listening Mock",
            "test_days_away": 9,
            "test_topic_idxs": [7],
        },
    ]

    for s in students:
        conn.execute(
            """
            INSERT OR REPLACE INTO students(id, name, grade, board, target_exam,
                daily_study_time_minutes, source)
            VALUES(?,?,NULL,NULL,?,?,?)
            """,
            (s["id"], s["name"], s["target_exam"], s["daily_minutes"], "ednet"),
        )

        # Attempts: 60 per student, drawn from the question pool. Accuracy
        # per question depends on whether its topics hit the strong or weak set.
        arng = random.Random(100 if s["id"] == "EN-u001" else 200)
        strong_topic_ids = {topic_ids[i] for i in s["strong_idxs"]}
        weak_topic_ids = {topic_ids[i] for i in s["weak_idxs"]}
        attempted = arng.sample(questions, 60)
        now = datetime.now(timezone.utc)
        for order_idx, (qid, q_topic_ids) in enumerate(attempted):
            if set(q_topic_ids) & strong_topic_ids:
                p_correct = 0.88
            elif set(q_topic_ids) & weak_topic_ids:
                p_correct = 0.34
            else:
                p_correct = 0.62
            correct = 1 if arng.random() < p_correct else 0
            elapsed = arng.randint(10_000, 45_000)
            when = now - timedelta(
                days=30 - order_idx // 2,
                seconds=arng.randint(0, 3600),
            )
            # A question with multiple tags records one attempt per tag so the
            # per-topic aggregate stays a clean avg(correct).
            for tid in q_topic_ids:
                conn.execute(
                    """
                    INSERT INTO attempts(student_id, topic_id, question_id,
                        correct, elapsed_ms, answered_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (s["id"], tid, qid, correct, elapsed, _iso(when)),
                )

        # Upcoming test.
        test_date = (date.today() + timedelta(days=s["test_days_away"])).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO tests(id, student_id, subject, name, scheduled_for) VALUES(?,?,?,?,?)",
            (s["test_id"], s["id"], "TOEIC Mock", s["test_name"], test_date),
        )
        for i in s["test_topic_idxs"]:
            conn.execute(
                "INSERT OR IGNORE INTO test_topics(test_id, topic_id) VALUES(?,?)",
                (s["test_id"], topic_ids[i]),
            )

    # Materials: one practice-pack per tag.
    content_types = ["notes", "video", "practice"]
    difficulties = ["beginner", "intermediate", "advanced"]
    for i, (name, _subject) in enumerate(tags, start=200):
        ct = content_types[i % 3]
        diff = difficulties[i % 3]
        conn.execute(
            """
            INSERT OR REPLACE INTO materials(id, topic_id, title, content_type,
                difficulty, estimated_minutes, description)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                f"M{i}",
                topic_ids[i - 200],
                f"{name.split(' - ', 1)[-1]} — {ct.capitalize()} Pack",
                ct,
                diff,
                15 + 5 * (i % 4),
                f"Targeted {ct} pack for mastering {name.lower()}.",
            ),
        )

    conn.commit()


def load_ednet_raw(conn: sqlite3.Connection, ednet_dir: Path) -> None:
    """Load EdNet KT1-format CSVs from ednet_dir.

    Expected layout (matches the public EdNet KT1 release, plus a couple of
    bookkeeping files we need that aren't in EdNet itself):

        <dir>/contents/questions.csv        columns: question_id, part, tags
        <dir>/contents/tags.csv             columns: tag_id, name, subject
        <dir>/users/<student_id>.csv        columns: timestamp, question_id,
                                                     correct, elapsed_time
        <dir>/students.csv                  columns: student_id, name, grade,
                                                     board, target_exam, daily_minutes
        <dir>/tests.csv     (optional)      columns: test_id, student_id, subject,
                                                     name, date, topics (';'-separated)
        <dir>/materials.csv (optional)      columns: material_id, topic, title,
                                                     content_type, difficulty,
                                                     estimated_minutes, description
    """
    contents = ednet_dir / "contents"
    tags = {
        r["tag_id"]: (r["name"], r.get("subject") or "TOEIC")
        for r in csv.DictReader((contents / "tags.csv").open())
    }
    topic_ids: dict[str, int] = {}
    for name, subject in tags.values():
        topic_ids[name] = _upsert_topic(conn, name, subject)

    question_topic_ids: dict[str, list[int]] = {}
    for row in csv.DictReader((contents / "questions.csv").open()):
        qid = row["question_id"]
        tag_ids = [t.strip() for t in (row.get("tags") or "").split(";") if t.strip()]
        question_topic_ids[qid] = [topic_ids[tags[t][0]] for t in tag_ids if t in tags]

    for row in csv.DictReader((ednet_dir / "students.csv").open()):
        conn.execute(
            """
            INSERT OR REPLACE INTO students(id, name, grade, board, target_exam,
                daily_study_time_minutes, source)
            VALUES(?,?,?,?,?,?,?)
            """,
            (row["student_id"], row["name"], int(row.get("grade") or 0) or None,
             row.get("board") or None, row.get("target_exam") or None,
             int(row.get("daily_minutes") or 60), "ednet"),
        )

    for user_file in sorted((ednet_dir / "users").glob("*.csv")):
        student_id = user_file.stem
        for row in csv.DictReader(user_file.open()):
            qid = row["question_id"]
            topics_for_q = question_topic_ids.get(qid) or []
            if not topics_for_q:
                continue
            correct = int(row.get("correct") or 0)
            elapsed = int(row.get("elapsed_time") or 0)
            ts_ms = int(row["timestamp"])
            when = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            for tid in topics_for_q:
                conn.execute(
                    """
                    INSERT INTO attempts(student_id, topic_id, question_id,
                        correct, elapsed_ms, answered_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (student_id, tid, qid, correct, elapsed, _iso(when)),
                )

    tests_csv = ednet_dir / "tests.csv"
    if tests_csv.exists():
        for row in csv.DictReader(tests_csv.open()):
            conn.execute(
                "INSERT OR REPLACE INTO tests(id, student_id, subject, name, scheduled_for) VALUES(?,?,?,?,?)",
                (row["test_id"], row["student_id"], row["subject"], row["name"], row["date"]),
            )
            for topic in (row.get("topics") or "").split(";"):
                topic = topic.strip()
                if not topic:
                    continue
                tid = topic_ids.get(topic) or _upsert_topic(conn, topic, row.get("subject"))
                conn.execute(
                    "INSERT OR IGNORE INTO test_topics(test_id, topic_id) VALUES(?,?)",
                    (row["test_id"], tid),
                )

    mats_csv = ednet_dir / "materials.csv"
    if mats_csv.exists():
        for row in csv.DictReader(mats_csv.open()):
            tid = topic_ids.get(row["topic"]) or _upsert_topic(conn, row["topic"], None)
            conn.execute(
                """
                INSERT OR REPLACE INTO materials(id, topic_id, title, content_type,
                    difficulty, estimated_minutes, description)
                VALUES(?,?,?,?,?,?,?)
                """,
                (row["material_id"], tid, row["title"], row.get("content_type"),
                 row.get("difficulty"), int(row.get("estimated_minutes") or 20),
                 row.get("description", "")),
            )

    conn.commit()


# ------------------------------------------------------------- extra roster

# Extra topics introduced by the expanded roster, each with its subject. The
# CBSE seed only covers a handful of Math/Science topics; these extend
# coverage into English, History, Biology, and JEE-level Physics/Chemistry.
EXTRA_TOPICS: list[tuple[str, str]] = [
    # CBSE Math
    ("Geometry",                           "Mathematics"),
    ("Trigonometry",                       "Mathematics"),
    ("Mensuration",                        "Mathematics"),
    ("Coordinate Geometry",                "Mathematics"),
    ("Statistics and Probability",         "Mathematics"),
    ("Calculus - Limits and Derivatives",  "Mathematics"),
    # CBSE Biology (Science)
    ("Life Processes",                     "Science"),
    ("Cell Structure",                     "Science"),
    ("Genetics and Heredity",              "Science"),
    # CBSE English
    ("English Grammar - Tenses",           "English"),
    ("English Grammar - Prepositions",     "English"),
    ("English Grammar - Articles",         "English"),
    ("English - Reading Comprehension",    "English"),
    # CBSE History
    ("Rise of Nationalism in Europe",      "History"),
    ("Civil Disobedience Movement",        "History"),
    ("World Wars and Their Impact",        "History"),
    # JEE Physics
    ("JEE Physics - Mechanics",            "JEE Physics"),
    ("JEE Physics - Electromagnetism",     "JEE Physics"),
    ("JEE Physics - Modern Physics",       "JEE Physics"),
    # JEE Chemistry
    ("JEE Chemistry - Organic",            "JEE Chemistry"),
    ("JEE Chemistry - Inorganic",          "JEE Chemistry"),
    ("JEE Chemistry - Physical",           "JEE Chemistry"),
]


# One or two materials per extra topic. material_id starts at M300 to avoid
# clashing with the CBSE JSON (M1xx) and the TOEIC pack (M2xx).
EXTRA_MATERIALS: list[dict] = [
    # Math
    {"id": "M301", "topic": "Geometry", "title": "Geometry — Triangles and Circles (Concept Notes)",
     "content_type": "notes", "difficulty": "beginner", "estimated_minutes": 25,
     "description": "Theorems on congruence, similarity, and properties of circles. CBSE Class 9-10 foundation."},
    {"id": "M302", "topic": "Trigonometry", "title": "Trigonometric Ratios — Worked Video",
     "content_type": "video", "difficulty": "intermediate", "estimated_minutes": 20,
     "description": "sin, cos, tan for standard angles; identities and height-distance applications."},
    {"id": "M303", "topic": "Mensuration", "title": "Mensuration Drill — Surface Area & Volume",
     "content_type": "practice", "difficulty": "intermediate", "estimated_minutes": 40,
     "description": "Mixed problems across cones, cylinders, spheres, and composite solids."},
    {"id": "M304", "topic": "Coordinate Geometry", "title": "Distance, Section and Area Formulas",
     "content_type": "notes", "difficulty": "intermediate", "estimated_minutes": 30,
     "description": "Distance formula, section formula, area of triangle via coordinates, with solved examples."},
    {"id": "M305", "topic": "Statistics and Probability", "title": "Mean, Median, Mode & Classical Probability",
     "content_type": "practice", "difficulty": "beginner", "estimated_minutes": 30,
     "description": "Descriptive stats drills and probability basics with mutually-exclusive events."},
    {"id": "M306", "topic": "Calculus - Limits and Derivatives", "title": "Limits and Derivatives — Pre-University Primer",
     "content_type": "video", "difficulty": "advanced", "estimated_minutes": 35,
     "description": "Intuition for limits, standard derivative rules, worked examples at class 12 / entrance level."},
    # Biology
    {"id": "M307", "topic": "Life Processes", "title": "Life Processes — Nutrition, Respiration, Transport",
     "content_type": "notes", "difficulty": "beginner", "estimated_minutes": 25,
     "description": "Class 10 biology chapter: how organisms perform life processes, with diagrams and key terms."},
    {"id": "M308", "topic": "Cell Structure", "title": "Cell Organelles — Animated Walkthrough",
     "content_type": "video", "difficulty": "beginner", "estimated_minutes": 18,
     "description": "Tour of the cell: membrane, nucleus, mitochondria, ER, Golgi, and their roles."},
    {"id": "M309", "topic": "Genetics and Heredity", "title": "Mendel's Laws — Worked Problems",
     "content_type": "practice", "difficulty": "intermediate", "estimated_minutes": 30,
     "description": "Punnett squares and monohybrid/dihybrid crosses with step-by-step solutions."},
    # English
    {"id": "M310", "topic": "English Grammar - Tenses", "title": "Tenses Crash Course (12 Tenses)",
     "content_type": "notes", "difficulty": "beginner", "estimated_minutes": 20,
     "description": "All 12 tenses with form, usage, and contrast examples. Includes common error traps."},
    {"id": "M311", "topic": "English Grammar - Prepositions", "title": "Prepositions Drill",
     "content_type": "practice", "difficulty": "beginner", "estimated_minutes": 15,
     "description": "Fill-in-the-blanks and sentence-correction drills targeting tricky prepositions."},
    {"id": "M312", "topic": "English Grammar - Articles", "title": "A, An, The — Usage Notes",
     "content_type": "notes", "difficulty": "beginner", "estimated_minutes": 12,
     "description": "Definite vs indefinite articles, zero-article cases, and CBSE Class 10 error spotting."},
    {"id": "M313", "topic": "English - Reading Comprehension", "title": "Unseen Passage Strategy — Worked Examples",
     "content_type": "video", "difficulty": "intermediate", "estimated_minutes": 22,
     "description": "Skimming, inference, tone, and vocab-in-context, walked through three passages."},
    # History
    {"id": "M314", "topic": "Rise of Nationalism in Europe", "title": "Nationalism in Europe — Timeline & Themes",
     "content_type": "notes", "difficulty": "intermediate", "estimated_minutes": 30,
     "description": "French Revolution → Unification of Italy & Germany → Balkans. Causes, key figures, outcomes."},
    {"id": "M315", "topic": "Civil Disobedience Movement", "title": "Civil Disobedience — Gandhi's Salt March",
     "content_type": "video", "difficulty": "intermediate", "estimated_minutes": 18,
     "description": "Events, participation across classes, and why Gandhi chose salt as a symbol."},
    {"id": "M316", "topic": "World Wars and Their Impact", "title": "WW1 vs WW2 — Comparative Notes",
     "content_type": "notes", "difficulty": "intermediate", "estimated_minutes": 28,
     "description": "Causes, alliances, theatres, outcomes, and impact on decolonisation."},
    # JEE Physics
    {"id": "M317", "topic": "JEE Physics - Mechanics", "title": "Kinematics and Dynamics — Problem Set",
     "content_type": "practice", "difficulty": "advanced", "estimated_minutes": 60,
     "description": "JEE-level problems on projectile motion, friction, pulley systems, and rotational dynamics."},
    {"id": "M318", "topic": "JEE Physics - Electromagnetism", "title": "EMF, Faraday, and Inductance",
     "content_type": "notes", "difficulty": "advanced", "estimated_minutes": 45,
     "description": "Electromagnetic induction concepts with worked JEE Main + Advanced style problems."},
    {"id": "M319", "topic": "JEE Physics - Modern Physics", "title": "Photoelectric Effect and Atomic Models",
     "content_type": "video", "difficulty": "advanced", "estimated_minutes": 30,
     "description": "Photons, work function, Bohr's model with numerical drills."},
    # JEE Chemistry
    {"id": "M320", "topic": "JEE Chemistry - Organic", "title": "GOC — General Organic Chemistry",
     "content_type": "notes", "difficulty": "advanced", "estimated_minutes": 50,
     "description": "Inductive, resonance, hyperconjugation effects; stability of intermediates. Foundation for organic JEE."},
    {"id": "M321", "topic": "JEE Chemistry - Organic", "title": "Named Reactions — Flashcard Pack",
     "content_type": "practice", "difficulty": "advanced", "estimated_minutes": 40,
     "description": "50 named reactions (Aldol, Cannizzaro, Hofmann, etc.) with mechanism prompts."},
    {"id": "M322", "topic": "JEE Chemistry - Inorganic", "title": "Periodic Trends — Concept Map",
     "content_type": "notes", "difficulty": "intermediate", "estimated_minutes": 25,
     "description": "Atomic radius, ionisation energy, electronegativity patterns with exceptions explained."},
    {"id": "M323", "topic": "JEE Chemistry - Physical", "title": "Thermodynamics and Equilibrium — Drills",
     "content_type": "practice", "difficulty": "advanced", "estimated_minutes": 55,
     "description": "ΔG, ΔS, ΔH, Kp/Kc calculations; Le Chatelier applied problems."},
]


# One extra student per entry. Each lists weak and strong topics by name plus
# a single upcoming test so the agent has urgency signal to reason about.
EXTRA_STUDENTS: list[dict] = [
    {
        "id": "S124", "name": "Ananya", "grade": 10, "board": "CBSE",
        "target_exam": "School Exams", "daily_minutes": 120,
        "weak_topics":   ["Geometry", "Trigonometry", "Mensuration"],
        "strong_topics": ["Algebra", "Quadratic Equations", "Chemical Reactions"],
        "neutral_topics": ["Coordinate Geometry", "Statistics and Probability"],
        "test": {"id": "T301", "subject": "Mathematics", "name": "Math Monthly Test",
                 "days_away": 10, "topics": ["Geometry", "Trigonometry"]},
    },
    {
        "id": "S125", "name": "Vikram", "grade": 9, "board": "CBSE",
        "target_exam": "School Exams", "daily_minutes": 75,
        "weak_topics":   ["Algebra", "Linear Equations", "Geometry", "Life Processes"],
        "strong_topics": [],
        "neutral_topics": ["English Grammar - Tenses", "Cell Structure"],
        "test": {"id": "T302", "subject": "Science", "name": "Science Half-Yearly",
                 "days_away": 18, "topics": ["Life Processes", "Cell Structure"]},
    },
    {
        "id": "S126", "name": "Rhea", "grade": 12, "board": "CBSE",
        "target_exam": "Pre-Board + JEE Main", "daily_minutes": 180,
        "weak_topics":   ["Calculus - Limits and Derivatives"],
        "strong_topics": ["Algebra", "Quadratic Equations", "Coordinate Geometry",
                          "Trigonometry", "Statistics and Probability"],
        "neutral_topics": ["English - Reading Comprehension"],
        "test": {"id": "T303", "subject": "Mathematics", "name": "Pre-Board Math",
                 "days_away": 21, "topics": ["Calculus - Limits and Derivatives", "Trigonometry"]},
    },
    {
        "id": "S127", "name": "Aarav", "grade": 10, "board": "CBSE",
        "target_exam": "School Exams", "daily_minutes": 90,
        "weak_topics":   ["English Grammar - Tenses", "English Grammar - Prepositions",
                          "English - Reading Comprehension"],
        "strong_topics": ["Rise of Nationalism in Europe", "Civil Disobedience Movement",
                          "World Wars and Their Impact"],
        "neutral_topics": ["English Grammar - Articles"],
        "test": {"id": "T304", "subject": "English", "name": "English Language Test",
                 "days_away": 7, "topics": ["English Grammar - Tenses",
                                            "English - Reading Comprehension"]},
    },
    {
        "id": "S128", "name": "Neha", "grade": 11, "board": "CBSE",
        "target_exam": "JEE Main + Advanced", "daily_minutes": 240,
        "weak_topics":   ["JEE Chemistry - Organic"],
        "strong_topics": ["JEE Physics - Mechanics", "JEE Physics - Electromagnetism",
                          "JEE Chemistry - Physical"],
        "neutral_topics": ["JEE Chemistry - Inorganic", "JEE Physics - Modern Physics"],
        "test": {"id": "T305", "subject": "Chemistry", "name": "JEE Main Test Series",
                 "days_away": 14, "topics": ["JEE Chemistry - Organic",
                                             "JEE Chemistry - Inorganic"]},
    },
]


# Accuracy target (probability-of-correct) by the student's relation to a topic.
_ACC = {"strong": 0.88, "neutral": 0.65, "weak": 0.34}
_ATTEMPTS_PER_TOPIC = 15


# Prerequisite graph: each tuple is (topic, prereq_topic, rationale).
# Curated by pedagogy, not exhaustive — the goal is to ensure the plan_study_week
# tool can refuse to recommend a topic whose foundations are still weak.
TOPIC_PREREQS: list[tuple[str, str, str]] = [
    # CBSE Math
    ("Quadratic Equations", "Algebra",
     "Quadratics require fluent algebraic manipulation (factoring, distribution, substitution)."),
    ("Algebra", "Linear Equations",
     "Linear-equation techniques underpin rearranging algebraic expressions."),
    ("Trigonometry", "Algebra",
     "Trig identities are solved via algebraic manipulation."),
    ("Trigonometry", "Geometry",
     "Trig builds on right-triangle and angle properties."),
    ("Coordinate Geometry", "Geometry",
     "Coordinate geometry encodes geometric objects as equations."),
    ("Coordinate Geometry", "Algebra",
     "Lines, circles and distance formulae require algebraic fluency."),
    ("Mensuration", "Geometry",
     "Surface-area and volume formulas build on geometric reasoning."),
    ("Calculus - Limits and Derivatives", "Algebra",
     "Calculus constantly manipulates algebraic expressions."),
    ("Calculus - Limits and Derivatives", "Trigonometry",
     "Derivatives of trig functions and limits of trig ratios are core."),
    ("Statistics and Probability", "Algebra",
     "Probability and descriptive-stats formulas require algebra."),
    # CBSE Science
    ("Light - Reflection and Refraction", "Algebra",
     "Mirror/lens formula problems reduce to algebraic substitution."),
    ("Genetics and Heredity", "Cell Structure",
     "Genetic inheritance is reasoned about at the cellular level."),
    # CBSE English
    ("English - Reading Comprehension", "English Grammar - Tenses",
     "Tense fluency unlocks complex sentence comprehension."),
    ("English - Reading Comprehension", "English Grammar - Articles",
     "Articles affect meaning precision in passages."),
    # CBSE History — chronology
    ("Civil Disobedience Movement", "Rise of Nationalism in Europe",
     "Nationalist ideas from Europe preceded Indian civil-disobedience."),
    ("World Wars and Their Impact", "Rise of Nationalism in Europe",
     "Nationalism is a primary cause of WW1 and WW2."),
    # JEE
    ("JEE Physics - Mechanics", "Algebra",
     "Mechanics problems reduce to simultaneous equations."),
    ("JEE Physics - Mechanics", "Trigonometry",
     "Vector resolution and projectile motion rely on trig."),
    ("JEE Physics - Electromagnetism", "JEE Physics - Mechanics",
     "EM forces are reasoned about in the Newtonian framework."),
    ("JEE Physics - Modern Physics", "JEE Physics - Mechanics",
     "Energy and momentum concepts transfer directly."),
    ("JEE Chemistry - Organic", "JEE Chemistry - Inorganic",
     "Mechanisms require understanding of electronic structure and bonding."),
    ("JEE Chemistry - Physical", "Algebra",
     "Thermodynamics and equilibrium problems are algebraic."),
    # TOEIC
    ("Reading - Inference Questions", "Reading - Detail Questions",
     "Inference requires fluent detail extraction first."),
    ("Vocabulary - Business Idioms", "Vocabulary - Academic Words",
     "Business idioms assume a baseline academic lexicon."),
    ("Grammar - Conditionals", "Grammar - Tense Consistency",
     "Conditionals compose multiple tenses; tense fluency is foundational."),
]


def seed_prereqs(conn: sqlite3.Connection) -> None:
    """Populate the topic_prerequisites graph. Idempotent (INSERT OR IGNORE).
    Topics that don't exist yet are created so the graph stays consistent even
    if a topic is referenced before it's seeded elsewhere.
    """
    for topic, prereq, rationale in TOPIC_PREREQS:
        topic_id = _upsert_topic(conn, topic, None)
        prereq_id = _upsert_topic(conn, prereq, None)
        if topic_id == prereq_id:
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO topic_prerequisites(topic_id, prereq_topic_id, rationale)
            VALUES(?,?,?)
            """,
            (topic_id, prereq_id, rationale),
        )
    conn.commit()


def seed_extra_roster(conn: sqlite3.Connection) -> None:
    """Insert the extended CBSE + JEE roster straight into SQLite."""
    # Ensure every extra topic has an id.
    topic_ids: dict[str, int] = {
        name: _upsert_topic(conn, name, subject) for name, subject in EXTRA_TOPICS
    }

    # Materials (insert first so tests and students can reference topics reliably).
    for m in EXTRA_MATERIALS:
        tid = topic_ids.get(m["topic"]) or _upsert_topic(conn, m["topic"], None)
        conn.execute(
            """
            INSERT OR REPLACE INTO materials(id, topic_id, title, content_type,
                difficulty, estimated_minutes, description)
            VALUES(?,?,?,?,?,?,?)
            """,
            (m["id"], tid, m["title"], m["content_type"], m["difficulty"],
             m["estimated_minutes"], m["description"]),
        )

    # Students + labels + attempts + test.
    rng = random.Random(2025)
    now = datetime.now(timezone.utc)

    for s in EXTRA_STUDENTS:
        conn.execute(
            """
            INSERT OR REPLACE INTO students(id, name, grade, board, target_exam,
                daily_study_time_minutes, source)
            VALUES(?,?,?,?,?,?,?)
            """,
            (s["id"], s["name"], s["grade"], s["board"], s["target_exam"],
             s["daily_minutes"], "cbse"),
        )

        for relation, label in (("weak_topics", "weak"), ("strong_topics", "strong")):
            for topic in s.get(relation, []):
                tid = topic_ids.get(topic) or _upsert_topic(conn, topic, None)
                conn.execute(
                    "INSERT OR IGNORE INTO student_topic_label(student_id, topic_id, label) VALUES(?,?,?)",
                    (s["id"], tid, label),
                )

        # Synthesise attempts across weak, neutral, strong so modelling has
        # enough evidence to cross WEAK_ACC_LOWER and STRONG_ACC_LOWER.
        for relation_key, acc_key in (("weak_topics", "weak"),
                                      ("strong_topics", "strong"),
                                      ("neutral_topics", "neutral")):
            target = _ACC[acc_key]
            for topic in s.get(relation_key, []):
                tid = topic_ids.get(topic) or _upsert_topic(conn, topic, None)
                for i in range(_ATTEMPTS_PER_TOPIC):
                    correct = 1 if rng.random() < target else 0
                    elapsed = rng.randint(15_000, 60_000)
                    when = now - timedelta(
                        days=rng.randint(1, 30),
                        minutes=rng.randint(0, 600),
                    )
                    conn.execute(
                        """
                        INSERT INTO attempts(student_id, topic_id, question_id,
                            correct, elapsed_ms, answered_at)
                        VALUES(?,?,?,?,?,?)
                        """,
                        (s["id"], tid, f"{s['id'].lower()}-{topic[:5]}-{i}".lower(),
                         correct, elapsed, _iso(when)),
                    )

        # Upcoming test.
        test = s["test"]
        test_date = (date.today() + timedelta(days=test["days_away"])).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO tests(id, student_id, subject, name, scheduled_for) VALUES(?,?,?,?,?)",
            (test["id"], s["id"], test["subject"], test["name"], test_date),
        )
        for topic in test["topics"]:
            tid = topic_ids.get(topic) or _upsert_topic(conn, topic, test.get("subject"))
            conn.execute(
                "INSERT OR IGNORE INTO test_topics(test_id, topic_id) VALUES(?,?)",
                (test["id"], tid),
            )

    conn.commit()
