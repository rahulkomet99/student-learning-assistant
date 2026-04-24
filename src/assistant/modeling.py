"""Student modelling: derive weakness signals from attempt history.

Why this module exists
----------------------
The original JSON profile listed `weak_topics` as a static string list.  With
an attempts table we can do better:

  * A topic is weak if the student's *recent* accuracy on it is low.
  * A topic with 2 attempts is noisier than one with 50 — we factor in sample
    size via Wilson-score lower bound (conservative estimate).
  * Topics without any attempt data fall back to the declared `weak_topics` /
    `strong_topics` labels in `student_topic_label` — so the system degrades
    gracefully on a brand-new student.

This lets the same `get_weak_topics` tool work against real EdNet interaction
data, the CBSE Arjun seed, or a student with mixed signals, with one code
path.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

# Wilson score lower bound for proportion estimates.  Gives conservative
# accuracy estimates when attempts is small (e.g. 0/2 -> ~0 rather than
# treating it as a hard zero).
def _wilson_lower(correct: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = correct / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (center - margin) / denom)


@dataclass
class TopicSignal:
    topic: str
    subject: str | None
    attempts: int
    accuracy_mean: float          # simple avg — stable, reported to the model
    accuracy_lower: float         # Wilson lower bound — used for ranking
    label: str | None             # declared 'weak' / 'strong' / None


# Thresholds for mapping accuracy -> label when enough evidence exists.
MIN_ATTEMPTS_FOR_MODEL = 5
WEAK_ACC_LOWER = 0.55
STRONG_ACC_LOWER = 0.75


def compute_topic_signals(
    conn: sqlite3.Connection,
    student_id: str,
    recent_window: int = 30,
) -> list[TopicSignal]:
    """Aggregate the `recent_window` most recent attempts per topic.

    Returns one TopicSignal per topic the student has *either* answered or
    has a declared label for.
    """
    # Per-topic recent accuracy: window is per topic, not global, so a
    # dormant topic doesn't get flushed out by a flurry of activity elsewhere.
    attempt_rows = conn.execute(
        """
        WITH ranked AS (
            SELECT a.topic_id, a.correct,
                   ROW_NUMBER() OVER (PARTITION BY a.topic_id ORDER BY a.answered_at DESC) AS rn
            FROM attempts a
            WHERE a.student_id = ?
        )
        SELECT t.name AS topic, t.subject, r.topic_id,
               SUM(r.correct) AS correct, COUNT(*) AS total
        FROM ranked r
        JOIN topics t ON t.id = r.topic_id
        WHERE r.rn <= ?
        GROUP BY r.topic_id
        """,
        (student_id, recent_window),
    ).fetchall()

    label_rows = conn.execute(
        """
        SELECT t.name AS topic, t.subject, stl.label
        FROM student_topic_label stl
        JOIN topics t ON t.id = stl.topic_id
        WHERE stl.student_id = ?
        """,
        (student_id,),
    ).fetchall()
    label_by_topic: dict[str, dict] = {r["topic"]: dict(r) for r in label_rows}

    signals: dict[str, TopicSignal] = {}
    for row in attempt_rows:
        correct = int(row["correct"] or 0)
        total = int(row["total"])
        signals[row["topic"]] = TopicSignal(
            topic=row["topic"],
            subject=row["subject"],
            attempts=total,
            accuracy_mean=round(correct / total, 3) if total else 0.0,
            accuracy_lower=round(_wilson_lower(correct, total), 3),
            label=(label_by_topic.get(row["topic"]) or {}).get("label"),
        )

    # Include declared labels even when we have no attempt data.
    for topic, info in label_by_topic.items():
        if topic not in signals:
            signals[topic] = TopicSignal(
                topic=topic,
                subject=info["subject"],
                attempts=0,
                accuracy_mean=0.0,
                accuracy_lower=0.0,
                label=info["label"],
            )

    return list(signals.values())


def rank_weak_topics(signals: list[TopicSignal], limit: int = 5) -> list[TopicSignal]:
    """Return topics ordered weakest-first. Blends evidence sources:
        * modelled topics (>= MIN_ATTEMPTS_FOR_MODEL) below WEAK_ACC_LOWER
        * explicitly-labelled 'weak' topics
    """
    modelled_weak = [
        s for s in signals
        if s.attempts >= MIN_ATTEMPTS_FOR_MODEL and s.accuracy_lower < WEAK_ACC_LOWER
    ]
    modelled_weak.sort(key=lambda s: (s.accuracy_lower, -s.attempts))

    declared_only = [
        s for s in signals
        if s.label == "weak" and s.attempts < MIN_ATTEMPTS_FOR_MODEL
    ]
    declared_only.sort(key=lambda s: s.topic)

    seen: set[str] = set()
    out: list[TopicSignal] = []
    for s in (*modelled_weak, *declared_only):
        if s.topic in seen:
            continue
        seen.add(s.topic)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def rank_strong_topics(signals: list[TopicSignal], limit: int = 5) -> list[TopicSignal]:
    modelled_strong = [
        s for s in signals
        if s.attempts >= MIN_ATTEMPTS_FOR_MODEL and s.accuracy_lower >= STRONG_ACC_LOWER
    ]
    modelled_strong.sort(key=lambda s: (-s.accuracy_lower, -s.attempts))

    declared_only = [
        s for s in signals
        if s.label == "strong" and s.attempts < MIN_ATTEMPTS_FOR_MODEL
    ]
    declared_only.sort(key=lambda s: s.topic)

    seen: set[str] = set()
    out: list[TopicSignal] = []
    for s in (*modelled_strong, *declared_only):
        if s.topic in seen:
            continue
        seen.add(s.topic)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def signal_to_dict(s: TopicSignal) -> dict:
    return {
        "topic": s.topic,
        "subject": s.subject,
        "attempts": s.attempts,
        "accuracy_mean_percentage": round(s.accuracy_mean * 100, 1),
        "accuracy_lower_bound_percentage": round(s.accuracy_lower * 100, 1),
        "declared_label": s.label,
    }


def weak_prerequisites_for(
    conn: sqlite3.Connection,
    student_id: str,
    topic: str,
    signals: list[TopicSignal] | None = None,
) -> list[dict]:
    """Return prerequisite topics that are themselves weaker than the focal
    topic — the student should shore these up first.

    Uses the same Wilson-lower signal as `rank_weak_topics`; a prereq is
    flagged when the student has enough evidence on it (attempts >=
    MIN_ATTEMPTS_FOR_MODEL) and its lower-bound accuracy is below
    WEAK_ACC_LOWER.
    """
    if signals is None:
        signals = compute_topic_signals(conn, student_id)
    signal_by_topic = {s.topic: s for s in signals}
    target = signal_by_topic.get(topic)
    target_acc = target.accuracy_lower if target else 0.0

    rows = conn.execute(
        """
        SELECT t_pre.name AS prereq_topic, tp.rationale
        FROM topic_prerequisites tp
        JOIN topics t      ON t.id = tp.topic_id
        JOIN topics t_pre  ON t_pre.id = tp.prereq_topic_id
        WHERE t.name = ?
        """,
        (topic,),
    ).fetchall()

    flagged: list[dict] = []
    for r in rows:
        prereq_topic = r["prereq_topic"]
        sig = signal_by_topic.get(prereq_topic)
        # Only flag prereqs we have evidence on, or ones explicitly declared weak.
        if sig is None:
            continue
        prereq_is_weak = (
            (sig.attempts >= MIN_ATTEMPTS_FOR_MODEL and sig.accuracy_lower < WEAK_ACC_LOWER)
            or sig.label == "weak"
        )
        # Worth flagging if the prereq is weak AND weaker than the focal topic
        # (shoring up the prereq is the higher-value move).
        if prereq_is_weak and sig.accuracy_lower <= target_acc + 0.05:
            flagged.append({
                "prereq_topic": prereq_topic,
                "prereq_accuracy_lower_percentage": round(sig.accuracy_lower * 100, 1),
                "rationale": r["rationale"],
            })
    return flagged
