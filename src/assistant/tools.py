"""Tools exposed to the Claude agent.

Design notes
------------
We keep a small, composable tool surface instead of one mega-tool:

  - get_weak_topics: structured lookup over topic_performance + profile.
  - get_upcoming_tests: structured lookup, filtered by days_ahead.
  - recommend_study_material: hybrid BM25+dense retrieval (semantic search).
  - plan_study_week: higher-level composer that prioritises topics by
    (weakness x test-urgency) and allocates the student's daily study budget.

Each handler returns a JSON-serialisable dict (never raw objects) so the agent
loop can pass results back to the model as tool_result blocks directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

from .data import DataStore
from .modeling import (
    compute_topic_signals,
    rank_strong_topics,
    rank_weak_topics,
    signal_to_dict,
)
from .retrieval import HybridRetriever

# --- Tool schemas (Anthropic tool-use format) ----------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_weak_topics",
        "description": (
            "Return the student's weak topics along with their topic-level scores and "
            "the subject-level performance they belong to. Use this to ground any "
            "recommendation about what the student should focus on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_upcoming_tests",
        "description": (
            "Return tests scheduled within the next `days_ahead` days (default 14), "
            "with topic coverage and days remaining. Use this whenever the student "
            "asks about prep, prioritisation, or what to study this week."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 90,
                    "description": "Look-ahead window in days. Defaults to 14.",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "recommend_study_material",
        "description": (
            "Semantic + keyword hybrid search over the student's study material "
            "library. Pass a natural-language `query` describing what to revise. "
            "Optionally constrain by `topic_filter` (exact topic names) and "
            "`top_k` (default 3). Returns material_ids that MUST be cited in the "
            "final answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of what to study.",
                },
                "topic_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of exact topic names to restrict results to.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "Number of results to return. Defaults to 3.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "plan_study_week",
        "description": (
            "Generate a day-by-day study plan for the next `days` days (default 7) "
            "that balances the student's weak topics against the urgency of their "
            "upcoming tests, respecting their daily_study_time_minutes budget. "
            "Returns a prioritised plan with time allocations and recommended "
            "materials per topic. Use this for 'what should I study this week' "
            "style queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Planning horizon. Defaults to 7.",
                }
            },
            "additionalProperties": False,
        },
    },
]


# --- Handlers ------------------------------------------------------------------

@dataclass
class ToolContext:
    """Everything the tool handlers need. Injected at construction time so tool
    calls stay pure functions of (context, inputs).

    `data` is duck-typed: may be a legacy `DataStore` (JSON-backed) or the new
    SQLite-backed `Store`. When a SQLite `conn` attribute is present, the
    weakness-ranking tools switch from static profile labels to dynamic
    modelling over the attempts table.
    """

    data: DataStore
    retriever: HybridRetriever
    today: date

    @classmethod
    def build(cls, data: DataStore, retriever: HybridRetriever, today: date | None = None) -> "ToolContext":
        return cls(data=data, retriever=retriever, today=today or datetime.now().date())

    def has_attempts_backend(self) -> bool:
        return getattr(self.data, "conn", None) is not None


def _days_until(ctx: ToolContext, iso_date: str) -> int:
    return (date.fromisoformat(iso_date) - ctx.today).days


def handle_get_weak_topics(ctx: ToolContext, _inputs: dict) -> dict:
    """Return weak (and strong) topics.

    When attempt data is available (SQLite-backed store), weakness is the
    Wilson lower-bound of recent accuracy — so a student with 2/3 correct is
    not treated as 66.7% strong. Falls back to the student's declared
    profile labels when there's no attempt history.
    """
    if ctx.has_attempts_backend():
        signals = compute_topic_signals(ctx.data.conn, ctx.data.student_id)
        weak = [signal_to_dict(s) for s in rank_weak_topics(signals)]
        strong = [signal_to_dict(s) for s in rank_strong_topics(signals)]
        return {
            "student_id": ctx.data.student_id,
            "weak_topics": weak,
            "strong_topics": strong,
            "subject_performance": ctx.data.performance.get("subject_performance", []),
            "source": "modelled_from_attempts",
        }

    # Legacy path — static labels from profile JSON.
    topic_rows = []
    for topic in ctx.data.profile.get("weak_topics", []):
        score = ctx.data.topic_score(topic)
        topic_rows.append({
            "topic": topic,
            "topic_score_percentage": score,
            "subject_score_percentage": _subject_for_topic(ctx, topic),
        })
    return {
        "student_id": ctx.data.student_id,
        "weak_topics": topic_rows,
        "strong_topics": ctx.data.profile.get("strong_topics", []),
        "subject_performance": ctx.data.performance.get("subject_performance", []),
        "source": "declared_labels",
    }


def _subject_for_topic(ctx: ToolContext, topic: str) -> float | None:
    for row in ctx.data.performance.get("topic_performance", []):
        if row["topic"] == topic:
            return ctx.data.subject_score(row["subject"])
    return None


def handle_get_upcoming_tests(ctx: ToolContext, inputs: dict) -> dict:
    days_ahead = int(inputs.get("days_ahead") or 14)
    rows = []
    for t in ctx.data.tests:
        delta = _days_until(ctx, t["date"])
        if 0 <= delta <= days_ahead:
            rows.append({**t, "days_until": delta})
    rows.sort(key=lambda r: r["days_until"])
    return {
        "student_id": ctx.data.student_id,
        "window_days": days_ahead,
        "today": ctx.today.isoformat(),
        "tests": rows,
    }


def handle_recommend_study_material(ctx: ToolContext, inputs: dict) -> dict:
    query = inputs["query"]
    topic_filter = inputs.get("topic_filter")
    top_k = int(inputs.get("top_k") or 3)
    hits = ctx.retriever.search(query, top_k=top_k, topic_filter=topic_filter)
    return {
        "query": query,
        "topic_filter": topic_filter,
        "results": [
            {
                **hit.material.to_dict(),
                "rrf_score": round(hit.score, 5),
                "dense_rank": hit.dense_rank,
                "bm25_rank": hit.bm25_rank,
            }
            for hit in hits
        ],
    }


def handle_plan_study_week(ctx: ToolContext, inputs: dict) -> dict:
    """Compose weak-topic + test-urgency + daily-budget into a concrete plan.

    Scoring per topic:
        base    = max(0, (70 - topic_score) / 70)    # weakness signal, 0..1
        urgency = max(0, (14 - min_days_to_test) / 14) if topic hits an upcoming
                  test within 14 days, else 0
        priority = base * (1 + 2 * urgency)

    We then normalise priorities into proportional time slices of the student's
    daily_study_time_minutes, round to the nearest 5 minutes, and attach the top
    retrieved material per topic as a concrete "what to do" anchor.
    """
    days = int(inputs.get("days") or 7)
    daily_minutes = int(ctx.data.profile.get("daily_study_time_minutes", 60))

    # Prefer modelled weaknesses (Wilson-lower of recent accuracy); fall back
    # to declared labels if there are no attempt rows yet. Each entry carries
    # an effective score% used by the priority blender below.
    weak_entries: list[tuple[str, float]] = []
    if ctx.has_attempts_backend():
        signals = compute_topic_signals(ctx.data.conn, ctx.data.student_id)
        for s in rank_weak_topics(signals, limit=6):
            score_pct = s.accuracy_lower * 100 if s.attempts else 42.0
            weak_entries.append((s.topic, score_pct))
    if not weak_entries:
        for topic in ctx.data.profile.get("weak_topics", []):
            weak_entries.append((topic, ctx.data.topic_score(topic) or 50.0))

    if not weak_entries:
        return {"days": days, "daily_minutes": daily_minutes, "plan": [], "note": "No weak topics on file."}

    test_urgency: dict[str, int] = {}
    for t in ctx.data.tests:
        delta = _days_until(ctx, t["date"])
        if delta < 0 or delta > 14:
            continue
        for topic in t["topics"]:
            prev = test_urgency.get(topic)
            if prev is None or delta < prev:
                test_urgency[topic] = delta

    topic_priorities: list[dict] = []
    for topic, score in weak_entries:
        base = max(0.0, (70.0 - score) / 70.0)
        min_days = test_urgency.get(topic)
        urgency = max(0.0, (14 - min_days) / 14) if min_days is not None else 0.0
        priority = base * (1 + 2 * urgency)
        topic_priorities.append({
            "topic": topic,
            "topic_score_percentage": round(score, 1),
            "days_to_next_test": min_days,
            "weakness_component": round(base, 3),
            "urgency_component": round(urgency, 3),
            "priority": round(priority, 3),
        })

    total = sum(tp["priority"] for tp in topic_priorities) or 1.0
    for tp in topic_priorities:
        raw = (tp["priority"] / total) * daily_minutes
        tp["daily_minutes"] = max(10, 5 * round(raw / 5))

    # Balance rounding drift back to the budget.
    drift = daily_minutes - sum(tp["daily_minutes"] for tp in topic_priorities)
    if topic_priorities and drift:
        topic_priorities.sort(key=lambda r: -r["priority"])
        topic_priorities[0]["daily_minutes"] = max(5, topic_priorities[0]["daily_minutes"] + drift)

    for tp in topic_priorities:
        hits = ctx.retriever.search(
            query=f"revise {tp['topic']} for upcoming test",
            top_k=2,
            topic_filter=[tp["topic"]],
        )
        tp["recommended_materials"] = [
            {**hit.material.to_dict(), "rrf_score": round(hit.score, 5)} for hit in hits
        ]

    topic_priorities.sort(key=lambda r: -r["priority"])

    return {
        "days": days,
        "daily_minutes": daily_minutes,
        "today": ctx.today.isoformat(),
        "plan": topic_priorities,
    }


HANDLERS: dict[str, Callable[[ToolContext, dict], dict]] = {
    "get_weak_topics": handle_get_weak_topics,
    "get_upcoming_tests": handle_get_upcoming_tests,
    "recommend_study_material": handle_recommend_study_material,
    "plan_study_week": handle_plan_study_week,
}


def dispatch(ctx: ToolContext, name: str, inputs: dict) -> dict:
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return handler(ctx, inputs or {})
    except Exception as exc:  # surface errors to the model so it can recover
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect_citations(tool_results: list[dict[str, Any]]) -> list[str]:
    """Extract unique material_ids across all tool results for downstream UI display."""
    seen: list[str] = []
    for result in tool_results:
        _walk_for_ids(result, seen)
    return seen


def _walk_for_ids(obj: Any, seen: list[str]) -> None:
    if isinstance(obj, dict):
        mid = obj.get("material_id")
        if isinstance(mid, str) and mid not in seen:
            seen.append(mid)
        for v in obj.values():
            _walk_for_ids(v, seen)
    elif isinstance(obj, list):
        for v in obj:
            _walk_for_ids(v, seen)
