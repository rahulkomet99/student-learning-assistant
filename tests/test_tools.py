"""Smoke tests for the SQLite-backed Store, modelling, tools, and retrieval.

These tests do not call the Anthropic API. They exercise the boundary between
the agent and the rest of the system: schema init, CBSE + EdNet seeders,
modelling rollups, tool handlers.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from assistant.modeling import (  # noqa: E402
    compute_topic_signals,
    rank_strong_topics,
    rank_weak_topics,
)
from assistant.retrieval import HybridRetriever  # noqa: E402
from assistant.store import Store  # noqa: E402
from assistant.schema import init_db  # noqa: E402
from assistant.seed import seed_cbse, seed_ednet_sample, seed_extra_roster, seed_prereqs  # noqa: E402
from assistant.tools import ToolContext, dispatch  # noqa: E402


FIXED_TODAY = date(2026, 4, 24)


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory) -> Path:
    db_path = tmp_path_factory.mktemp("sla") / "db.sqlite3"
    conn = init_db(db_path)
    seed_cbse(conn)
    seed_ednet_sample(conn)
    seed_extra_roster(conn)
    seed_prereqs(conn)
    conn.close()
    return db_path


@pytest.fixture()
def cbse_ctx(seeded_db):
    store = Store.open("S123", db_path=seeded_db)
    return ToolContext.build(store, HybridRetriever(store.materials), today=FIXED_TODAY)


@pytest.fixture()
def ednet_ctx(seeded_db):
    store = Store.open("EN-u001", db_path=seeded_db)
    return ToolContext.build(store, HybridRetriever(store.materials), today=FIXED_TODAY)


# --- retrieval -----------------------------------------------------------------

def test_retriever_finds_algebra(cbse_ctx):
    hits = cbse_ctx.retriever.search("algebra factorization", top_k=3)
    assert hits
    assert any(h.material.topic == "Algebra" for h in hits)


def test_retriever_topic_filter(cbse_ctx):
    hits = cbse_ctx.retriever.search(
        "revision", top_k=5,
        topic_filter=["Light - Reflection and Refraction"],
    )
    assert hits
    assert all(h.material.topic == "Light - Reflection and Refraction" for h in hits)


def test_recommend_returns_valid_ids(cbse_ctx):
    result = dispatch(cbse_ctx, "recommend_study_material", {"query": "Snell's law refractive index"})
    assert result["results"]
    assert any(r["material_id"] == "M112" for r in result["results"])


# --- tools (CBSE) --------------------------------------------------------------

def test_get_weak_topics_uses_attempts_backend(cbse_ctx):
    result = dispatch(cbse_ctx, "get_weak_topics", {})
    assert result["source"] == "modelled_from_attempts"
    topics = [t["topic"] for t in result["weak_topics"]]
    # Arjun's declared weak topics should surface via the synthesised attempts.
    assert {"Algebra", "Quadratic Equations"} & set(topics)
    # Each signal should report attempt count + both accuracy metrics.
    assert all("attempts" in t for t in result["weak_topics"])
    assert all("accuracy_lower_bound_percentage" in t for t in result["weak_topics"])


def test_get_upcoming_tests_window(cbse_ctx):
    result = dispatch(cbse_ctx, "get_upcoming_tests", {"days_ahead": 10})
    assert result["tests"], "should find T201 on 2026-04-29 within 10 days of 2026-04-24"
    assert result["tests"][0]["test_id"] == "T201"
    assert result["tests"][0]["days_until"] == 5


def test_plan_study_week_respects_budget(cbse_ctx):
    result = dispatch(cbse_ctx, "plan_study_week", {"days": 7})
    total = sum(p["daily_minutes"] for p in result["plan"])
    assert total == result["daily_minutes"], f"plan totals {total}, expected {result['daily_minutes']}"
    assert all(p.get("recommended_materials") for p in result["plan"])


def test_plan_prioritises_urgent_weak_topic(cbse_ctx):
    result = dispatch(cbse_ctx, "plan_study_week", {"days": 7})
    # Algebra or Quadratic Equations (tested in 5 days) should rank above Light
    # (tested in 12 days).
    topics_in_order = [p["topic"] for p in result["plan"]]
    assert topics_in_order[0] in {"Algebra", "Quadratic Equations"}


# --- modelling -----------------------------------------------------------------

def test_modelling_flips_strong_weak_for_different_students(seeded_db):
    cbse_store = Store.open("S123", db_path=seeded_db)
    ednet_store = Store.open("EN-u001", db_path=seeded_db)

    cbse_signals = compute_topic_signals(cbse_store.conn, "S123")
    ednet_signals = compute_topic_signals(ednet_store.conn, "EN-u001")

    cbse_weak = {s.topic for s in rank_weak_topics(cbse_signals)}
    ednet_weak = {s.topic for s in rank_weak_topics(ednet_signals)}

    # The two students live in different subject spaces; their weak topics
    # should not overlap.
    assert cbse_weak and ednet_weak
    assert not (cbse_weak & ednet_weak)
    # EdNet u001 was seeded with weak tags t3 (Conditionals) + t4 (Business Idioms).
    assert any("Conditionals" in t or "Business Idioms" in t for t in ednet_weak)


# --- ednet end-to-end -----------------------------------------------------------

def test_ednet_student_has_tests_and_weak_topics(ednet_ctx):
    weak = dispatch(ednet_ctx, "get_weak_topics", {})
    assert weak["source"] == "modelled_from_attempts"
    assert weak["weak_topics"]

    tests = dispatch(ednet_ctx, "get_upcoming_tests", {"days_ahead": 30})
    assert tests["tests"]
    assert tests["tests"][0]["test_id"].startswith("EN-T")


def test_ednet_plan_has_recommendations(ednet_ctx):
    plan = dispatch(ednet_ctx, "plan_study_week", {"days": 7})
    assert plan["plan"]
    assert any(p.get("recommended_materials") for p in plan["plan"])


# --- prerequisite graph ----------------------------------------------------

def test_plan_surfaces_weak_prerequisites(seeded_db):
    """Ananya is declared weak on Geometry + Trigonometry + Mensuration.
    Geometry is a prereq for both of the others; the plan should flag it."""
    store = Store.open("S124", db_path=seeded_db)
    ctx = ToolContext.build(store, HybridRetriever(store.materials), today=FIXED_TODAY)
    plan = dispatch(ctx, "plan_study_week", {"days": 7})

    trig = next((p for p in plan["plan"] if p["topic"] == "Trigonometry"), None)
    assert trig is not None, "Trigonometry should appear in Ananya's plan"
    weak_prereqs = trig.get("weak_prerequisites", [])
    assert any(wp["prereq_topic"] == "Geometry" for wp in weak_prereqs), (
        f"expected Geometry flagged as weak prereq of Trigonometry, got {weak_prereqs}"
    )


def test_plan_skips_strong_prerequisite(seeded_db):
    """Rhea is weak on Calculus but strong on Algebra (a Calculus prereq).
    Algebra should NOT be flagged as a weak prereq."""
    store = Store.open("S126", db_path=seeded_db)
    ctx = ToolContext.build(store, HybridRetriever(store.materials), today=FIXED_TODAY)
    plan = dispatch(ctx, "plan_study_week", {"days": 7})
    calc = next((p for p in plan["plan"] if "Calculus" in p["topic"]), None)
    assert calc is not None
    for wp in calc.get("weak_prerequisites", []):
        assert wp["prereq_topic"] != "Algebra", (
            f"Algebra is strong for Rhea; should not be a weak prereq. Got: {wp}"
        )
