"""Agent-loop tests with a mocked Anthropic client.

These exercise the control flow in `Agent.run`:
  - termination on end_turn
  - tool_use dispatch loop with tool_result turnaround
  - MAX_TURNS exhaustion safety net
  - citation audit emitting a guardrail warning when tools returned
    material_ids but the final answer cited none

We don't hit the real API. A tiny `FakeClient` mimics the
`client.messages.stream(...)` context manager so the agent's streaming path
runs end-to-end against deterministic fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from assistant.agent import MAX_TURNS, Agent, AgentEvent  # noqa: E402
from assistant.retrieval import HybridRetriever  # noqa: E402
from assistant.schema import init_db  # noqa: E402
from assistant.seed import (  # noqa: E402
    seed_cbse,
    seed_ednet_sample,
    seed_extra_roster,
    seed_prereqs,
)
from assistant.store import Store  # noqa: E402
from assistant.tracing import TraceLogger  # noqa: E402


# --- fakes -------------------------------------------------------------------

def text_delta_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(block_id: str, name: str, inp: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=inp)


def fake_final(content: list, stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=10,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


class FakeStream:
    def __init__(self, events: list, final: SimpleNamespace):
        self._events = events
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


class FakeMessages:
    def __init__(self, turns: list[tuple[list, SimpleNamespace]]):
        self._turns = iter(turns)

    def stream(self, **_kwargs):
        events, final = next(self._turns)
        return FakeStream(events, final)


class FakeClient:
    """Duck-types anthropic.Anthropic for Agent.run's code path."""

    def __init__(self, turns: list[tuple[list, SimpleNamespace]]):
        self.messages = FakeMessages(turns)


# --- fixtures ---------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory) -> Path:
    db_path = tmp_path_factory.mktemp("sla-agent") / "db.sqlite3"
    conn = init_db(db_path)
    seed_cbse(conn)
    seed_ednet_sample(conn)
    seed_extra_roster(conn)
    seed_prereqs(conn)
    conn.close()
    return db_path


@pytest.fixture
def make_agent(seeded_db):
    def _build(client):
        store = Store.open("S123", db_path=seeded_db)
        retriever = HybridRetriever(store.materials)
        return Agent(client=client, data=store, retriever=retriever)
    return _build


def _collect(agent_gen) -> list[AgentEvent]:
    return list(agent_gen)


# --- tests ------------------------------------------------------------------

def test_terminates_on_end_turn_with_no_tools(make_agent):
    """Single-turn response: text streams, then done event with empty citations."""
    turns = [
        (
            [text_delta_event("Hi "), text_delta_event("there")],
            fake_final([text_block("Hi there")], stop_reason="end_turn"),
        ),
    ]
    agent = make_agent(FakeClient(turns))
    events = _collect(agent.run("hi"))

    kinds = [e.kind for e in events]
    assert "text_delta" in kinds
    assert kinds[-1] == "done"
    done = events[-1]
    assert done.data["stop_reason"] == "end_turn"
    assert done.data["citations"] == []
    assert all(k != "tool_use" for k in kinds), "expected no tool calls"


def test_tool_use_loop_roundtrips_and_collects_citations(make_agent):
    """Turn 1: text + tool_use → dispatch → tool_result.
       Turn 2: final text cites [M101] → done with M101 in citations."""
    turns = [
        (
            [text_delta_event("Let me check…")],
            fake_final(
                [
                    text_block("Let me check…"),
                    tool_use_block("t1", "recommend_study_material", {"query": "algebra"}),
                ],
                stop_reason="tool_use",
            ),
        ),
        (
            [text_delta_event("Start with "), text_delta_event("[M101]")],
            fake_final(
                [text_block("Start with [M101]")],
                stop_reason="end_turn",
            ),
        ),
    ]
    agent = make_agent(FakeClient(turns))
    events = _collect(agent.run("algebra help"))

    kinds = [e.kind for e in events]
    assert kinds.count("tool_use") == 1, f"expected 1 tool call, got {kinds}"
    assert kinds.count("tool_result") == 1
    # Tool result must include the retrieved material_ids
    tool_result = next(e for e in events if e.kind == "tool_result")
    assert tool_result.data["name"] == "recommend_study_material"
    # Final done event should carry citations collected across turns
    done = events[-1]
    assert done.kind == "done"
    assert "M101" in done.data["citations"]


def test_max_turns_exhaustion_surfaces_error(make_agent):
    """If Claude keeps returning tool_use forever, the loop must bail out."""
    always_tool = (
        [text_delta_event("working…")],
        fake_final(
            [tool_use_block("t1", "get_weak_topics", {})],
            stop_reason="tool_use",
        ),
    )
    turns = [always_tool] * (MAX_TURNS + 2)  # more than the agent should ever consume
    agent = make_agent(FakeClient(turns))
    events = _collect(agent.run("never ends"))

    assert events[-1].kind == "error"
    assert "max_turns" in events[-1].data["message"].lower()
    # And exactly MAX_TURNS tool calls, not more.
    assert sum(1 for e in events if e.kind == "tool_use") == MAX_TURNS


def test_citation_audit_warns_when_materials_surfaced_but_not_cited(make_agent, tmp_path):
    """Tool returned material_ids, but the final answer has no [M###] — the
    guardrail should log a `missing_citation` warning to the trace JSONL."""
    turns = [
        (
            [],
            fake_final(
                [tool_use_block("t1", "recommend_study_material", {"query": "algebra"})],
                stop_reason="tool_use",
            ),
        ),
        (
            [text_delta_event("Just study more, no refs.")],
            fake_final(
                [text_block("Just study more, no refs.")],
                stop_reason="end_turn",
            ),
        ),
    ]
    agent = make_agent(FakeClient(turns))
    trace = TraceLogger(trace_dir=tmp_path)
    _collect(agent.run("algebra help", trace=trace))

    lines = trace.path.read_text().splitlines()
    warnings = [
        json.loads(line) for line in lines
        if line and json.loads(line).get("kind") == "guardrail_warning"
    ]
    assert warnings, "expected a guardrail_warning entry"
    assert warnings[0].get("note", "").startswith("Tools returned material_ids")


def test_citation_audit_silent_when_cited(make_agent, tmp_path):
    """When the model does cite, no guardrail warning fires."""
    turns = [
        (
            [],
            fake_final(
                [tool_use_block("t1", "recommend_study_material", {"query": "algebra"})],
                stop_reason="tool_use",
            ),
        ),
        (
            [text_delta_event("See [M101].")],
            fake_final(
                [text_block("See [M101].")],
                stop_reason="end_turn",
            ),
        ),
    ]
    agent = make_agent(FakeClient(turns))
    trace = TraceLogger(trace_dir=tmp_path)
    _collect(agent.run("algebra help", trace=trace))

    lines = trace.path.read_text().splitlines()
    assert not any(
        json.loads(line).get("kind") == "guardrail_warning"
        for line in lines if line
    )
