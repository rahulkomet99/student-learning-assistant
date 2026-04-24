"""Capture the assignment's 4 sample queries end-to-end into markdown.

Runs the agent against the default student (Arjun — the CBSE grade-10 student
from the assignment) and writes each query's tool calls, cited material_ids,
and final answer to `evals/assignment_queries.md`. This is the artifact graders
can read without running the server themselves.

Usage:
    python -m evals.capture_assignment_responses            # default: S123
    python -m evals.capture_assignment_responses --student S128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from assistant.agent import Agent  # noqa: E402
from assistant.retrieval import HybridRetriever  # noqa: E402
from assistant.store import Store, ensure_seeded  # noqa: E402
from assistant.tracing import TraceLogger  # noqa: E402

OUT_PATH = ROOT / "evals" / "assignment_queries.md"

ASSIGNMENT_QUERIES: list[str] = [
    "I am weak in Algebra. What should I do next?",
    "What should I study this week?",
    "Which topic should I prioritize first?",
    "I have a Maths test coming up. Help me prepare.",
]


def run_and_capture(agent: Agent, trace: TraceLogger, query: str) -> dict:
    """Run one query. Collect text, tool calls, and citations."""
    t0 = time.time()
    text_parts: list[str] = []
    tools_called: list[dict] = []
    citations: list[str] = []
    usage: dict = {}

    for event in agent.run(query, trace=trace):
        if event.kind == "text_delta":
            text_parts.append(event.data["text"])
        elif event.kind == "tool_use":
            tools_called.append({
                "name": event.data["name"],
                "input": event.data.get("input") or {},
            })
        elif event.kind == "done":
            citations = event.data.get("citations", [])
            usage = event.data.get("usage", {})
        elif event.kind == "error":
            text_parts.append(f"\n\n[error] {event.data.get('message')}")

    return {
        "query": query,
        "final_text": "".join(text_parts).strip(),
        "tools_called": tools_called,
        "citations": citations,
        "usage": usage,
        "latency_seconds": round(time.time() - t0, 2),
    }


def render_markdown(student: dict, results: list[dict]) -> str:
    """Render the captured run as a single markdown artifact for the README."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# How the assistant answers the assignment's sample queries",
        "",
        f"_Captured run against student **{student['name']}** "
        f"({student['student_id']}, grade {student.get('grade') or '—'}, "
        f"{student.get('board') or student.get('target_exam') or 'general'}, "
        f"{student['daily_study_time_minutes']}-min daily budget) on {now}._",
        "",
        "Each section below shows the verbatim query, the tools the agent chose "
        "to call and with what inputs, the material_ids it cited, and the final "
        "streamed answer. Nothing here is hand-edited — re-run "
        "`python -m evals.capture_assignment_responses` to refresh.",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(results, start=1):
        lines.append(f"## {i}. `{r['query']}`")
        lines.append("")

        if r["tools_called"]:
            lines.append("**Tools called:**")
            for call in r["tools_called"]:
                args = ", ".join(f"{k}={v!r}" for k, v in call["input"].items())
                lines.append(f"- `{call['name']}({args})`")
            lines.append("")

        if r["citations"]:
            lines.append(f"**Cited materials:** {', '.join(r['citations'])}")
            lines.append("")

        usage = r.get("usage", {})
        if usage:
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)
            lines.append(
                f"**Usage:** input={usage.get('input_tokens', 0)} · "
                f"output={usage.get('output_tokens', 0)} · "
                f"cache_read={cache_read} · cache_write={cache_write} · "
                f"{r['latency_seconds']}s"
            )
            lines.append("")

        lines.append("**Answer:**")
        lines.append("")
        lines.append("> " + r["final_text"].replace("\n", "\n> "))
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(prog="capture_assignment_responses")
    parser.add_argument("--student", default="S123",
                        help="Student id to run the 4 sample queries against.")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set — cannot call the agent.", file=sys.stderr)
        return 1

    ensure_seeded()
    try:
        store = Store.open(args.student)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    retriever = HybridRetriever(store.materials)
    client = Anthropic()
    trace = TraceLogger()
    agent = Agent(client=client, data=store, retriever=retriever)
    store.start_session(trace.session_id)

    print(f"Running {len(ASSIGNMENT_QUERIES)} queries as {store.profile['name']} "
          f"({args.student}) — trace: {trace.path}")
    results: list[dict] = []
    for query in ASSIGNMENT_QUERIES:
        print(f"  · {query}")
        results.append(run_and_capture(agent, trace, query))

    OUT_PATH.write_text(render_markdown(store.profile, results))
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
