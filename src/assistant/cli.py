"""Minimal CLI: one command, REPL, streams tokens as they arrive.

Usage:
  assistant-cli                    # REPL
  assistant-cli -q "what to do?"   # one-shot
  assistant-cli --trace            # print tool-call summaries alongside text
"""

from __future__ import annotations

import argparse
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .agent import Agent
from .retrieval import HybridRetriever
from .store import Store, ensure_seeded
from .tracing import TraceLogger

DEFAULT_STUDENT_ID = os.environ.get("DEFAULT_STUDENT_ID", "S123")

console = Console()


def _run_once(agent: Agent, message: str, trace_logger: TraceLogger, show_trace: bool) -> None:
    console.print()
    console.print(Text("you  ", style="bold cyan"), end="")
    console.print(message)
    console.print(Text("coach", style="bold yellow"), end="  ")

    started_text = False
    for event in agent.run(message, trace=trace_logger):
        if event.kind == "text_delta":
            if not started_text:
                started_text = True
            console.print(event.data["text"], end="", markup=False, highlight=False, soft_wrap=True)
        elif event.kind == "tool_use" and show_trace:
            console.print()
            console.print(
                Text(f"  ↳ tool: {event.data['name']}", style="dim"),
                Text(f" {event.data['input']}", style="dim italic"),
            )
        elif event.kind == "tool_result" and show_trace:
            preview = str(event.data["result"])[:180]
            console.print(Text(f"  ← {preview}", style="dim"))
        elif event.kind == "done":
            console.print()
            citations = event.data.get("citations", [])
            if citations:
                console.print(Text(f"  sources: {', '.join(citations)}", style="dim"))
            usage = event.data.get("usage", {})
            if show_trace and usage:
                console.print(Text(f"  usage: {usage}", style="dim"))
        elif event.kind == "error":
            console.print()
            console.print(Text(f"[error] {event.data.get('message')}", style="red"))


def main() -> int:
    parser = argparse.ArgumentParser(prog="assistant-cli", description="Student Learning Assistant (CLI)")
    parser.add_argument("-q", "--query", help="Single-shot query; skip REPL.")
    parser.add_argument("--trace", action="store_true", help="Print tool calls and usage inline.")
    parser.add_argument(
        "--student", default=DEFAULT_STUDENT_ID,
        help=f"Student id to act on. Defaults to {DEFAULT_STUDENT_ID}. Use --list to see all.",
    )
    parser.add_argument("--list", action="store_true", help="List known students and exit.")
    args = parser.parse_args()

    load_dotenv()
    ensure_seeded()
    if args.list:
        for s in Store.list_students():
            console.print(f"  [cyan]{s['id']}[/]  {s['name']}  ({s['source']})")
        return 0
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY is not set. Create a .env from .env.example.[/]")
        return 1

    try:
        store = Store.open(args.student)
    except KeyError as exc:
        console.print(f"[red]{exc}[/]. Run with --list to see available student ids.")
        return 2
    retriever = HybridRetriever(store.materials)
    client = Anthropic()
    agent = Agent(client=client, data=store, retriever=retriever)
    trace = TraceLogger()
    store.start_session(trace.session_id)

    weak_labels = store.profile.get("weak_topics") or [r["topic"] for r in store.performance.get("topic_performance", [])[:3]]
    console.print(Panel.fit(
        Text.from_markup(
            f"[bold]Student Learning Assistant[/]\n"
            f"Student: [bold]{store.profile['name']}[/] ({store.profile['source']})"
            f"{' · grade ' + str(store.profile['grade']) if store.profile['grade'] else ''}"
            f"{' · ' + store.profile['board'] if store.profile.get('board') else ''}\n"
            f"Weak: [red]{', '.join(weak_labels[:3])}[/]\n"
            f"Daily budget: {store.profile['daily_study_time_minutes']} min  ·  trace: [cyan]{trace.path}[/]\n"
            f"Type your question, or [bold]/exit[/] to quit."
        ),
        border_style="yellow",
    ))

    if args.query:
        _run_once(agent, args.query, trace, args.trace)
        console.print()
        return 0

    while True:
        try:
            msg = console.input("\n[bold cyan]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not msg:
            continue
        if msg in {"/exit", "/quit", ":q"}:
            return 0
        _run_once(agent, msg, trace, args.trace)


if __name__ == "__main__":
    sys.exit(main())
