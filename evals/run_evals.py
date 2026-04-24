"""Eval harness for the Student Learning Assistant.

For each golden case we:
  1. Run the agent end-to-end, collecting (tools_called, citations, final_text).
  2. Check hard assertions: expected_tools_any_of, must_not_tools.
  3. Run an LLM-as-judge over the rubric, with the final text + tool trace as
     context. Judge returns a 1-5 score per rubric bullet plus a verdict.
  4. Aggregate: per-case pass/fail, avg rubric score, tool-coverage rate.

Run:
  python -m evals.run_evals                  # run all
  python -m evals.run_evals --id algebra-weak  # one case
  python -m evals.run_evals --limit 3        # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from assistant.agent import Agent  # noqa: E402
from assistant.data import load_data  # noqa: E402
from assistant.retrieval import HybridRetriever  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"
# Use a smaller / different-tier model as judge to avoid same-family bias
# (the agent runs on Opus 4.7, so we judge with Haiku 4.5).
JUDGE_MODEL = "claude-haiku-4-5-20251001"


JUDGE_SYSTEM = """You evaluate a student-learning-assistant agent.

Given a user query, the tools the agent called, the final answer, and a list of
rubric criteria, score each criterion on a 1-5 integer scale:
  5 = fully satisfied, unambiguously
  4 = mostly satisfied
  3 = partially satisfied
  2 = weakly attempted
  1 = absent or wrong

Return ONLY JSON on a single line, no prose, in the exact shape:
{"scores": [int, ...], "notes": "one short sentence per criterion, semicolon-separated"}

The `scores` array length must equal the number of rubric criteria provided.
"""


@dataclass
class RunResult:
    case_id: str
    query: str
    final_text: str
    tools_called: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    error: str | None = None


def run_agent(agent: Agent, query: str) -> RunResult:
    result = RunResult(case_id="", query=query, final_text="")
    text_parts: list[str] = []
    try:
        for event in agent.run(query):
            if event.kind == "text_delta":
                text_parts.append(event.data["text"])
            elif event.kind == "tool_use":
                result.tools_called.append(event.data["name"])
            elif event.kind == "done":
                result.citations = event.data.get("citations", [])
            elif event.kind == "error":
                result.error = event.data.get("message")
    except Exception as exc:  # pragma: no cover - defensive
        result.error = f"{type(exc).__name__}: {exc}"
    result.final_text = "".join(text_parts)
    return result


def judge(client: Anthropic, query: str, result: RunResult, rubric: list[str]) -> dict:
    rubric_block = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric))
    user_prompt = (
        f"Query: {query}\n\n"
        f"Tools called (in order): {result.tools_called or 'none'}\n"
        f"Citations: {result.citations or 'none'}\n\n"
        f"Final answer:\n---\n{result.final_text}\n---\n\n"
        f"Rubric ({len(rubric)} criteria):\n{rubric_block}"
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Judge went off-format — record raw for debugging.
        return {"scores": [0] * len(rubric), "notes": f"judge returned non-JSON: {text[:200]}"}


def check_tool_expectations(result: RunResult, case: dict) -> tuple[bool, str]:
    expected_any = set(case.get("expected_tools_any_of", []))
    must_not = set(case.get("must_not_tools", []))
    called = set(result.tools_called)
    if expected_any and not (called & expected_any):
        return False, f"expected any of {sorted(expected_any)}, called {sorted(called)}"
    if must_not & called:
        return False, f"must-not tools called: {sorted(must_not & called)}"
    return True, "ok"


def load_cases(id_filter: str | None = None) -> list[dict]:
    cases = []
    for line in GOLDEN_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        if id_filter and case["id"] != id_filter:
            continue
        cases.append(case)
    return cases


def print_case_report(case: dict, result: RunResult, ok: bool, reason: str, verdict: dict) -> None:
    scores = verdict.get("scores", [])
    avg = mean(scores) if scores else 0.0
    status = "PASS" if ok and avg >= 3.5 else ("SOFT" if ok else "FAIL")
    print(f"\n[{status}] {case['id']}  avg_rubric={avg:.2f}  tool_check={reason}")
    print(f"  query: {case['query']}")
    print(f"  tools: {result.tools_called or '[]'}  citations: {result.citations or '[]'}")
    if result.error:
        print(f"  error: {result.error}")
    for crit, score in zip(case["rubric"], scores):
        print(f"   - [{score}/5] {crit}")
    notes = verdict.get("notes")
    if notes:
        print(f"   notes: {notes}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_evals")
    parser.add_argument("--id", help="Run a single case by id")
    parser.add_argument("--limit", type=int, help="Run only the first N cases")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge (tool checks only)")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1

    data = load_data()
    retriever = HybridRetriever(data.materials)
    client = Anthropic()
    agent = Agent(client=client, data=data, retriever=retriever)

    cases = load_cases(args.id)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No matching cases.", file=sys.stderr)
        return 1

    tool_pass = 0
    rubric_scores: list[float] = []
    for case in cases:
        result = run_agent(agent, case["query"])
        result.case_id = case["id"]
        ok, reason = check_tool_expectations(result, case)
        if ok:
            tool_pass += 1
        verdict = (
            {"scores": [0] * len(case["rubric"]), "notes": "judge skipped"}
            if args.no_judge
            else judge(client, case["query"], result, case["rubric"])
        )
        if verdict.get("scores"):
            rubric_scores.append(mean(verdict["scores"]))
        print_case_report(case, result, ok, reason, verdict)

    print("\n" + "=" * 70)
    print(f"Cases run        : {len(cases)}")
    print(f"Tool-check pass  : {tool_pass}/{len(cases)}  ({100*tool_pass/len(cases):.0f}%)")
    if rubric_scores:
        print(f"Avg rubric score : {mean(rubric_scores):.2f}/5  across {len(rubric_scores)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
