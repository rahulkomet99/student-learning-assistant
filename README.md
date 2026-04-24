# Student Learning Assistant

[![tests](https://github.com/rahulkomet99/student-learning-assistant/actions/workflows/test.yml/badge.svg)](https://github.com/rahulkomet99/student-learning-assistant/actions/workflows/test.yml)

An AI study coach that answers a student's questions — *"what should I study this week?"*, *"I'm weak in Algebra, what next?"* — grounded in their profile, attempt-level performance, upcoming tests, and study library.

One FastAPI process. One SQLite file. Streaming chat UI with math rendering, citations, and an LLM-as-judge eval harness.

## Demo

<!-- Record a ~30s screencast and drop it at docs/demo.gif -->
![demo](docs/demo.gif)

Live captures of the assignment's 4 sample queries (tools called, citations, final answers) → [`evals/assignment_queries.md`](evals/assignment_queries.md).

## What the brief asked for — and where to look

The assignment asks for four things. All four are present; this section tells you where.

| Requirement | Where it lives |
|---|---|
| **Retrieval** (embeddings / semantic search or structured filtering) | Both. Hybrid BM25 + TF-IDF with RRF fusion in [`retrieval.py`](src/assistant/retrieval.py); structured SQL filtering in [`tools.py`](src/assistant/tools.py). |
| **Tool usage** (at least one of `get_weak_topics`, `get_upcoming_tests`, `recommend_study_material`) | All four, plus `plan_study_week` as a composer — see [`tools.py`](src/assistant/tools.py). |
| **Context-aware responses** | Each tool reads a specific student's profile, performance, and tests from SQLite; the system prompt also pins a per-student snapshot with prompt caching — see [`agent.py`](src/assistant/agent.py). |
| **Simple API / CLI / UI** | FastAPI SSE API + a minimal chat UI + a CLI REPL, one uvicorn process — see [`server.py`](src/assistant/server.py) and [`cli.py`](src/assistant/cli.py). |

Everything beyond that list — multi-student store, Wilson-lower modelling, prereq graph, guardrails, evals, CI — is ***intentional over-scope*** to show architectural reasoning. If you want to verify just the brief, run `python -m evals.capture_assignment_responses` and read the generated markdown.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env      # add ANTHROPIC_API_KEY

assistant-server          # open http://localhost:8000
assistant-cli --list      # list seeded students
assistant-cli -q "What should I study this week?" --trace
```

First boot creates `data/db.sqlite3` and seeds 9 students (CBSE, JEE, TOEIC) + ~30 topics + prereq graph + study materials. Delete the DB to re-seed.

## What's in it

- **Tool-using agent** (Claude Opus 4.7) with 4 tools: `get_weak_topics`, `get_upcoming_tests`, `recommend_study_material`, `plan_study_week`.
- **Hybrid retrieval**: BM25 + TF-IDF fused via Reciprocal Rank Fusion, in-house numpy (no PyTorch).
- **Dynamic student modelling**: weakness ranked by Wilson lower-bound over recent attempts, falling back to declared labels.
- **Curriculum-aware planning**: a topic-prerequisite graph lets the planner refuse to recommend Quadratics when Algebra is still weak.
- **Streaming SSE + async agent loop**: tokens flush token-by-token; tool calls render as inline status lines.
- **Markdown rendering** via `marked` + `DOMPurify`, with custom extensions for citation pills (`[M103]`) and KaTeX math (`$...$`, `$$...$$`).
- **Guardrails**: prompt-injection defence (user input XML-tagged), citation audit in the trace log.
- **Prompt caching** on the system prompt + per-student snapshot.
- **Eval harness** with golden queries + LLM-as-judge rubric scoring.
- **9 seeded students** across grades 9–12 (CBSE school + JEE prep) and TOEIC prep.
- **Chat UX**: student switcher, subject-aware suggestions, copy/regenerate/stop on every response, pill composer.

No PyTorch, no vector DB, no Node, no build step. Install size < 50 MB.

## Architecture

```
                    ┌──────────────────────────────────────────┐
  browser chat UI   │  index.html + styles.css + app.js         │
  ──── SSE chat ────│  marked + DOMPurify + KaTeX               │
                    │  student switcher · copy · regenerate     │
                    └──────────────────┬───────────────────────┘
                                       │
                     ┌─────────────────▼────────────────────────┐
                     │  FastAPI (server.py)                      │
                     │   POST /api/chat       SSE stream         │
                     │   GET  /api/students   switcher data      │
                     │   GET  /api/student    snapshot           │
                     │   GET  /healthz                          │
                     └─────────────────┬────────────────────────┘
                                       │
                   ┌───────────────────▼───────────────────────┐
                   │  Agent (agent.py)                          │
                   │   AsyncAnthropic streaming (async gen)     │
                   │   Prompt caching · citation audit          │
                   │   User content wrapped <user_message>…     │
                   │   Trace JSONL per session                  │
                   └──┬──────────────────────────────────┬─────┘
                      │                                  │
       ┌──────────────▼──────────┐       ┌───────────────▼──────────┐
       │  tools.py (4 tools)     │──────▶│  retrieval.py            │
       │   get_weak_topics       │       │   HybridRetriever         │
       │   get_upcoming_tests    │       │    BM25 Okapi             │
       │   recommend_study_mat.  │       │    TF-IDF + cosine        │
       │   plan_study_week       │       │    RRF fusion             │
       └──┬──────────────────────┘       └──────────────────────────┘
          │         modelling               store            seed
          ▼           ▼                      ▼                 ▼
       ┌────────────────────┐     ┌──────────────────┐   ┌──────────────────┐
       │ modeling.py        │     │ store.py         │   │ seed.py          │
       │ Wilson-lower rank  │◀────│ Store.open(sid)  │◀──│ seed_cbse()      │
       │ weak_prereqs_for() │     │ aggregates attmpts│   │ seed_ednet()     │
       └────────────────────┘     └────────┬─────────┘   │ seed_extra()     │
                                           │             │ seed_prereqs()   │
                                  ┌────────▼──────────┐  └──────────────────┘
                                  │ SQLite schema     │
                                  │  students         │
                                  │  topics + prereqs │
                                  │  attempts         │
                                  │  materials        │
                                  │  tests            │
                                  │  sessions + msgs  │
                                  └───────────────────┘
```

## Key decisions

- **SQLite, not JSON** — real relational store; bundled JSON (CBSE) and EdNet-style synthetic data both seed into it on first boot. Drop real EdNet KT1 CSVs into `data/ednet_raw/` and they override the sample.
- **Async everywhere on the server path** — `AsyncAnthropic` + async generator so SSE deltas actually flush. The sync agent is kept for the CLI.
- **Hybrid BM25 + TF-IDF** — dropped `sentence-transformers` (pulls ~500 MB of torch) for a ~30-line numpy TF-IDF. At this corpus size, recall is indistinguishable; upgrade path is a one-class swap behind `HybridRetriever`.
- **4 composable tools** — kept small and independent so the agent's tool trace is itself a debugging artifact. `plan_study_week` is the higher-level composer.
- **Wilson-lower, not raw accuracy** — a student with 2 / 3 correct is not 66 % strong. Conservative estimates tighten as evidence accumulates.
- **Prerequisite graph** — ~25 hand-curated edges across CBSE / JEE / TOEIC. A topic's weak prereqs get attached to its plan entry so the agent can recommend shoring up fundamentals first.
- **Prompt-injection defence at the agent layer** — every user turn is wrapped in `<user_message>…</user_message>`. The system prompt treats that content as data.
- **Marked + DOMPurify for rendering**, with custom extensions for citations and math — `renderMarkdown` is effectively one line.
- **Judge model ≠ agent model** — agent runs on Opus 4.7; the LLM-as-judge runs on Haiku 4.5 to reduce same-family scoring bias.

## Student modelling

```
For each student × topic:
    up to N = 30 most recent attempts
    acc_lower = Wilson(correct, total, z = 1.96)

Rank weak-first by acc_lower when attempts ≥ 5;
otherwise fall back to declared 'weak'/'strong' labels.
```

Example (from the seeded data): Ananya's Trigonometry, Mensuration, and Coordinate Geometry are all weak, but they share a weaker foundation (Geometry) — the plan flags Geometry as the weak prerequisite under each.

## Guardrails

- **Prompt-injection defence**: user input wrapped in `<user_message>` XML tags; system prompt instructs the model to treat that content as data and to ignore any override attempts inside. Literal `</user_message>` in user text is neutralised.
- **Citation audit**: after the agent finishes, we scan for `[M###]` tokens against the material_ids the tools returned. If anything was retrieved and nothing was cited, a `guardrail_warning` event lands in the trace JSONL. No silent retries.
- **Bounded loop**: `MAX_TURNS = 6`. Tool results trimmed to ~200-char summaries on the SSE wire; full payloads live in the trace.

## Evals

```bash
python -m evals.run_evals                        # full run (LLM judge)
python -m evals.run_evals --no-judge             # tool-coverage only
python -m evals.capture_assignment_responses     # capture the 4 assignment queries
```

`evals/golden.jsonl` has 8 cases. Each run does a hard tool-coverage check plus Claude-as-judge scoring against a domain-specific rubric (budget respected? top priority justified? citations surfaced?).

## Data

| Table | Purpose |
|---|---|
| `students` | Profile (name, grade, board, target_exam, daily budget, source) |
| `topics` | Topic catalogue with subject |
| `student_topic_label` | Declared weak/strong labels (fallback when attempts sparse) |
| `attempts` | One row per answered question — the grain modelling works over |
| `materials` | Study library (title, type, difficulty, description) |
| `tests` + `test_topics` | Upcoming tests and their topic coverage |
| `topic_prerequisites` | Hand-curated *"A needs B"* graph |
| `sessions` + `messages` | Conversation persistence per student |

## Limitations

- **No auth** — student switcher goes by id; fine for a demo, not prod.
- **Single box** — stateless agent, stateful SQLite. Scaling is a Postgres swap.
- **No retry** on Anthropic 5xx / timeouts.
- **Judge is still one LLM** — Haiku-on-Opus reduces family bias, but a fully rigorous harness needs human-labelled goldens or multi-vendor judging.
- **EdNet sample is synthetic** — the real dataset is license-gated. The CSV loader is the real asset.

## Next improvements

1. Auth + per-student isolation at the Store layer.
2. Spaced-repetition scheduling inside `plan_study_week` (SM-2-style revisits).
3. Haiku routing for simple queries; Sonnet/Opus for composite planning.
4. Retry + timeout middleware on the Anthropic client.
5. OpenTelemetry spans around each tool call + agent turn.
6. Grow the eval set to ~50 cases with per-criterion histograms.
7. Structured citation output via Claude's JSON mode so the UI renders authoritatively.
