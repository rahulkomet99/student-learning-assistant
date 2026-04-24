# Student Learning Assistant

An AI study coach that answers a student's questions like "what should I study this week?" or
"I'm weak in Algebra, what next?" — grounded in their **actual profile, attempt-level performance,
upcoming tests and study-material library**, held in a real relational store.

One FastAPI process. One SQLite file. A streaming chat UI with student switching, tool-call
indicators, math rendering (KaTeX), citations, and an LLM-as-judge eval harness.

---

## TL;DR

- **One command, one server, one browser tab.** FastAPI serves the UI + the SSE API from the same process.
- **Real RAG:** hybrid retrieval (BM25 + TF-IDF, fused via Reciprocal Rank Fusion) with citations.
- **Tool-augmented agent** with 4 tools (not just a prompt). Claude streams tokens and calls tools until it answers.
- **Real relational data** — SQLite schema with students / topics / attempts / materials / tests / sessions.
- **Student modelling** — weak topics are inferred dynamically from attempt history using the Wilson-lower bound; declared labels are a fallback.
- **Works on EdNet-format data.** Bundled sample covers two TOEIC students alongside the CBSE student from the assignment. Drop real EdNet KT1 CSVs into `data/ednet_raw/` and they ingest on first boot.
- **Prod hygiene:** prompt-injection defence, citation auditing, prompt caching, session logging, `/healthz`, and an eval harness.
- **No heavy deps.** No PyTorch, no vector DB, no Node, no build step. Total install < 50 MB.

---

## Setup

```bash
cd student-learning-assistant

python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# set ANTHROPIC_API_KEY=sk-ant-...

# Run the server + UI:
assistant-server
# open http://localhost:8000

# Or use the CLI:
assistant-cli --list                     # see all students
assistant-cli --student S123 -q "What should I study this week?" --trace
assistant-cli --student EN-u001           # TOEIC student REPL

# Run the eval harness:
python -m evals.run_evals
```

Python 3.10+. On first run the server creates `data/db.sqlite3` and seeds it from the bundled JSON (CBSE) + the EdNet-schema sample in `data/ednet_sample/`. Delete the DB file to re-seed.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
  browser chat UI   │  index.html + styles.css + app.js         │
  ─── SSE /api/chat │  KaTeX-rendered math · tool chips ·        │
                    │  citation pills · student switcher         │
                    └──────────────────┬───────────────────────┘
                                       │
                     ┌─────────────────▼────────────────────────┐
                     │  FastAPI (server.py)                      │
                     │   POST /api/chat      SSE stream          │
                     │   GET  /api/students  switcher data       │
                     │   GET  /api/student   snapshot            │
                     │   GET  /healthz                          │
                     └─────────────────┬────────────────────────┘
                                       │
                   ┌───────────────────▼───────────────────────┐
                   │  Agent (agent.py)                          │
                   │   - AsyncAnthropic streaming (async gen)   │
                   │   - Prompt caching on system + snapshot    │
                   │   - User content wrapped in <user_message> │
                   │   - Citation audit post-generation         │
                   │   - JSONL trace per session                │
                   └──┬──────────────────────────────────┬─────┘
                      │                                  │
       ┌──────────────▼──────────┐       ┌───────────────▼──────────┐
       │  tools.py (4 tools)     │       │  retrieval.py            │
       │   get_weak_topics       │       │   HybridRetriever         │
       │   get_upcoming_tests    │──────▶│    BM25 Okapi             │
       │   recommend_study_mat.  │       │    TF-IDF + cosine        │
       │   plan_study_week       │       │    RRF fusion             │
       └──┬──────────────────────┘       └──────────────────────────┘
          │
          │   modelling                store                seed
          ▼                            ▼                    ▼
  ┌────────────────────┐     ┌──────────────────┐   ┌──────────────────┐
  │ modeling.py        │     │ store.py         │   │ seed.py          │
  │ compute_signals()  │     │ Store.open(sid)  │   │ seed_cbse()      │
  │ rank_weak/strong() │◀────│ reads SQLite     │◀──│ seed_ednet(...)  │
  │ Wilson-lower bound │     │ aggregates attmpts│   │ generates sample │
  └────────────────────┘     └────────┬─────────┘   └──────────────────┘
                                      │
                      ┌───────────────▼────────────────┐
                      │  SQLite (schema.py)            │
                      │    students  · topics          │
                      │    attempts  · materials       │
                      │    tests     · test_topics     │
                      │    sessions  · messages        │
                      └────────────────────────────────┘
```

---

## How a query flows through the system

Example: **"What should I study this week?"** as Priya (TOEIC prep).

1. **UI → API.** Browser POSTs `{message, history, student_id: "EN-u001"}` to `/api/chat`. Server opens an SSE stream.
2. **Agent run.** The user's message is wrapped in `<user_message>` and placed after a cached system prompt that includes Priya's full profile snapshot. Four tool schemas are exposed.
3. **Claude picks tools.** For a weekly-plan question it calls `plan_study_week` (which itself composes `get_weak_topics` + `get_upcoming_tests` internally).
4. **Tools execute locally against SQLite.** `plan_study_week`:
   - Pulls Priya's weak topics from the **modelling layer** — Wilson-lower-bound of recent accuracy on the `attempts` table. Priya comes out weak on "Vocabulary - Business Idioms" (8%) and "Reading - Inference Questions" (19%).
   - Computes test urgency for the TOEIC mock in 6 days.
   - Scores each topic `priority = weakness × (1 + 2·urgency)`.
   - Allocates Priya's 60-minute daily budget proportionally, rounded to 5-min blocks.
   - Attaches the top-retrieved material per topic (hybrid BM25 + TF-IDF, filtered by topic).
5. **UI shows tool chips in real time** (`Looking up upcoming tests…` → `Fetched upcoming tests (1)`).
6. **Final text streams token-by-token.** Math expressions render as KaTeX. Citations like `[M204]` render as pills with hover tooltips showing the material title + topic.
7. **Session logged** to SQLite (`sessions` + `messages` tables) and to `traces/<timestamp>.jsonl`. If the model returned material_ids but didn't cite them, a `guardrail_warning` event lands in the trace.

---

## Student modelling

The original JSON profile had a static `weak_topics` list. That doesn't survive contact with real data, where weakness is a rolling function of recent accuracy.

```
For each student × topic:
    take up to N most recent attempts (default N=30)
    acc_mean  = correct / total                      # simple average
    acc_lower = Wilson(correct, total, z=1.96)       # lower 95% CI bound

Rank weak-first by acc_lower when attempts ≥ 5; otherwise fall back to the
student's declared 'weak' / 'strong' labels.
```

Why Wilson's lower bound? A student with 0/2 wrong isn't 0% weak — they have one data point. Wilson gives us a conservative estimate that tightens as evidence accumulates. 15/15 correct reads as ~80% lower-bound, not 100%.

This is why the same tool works against the CBSE student (with synthesised attempts consistent with his declared labels) and the EdNet students (with real-format attempt logs and no declared labels).

---

## Data layer

SQLite, single file, auto-created on first run. Seeded from:

1. **Bundled CBSE JSON** (`data/student_profile.json`, etc.) — one student, Arjun, grade 10. We synthesise ~15 attempts per declared topic so the modelling layer has real data to aggregate over; accuracies (~40% weak, ~85% strong) stay consistent with the declared score numbers.
2. **Bundled EdNet-schema sample** (`data/ednet_sample/`) — two TOEIC-prep students, 60 attempts each, two synthetic upcoming mock tests, and a practice-pack material per tag. This is structurally identical to EdNet KT1; the numbers are generated so CI runs offline.
3. **Optional: real EdNet KT1** — drop your licensed EdNet CSVs into `data/ednet_raw/` in the layout documented in `seed.py` (`contents/questions.csv`, `contents/tags.csv`, `users/<id>.csv`, `students.csv`, `tests.csv`, `materials.csv`). They take precedence over the synthetic sample on the next cold-boot.

Schema:

| Table | Purpose |
|---|---|
| `students` | Profile — id, name, grade, board, target_exam, daily budget, `source` ('cbse' \| 'ednet' \| …) |
| `topics` | Topic catalogue with optional subject |
| `student_topic_label` | Declared weak/strong labels — fallback when attempts are sparse |
| `attempts` | One row per answered question — the grain modelling works over |
| `materials` | Study library — title, content_type, difficulty, description |
| `tests` + `test_topics` | Upcoming tests and their topic coverage |
| `sessions` + `messages` | Conversation persistence per student |

---

## Guardrails

**Prompt-injection defence.** Every user message is wrapped in `<user_message>…</user_message>` before it reaches Claude. The system prompt explicitly instructs Claude to treat that content as data, not as instructions. Literal `</user_message>` tags in user input are neutralised so the student can't close the tag and escape.

**Citation audit.** After the agent produces its final text, we check whether any tool returned `material_id` values. If yes and the response cites none of them, a structured `guardrail_warning` event with `kind: missing_citation` lands in the trace JSONL. No auto-retry (silent retries hide model behaviour from evals); the signal is visible to whoever reads the trace.

**Blast-radius limits.** The agent loop is bounded by `MAX_TURNS=6`. Tool results cap at ~200 chars in SSE summaries; full payloads live in the trace.

---

## Evals

`evals/golden.jsonl` holds 8 cases. For each:

1. Run the agent end-to-end; collect `(tools_called, citations, final_text)`.
2. **Hard check:** tool expectations — e.g. "must have called `get_upcoming_tests`".
3. **LLM-as-judge:** Claude rates the response 1-5 against a rubric of domain-specific criteria ("does it respect the 90-minute budget?", "does it pick one top priority?", etc.).

```bash
python -m evals.run_evals                      # full run (judge included)
python -m evals.run_evals --no-judge           # tool checks only (fast)
python -m evals.run_evals --id algebra-weak    # one case
```

---

## Key decisions & tradeoffs

### Hybrid TF-IDF + BM25 (no transformer embeddings)

| | sentence-transformers | TF-IDF + BM25 (chosen) |
|---|---|---|
| Install size | ~500 MB (pulls PyTorch) | ~30 MB |
| First-run latency | 1-3 s to load model | < 50 ms |
| Recall on this corpus | Marginally better | Indistinguishable |
| Upgrade path | — | `TfidfIndex` swaps for an embedding backend; RRF stays |

For a ~20-document corpus, dense embeddings are overkill. Retrieval is isolated behind `HybridRetriever`, so swapping in sentence-transformers or Voyage later is ~20 lines.

### Async streaming, not sync

The agent exposes both `run()` (sync, for the CLI) and `run_async()` (async, used by the FastAPI SSE endpoint). The first iteration used sync inside async and had tokens batch at the end — an easy trap. The fix was to use `AsyncAnthropic` throughout the server path so each text delta flushes to the socket immediately. Headers `X-Accel-Buffering: no` + `Cache-Control: no-cache, no-transform` are set on the SSE response so any upstream proxy behaves.

### Prompt caching

System instructions + the student snapshot are both cache-anchored (`cache_control: ephemeral`). Multi-turn sessions hit the cache instead of reprocessing profile JSON every turn. Small win on single-turn demos; real win in production.

### Modelling vs declared labels

The weakness signal comes from attempts when available, and falls back to declared labels when not. This means:
- The same codepath works for real EdNet users (no declared labels, lots of attempts).
- The CBSE student still reads sensibly on day 1 (declared labels, before any real practice is logged).
- A student with 3 attempts isn't labelled 100% strong on a topic.

---

## Project layout

```
student-learning-assistant/
├── data/
│   ├── student_profile.json         # CBSE seed (as given in the assignment)
│   ├── performance_history.json
│   ├── study_materials.json
│   ├── upcoming_tests.json
│   ├── ednet_sample/                # bundled EdNet-schema sample (2 users)
│   ├── ednet_raw/                   # optional: drop real EdNet CSVs here
│   └── db.sqlite3                   # auto-created; gitignored
├── src/assistant/
│   ├── agent.py                     # streaming + sync tool-use loop, guardrails
│   ├── server.py                    # FastAPI app (UI + SSE /chat + student switch)
│   ├── cli.py                       # REPL / one-shot query; respects --student
│   ├── tools.py                     # 4 tool schemas + handlers
│   ├── retrieval.py                 # in-house BM25 + TF-IDF + RRF
│   ├── modeling.py                  # Wilson-lower weakness ranking
│   ├── store.py                     # SQLite-backed data access per student
│   ├── schema.py                    # schema DDL + init
│   ├── seed.py                      # CBSE + EdNet seeders
│   ├── tracing.py                   # JSONL session logger
│   ├── data.py                      # legacy JSON loader (still used by data.Material)
│   └── static/                      # index.html · styles.css · app.js
├── evals/
│   ├── golden.jsonl                 # 8 graded queries
│   └── run_evals.py                 # harness + LLM-as-judge
├── tests/
│   └── test_tools.py                # 10 tests: retrieval, modelling, tools, store
├── traces/                          # session JSONL logs (git-ignored)
├── pyproject.toml
└── README.md
```

---

## Limitations (known)

- **No auth.** Student switching goes by id; anyone on the box can query any student. Next step is magic-link email login with per-student data isolation enforced at the Store layer.
- **Single server, single box.** Stateless agent, stateful SQLite. Scaling reads is one Postgres swap away; scaling writes needs work.
- **No retries on Anthropic errors.** 5xx or timeouts bubble to the user as `error` events. Needs exponential-backoff retry middleware.
- **Eval judge is Claude-on-Claude.** A single-model judge is cheap but biased. A different model (or labelled human data) would be stronger.
- **EdNet sample is generated.** Real EdNet is license-gated and not committed. The loader is the real asset — point it at licensed CSVs and it ingests on first boot.
- **No rate limiting.** Fine for a demo, not for multi-user prod.

---

## Next improvements (in priority order)

1. **Auth + per-student isolation** enforced at the Store layer (not just at the UI).
2. **Spaced repetition** in `plan_study_week` — interleave revisits of recently-learned topics on an SM-2 schedule.
3. **Haiku routing.** Simple queries ("what are my weak areas") don't need Sonnet; route to Haiku 4.5 for 10× cheaper, 3× faster.
4. **Retry + timeout middleware** around the Anthropic client.
5. **OpenTelemetry spans** around each tool call and agent turn; wire to Jaeger for local debugging.
6. **Eval scale + dashboard.** Grow the golden set to ~50 cases, add per-criterion score histograms, run judge as a nightly job (too slow for every PR).
7. **Structured citation output.** Have the model emit `{"text": "...", "citations": [...]}` via Claude's structured-output APIs so the UI renders citations authoritatively instead of regex-matching.
8. **Content chunking for direct Q&A.** Right now each material is a row; chunking actual lesson content would let the assistant answer content questions ("what is the discriminant?") directly, not just point to materials.
