"""FastAPI app: streaming chat UI + SSE endpoint + student switching.

Routes:
  GET  /                            static single-page chat UI
  GET  /api/students                list all students in the DB
  GET  /api/student?id=S123         snapshot for one student (defaults to
                                    DEFAULT_STUDENT_ID)
  GET  /api/materials/{id}          material detail (for UI citation tooltips)
  POST /api/chat                    streaming SSE endpoint. Body may include
                                    `student_id` to switch context.

The SSE protocol is one JSON object per `data:` line, with event kinds:
  session | text | tool_use | tool_result | done | error
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import Agent
from .retrieval import HybridRetriever
from .store import Store, ensure_seeded
from .tracing import TraceLogger

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_STUDENT_ID = os.environ.get("DEFAULT_STUDENT_ID", "S123")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    student_id: str | None = None
    # Optional: resume an existing session so messages persist into the same
    # `sessions` row instead of creating a new one. The sidebar uses this
    # when the user clicks a past conversation.
    session_id: str | None = None


def _materials_index(store: Store) -> dict:
    return {
        m.material_id: m.to_dict() | {"description": m.description}
        for m in store.materials
    }


def create_app() -> FastAPI:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Create a .env from .env.example before starting the server."
        )

    app = FastAPI(title="Student Learning Assistant")

    # One-time DB seed at startup. Idempotent.
    ensure_seeded()

    client = AsyncAnthropic()

    # Retrievers + materials indexes are per-student so students with
    # different catalogues don't leak into each other's searches. We cache
    # them to amortise the TF-IDF + BM25 index build across requests.
    _retrievers: dict[str, HybridRetriever] = {}
    _materials_by_id: dict[str, dict] = {}

    def _load(student_id: str) -> tuple[Store, HybridRetriever, dict]:
        store = Store.open(student_id)
        retriever = _retrievers.get(student_id)
        if retriever is None:
            retriever = HybridRetriever(store.materials)
            _retrievers[student_id] = retriever
            _materials_by_id[student_id] = _materials_index(store)
        return store, retriever, _materials_by_id[student_id]

    @app.get("/api/students")
    def list_students() -> dict:
        return {"students": Store.list_students(), "default": DEFAULT_STUDENT_ID}

    @app.get("/api/student")
    def student_snapshot(id: str = Query(default=DEFAULT_STUDENT_ID)) -> JSONResponse:
        try:
            store = Store.open(id)
        except KeyError:
            return JSONResponse({"error": f"unknown student: {id}"}, status_code=404)
        return JSONResponse({
            "profile": store.profile,
            "subject_performance": store.performance.get("subject_performance", []),
            "topic_performance": store.performance.get("topic_performance", []),
            "upcoming_tests": store.tests,
        })

    @app.get("/api/sessions")
    def list_sessions(
        student_id: str = Query(default=DEFAULT_STUDENT_ID),
        limit: int = Query(default=30, ge=1, le=100),
    ) -> JSONResponse:
        try:
            store = Store.open(student_id)
        except KeyError:
            return JSONResponse({"error": f"unknown student: {student_id}"}, status_code=404)
        return JSONResponse({"sessions": store.recent_sessions(limit=limit)})

    @app.get("/api/sessions/{session_id}")
    def get_session(
        session_id: str,
        student_id: str = Query(default=DEFAULT_STUDENT_ID),
    ) -> JSONResponse:
        try:
            store = Store.open(student_id)
        except KeyError:
            return JSONResponse({"error": f"unknown student: {student_id}"}, status_code=404)
        msgs = store.session_messages(session_id)
        if msgs is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"session_id": session_id, "messages": msgs})

    @app.delete("/api/sessions/{session_id}")
    def delete_session(
        session_id: str,
        student_id: str = Query(default=DEFAULT_STUDENT_ID),
    ) -> JSONResponse:
        try:
            store = Store.open(student_id)
        except KeyError:
            return JSONResponse({"error": f"unknown student: {student_id}"}, status_code=404)
        if not store.delete_session(session_id):
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"ok": True})

    @app.get("/api/materials/{material_id}")
    def material_detail(material_id: str) -> JSONResponse:
        # Check across all cached students; material IDs are unique.
        for idx in _materials_by_id.values():
            if material_id in idx:
                return JSONResponse(idx[material_id])
        return JSONResponse({"error": "not found"}, status_code=404)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        student_id = req.student_id or DEFAULT_STUDENT_ID
        try:
            store, retriever, materials_by_id = _load(student_id)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

        agent = Agent(client=client, data=store, retriever=retriever)
        trace = TraceLogger()
        # Reuse an existing session row when the client is resuming a past
        # conversation; otherwise start a fresh one. We use the trace's
        # session_id as the key in the latter case so the on-disk JSONL
        # trace and the DB row share an identifier.
        session_id = req.session_id or trace.session_id
        store.start_session(session_id)
        history = [{"role": m.role, "content": m.content} for m in req.history]

        async def event_stream() -> AsyncIterator[bytes]:
            yield _sse({"kind": "session", "session_id": session_id, "student_id": student_id})
            store.log_message(session_id, "user", req.message)
            assistant_parts: list[str] = []
            try:
                async for event in agent.run_async(req.message, history=history, trace=trace):
                    if event.kind == "text_delta":
                        assistant_parts.append(event.data.get("text", ""))
                    yield _sse(_event_to_sse(event, materials_by_id))
            except Exception as exc:  # pragma: no cover - surfaces to client
                trace.error("server", f"{type(exc).__name__}: {exc}")
                yield _sse({"kind": "error", "message": str(exc)})
            finally:
                # Persist the assistant turn too so the sidebar can replay
                # the full conversation, not just the user side.
                final_text = "".join(assistant_parts).strip()
                if final_text:
                    store.log_message(session_id, "assistant", final_text)

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, default=str)}\n\n".encode("utf-8")


def _event_to_sse(event, materials_by_id: dict) -> dict:
    kind = event.kind
    if kind == "text_delta":
        return {"kind": "text", "delta": event.data["text"]}
    if kind == "tool_use":
        return {"kind": "tool_use", "name": event.data["name"], "input": event.data["input"]}
    if kind == "tool_result":
        return {
            "kind": "tool_result",
            "name": event.data["name"],
            "summary": _summarise_result(event.data["name"], event.data["result"]),
        }
    if kind == "done":
        citations = [
            {"material_id": mid, **materials_by_id.get(mid, {})}
            for mid in event.data.get("citations", [])
        ]
        return {"kind": "done", "citations": citations, "usage": event.data.get("usage", {})}
    if kind == "error":
        return {"kind": "error", "message": event.data.get("message", "unknown error")}
    return {"kind": kind, "data": event.data}


def _summarise_result(name: str, result: Any) -> dict:
    if not isinstance(result, dict):
        return {"preview": str(result)[:200]}
    if name == "recommend_study_material":
        return {
            "query": result.get("query"),
            "count": len(result.get("results", [])),
            "material_ids": [r.get("material_id") for r in result.get("results", [])],
        }
    if name == "get_upcoming_tests":
        return {
            "count": len(result.get("tests", [])),
            "next": result.get("tests", [{}])[0] if result.get("tests") else None,
        }
    if name == "get_weak_topics":
        return {
            "count": len(result.get("weak_topics", [])),
            "topics": [r["topic"] for r in result.get("weak_topics", [])],
            "source": result.get("source"),
        }
    if name == "plan_study_week":
        return {
            "topics": [r["topic"] for r in result.get("plan", [])],
            "daily_minutes": result.get("daily_minutes"),
        }
    return {"keys": list(result.keys())}


def run() -> None:
    import uvicorn

    uvicorn.run(
        "assistant.server:create_app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        factory=True,
        reload=bool(os.environ.get("RELOAD")),
    )


if __name__ == "__main__":
    run()
