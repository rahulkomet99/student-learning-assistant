"""JSONL trace logger for agent turns and tool calls.

One file per session (timestamp-named). Each line is a JSON object with a `kind`
discriminator, so downstream analysis / eval can just stream-read.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_DIR = Path(__file__).resolve().parent.parent.parent / "traces"


class TraceLogger:
    def __init__(self, session_id: str | None = None, trace_dir: Path = TRACE_DIR):
        trace_dir.mkdir(parents=True, exist_ok=True)
        stamp = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = trace_dir / f"{stamp}.jsonl"
        self.session_id = stamp
        self._start = time.time()

    def log(self, kind: str, **fields: Any) -> None:
        record = {
            "ts": round(time.time() - self._start, 3),
            "kind": kind,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # Convenience wrappers --------------------------------------------------

    def user_message(self, text: str) -> None:
        self.log("user_message", text=text)

    def tool_use(self, name: str, inputs: dict) -> None:
        self.log("tool_use", name=name, inputs=inputs)

    def tool_result(self, name: str, result: dict) -> None:
        self.log("tool_result", name=name, result=result)

    def assistant_message(self, text: str, usage: dict | None = None) -> None:
        self.log("assistant_message", text=text, usage=usage or {})

    def error(self, where: str, message: str) -> None:
        self.log("error", where=where, message=message)
