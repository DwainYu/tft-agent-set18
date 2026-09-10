"""Structured traces: without them a failed run is unreproducible.

Every event becomes one JSON line. Reading the file back is how you answer the
only question that matters during debugging: what did the model actually see?
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

_counter = itertools.count(1)


class Trace:
    """Appends events to a JSONL file and keeps them in memory for printing."""

    def __init__(self, path: Optional[Union[str, Path]] = None, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{next(_counter)}"
        self.path = Path(path) if path else None
        self.events: Dict[str, Any] = {"run_id": self.run_id, "started_at": time.time(), "events": []}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **payload: Any) -> None:
        record = {"at": round(time.time() - self.events["started_at"], 3), "kind": kind, **payload}
        self.events["events"].append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self, **summary: Any) -> Dict[str, Any]:
        self.events["summary"] = summary
        self.events["duration_seconds"] = round(time.time() - self.events["started_at"], 3)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "finish", **self.events["summary"]}, ensure_ascii=False) + "\n")
        return self.events

    def timeline(self) -> str:
        rows = []
        for record in self.events["events"]:
            label = record["kind"]
            detail = record.get("content") or record.get("result") or record.get("error") or record.get("message") or ""
            name = record.get("name")
            rows.append(f"{record['at']:>7.3f}s {label:<16} {(name + ' ') if name else ''}{detail}".rstrip())
        return "\n".join(rows)

    def handler(self):  # noqa: ANN201 - returns the Agent's on_event callable
        def on_event(kind: str, payload: Dict[str, Any]) -> None:
            self.event(kind, **payload)

        return on_event
