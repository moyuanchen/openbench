"""Per-request metrics, written inside the serve container to the `lab-metrics` Volume.

One JSON line per completion request in <dir>/<model>/<UTC date>-<container>.jsonl. Each container writes its
own file, so containers never contend for the same file. `lab metrics` downloads and summarizes them.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class Recorder:
    def __init__(self, directory: str, spec: dict, commit, interval: float = 30.0):
        self.dir = Path(directory) / spec["name"]
        self.container = os.environ.get("MODAL_TASK_ID", "local")
        self.static = {
            "model": spec["name"],
            "backend": spec["backend"],
            "gpu": spec["modal_gpu"],
            "container": self.container,
        }
        self._commit = commit
        self._lock = threading.Lock()
        self._dirty = False
        self.in_flight = 0
        self.served = 0
        threading.Thread(target=self._commit_loop, args=(interval,), daemon=True).start()

    def write(self, record: dict) -> None:
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **self.static, **record})
        path = self.dir / f"{datetime.now(timezone.utc):%Y-%m-%d}-{self.container}.jsonl"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(line + "\n")
            self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
        try:
            self._commit()
        except Exception as exc:  # never let metrics break serving
            print(f"metrics commit failed: {exc!r}", flush=True)
            self._dirty = True

    def _commit_loop(self, interval: float) -> None:
        while True:
            time.sleep(interval)
            self.flush()


class StreamStats:
    """Reads an OpenAI-style response as it streams past, without changing it."""

    def __init__(self, streaming: bool):
        self.streaming = streaming
        self.first_byte: float | None = None
        self.usage: dict | None = None
        self.finish_reason: str | None = None
        self._tail = b""
        self._body = bytearray()

    def feed(self, chunk: bytes) -> None:
        if self.first_byte is None:
            self.first_byte = time.monotonic()
        if not self.streaming:
            if len(self._body) < 8 * 1024 * 1024:
                self._body.extend(chunk)
            return
        lines = (self._tail + chunk).split(b"\n")
        self._tail = lines.pop()
        for line in lines:
            self._parse_event(line.strip())

    def finish(self) -> None:
        if self.streaming:
            self._parse_event(self._tail.strip())
        else:
            try:
                self._take(json.loads(self._body))
            except ValueError:
                pass

    def _parse_event(self, line: bytes) -> None:
        if not line.startswith(b"data:") or line == b"data: [DONE]":
            return
        try:
            self._take(json.loads(line[5:]))
        except ValueError:
            pass

    def _take(self, event) -> None:
        if not isinstance(event, dict):
            return
        if event.get("usage"):
            self.usage = event["usage"]
        for choice in event.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
