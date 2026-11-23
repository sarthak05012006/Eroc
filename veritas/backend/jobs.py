"""Background jobs (conflict scan, ...) with progress, errors and a result. One running job per kind."""
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone

from . import logs

log = logs.get("jobs")


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.state = "queued"           # queued | running | done | failed
        self.done = 0
        self.total = 0
        self.message = ""
        self.errors: list[str] = []     # non-fatal problems (e.g. a model call failed for one group)
        self.fatal: str | None = None
        self.started: float | None = None
        self.finished: float | None = None

    def progress(self, done: int, total: int, message: str = ""):
        self.done, self.total = done, total
        if message:
            self.message = message

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state, "running": self.state in ("queued", "running"),
                "done": self.done, "total": self.total, "message": self.message, "errors": self.errors[-10:],
                "error_count": len(self.errors), "error": self.fatal,
                "seconds": round((self.finished or time.time()) - self.started, 1) if self.started else 0.0,
                "finished": datetime.fromtimestamp(self.finished, timezone.utc).isoformat() if self.finished else None}


class JobRegistry:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def latest(self, kind: str) -> Job | None:
        with self._lock:
            mine = [j for j in self._jobs.values() if j.kind == kind]
        return max(mine, key=lambda j: j.started or 0, default=None)

    def start(self, kind: str, target) -> Job:
        """Run target(job) in a thread. If a job of this kind is already running, return it instead."""
        with self._lock:
            for j in self._jobs.values():
                if j.kind == kind and j.state in ("queued", "running"):
                    return j
            job = Job(kind)
            job.state, job.started = "running", time.time()
            self._jobs[job.id] = job

        def run():
            try:
                target(job)
                job.state = "done"
            except Exception as e:          # never let a job die silently
                log.exception("job %s (%s) failed", job.id, kind)
                job.fatal, job.state = str(e), "failed"
            finally:
                job.finished = time.time()

        threading.Thread(target=run, daemon=True, name=f"job-{kind}").start()
        return job
