"""Analysis jobs for the MCP service: one per project name, run in a small thread pool.

The MCP service analyses nothing itself. A job hands the audio to the web app's own endpoint
(backend.py), so Flask, whose project lock only holds inside its own process (pipeline.py),
stays the only writer of uploads/ (docs/adr/0012).
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

log = logging.getLogger("beat-analyser.jobs")

PENDING = ("queued", "running")


@dataclass
class Job:
    name: str
    status: str = "queued"
    error: str | None = None


class JobsFull(Exception):
    """More jobs are pending than the registry accepts."""


class Jobs:
    def __init__(
        self,
        submit: Callable[[Callable[[], None]], object] | None = None,
        max_pending: int = 8,
        workers: int = 2,
    ) -> None:
        # Two workers: the web app runs one gunicorn worker with four threads, and each
        # analysis holds one of them for minutes.
        if submit is None:
            submit = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job").submit
        self._submit = submit
        self._max_pending = max_pending
        self._jobs: dict[str, Job] = {}
        self._guard = threading.Lock()

    def _refuse_if_full(self, name: str) -> Job | None:
        """Return the pending job for name; raise JobsFull when none is and no room is left.
        The caller holds the guard."""
        current = self._jobs.get(name)
        if current is not None and current.status in PENDING:
            return current
        pending = sum(job.status in PENDING for job in self._jobs.values())
        if pending >= self._max_pending:
            raise JobsFull(f"{pending} analyses pending; try again when one finishes")
        return None

    def check_room(self, name: str) -> None:
        """Raise JobsFull when start(name) would be refused, so a caller can refuse before
        it decodes a payload."""
        with self._guard:
            self._refuse_if_full(name)

    def start(self, name: str, work: Callable[[], None]) -> Job:
        """Queue work for name, or return the job already queued or running for it."""
        with self._guard:
            current = self._refuse_if_full(name)
            if current is not None:
                return current
            job = Job(name)
            self._jobs[name] = job
        self._submit(lambda: self._run(job, work))
        return job

    def get(self, name: str) -> Job | None:
        with self._guard:
            return self._jobs.get(name)

    @staticmethod
    def _run(job: Job, work: Callable[[], None]) -> None:
        job.status = "running"
        try:
            work()
        except Exception as exc:
            # Recorded, never raised: the pool would swallow it and the agent would poll a
            # job stuck in "running".
            log.exception("job %s failed", job.name)
            job.error = str(exc) or type(exc).__name__
            job.status = "failed"
            return
        job.status = "done"
