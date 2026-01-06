from __future__ import annotations

from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


class ProgressEmitter:
    """
    Lightweight progress emitter used by samplers.

    Acts as a no-op if Rich is unavailable or progress is disabled.
    """

    def __init__(self, *, total: int | None, description: str, enabled: bool = True):
        self._enabled = enabled and Progress is not None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._total = total
        self._description = description

    def __enter__(self):
        if not self._enabled:
            return self

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=True,
        )
        self._progress.__enter__()
        self._task_id = self._progress.add_task(
            self._description,
            total=self._total,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and self._progress is not None:
            self._progress.__exit__(exc_type, exc, tb)

    def tick(self, n: int = 1):
        if not self._enabled or self._progress is None or self._task_id is None:
            return
        self._progress.advance(self._task_id, n)


class LazyProgress:
    """
    Lazily-instantiated progress bar.

    No output is produced unless `.tick()` is called at least once.
    """

    def __init__(
        self,
        *,
        total: int | None,
        description: str,
        enabled: bool = True,
        persist: bool = False,
    ):
        self._enabled = enabled and Progress is not None
        self._total = total
        self._description = description
        self._persist = persist

        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._started = False

    # ================================================
    # Internal
    # ================================================
    def _ensure_started(self):
        if not self._enabled or self._started:
            return

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=not self._persist,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            self._description,
            total=self._total,
        )
        self._started = True

    def _get_task(self):
        if not self._started or self._progress is None:
            return None
        return self._progress.tasks[self._task_id]

    # ================================================
    # Public API
    # ================================================
    def set_total(self, total: int) -> None:
        self._total = int(total)

        if not self._enabled:
            return

        if self._started:
            # Update live task
            self._progress.update(self._task_id, total=self._total)

    def tick(self, n: int = 1):
        if not self._enabled:
            return

        self._ensure_started()
        if (self._progress is not None) and (self._task_id is not None):
            self._progress.advance(self._task_id, n)

    def close(self):
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    @property
    def completed(self) -> int | None:
        task = self._get_task()
        return None if task is None else int(task.completed)

    @property
    def remaining(self) -> int | None:
        task = self._get_task()
        if task is None or task.total is None:
            return None
        return int(task.total - task.completed)

    @property
    def fraction(self) -> float | None:
        task = self._get_task()
        if task is None or task.total is None:
            return None
        return float(task.completed / task.total)

    @property
    def percent(self) -> float | None:
        frac = self.fraction
        return None if frac is None else 100.0 * frac
