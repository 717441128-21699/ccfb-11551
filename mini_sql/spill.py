"""Memory accounting, spill thresholds and temporary-file lifecycle.

The engine never pretends to have a hard allocator hook into numpy; instead
every operator reports the *materialised* bytes it owns to :class:`MemTracker`.
Sort and hash aggregate operators spill when their resident footprint crosses
the configured fraction of the query's memory budget.

:class:`TempManager` owns one root directory per query.  All spill files live
under it and are removed deterministically on query completion, cancellation or
error (plus a best-effort sweep of stale roots from crashed processes).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .types import Batch, CancelledError


class CancelToken:
    """Thread-safe cancellation flag checked between every batch."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise CancelledError("query cancelled by user")


class MemTracker:
    """Process-local estimate of memory currently held by query operators."""

    def __init__(self, budget_bytes: int, spill_ratio: float = 0.75):
        self.budget = int(budget_bytes)
        self.spill_ratio = float(spill_ratio)
        self._current = 0
        self.peak = 0
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        return self._current

    def spill_threshold(self) -> int:
        return int(self.budget * self.spill_ratio)

    def should_spill(self) -> bool:
        return self._current >= self.spill_threshold()

    def add(self, n: int) -> None:
        with self._lock:
            self._current += int(n)
            if self._current > self.peak:
                self.peak = self._current

    def free(self, n: int) -> None:
        with self._lock:
            self._current -= int(n)
            if self._current < 0:
                self._current = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "budget_bytes": self.budget,
                "current_bytes": self._current,
                "peak_bytes": self.peak,
                "spill_threshold_bytes": self.spill_threshold(),
                "over_budget": self._current > self.budget,
            }


class TempManager:
    """Owns the spill directory of one query and every run file inside it."""

    def __init__(self, base_dir: str | None = None, prefix: str = "minisql_"):
        base = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
        base.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=prefix, dir=str(base)))
        self._files: list[Path] = []
        self.closed = False
        self.bytes_written = 0
        self.bytes_read = 0
        self._lock = threading.Lock()

    def new_file(self, name: str) -> Path:
        p = self.root / name
        with self._lock:
            self._files.append(p)
        return p

    def subdir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record_write(self, n: int) -> None:
        self.bytes_written += int(n)

    def record_read(self, n: int) -> None:
        self.bytes_read += int(n)

    def cleanup(self) -> None:
        """Deterministic removal; safe to call multiple times."""
        if self.closed:
            return
        self.closed = True
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()

    @staticmethod
    def sweep_stale(base_dir: str | None = None, max_age_hours: float = 24.0) -> int:
        """Remove leftover roots from crashed processes; returns count removed."""
        base = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        if not base.exists():
            return 0
        for child in base.glob("minisql_*"):
            try:
                if child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed


def save_batch(path: Path, batch: Batch, temp: TempManager | None) -> int:
    """Persist one batch to an .npz run (handles dict-encoded strings)."""
    payload = {}
    for name, col in batch.cols.items():
        key = _safe_key(name)
        payload[f"{key}__data"] = col.data
        payload[f"{key}__mask"] = col.mask
        payload[f"{key}__dtype"] = np.array(col.dtype)
        if col.dictionary is not None:
            payload[f"{key}__dict"] = np.array(col.dictionary, dtype=object)
    np.savez(path, **payload)
    size = path.stat().st_size
    if temp is not None:
        temp.record_write(size)
    return size


def _safe_key(name: str) -> str:
    # npz keys are identifiers; keep the column name but tag it so suffix
    # parsing is unambiguous regardless of underscores in the name
    return "col_" + name.replace(".", "_dot_")


def _key_to_name(key: str) -> str:
    base = key[len("col_"):]
    return base.replace("_dot_", ".")


def load_batch(path: Path, temp: TempManager | None) -> Batch:
    # Spill files are written exclusively by this engine; object (string and
    # dictionary) arrays require allow_pickle, so enable it for our own
    # trusted run files.
    with np.load(path, allow_pickle=True) as z:
        names = sorted({
            _key_to_name(k[: -len("__dtype")])
            for k in z.files if k.endswith("__dtype")
        })
        cols = {}
        for name in names:
            key = _safe_key(name)
            dtype = str(z[f"{key}__dtype"])
            dictionary = None
            if f"{key}__dict" in z.files:
                dictionary = [str(v) for v in z[f"{key}__dict"]]
            cols[name] = Col_like(
                np.array(z[f"{key}__data"], copy=True),
                np.array(z[f"{key}__mask"], copy=True),
                dtype, dictionary)
    if temp is not None:
        temp.record_read(path.stat().st_size)
    return Batch(cols)


def Col_like(data: np.ndarray, mask: np.ndarray, dtype: str,
             dictionary=None):
    from .types import Col

    return Col(data, mask, dtype, dictionary)


def write_run(path: Path, batches: list[Batch], temp: TempManager) -> int:
    """Write several batches as one run file and return bytes written."""
    merged = Batch.concat_batches(batches)
    return save_batch(path, merged, temp)
