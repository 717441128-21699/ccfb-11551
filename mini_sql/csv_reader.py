"""Parallel chunked CSV parser with sampled type inference and dirty-row stats.

The file is split into byte ranges aligned to newline boundaries (a handful of
seeks in the parent process).  Each range is parsed in a worker process which
converts text rows straight into columnar numpy arrays, so only the columns the
query needs are materialised (projection push-down).  Rows whose field count
does not match the header or whose values violate the inferred column type are
skipped and counted instead of aborting the query.
"""

from __future__ import annotations

import csv
import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from .types import (
    FLOAT64,
    INT64,
    Col,
    Batch,
    SQLError,
    infer_type,
)

DEFAULT_NULL_TOKENS = ("",)
MIN_PARALLEL_BYTES = 4 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
SAMPLE_ROWS = 2000
# Upper bound on rows per emitted batch (kept in step with the execution layer).
BATCH_ROWS = 65_536


@dataclass
class CsvSource:
    """A mounted CSV file: path, dialect, inferred schema and parse stats."""

    name: str
    path: str
    delimiter: str = "|"
    has_header: bool = True
    quotechar: str = '"'
    encoding: str = "utf-8"
    null_tokens: tuple[str, ...] = DEFAULT_NULL_TOKENS
    column_names: list[str] = field(default_factory=list)
    column_types: list[str] = field(default_factory=list)
    rows_seen: int = 0
    rows_dirty: int = 0
    rows_ok: int = 0

    @property
    def schema(self) -> list[tuple[str, str]]:
        return list(zip(self.column_names, self.column_types))

    def type_of(self, column: str) -> str:
        try:
            return self.column_types[self.column_names.index(column)]
        except ValueError as exc:
            raise SQLError(f"unknown column {column!r} in table {self.name!r}") from exc

    def infer_schema(self, sample_rows: int = SAMPLE_ROWS) -> None:
        """Read the header and up to ``sample_rows`` data rows to guess types."""
        with open(self.path, "r", encoding=self.encoding, newline="") as fh:
            first = fh.readline()
            if not first:
                raise SQLError(f"CSV file is empty: {self.path}")
            if self.has_header:
                header = next(csv.reader([first], delimiter=self.delimiter,
                                         quotechar=self.quotechar))
                self.column_names = [h.strip() for h in header]
            data_start = fh.tell() if self.has_header else 0
            samples: list[list[str | None]] = []
            if self.has_header:
                for i, row in enumerate(csv.reader(fh, delimiter=self.delimiter,
                                                    quotechar=self.quotechar)):
                    if i >= sample_rows:
                        break
                    samples.append(row)
            else:
                with open(self.path, "r", encoding=self.encoding, newline="") as fh2:
                    for i, row in enumerate(csv.reader(fh2, delimiter=self.delimiter,
                                                        quotechar=self.quotechar)):
                        if i >= sample_rows:
                            break
                        samples.append(row)
        if not self.has_header:
            width = max((len(r) for r in samples), default=0)
            if width == 0:
                raise SQLError(f"could not determine width of {self.path}")
            self.column_names = [f"c{i}" for i in range(width)]
        width = len(self.column_names)
        cols: list[list[str | None]] = [[] for _ in range(width)]
        for row in samples[:sample_rows]:
            if len(row) != width:
                continue  # dirty rows do not participate in type inference
            for i, v in enumerate(row):
                cols[i].append(None if v in self.null_tokens else v)
        self.column_types = [infer_type(c) for c in cols]

    # ------------------------------------------------------------------
    def scan(self, projected: list[str] | None = None, parallel: int = 1,
             chunk_bytes: int = DEFAULT_CHUNK_BYTES,
             max_batch_rows: int = BATCH_ROWS):
        """Yield :class:`Batch` objects containing only ``projected`` columns.

        Raw parse chunks are additionally split to at most ``max_batch_rows``
        so a single resident batch never dwarfs the query memory budget.
        """
        if not self.column_names:
            self.infer_schema()
        if projected is None:
            projected = list(self.column_names)
        missing = [c for c in projected if c not in self.column_names]
        if missing:
            raise SQLError(f"unknown columns {missing} in table {self.name!r}")
        proj_idx = [self.column_names.index(c) for c in projected]
        proj_types = [self.column_types[i] for i in proj_idx]

        size = os.path.getsize(self.path)
        ranges = split_ranges(self.path, size, chunk_bytes)
        header_len = self._header_byte_length()
        if ranges and ranges[0][0] == 0:
            first_start, first_end = ranges[0]
            ranges[0] = (header_len, first_end)
        ranges = [r for r in ranges if r[1] > r[0]]

        args = [
            (
                self.path, start, end, self.delimiter, self.quotechar,
                self.encoding, self.null_tokens, self.column_names,
                self.column_types, proj_idx, proj_types,
                start == 0 and self.encoding.startswith("utf-8"),
            )
            for start, end in ranges
        ]

        use_pool = parallel > 1 and size >= MIN_PARALLEL_BYTES and len(args) > 1
        self.rows_seen = self.rows_dirty = self.rows_ok = 0
        if use_pool:
            yield from self._emit(
                self._scan_pool(args, parallel, projected, proj_types),
                projected, proj_types, max_batch_rows)
        else:
            def serial():
                for a in args:
                    yield _parse_chunk(a)
            yield from self._emit(serial(), projected, proj_types, max_batch_rows)

    def _emit(self, payload_iter, projected, proj_types, max_batch_rows):
        """Translate worker payloads to batches, splitting oversize ones."""
        for payload, seen, dirty in payload_iter:
            self._accumulate(seen, dirty)
            if payload is None:
                continue
            total = len(payload[0][0])
            for start in range(0, total, max_batch_rows):
                end = min(start + max_batch_rows, total)
                piece = [
                    (data[start:end], mask[start:end], dtype, dictionary)
                    for data, mask, dtype, dictionary in payload
                ]
                yield _payload_to_batch(piece, projected, proj_types)

    def _scan_pool(self, args, parallel, projected, proj_types):
        # Keep at most one chunk in flight per worker so the parent process
        # never buffers more than ``parallel`` parsed batches at once; this is
        # what lets the query-level memory budget actually bound the scan.
        in_flight = max(1, parallel)
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            pending = {}
            it = iter(args)
            for a in args[:in_flight]:
                pending[pool.submit(_parse_chunk, a)] = a
            next_i = in_flight
            while pending:
                for fut in as_completed(list(pending)):
                    break
                result = fut.result()
                del pending[fut]
                yield result
                if next_i < len(args):
                    pending[pool.submit(_parse_chunk, args[next_i])] = args[next_i]
                    next_i += 1

    def _accumulate(self, seen: int, dirty: int) -> None:
        self.rows_seen += seen
        self.rows_dirty += dirty
        self.rows_ok += seen - dirty

    def _header_byte_length(self) -> int:
        if not self.has_header:
            return 0
        with open(self.path, "rb") as fh:
            line = fh.readline()
        return len(line)


def split_ranges(path: str, size: int, chunk_bytes: int) -> list[tuple[int, int]]:
    """Split [0, size) into ranges whose edges sit on newline boundaries."""
    if size == 0:
        return []
    if size <= chunk_bytes:
        return [(0, size)]
    n = max(1, int(np.ceil(size / chunk_bytes)))
    target = size // n
    boundaries = [0]
    with open(path, "rb") as fh:
        for _ in range(n - 1):
            pos = boundaries[-1] + target
            if pos >= size:
                break
            fh.seek(pos)
            fh.readline()  # advance to end of the line straddling pos
            actual = fh.tell()
            if actual <= boundaries[-1] or actual >= size:
                break
            boundaries.append(actual)
    boundaries.append(size)
    return list(zip(boundaries[:-1], boundaries[1:]))


def _parse_chunk(args):  # noqa: C901 - worker entry point, must be top-level
    (path, start, end, delimiter, quotechar, encoding, null_tokens,
     all_names, all_types, proj_idx, proj_types, strip_bom) = args

    values: list[list] = [[] for _ in proj_idx]
    masks: list[list] = [[] for _ in proj_idx]
    seen = dirty = 0
    expected = len(all_names)

    with open(path, "rb") as fh:
        fh.seek(start)
        raw = fh.read(end - start)
    if strip_bom and raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode(encoding)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter,
                        quotechar=quotechar)
    for row in reader:
        if not row:
            continue
        seen += 1
        if len(row) != expected:
            dirty += 1
            continue
        # Dirty-row validation must cover EVERY column, not only the projected
        # ones: a malformed token in a column this query ignores still makes
        # the whole row invalid, so it is counted and skipped atomically.
        row_ok = True
        for src_i, token in enumerate(row):
            if token in null_tokens:
                continue
            dtype = all_types[src_i]
            try:
                if dtype == INT64:
                    iv = int(token)
                    if not (-(2**63) <= iv <= 2**63 - 1):
                        raise ValueError
                elif dtype == FLOAT64:
                    if not np.isfinite(float(token)):
                        raise ValueError
            except ValueError:
                row_ok = False
                break
        if not row_ok:
            dirty += 1
            continue
        # row is fully valid; materialise only the projected columns
        parsed = []
        for out_i, src_i in enumerate(proj_idx):
            token = row[src_i]
            if token in null_tokens:
                parsed.append((None, 0))
                continue
            dtype = proj_types[out_i]
            if dtype == INT64:
                v = int(token)
            elif dtype == FLOAT64:
                v = float(token)
            else:
                v = token
            parsed.append((v, 1))
        for out_i, (v, m) in enumerate(parsed):
            values[out_i].append(v)
            masks[out_i].append(m)

    if not values or not values[0]:
        return None, seen, dirty

    payload = []
    for out_i, dtype in enumerate(proj_types):
        mask = np.array(masks[out_i], dtype=np.uint8)
        if dtype == INT64:
            # NULL slots carry None in ``values``; pre-fill zeros and only
            # copy valid entries so np.array never sees a None
            data = np.zeros(len(values[out_i]), dtype=np.int64)
            valid_idx = np.where(mask == 1)[0]
            if valid_idx.size:
                data[valid_idx] = np.array(
                    [values[out_i][j] for j in valid_idx], dtype=np.int64)
            payload.append((data, mask, dtype, None))
        elif dtype == FLOAT64:
            data = np.zeros(len(values[out_i]), dtype=np.float64)
            valid_idx = np.where(mask == 1)[0]
            if valid_idx.size:
                data[valid_idx] = np.array(
                    [values[out_i][j] for j in valid_idx], dtype=np.float64)
            payload.append((data, mask, dtype, None))
        else:
            # dictionary-encode this chunk's strings: keep one Python copy of
            # each distinct value and store int32 codes instead of an object
            # array with one str object per row
            codes, dictionary = _encode_dictionary(values[out_i], masks[out_i])
            payload.append((codes, mask, dtype, dictionary))
    return payload, seen, dirty


def _encode_dictionary(values, masks):
    lookup: dict[str, int] = {}
    dictionary: list[str] = []
    codes = np.empty(len(values), dtype=np.int32)
    for i, v in enumerate(values):
        if masks[i] == 0 or v is None:
            codes[i] = 0
            continue
        code = lookup.get(v)
        if code is None:
            code = len(dictionary)
            lookup[v] = code
            dictionary.append(v)
        codes[i] = code
    return codes, dictionary


def _payload_to_batch(payload, projected, proj_types) -> Batch:
    cols = {}
    for name, (data, mask, dtype, dictionary) in zip(projected, payload):
        cols[name] = Col(data, mask, dtype, dictionary)
    return Batch(cols, len(data))
