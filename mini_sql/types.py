"""Column types, nullable columns, batches and three-valued logic helpers.

NULL representation
-------------------
Every :class:`Col` pairs a numpy ``data`` array with a uint8 ``mask`` where
``1`` means *valid* and ``0`` means *NULL*.  Boolean results of predicates are
never columns of Python bools: they use int8 tri-state codes (1 = TRUE,
0 = FALSE, -1 = UNKNOWN) implementing SQL three-valued logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

INT64 = "int64"
FLOAT64 = "float64"
STRING = "string"

ALL_TYPES = (INT64, FLOAT64, STRING)

# Kleene tri-state codes.
TRI_TRUE = 1
TRI_FALSE = 0
TRI_UNKNOWN = -1

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


class SQLError(Exception):
    """Any semantic, syntax or execution error raised by the engine."""


class CancelledError(SQLError):
    """Raised when the query cancellation flag is observed."""


def infer_type(samples: list[str | None]) -> str:
    """Infer a column type from sampled text values.

    Types are chosen by majority vote rather than veto, so a small number of
    dirty tokens (e.g. an injected ``NOT_AN_INT``) cannot flip an otherwise
    numeric column to STRING.  Preference on a tie is INT64 -> FLOAT64 ->
    STRING; rows that violate the chosen type are later counted as dirty and
    skipped during parsing.  A column that is entirely NULL defaults to STRING.
    """
    n_int = n_float = n_other = 0
    for v in samples:
        if v is None or v == "":
            continue
        try:
            int(v)
            n_int += 1
            continue
        except (ValueError, TypeError):
            pass
        try:
            if math.isfinite(float(v)):
                n_float += 1
                continue
        except (ValueError, TypeError):
            pass
        n_other += 1
    total = n_int + n_float + n_other
    if total == 0:
        return STRING
    # numeric columns: ints and decimals are the same family; only treat the
    # column as text when non-numeric tokens are the plurality
    numeric = n_int + n_float
    if n_other > numeric:
        return STRING
    if n_float > 0 and n_float >= n_int:
        return FLOAT64
    if n_int > 0:
        return INT64
    return STRING


def parse_int64(text: str) -> int:
    v = int(text)
    if not INT64_MIN <= v <= INT64_MAX:
        raise SQLError(f"integer value out of range: {text}")
    return v


def build_col(py_values: list, dtype: str):
    """Build a :class:`Col` from a list of Python values (``None`` = NULL)."""
    n = len(py_values)
    mask = np.ones(n, dtype=np.uint8)
    if dtype == INT64:
        data = np.zeros(n, dtype=np.int64)
        for i, v in enumerate(py_values):
            if v is None:
                mask[i] = 0
            else:
                data[i] = v if isinstance(v, int) else parse_int64(str(v))
    elif dtype == FLOAT64:
        data = np.zeros(n, dtype=np.float64)
        for i, v in enumerate(py_values):
            if v is None:
                mask[i] = 0
            else:
                data[i] = v if isinstance(v, (int, float)) else float(v)
    elif dtype == STRING:
        data = np.empty(n, dtype=object)
        for i, v in enumerate(py_values):
            if v is None:
                mask[i] = 0
                data[i] = None
            else:
                data[i] = str(v)
    else:  # pragma: no cover - internal
        raise SQLError(f"unknown type {dtype}")
    return Col(data, mask, dtype)


@dataclass
class Col:
    """A nullable columnar array.

    String columns may be *dictionary encoded*: ``data`` then holds int32
    codes into ``dictionary`` (a list of Python strings).  Low-cardinality
    text columns (dates, flags, statuses, names) become compact integer
    arrays this way.  Any string operation can call :meth:`materialised` to
    obtain a plain object-array view; numeric columns are unaffected.
    """

    data: np.ndarray
    mask: np.ndarray  # uint8, 1 = valid
    dtype: str
    dictionary: list | None = None

    def __len__(self) -> int:
        return len(self.data)

    @property
    def is_dict(self) -> bool:
        return self.dtype == STRING and self.dictionary is not None

    @property
    def null_count(self) -> int:
        return int((self.mask == 0).sum())

    def memory_bytes(self) -> int:
        base = int(self.data.nbytes) + int(self.mask.nbytes)
        if self.dtype == STRING:
            if self.dictionary is not None:
                base += sum(49 + len(s) * 2 for s in self.dictionary)
            else:
                total = 49 * int((self.mask == 1).sum())
                for v in self.data:
                    if v is not None:
                        total += len(v) * 2
                base += total
        return base

    def to_python(self, i: int):
        if self.mask[i] == 0:
            return None
        if self.is_dict:
            return self.dictionary[int(self.data[i])]
        v = self.data[i]
        if self.dtype == STRING:
            return str(v)
        if self.dtype == FLOAT64:
            return float(v)
        return int(v)

    def slice(self, idx: np.ndarray) -> "Col":
        return Col(
            self.data[idx], self.mask[idx], self.dtype,
            list(self.dictionary) if self.dictionary is not None else None)

    def concat(self, other: "Col") -> "Col":
        if self.dtype != other.dtype:
            raise SQLError("cannot concat columns of different types")
        if self.is_dict and other.is_dict and self.dictionary == other.dictionary:
            return Col(np.concatenate([self.data, other.data]),
                       np.concatenate([self.mask, other.mask]),
                       STRING, list(self.dictionary))
        if self.dtype == STRING:
            a, b = self.materialised(), other.materialised()
            return Col(np.concatenate([a.data, b.data]),
                       np.concatenate([a.mask, b.mask]), STRING)
        return Col(
            np.concatenate([self.data, other.data]),
            np.concatenate([self.mask, other.mask]),
            self.dtype,
        )

    def materialised(self) -> "Col":
        """Plain object-array view of a (possibly dict-encoded) string col."""
        if not self.is_dict:
            return self
        out = np.empty(len(self), dtype=object)
        for i in range(len(self)):
            out[i] = None if self.mask[i] == 0 \
                else self.dictionary[int(self.data[i])]
        return Col(out, self.mask, STRING)


class Batch:
    """An ordered, equal-length set of named columns."""

    def __init__(self, cols: dict[str, Col], rowcount: int | None = None):
        self.cols = dict(cols)
        if rowcount is None:
            rowcount = len(next(iter(self.cols.values()))) if self.cols else 0
        self.rowcount = rowcount

    def names(self) -> list[str]:
        return list(self.cols.keys())

    def get(self, name: str) -> Col:
        return self.cols[name]

    def has(self, name: str) -> bool:
        return name in self.cols

    def keep(self, names: list[str]) -> "Batch":
        return Batch({n: self.cols[n] for n in names}, self.rowcount)

    def project(self, idx: np.ndarray) -> "Batch":
        if len(idx) == self.rowcount:
            return self
        return Batch({n: c.slice(idx) for n, c in self.cols.items()}, len(idx))

    def memory_bytes(self) -> int:
        return sum(c.memory_bytes() for c in self.cols.values())

    @staticmethod
    def concat_batches(batches: list["Batch"]) -> "Batch":
        if not batches:
            raise SQLError("cannot concat empty batch list")
        names = batches[0].names()
        cols = {
            n: _concat_col([b.cols[n] for b in batches])
            for n in names
        }
        return Batch(cols, sum(b.rowcount for b in batches))

    @staticmethod
    def empty_like(schema: list[tuple[str, str]]) -> "Batch":
        cols = {}
        for name, dtype in schema:
            if dtype == STRING:
                data = np.empty(0, dtype=object)
            elif dtype == FLOAT64:
                data = np.empty(0, dtype=np.float64)
            else:
                data = np.empty(0, dtype=np.int64)
            cols[name] = Col(data, np.empty(0, dtype=np.uint8), dtype)
        return Batch(cols, 0)


def _concat_col(cs: list[Col]) -> Col:
    out = cs[0]
    for c in cs[1:]:
        out = out.concat(c)
    return out


# ---------------------------------------------------------------------------
# Three-valued logic
# ---------------------------------------------------------------------------

def tri_not(v: np.ndarray) -> np.ndarray:
    """Kleene NOT: TRUE<->FALSE, UNKNOWN stays UNKNOWN."""
    out = np.empty_like(v)
    out[v == TRI_TRUE] = TRI_FALSE
    out[v == TRI_FALSE] = TRI_TRUE
    out[v == TRI_UNKNOWN] = TRI_UNKNOWN
    return out


def tri_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised Kleene AND."""
    out = np.full(len(a), TRI_UNKNOWN, dtype=np.int8)
    af, bf = a == TRI_FALSE, b == TRI_FALSE
    at, bt = a == TRI_TRUE, b == TRI_TRUE
    out[af | bf] = TRI_FALSE
    out[at & bt] = TRI_TRUE
    return out


def tri_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised Kleene OR."""
    out = np.full(len(a), TRI_UNKNOWN, dtype=np.int8)
    at, bt = a == TRI_TRUE, b == TRI_TRUE
    af, bf = a == TRI_FALSE, b == TRI_FALSE
    out[at | bt] = TRI_TRUE
    out[af & bf] = TRI_FALSE
    return out


def checked_int_binop(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    """SQL exact-integer arithmetic; raises on int64 overflow.

    TPC-H arithmetic is dominated by decimal (float64) columns; exact integer
    op inputs are rare and small, so the element-wise Python check is adequate
    and keeps overflow semantics strictly correct.
    """
    res = np.empty(len(a), dtype=np.int64)
    for i in range(len(a)):
        x, y = int(a[i]), int(b[i])
        if op == "+":
            r = x + y
        elif op == "-":
            r = x - y
        elif op == "*":
            r = x * y
        else:  # pragma: no cover
            raise SQLError(f"unknown integer op {op}")
        if not INT64_MIN <= r <= INT64_MAX:
            raise SQLError(
                "data exception - numeric value out of range "
                f"({x} {op} {y} overflows bigint)"
            )
        res[i] = r
    return res


def literal_col(value, dtype: str, n: int) -> Col:
    """A constant column of ``n`` rows (NULL when ``value is None``)."""
    mask = np.ones(n, dtype=np.uint8)
    if value is None:
        mask[:] = 0
    if dtype == INT64:
        data = np.full(n, 0 if value is None else int(value), dtype=np.int64)
    elif dtype == FLOAT64:
        data = np.full(n, 0.0 if value is None else float(value), dtype=np.float64)
    else:
        data = np.full(n, None if value is None else str(value), dtype=object)
    return Col(data, mask, dtype)
