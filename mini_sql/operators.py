"""Pull-based, columnar, batched physical operators (scan/filter/join/sort/...).

All operators consume and produce :class:`~mini_sql.types.Batch` objects.
Rows flow in fixed-size batches and expression work is vectorised through
numpy (see :mod:`mini_sql.expr`).  Every operator records the rows it accepted
in and emitted out so the EXPLAIN-style plan can show cardinality per node.

Spill
-----
* :class:`HashJoinOp` builds in memory while the footprint stays under the
  spill threshold; once crossed it switches to a Grace partition join,
  routing both build and probe rows through on-disk partition runs.
* :class:`SortOp` sorts each in-memory run, spills it, and does an n-way merge
  of the run files.  Hash aggregation lives in :mod:`mini_sql.aggregate`.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from . import expr as expr_mod
from .parser import (
    AggCall,
    BinOp,
    ColumnRef,
    FuncExpr,
    OrderKey,
    Star,
)
from .plan import (
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalProject,
    LogicalScan,
    LogicalSort,
)
from .spill import MemTracker, TempManager, load_batch, save_batch
from .types import (
    FLOAT64,
    STRING,
    Batch,
    Col,
    SQLError,
    TRI_TRUE,
    literal_col,
)

BATCH_ROWS = 65_536


@dataclass
class OpStats:
    node_type: str
    detail: str
    rows_in: int = 0
    rows_out: int = 0
    batches_in: int = 0
    batches_out: int = 0
    elapsed_ms: float = 0.0
    spilled: bool = False
    spill_runs: int = 0
    spill_bytes: int = 0
    children: list["OpStats"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node": self.node_type,
            "detail": self.detail,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "batches_in": self.batches_in,
            "batches_out": self.batches_out,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "spilled": self.spilled,
            "spill_runs": self.spill_runs,
            "spill_bytes": self.spill_bytes,
            "children": [c.to_dict() for c in self.children],
        }


class Operator:
    def __init__(self, mem: MemTracker, temp: TempManager, cancel):
        self.mem = mem
        self.temp = temp
        self.cancel = cancel
        self.stats = OpStats(self.__class__.__name__.replace("Op", ""), "")
        self._t0 = 0.0

    def open(self) -> None:
        self._t0 = time.perf_counter()

    def close(self) -> None:
        self.stats.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0

    def __iter__(self):
        self.open()
        try:
            yield from self.execute()
        finally:
            self.close()

    def attach_child_stats(self, *children: "Operator") -> None:
        self.stats.children.extend(c.stats for c in children)

    def execute(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def check_cancel(self) -> None:
        if self.cancel is not None:
            self.cancel.check()


def _py_value(col: Col, i: int):
    v = col.data[i]
    if col.dtype == STRING:
        return str(v)
    if col.dtype == FLOAT64:
        return float(v)
    return int(v)


# ---------------------------------------------------------------------------
# Scan / Filter
# ---------------------------------------------------------------------------

class ScanOp(Operator):
    def __init__(self, logical: LogicalScan, source, parallel: int,
                 mem, temp, cancel, schema_factory):
        super().__init__(mem, temp, cancel)
        self.logical = logical
        self.source = source
        self.parallel = parallel
        self.schema_factory = schema_factory
        self.stats.node_type = "SeqScan"
        pushed = logical.pushed_filter
        self.stats.detail = (
            f"{logical.table} AS {logical.alias} "
            f"(columns: {', '.join(logical.required_columns) or '<none>'}"
            + (", pushed predicate" if pushed is not None else "") + ")"
        )

    def _max_batch_rows(self) -> int:
        """Bound a batch so a string-heavy wide batch stays under ~1/16 of
        the spill threshold.  Numeric columns are ~8-16 bytes/value; object
        (string) columns keep a Python str plus a pointer, so budget a
        conservative ~120 bytes/value for them."""
        n_cols = max(1, len(self.logical.required_columns))
        string_cols = 0
        try:
            schema = dict(self.source.schema)
            string_cols = sum(
                1 for c in self.logical.required_columns
                if schema.get(c) == STRING)
        except Exception:
            string_cols = n_cols
        per_row = (n_cols - string_cols) * 16 + string_cols * 120
        target_bytes = max(1, self.mem.spill_threshold() // 16)
        rows = target_bytes // max(1, per_row)
        return max(512, min(BATCH_ROWS, int(rows)))

    def execute(self):
        alias = self.logical.alias
        wanted = self.logical.required_columns
        max_rows = self._max_batch_rows()
        for raw in self.source.scan(
                projected=wanted, parallel=self.parallel,
                max_batch_rows=max_rows):
            self.check_cancel()
            self.stats.batches_in += 1
            self.stats.rows_in += raw.rowcount
            prefixed = {f"{alias}.{name}": col for name, col in raw.cols.items()}
            batch = Batch(prefixed, raw.rowcount)
            if self.logical.pushed_filter is not None:
                schema = self.schema_factory()
                tri = expr_mod.evaluate(
                    self.logical.pushed_filter, batch, schema
                )
                idx = np.where(tri.data == TRI_TRUE)[0]
                batch = batch.project(idx)
            # The scan owns no materialised state across batches; downstream
            # operators (agg/sort/join) account for the rows they retain.
            self.stats.batches_out += 1
            self.stats.rows_out += batch.rowcount
            yield batch


class FilterOp(Operator):
    def __init__(self, logical: LogicalFilter, child: Operator,
                 mem, temp, cancel, schema_factory):
        super().__init__(mem, temp, cancel)
        self.cond = logical.cond
        self.child = child
        self.schema_factory = schema_factory
        self.stats.node_type = "Filter"
        self.stats.detail = expr_text(self.cond)

    def execute(self):
        schema = self.schema_factory()
        for batch in self.child:
            self.check_cancel()
            self.stats.batches_in += 1
            self.stats.rows_in += batch.rowcount
            tri = expr_mod.evaluate(self.cond, batch, schema)
            idx = np.where(tri.data == TRI_TRUE)[0]
            out = batch.project(idx)
            self.stats.batches_out += 1
            self.stats.rows_out += out.rowcount
            yield out


# ---------------------------------------------------------------------------
# Hash join (in-memory with Grace-partition spill fallback)
# ---------------------------------------------------------------------------

class HashJoinOp(Operator):
    def __init__(self, logical: LogicalJoin, build_child: Operator,
                 probe_child: Operator, mem, temp, cancel, schema_factory):
        super().__init__(mem, temp, cancel)
        self.join_type = logical.join_type
        self.on = logical.cond
        self.build_child = build_child
        self.probe_child = probe_child
        self.schema_factory = schema_factory
        self.stats.node_type = "HashJoin"
        self.stats.detail = f"{self.join_type} JOIN ON {expr_text(self.on)}"
        self.left_keys, self.right_keys = _split_join_keys(self.on)
        self._build_batch: Batch | None = None

    def execute(self):
        build_batches = list(self.build_child)
        build_bytes = sum(b.memory_bytes() for b in build_batches)
        self.stats.rows_in += sum(b.rowcount for b in build_batches)
        self.mem.add(build_bytes)
        try:
            if build_bytes <= self.mem.spill_threshold():
                yield from self._probe_in_memory(build_batches)
            else:
                yield from self._grace_join(build_batches, build_bytes)
        finally:
            self.mem.free(build_bytes)

    # -- in-memory ---------------------------------------------------------
    def _probe_in_memory(self, build_batches):
        build = Batch.concat_batches(build_batches) if build_batches else None
        self._build_batch = build
        table, right_names = self._build_hash(build)
        schema = self.schema_factory()
        for pb in self.probe_child:
            self.check_cancel()
            self.stats.batches_in += 1
            self.stats.rows_in += pb.rowcount
            out = self._probe_batch(pb, table, right_names, schema, build)
            self.stats.batches_out += 1
            self.stats.rows_out += out.rowcount
            yield out

    def _build_hash(self, build: Batch | None):
        table: dict[tuple, list[int]] = {}
        if build is None or build.rowcount == 0:
            return table, []
        right_names = build.names()
        key_cols = [build.cols[k] for k in self.right_keys]
        for i in range(build.rowcount):
            key = tuple(None if c.mask[i] == 0 else _py_value(c, i)
                        for c in key_cols)
            # NULL join keys never match
            if any(v is None for v in key):
                continue
            table.setdefault(key, []).append(i)
        return table, right_names

    def _probe_batch(self, pb, table, right_names, schema, build) -> Batch:
        probe_keys = [pb.cols[k] for k in self.left_keys]
        n = pb.rowcount
        left_idx: list[int] = []
        right_idx: list[int] = []
        matched = np.zeros(n, dtype=bool)
        for i in range(n):
            if any(c.mask[i] == 0 for c in probe_keys):
                continue
            key = tuple(_py_value(c, i) for c in probe_keys)
            hits = table.get(key)
            if not hits:
                continue
            matched[i] = True
            left_idx.extend([i] * len(hits))
            right_idx.extend(hits)
        li = np.asarray(left_idx, dtype=np.int64)
        ri = np.asarray(right_idx, dtype=np.int64)
        out_cols = {name: col.slice(li) for name, col in pb.cols.items()}
        if right_names and build is not None:
            for name in right_names:
                out_cols[name] = build.cols[name].slice(ri)
        if self.join_type == "LEFT":
            unmatched = np.where(~matched)[0]
            out_cols = self._pad_left_unmatched(out_cols, pb, unmatched,
                                                right_names, build)
        length = len(next(iter(out_cols.values()))) if out_cols else 0
        return Batch(out_cols, length)

    def _pad_left_unmatched(self, out_cols, pb, unmatched, right_names, build):
        if len(unmatched) == 0:
            return out_cols
        add = {name: col.slice(unmatched) for name, col in pb.cols.items()}
        for name in right_names:
            dtype = build.cols[name].dtype
            add[name] = literal_col(None, dtype, len(unmatched))
        return {name: col.concat(add[name]) for name, col in out_cols.items()}

    # -- Grace spill -------------------------------------------------------
    def _grace_join(self, build_batches, build_bytes):
        self.stats.spilled = True
        partitions = max(4, int(np.ceil(
            build_bytes / max(1, self.mem.spill_threshold())
        )))
        jdir = self.temp.subdir(f"join_{id(self)}")
        build_paths = self._partition_batches(
            build_batches, self.right_keys, partitions, jdir, "build")
        probe_paths = self._partition_stream(partitions, jdir)
        self.stats.spill_runs = sum(p is not None for p in build_paths) \
            + sum(p is not None for p in probe_paths)
        schema = self.schema_factory()
        for p in range(partitions):
            self.check_cancel()
            bpath, ppath = build_paths[p], probe_paths[p]
            part_build = load_batch(bpath, self.temp) if bpath else None
            self._build_batch = part_build
            table, right_names = self._build_hash(part_build)
            if ppath is None:
                continue
            probe = load_batch(ppath, self.temp)
            for start in range(0, probe.rowcount, BATCH_ROWS):
                ids = np.arange(start, min(start + BATCH_ROWS, probe.rowcount),
                                dtype=np.int64)
                pb = Batch({n: c.slice(ids) for n, c in probe.cols.items()},
                           len(ids))
                self.stats.batches_in += 1
                self.stats.rows_in += pb.rowcount
                out = self._probe_batch(pb, table, right_names, schema,
                                        part_build)
                self.stats.batches_out += 1
                self.stats.rows_out += out.rowcount
                yield out

    def _partition_batches(self, batches, key_names, partitions, jdir, tag):
        paths: list = [None] * partitions
        buffers: list[list[Batch]] = [[] for _ in range(partitions)]
        for batch in batches:
            key_cols = [batch.cols[k] for k in key_names]
            part = self._partition_ids(key_cols, partitions)
            for p in range(partitions):
                idx = np.where(part == p)[0]
                if len(idx):
                    buffers[p].append(batch.project(idx))
        for p, parts in enumerate(buffers):
            if not parts:
                continue
            merged = Batch.concat_batches(parts)
            path = jdir / f"{tag}_p{p}.npz"
            self.stats.spill_bytes += save_batch(path, merged, self.temp)
            paths[p] = path
        return paths

    def _partition_stream(self, partitions, jdir):
        paths: list = [None] * partitions
        buffers: list[list[Batch]] = [[] for _ in range(partitions)]
        for batch in self.probe_child:
            key_cols = [batch.cols[k] for k in self.left_keys]
            part = self._partition_ids(key_cols, partitions)
            for p in range(partitions):
                idx = np.where(part == p)[0]
                if len(idx):
                    buffers[p].append(batch.project(idx))
        for p, parts in enumerate(buffers):
            if not parts:
                continue
            merged = Batch.concat_batches(parts)
            path = jdir / f"probe_p{p}.npz"
            self.stats.spill_bytes += save_batch(path, merged, self.temp)
            paths[p] = path
        return paths

    @staticmethod
    def _partition_ids(key_cols, partitions) -> np.ndarray:
        n = len(key_cols[0])
        h = np.zeros(n, dtype=np.uint64)
        for col in key_cols:
            for i in range(n):
                v = None if col.mask[i] == 0 else _py_value(col, i)
                h[i] = (h[i] * 1_000_003) ^ (hash(v) & 0xFFFFFFFF)
        return (h % partitions).astype(np.int64)


def _split_join_keys(on):
    """Extract equi-join key column lists from a conjunction of ``a = b``."""
    left, right = [], []

    def walk(node):
        if isinstance(node, BinOp) and node.op == "AND":
            walk(node.left)
            walk(node.right)
            return
        if not isinstance(node, BinOp) or node.op != "=":
            raise SQLError("only equi-joins (with optional AND conditions) are supported")
        l, r = node.left, node.right
        if not (isinstance(l, ColumnRef) and isinstance(r, ColumnRef)):
            raise SQLError("join conditions must compare columns")
        left.append(f"{l.qualifier}.{l.name}")
        right.append(f"{r.qualifier}.{r.name}")

    walk(on)
    return left, right


# ---------------------------------------------------------------------------
# External sort
# ---------------------------------------------------------------------------

class SortOp(Operator):
    def __init__(self, logical: LogicalSort, child: Operator,
                 mem, temp, cancel, schema_factory):
        super().__init__(mem, temp, cancel)
        self.order: list[OrderKey] = logical.keys
        self.child = child
        self.schema_factory = schema_factory
        self.stats.node_type = "Sort"
        self.stats.detail = ", ".join(
            f"{expr_text(k.expr)} {'DESC' if k.descending else 'ASC'}"
            for k in self.order
        )

    def execute(self):
        runs: list = []
        buffered: list[Batch] = []
        for batch in self.child:
            self.check_cancel()
            self.stats.batches_in += 1
            self.stats.rows_in += batch.rowcount
            buffered.append(batch)
            self.mem.add(batch.memory_bytes())
            if self.mem.should_spill():
                path = self._spill_run(buffered)
                runs.append(path)
                for b in buffered:
                    self.mem.free(b.memory_bytes())
                buffered = []
        if not runs:
            if buffered:
                yield from self._sort_in_memory(buffered)
                for b in buffered:
                    self.mem.free(b.memory_bytes())
            return
        if buffered:
            runs.append(self._spill_run(buffered))
            for b in buffered:
                self.mem.free(b.memory_bytes())
        yield from self._merge_runs(runs)

    def _sort_indices(self, batch: Batch, schema) -> np.ndarray:
        """Stable lexicographic sort over all ORDER BY keys.

        Each row maps to a tuple of ``(null_rank, sortable_value)`` pairs where
        null_rank places NULLs first or last per SQL/DuckDB defaults.  Python's
        stable sort then gives a correct total order for mixed string/numeric
        keys without object-array argsort ambiguity.
        """
        n = batch.rowcount
        key_specs = []
        for k in self.order:
            col = expr_mod.evaluate(k.expr, batch, schema)
            nulls_first = k.nulls_first
            if nulls_first is None:
                nulls_first = bool(k.descending)  # DuckDB default
            vals = [None] * n
            null_rank = np.ones(n, dtype=np.int64)
            for i in range(n):
                if col.mask[i] == 0:
                    null_rank[i] = 0 if nulls_first else 2
                else:
                    vals[i] = _py_value(col, i)
            key_specs.append((vals, null_rank, k.descending))

        def sort_key(i: int):
            parts = []
            for vals, null_rank, descending in key_specs:
                rank = int(null_rank[i])
                v = vals[i]
                if v is None:
                    # NULL ordering is fully decided by rank; use a constant
                    parts.append((rank, 0))
                elif descending:
                    parts.append((rank, _Rev(v)))
                else:
                    parts.append((rank, v))
            return tuple(parts)

        return np.array(sorted(range(n), key=sort_key),
                        dtype=np.int64)

    def _spill_run(self, batches):
        schema = self.schema_factory()
        merged = Batch.concat_batches(batches)
        idx = self._sort_indices(merged, schema)
        sorted_batch = Batch(
            {n: c.slice(idx) for n, c in merged.cols.items()}, merged.rowcount
        )
        path = self.temp.new_file(f"sort_run_{self.stats.spill_runs}.npz")
        size = save_batch(path, sorted_batch, self.temp)
        self.stats.spill_runs += 1
        self.stats.spill_bytes += size
        self.stats.spilled = True
        return path

    def _sort_in_memory(self, buffered):
        schema = self.schema_factory()
        merged = Batch.concat_batches(buffered)
        idx = self._sort_indices(merged, schema)
        sorted_batch = Batch(
            {n: c.slice(idx) for n, c in merged.cols.items()}, merged.rowcount
        )
        for start in range(0, sorted_batch.rowcount, BATCH_ROWS):
            ids = np.arange(start, min(start + BATCH_ROWS, sorted_batch.rowcount))
            out = Batch({n: c.slice(ids) for n, c in sorted_batch.cols.items()},
                        len(ids))
            self.stats.batches_out += 1
            self.stats.rows_out += out.rowcount
            yield out

    def _merge_runs(self, runs):
        loaded = [load_batch(p, self.temp) for p in runs]
        names = loaded[0].names()
        cursors = [0] * len(runs)
        heap: list = []
        for ri, batch in enumerate(loaded):
            if batch.rowcount:
                heapq.heappush(heap, (self._row_key(batch, 0), ri))
        parts: dict[str, list[Col]] = {n: [] for n in names}
        emitted = 0
        while heap:
            self.check_cancel()
            _, ri = heapq.heappop(heap)
            batch = loaded[ri]
            i = cursors[ri]
            for n in names:
                parts[n].append(batch.cols[n].slice(np.array([i])))
            cursors[ri] += 1
            if cursors[ri] < batch.rowcount:
                heapq.heappush(heap, (self._row_key(batch, cursors[ri]), ri))
            emitted += 1
            if emitted % BATCH_ROWS == 0:
                yield self._flush_merge(parts, names)
        if any(parts[n] for n in names):
            yield self._flush_merge(parts, names)

    def _flush_merge(self, parts, names):
        cols = {}
        for n in names:
            col = parts[n][0]
            for c in parts[n][1:]:
                col = col.concat(c)
            cols[n] = col
            parts[n] = []
        batch = Batch(cols, len(next(iter(cols.values()))))
        self.stats.batches_out += 1
        self.stats.rows_out += batch.rowcount
        return batch

    def _row_key(self, batch: Batch, i: int):
        schema = self.schema_factory()
        key = []
        for k in self.order:
            col = expr_mod.evaluate(k.expr, batch, schema)
            nulls_first = k.nulls_first
            if nulls_first is None:
                nulls_first = bool(k.descending)
            if col.mask[i] == 0:
                null_group = 0 if nulls_first else 2
                key.append((null_group, 0))
            else:
                v = _py_value(col, i)
                key.append((1, _Rev(v) if k.descending else v))
        return tuple(key)


class _Rev:
    """Wrapper that reverses comparison order for DESC keys."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value


# ---------------------------------------------------------------------------
# Project / Limit
# ---------------------------------------------------------------------------

class ProjectOp(Operator):
    def __init__(self, logical: LogicalProject, child: Operator,
                 output_names: list[str], mem, temp, cancel, schema_factory):
        super().__init__(mem, temp, cancel)
        self.items = logical.items
        self.child = child
        self.output_names = output_names
        self.schema_factory = schema_factory
        self.stats.node_type = "Project"
        self.stats.detail = ", ".join(output_names)

    def execute(self):
        schema = self.schema_factory()
        for batch in self.child:
            self.check_cancel()
            self.stats.batches_in += 1
            self.stats.rows_in += batch.rowcount
            if any(isinstance(it.expr, Star) for it in self.items):
                # star keeps internal qualified names; the engine strips the
                # alias prefix only when materialising final output rows
                out = dict(batch.cols)
            else:
                out = {}
                for name, item in zip(self.output_names, self.items):
                    out[name] = expr_mod.evaluate(item.expr, batch, schema)
            nb = Batch(out, batch.rowcount)
            self.stats.batches_out += 1
            self.stats.rows_out += nb.rowcount
            yield nb


class LimitOp(Operator):
    def __init__(self, logical: LogicalLimit, child: Operator,
                 mem, temp, cancel):
        super().__init__(mem, temp, cancel)
        self.limit = logical.limit
        self.offset = logical.offset
        self.child = child
        self.stats.node_type = "Limit"
        self.stats.detail = (
            f"offset={self.offset}"
            + (f", limit={self.limit}" if self.limit is not None else "")
        )

    def execute(self):
        skipped = emitted = 0
        capped = self.limit is not None
        for batch in self.child:
            self.check_cancel()
            self.stats.batches_in += 1
            self.stats.rows_in += batch.rowcount
            if skipped < self.offset:
                take = min(batch.rowcount, self.offset - skipped)
                skipped += take
                batch = batch.project(np.arange(take, batch.rowcount))
            if batch.rowcount == 0:
                continue
            if capped:
                remain = self.limit - emitted
                if remain <= 0:
                    break
                if batch.rowcount > remain:
                    batch = batch.project(np.arange(remain))
            emitted += batch.rowcount
            self.stats.batches_out += 1
            self.stats.rows_out += batch.rowcount
            yield batch
            if capped and emitted >= self.limit:
                break


def expr_text(node) -> str:
    if isinstance(node, ColumnRef):
        return f"{node.qualifier}.{node.name}" if node.qualifier else node.name
    if isinstance(node, BinOp):
        return f"({expr_text(node.left)} {node.op} {expr_text(node.right)})"
    if isinstance(node, AggCall):
        if node.star:
            return "COUNT(*)"
        return f"{node.name.upper()}({expr_text(node.arg)})"
    if isinstance(node, FuncExpr):
        return f"{node.name.upper()}({', '.join(expr_text(a) for a in node.args)})"
    from .parser import CaseExpr
    if isinstance(node, CaseExpr):
        return "CASE ... END"
    return str(node)
