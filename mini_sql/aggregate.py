"""Hash aggregation with partitioned spill and associative state merging.

Per-group state
---------------
* ``count``        -> ``{"v": int}``
* ``sum`` / ``avg`` -> ``{"sum": float, "n": int}`` (SQL SUM/AVG ignore NULLs;
  AVG over zero non-null rows yields NULL)
* ``min`` / ``max`` -> ``{"v": value, "dtype": dtype}``

Spill model
-----------
While resident state stays under the memory threshold, groups accumulate in a
plain Python dict.  On the first threshold crossing the existing groups are
split into ``P`` partitions and written as JSONL partial-state runs; every
later batch is aggregated per partition and flushed the same way.  Finalisation
reads one partition at a time, merging all of its partial runs (each
aggregate's state is an associative monoid, so partials combine exactly), then
emits the finished groups.
"""

from __future__ import annotations

import json

import numpy as np

from . import expr as expr_mod
from .operators import BATCH_ROWS, Operator
from .parser import AggCall
from .plan import LogicalAggregate
from .spill import MemTracker, TempManager
from .types import (
    FLOAT64,
    INT64,
    STRING,
    Batch,
    Col,
)


class HashAggregateOp(Operator):
    def __init__(self, logical: LogicalAggregate, child: Operator,
                 group_output_names: list[str], agg_output_names: list[str],
                 mem: MemTracker, temp: TempManager, cancel, schema_factory):
        super().__init__(mem, temp, cancel)
        self.group_exprs = logical.group_exprs
        self.aggs: list[AggCall] = logical.aggs
        self.child = child
        self.group_output_names = group_output_names
        self.agg_output_names = agg_output_names
        self.schema_factory = schema_factory
        self.stats.node_type = "HashAggregate"
        self.stats.detail = (
            f"groups: {len(self.group_exprs)}, "
            f"aggregates: {', '.join(a.name for a in self.aggs) or '<none>'}"
        )
        self.partition_count = 8
        self.partition_files: dict[int, list[str]] = {}
        self.spill_dir: str | None = None
        self._resident = 0
        self._state_bytes = 0

    def _state_footprint(self, groups, order) -> int:
        """Estimate the real resident footprint of the Python group table.

        Each distinct group owns a key tuple (one entry per grouping
        expression) plus a list of per-aggregate state dicts.  String keys and
        min/max values carry their character payload; numeric keys are fixed
        width.  This is an estimate, but it tracks the dominant cost (one
        Python object graph per distinct group) closely enough to trigger
        spill before the dict blows past the query budget.
        """
        n_groups = len(order)
        if n_groups == 0:
            return 0
        per_key = max(1, len(self.group_exprs))
        # tuple + per-key object (int/float pointer or str header+payload)
        key_cost = per_key * 64
        # state: one dict per aggregate with a couple of boxed Python numbers
        state_cost = len(self.aggs) * 120
        # sample real string payloads from a handful of keys/values
        str_payload = 0
        sample = order[: min(n_groups, 256)]
        for key in sample:
            for v in key:
                if isinstance(v, str):
                    str_payload += 50 + len(v) * 2
        str_payload = int(str_payload / min(n_groups, len(sample))) * n_groups \
            if sample else 0
        return n_groups * (key_cost + state_cost) + str_payload + 64 * n_groups

    # ------------------------------------------------------------------
    def execute(self):
        schema = self.schema_factory()
        groups: dict[tuple, list[dict]] = {}
        order: list[tuple] = []
        spilled = False

        for batch in self.child:
            self.check_cancel()
            self.stats.batches_in += 1
            self.stats.rows_in += batch.rowcount
            keys = self._group_keys(batch, schema)
            arg_cols = self._eval_args(batch, schema)
            # count the transient input batch while folding it in, then
            # release: retained information lives in the group state below
            batch_bytes = batch.memory_bytes()
            self.mem.add(batch_bytes)
            self._update_groups(groups, order, keys, arg_cols)
            self.mem.free(batch_bytes)
            # keep the (often dominant) Python group-table cost in the budget
            new_state = self._state_footprint(groups, order)
            self.mem.add(max(0, new_state - self._state_bytes))
            self._state_bytes = new_state
            if not spilled and self.mem.should_spill():
                # the current batch is already folded into ``groups``; flush
                # everything once and switch later batches to partition files
                spilled = True
                self.stats.spilled = True
                self.spill_dir = str(self.temp.subdir(f"agg_{id(self)}"))
                self._flush_groups(groups, order)
                groups, order = {}, []
                self.mem.free(self._state_bytes)
                self._state_bytes = 0
                continue
            if spilled:
                self._spill_batch(keys, arg_cols)

        if spilled:
            yield from self._merge_and_emit()
        else:
            yield from self._emit(groups, order)

    # ------------------------------------------------------------------
    def _new_state(self) -> list[dict]:
        return [{} for _ in self.aggs]

    def _group_keys(self, batch, schema):
        if not self.group_exprs:
            return [()] * batch.rowcount
        key_cols = [expr_mod.evaluate(g, batch, schema) for g in self.group_exprs]
        n = batch.rowcount
        keys = [None] * n
        for i in range(n):
            keys[i] = tuple(
                None if c.mask[i] == 0 else _py_value(c, i) for c in key_cols
            )
        return keys

    def _eval_args(self, batch, schema) -> dict[int, Col | None]:
        out: dict[int, Col | None] = {}
        for k, agg in enumerate(self.aggs):
            out[k] = None if agg.star else expr_mod.evaluate(agg.arg, batch, schema)
        return out

    def _update_groups(self, groups, order, keys, arg_cols) -> None:
        n = len(keys)
        for i in range(n):
            key = keys[i]
            grp = groups.get(key)
            if grp is None:
                grp = self._new_state()
                groups[key] = grp
                order.append(key)
            for k, agg in enumerate(self.aggs):
                _update_state(grp[k], agg, arg_cols[k], i)

    def _spill_batch(self, keys, arg_cols) -> None:
        per_part_groups: dict[int, dict] = {}
        per_part_order: dict[int, list] = {}
        for i, key in enumerate(keys):
            p = _hash_key(key, self.partition_count)
            groups = per_part_groups.setdefault(p, {})
            order = per_part_order.setdefault(p, [])
            grp = groups.get(key)
            if grp is None:
                grp = self._new_state()
                groups[key] = grp
                order.append(key)
            for k, agg in enumerate(self.aggs):
                _update_state(grp[k], agg, arg_cols[k], i)
        for p, groups in per_part_groups.items():
            self._write_partial(p, [(key, groups[key]) for key in per_part_order[p]])

    def _flush_groups(self, groups, order) -> None:
        per_part: dict[int, list] = {}
        for key in order:
            p = _hash_key(key, self.partition_count)
            per_part.setdefault(p, []).append((key, groups[key]))
        for p, entries in per_part.items():
            self._write_partial(p, entries)

    def _write_partial(self, partition: int, entries: list[tuple]) -> None:
        idx = len(self.partition_files.get(partition, []))
        path = f"{self.spill_dir}/p{partition}_{idx}.jsonl"
        size = 0
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for key, state in entries:
                payload = {"k": list(key), "s": state}
                line = json.dumps(payload, default=str) + "\n"
                fh.write(line)
                size += len(line.encode("utf-8"))
        self.partition_files.setdefault(partition, []).append(path)
        self.stats.spill_runs += 1
        self.stats.spill_bytes += size
        self.temp.record_write(size)

    def _merge_and_emit(self):
        for p in range(self.partition_count):
            self.check_cancel()
            files = self.partition_files.get(p)
            if not files:
                continue
            merged: dict[tuple, list[dict]] = {}
            order: list[tuple] = []
            for path in files:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        payload = json.loads(line)
                        key = tuple(payload["k"])
                        grp = merged.get(key)
                        if grp is None:
                            grp = self._new_state()
                            merged[key] = grp
                            order.append(key)
                        for k, raw in enumerate(payload["s"]):
                            _merge_state(grp[k], raw, self.aggs[k])
            yield from self._emit(merged, order)

    # ------------------------------------------------------------------
    def _emit(self, groups, order):
        if not order:
            if self.group_exprs:
                return
            order = [()]
            groups = {(): self._new_state()}
        for start in range(0, len(order), BATCH_ROWS):
            chunk = order[start:start + BATCH_ROWS]
            cols = self._build_output(groups, chunk)
            batch = Batch(cols, len(chunk))
            self.stats.batches_out += 1
            self.stats.rows_out += batch.rowcount
            yield batch

    def _build_output(self, groups, keys):
        cols = {}
        for gi, out_name in enumerate(self.group_output_names):
            sample = next((key[gi] for key in keys if key[gi] is not None), None)
            if isinstance(sample, str):
                dtype = STRING
            elif isinstance(sample, float):
                dtype = FLOAT64
            else:
                dtype = INT64
            col = expr_mod.literal_col(None, dtype, len(keys))
            for i, key in enumerate(keys):
                if key[gi] is not None:
                    col.data[i] = key[gi]
                    col.mask[i] = 1
            cols[out_name] = col
        for ai, (agg, out_name) in enumerate(
                zip(self.aggs, self.agg_output_names)):
            vals, masks, dtype = self._finalize(groups, keys, ai, agg)
            if dtype == STRING:
                data = np.array(vals, dtype=object)
            elif dtype == FLOAT64:
                data = np.array(vals, dtype=np.float64)
            else:
                data = np.array(vals, dtype=np.int64)
            cols[out_name] = Col(data, np.array(masks, dtype=np.uint8), dtype)
        return cols

    def _finalize(self, groups, keys, ai, agg):
        vals, masks = [], []
        dtype = INT64
        for key in keys:
            st = groups[key][ai]
            if agg.name == "count":
                vals.append(int(st.get("v", 0)))
                masks.append(1)
            elif agg.name in ("sum", "avg"):
                n = int(st.get("n", 0))
                if n == 0:
                    vals.append(0.0)
                    masks.append(0)
                elif agg.name == "sum":
                    vals.append(float(st["sum"]))
                    masks.append(1)
                else:
                    vals.append(float(st["sum"]) / n)
                    masks.append(1)
                dtype = FLOAT64
            else:  # min/max
                if "v" not in st:
                    dtype = st.get("dtype", STRING)
                    vals.append(None if dtype == STRING else 0)
                    masks.append(0)
                else:
                    dtype = st.get("dtype", STRING)
                    vals.append(st["v"])
                    masks.append(1)
        return vals, masks, dtype


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------

def _py_value(col: Col, i: int):
    v = col.data[i]
    if col.dtype == STRING:
        return str(v)
    if col.dtype == FLOAT64:
        return float(v)
    return int(v)


def _hash_key(key: tuple, partitions: int) -> int:
    h = 0
    for v in key:
        h = (h * 1_000_003) ^ (hash(v) & 0xFFFFFFFF)
    return h % partitions


def _update_state(state: dict, agg: AggCall, arg: Col | None, i: int) -> None:
    name = agg.name
    if name == "count":
        if agg.star or (arg is not None and arg.mask[i] == 1):
            state["v"] = int(state.get("v", 0)) + 1
        return
    if name in ("sum", "avg"):
        if arg is not None and arg.mask[i] == 1:
            state["sum"] = float(state.get("sum", 0.0)) + float(arg.data[i])
            state["n"] = int(state.get("n", 0)) + 1
        return
    # min / max
    if arg is None or arg.mask[i] == 0:
        return
    v = _py_value(arg, i)
    if "v" not in state or _minmax_hit(v, state["v"], name):
        state["v"] = v
        state["dtype"] = arg.dtype


def _minmax_hit(a, b, name: str) -> bool:
    return a < b if name == "min" else a > b


def _merge_state(state: dict, raw: dict, agg: AggCall) -> None:
    name = agg.name
    if name == "count":
        if "v" in raw:
            state["v"] = int(state.get("v", 0)) + int(raw["v"])
        return
    if name in ("sum", "avg"):
        if "sum" in raw:
            state["sum"] = float(state.get("sum", 0.0)) + float(raw["sum"])
            state["n"] = int(state.get("n", 0)) + int(raw["n"])
        return
    if "v" not in raw:
        return
    dtype = raw.get("dtype", STRING)
    v = raw["v"]
    if dtype != STRING:
        v = float(v) if dtype == FLOAT64 else int(v)
    if "v" not in state or _minmax_hit(v, state["v"], name):
        state["v"] = v
        state["dtype"] = dtype
