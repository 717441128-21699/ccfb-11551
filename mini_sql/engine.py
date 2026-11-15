"""The query engine: catalog, logical->physical planning and execution driver."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .aggregate import HashAggregateOp
from .csv_reader import CsvSource
from .expr import Schema
from .operators import (
    FilterOp,
    HashJoinOp,
    LimitOp,
    Operator,
    ProjectOp,
    ScanOp,
    SortOp,
)
from .parser import (
    AggCall,
    BinOp,
    ColumnRef,
    SelectItem,
    SelectStmt,
    Star,
    parse_sql,
)
from .plan import (
    LogicalAggregate,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalProject,
    LogicalScan,
    LogicalSort,
    Planner,
)
from .spill import CancelToken, MemTracker, TempManager
from .types import Batch, SQLError


@dataclass
class ExecutionStats:
    elapsed_ms: float
    rows_out: int
    memory: dict
    plan: dict
    sources: dict
    spill_temp_dir: str | None
    spill_bytes_written: int
    spill_bytes_read: int
    cancelled: bool = False


@dataclass
class QueryResult:
    columns: list[str]
    types: list[str]
    rows: list[tuple]
    stats: ExecutionStats
    sql: str
    plan_tree: str

    def to_csv_bytes(self) -> bytes:
        import io
        import csv as _csv

        buf = io.StringIO(newline="")
        writer = _csv.writer(buf, delimiter="|", lineterminator="\n")
        writer.writerow(self.columns)
        for row in self.rows:
            writer.writerow(["" if v is None else v for v in row])
        return buf.getvalue().encode("utf-8")


class Engine:
    """Holds mounted data sources and runs queries."""

    def __init__(self, memory_limit_bytes: int = 2 * 1024**3,
                 parallel: int = 1, spill_dir: str | None = None,
                 batch_rows: int | None = None):
        self.sources: dict[str, CsvSource] = {}
        self.memory_limit = int(memory_limit_bytes)
        self.parallel = max(1, int(parallel))
        self.spill_dir = spill_dir
        if batch_rows is not None:  # pragma: no cover - tuning hook
            from . import operators
            operators.BATCH_ROWS = batch_rows

    # ------------------------------------------------------------------
    def mount(self, name: str, path: str, delimiter: str = "|",
              has_header: bool = True, encoding: str = "utf-8",
              null_tokens=("",), infer: bool = True) -> CsvSource:
        src = CsvSource(
            name=name, path=path, delimiter=delimiter,
            has_header=has_header, encoding=encoding,
            null_tokens=tuple(null_tokens),
        )
        if infer:
            src.infer_schema()
        self.sources[name] = src
        return src

    def unmount(self, name: str) -> None:
        self.sources.pop(name, None)

    def list_sources(self) -> dict:
        out = {}
        for name, src in self.sources.items():
            out[name] = {
                "path": src.path,
                "schema": [{"name": n, "type": t} for n, t in src.schema],
                "rows_seen": src.rows_seen,
                "rows_ok": src.rows_ok,
                "rows_dirty": src.rows_dirty,
            }
        return out

    # ------------------------------------------------------------------
    def explain(self, sql: str) -> tuple[object, str]:
        stmt = parse_sql(sql)
        table_schemas = {name: src.schema for name, src in self.sources.items()}
        logical = Planner(stmt, table_schemas).plan()
        return logical, render_plan(logical)

    def run(self, sql: str, cancel: CancelToken | None = None,
            max_rows: int | None = None) -> QueryResult:
        started = time.perf_counter()
        stmt = parse_sql(sql)
        table_schemas = {name: src.schema for name, src in self.sources.items()}
        logical = Planner(stmt, table_schemas).plan()
        rewritten = self._rewrite_aggregate_outputs(stmt, logical)

        mem = MemTracker(self.memory_limit)
        temp = TempManager(self.spill_dir)
        if cancel is None:
            cancel = CancelToken()
        try:
            root, output_names = self._build(
                logical, rewritten, mem, temp, cancel)
            columns: list[str] = []
            types: list[str] = []
            rows: list[tuple] = []
            for batch in root:
                cancel.check()
                if not columns:
                    columns = [self._display_name(n) for n in output_names]
                    types = [batch.cols[n].dtype for n in output_names]
                for i in range(batch.rowcount):
                    if max_rows is not None and len(rows) >= max_rows:
                        break
                    rows.append(tuple(
                        batch.cols[n].to_python(i) for n in output_names
                    ))
                if max_rows is not None and len(rows) >= max_rows:
                    break
            if not columns:
                columns = [self._display_name(n) for n in output_names]
                types = self._empty_types(logical, output_names)
            stats = ExecutionStats(
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                rows_out=len(rows),
                memory=mem.snapshot(),
                plan=root.stats.to_dict(),
                sources={
                    name: {"rows_seen": s.rows_seen, "rows_ok": s.rows_ok,
                           "rows_dirty": s.rows_dirty}
                    for name, s in self.sources.items()
                },
                spill_temp_dir=str(temp.root),
                spill_bytes_written=temp.bytes_written,
                spill_bytes_read=temp.bytes_read,
            )
            return QueryResult(columns, types, rows, stats, sql,
                               render_plan(logical))
        finally:
            temp.cleanup()

    # ------------------------------------------------------------------
    def _schema_factory(self):
        return Schema([])

    def _build(self, logical, rewritten, mem, temp, cancel):
        root = self._build_node(logical, mem, temp, cancel)
        output_names = rewritten["output_names"]
        return root, output_names

    def _build_node(self, node, mem, temp, cancel) -> Operator:
        t = type(node)
        if t is LogicalScan:
            op = ScanOp(node, self.sources[node.table], self.parallel,
                        mem, temp, cancel, self._schema_factory)
            return op
        if t is LogicalFilter:
            child = self._build_node(node.child, mem, temp, cancel)
            op = FilterOp(node, child, mem, temp, cancel, self._schema_factory)
            op.attach_child_stats(child)
            return op
        if t is LogicalJoin:
            build = self._build_node(node.right, mem, temp, cancel)
            probe = self._build_node(node.left, mem, temp, cancel)
            op = HashJoinOp(node, build, probe, mem, temp, cancel,
                            self._schema_factory)
            op.attach_child_stats(probe, build)
            return op
        if t is LogicalAggregate:
            child = self._build_node(node.child, mem, temp, cancel)
            op = HashAggregateOp(
                node, child, node._group_out_names, node._agg_out_names,
                mem, temp, cancel, self._schema_factory)
            op.attach_child_stats(child)
            return op
        if t is LogicalProject:
            child = self._build_node(node.child, mem, temp, cancel)
            op = ProjectOp(node, child, node._output_names, mem, temp, cancel,
                           self._schema_factory)
            op.attach_child_stats(child)
            return op
        if t is LogicalSort:
            child = self._build_node(node.child, mem, temp, cancel)
            op = SortOp(node, child, mem, temp, cancel, self._schema_factory)
            op.attach_child_stats(child)
            return op
        if t is LogicalLimit:
            child = self._build_node(node.child, mem, temp, cancel)
            op = LimitOp(node, child, mem, temp, cancel)
            op.attach_child_stats(child)
            return op
        raise SQLError(f"cannot build node {t.__name__}")  # pragma: no cover

    # ------------------------------------------------------------------
    def _rewrite_aggregate_outputs(self, stmt: SelectStmt, logical) -> dict:
        """Assign internal names and rewrite aggregates/group refs in project."""
        import copy

        project = _find_first(logical, LogicalProject)
        agg_node = _find_first(logical, LogicalAggregate)
        sort = _find_first(logical, LogicalSort)

        # ORDER BY alias resolution works for both aggregate and plain
        # queries: replace an aliased output reference with a deep copy of the
        # corresponding SELECT expression, which is evaluable pre-projection.
        alias_expr: dict[str, object] = {}
        for item in project.items:
            if item.alias is not None and not isinstance(item.expr, Star):
                alias_expr[item.alias] = item.expr

        output_names: list[str] = []
        if agg_node is None:
            for i, item in enumerate(project.items):
                output_names.append(item.alias or f"col{i + 1}")
            project._output_names = output_names
            if sort is not None:
                for key in sort.keys:
                    if isinstance(key.expr, ColumnRef) and key.expr.qualifier is None \
                            and key.expr.name in alias_expr:
                        key.expr = copy.deepcopy(alias_expr[key.expr.name])
            return {"output_names": output_names}

        group_out, agg_out = [], []
        for gi, gexpr in enumerate(agg_node.group_exprs):
            name = f"__group_{gi}"
            group_out.append(name)
        for ai, agg in enumerate(agg_node.aggs):
            agg_out.append(f"__agg_{ai}")
        agg_node._group_out_names = group_out
        agg_node._agg_out_names = agg_out

        # HAVING filters sit above the aggregate; rewrite their aggregate
        # calls and group references to the internal output columns
        having_filter = _find_having(logical)
        if having_filter is not None:
            having_filter.cond = self._rewrite_expr(
                having_filter.cond, agg_node, group_out, agg_out)

        # map each original select expression to either an internal ref or a
        # scalar expression evaluated over the aggregate output
        new_items: list[SelectItem] = []
        for i, item in enumerate(project.items):
            display = item.alias or self._default_name(item.expr, i)
            output_names.append(display)
            gi = _index_of_expr(agg_node.group_exprs, item.expr)
            if gi is not None:
                new_items.append(
                    SelectItem(ColumnRef(None, group_out[gi]), display))
                continue
            ai = _index_of_agg(agg_node.aggs, item.expr)
            if ai is not None:
                new_items.append(
                    SelectItem(ColumnRef(None, agg_out[ai]), display))
                continue
            rewritten = self._rewrite_expr(item.expr, agg_node,
                                           group_out, agg_out)
            new_items.append(SelectItem(rewritten, display))
        project.items = new_items
        project._output_names = output_names

        # ORDER BY: alias -> underlying expression (then aggregate rewrite);
        # any other expression is rewritten against aggregate outputs too
        if sort is not None:
            for key in sort.keys:
                if isinstance(key.expr, ColumnRef) and key.expr.qualifier is None \
                        and key.expr.name in alias_expr:
                    key.expr = copy.deepcopy(alias_expr[key.expr.name])
                key.expr = self._rewrite_expr(
                    key.expr, agg_node, group_out, agg_out)
        return {"output_names": output_names}

    def _rewrite_expr(self, node, agg_node, group_out, agg_out):
        gi = _index_of_expr(agg_node.group_exprs, node)
        if gi is not None:
            return ColumnRef(None, group_out[gi])
        ai = _index_of_agg(agg_node.aggs, node)
        if ai is not None:
            return ColumnRef(None, agg_out[ai])
        from .parser import BinOp, UnaryOp, FuncExpr, CaseExpr, LikeExpr, InList
        if isinstance(node, BinOp):
            return BinOp(node.op,
                         self._rewrite_expr(node.left, agg_node, group_out, agg_out),
                         self._rewrite_expr(node.right, agg_node, group_out, agg_out))
        if isinstance(node, UnaryOp):
            return UnaryOp(node.op,
                           self._rewrite_expr(node.operand, agg_node, group_out, agg_out))
        if isinstance(node, FuncExpr):
            return FuncExpr(node.name,
                            [self._rewrite_expr(a, agg_node, group_out, agg_out)
                             for a in node.args], dict(node.kwargs))
        if isinstance(node, CaseExpr):
            return CaseExpr(
                [(self._rewrite_expr(c, agg_node, group_out, agg_out),
                  self._rewrite_expr(r, agg_node, group_out, agg_out))
                 for c, r in node.whens],
                self._rewrite_expr(node.else_expr, agg_node, group_out, agg_out)
                if node.else_expr else None)
        if isinstance(node, LikeExpr):
            return LikeExpr(
                self._rewrite_expr(node.expr, agg_node, group_out, agg_out),
                self._rewrite_expr(node.pattern, agg_node, group_out, agg_out),
                node.negated)
        if isinstance(node, InList):
            return InList(
                self._rewrite_expr(node.expr, agg_node, group_out, agg_out),
                [self._rewrite_expr(v, agg_node, group_out, agg_out)
                 for v in node.values], node.negated)
        return node

    @staticmethod
    def _default_name(expr, i) -> str:
        if isinstance(expr, ColumnRef):
            return expr.name
        if isinstance(expr, AggCall):
            return f"{expr.name}_{i + 1}"
        return f"col{i + 1}"

    @staticmethod
    def _display_name(internal: str) -> str:
        # star projection keeps qualified names; user-visible headers strip
        # the table alias prefix
        return internal.split(".", 1)[1] if "." in internal else internal

    def _empty_types(self, logical, output_names) -> list[str]:
        project = _find_first(logical, LogicalProject)
        types = []
        for item in project.items:
            from .parser import Literal
            if isinstance(item.expr, ColumnRef):
                types.append("string")
            else:
                types.append("string")
        return types or ["string"] * len(output_names)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_first(node, cls):
    if isinstance(node, cls):
        return node
    if not _is_dc(node):
        return None
    for child in node.__dict__.values():
        if child is None:
            continue
        if isinstance(child, list):
            for c in child:
                found = _find_first(c, cls)
                if found is not None:
                    return found
        elif _is_dc(child):
            found = _find_first(child, cls)
            if found is not None:
                return found
    return None


def _is_dc(obj):
    from dataclasses import is_dataclass

    return is_dataclass(obj) and not isinstance(obj, type)


def _find_having(node):
    if isinstance(node, LogicalFilter) and node.is_having:
        return node
    if not _is_dc(node):
        return None
    for child in node.__dict__.values():
        if _is_dc(child):
            found = _find_having(child)
            if found is not None:
                return found
        elif isinstance(child, list):
            for c in child:
                found = _find_having(c)
                if found is not None:
                    return found
    return None


def _index_of_expr(exprs, target):
    for i, e in enumerate(exprs):
        if _expr_equal(e, target):
            return i
    return None


def _index_of_agg(aggs, target):
    if not isinstance(target, AggCall):
        return None
    for i, a in enumerate(aggs):
        if a.name == target.name and a.star == target.star \
                and _expr_equal(a.arg, target.arg):
            return i
    return None


def _expr_equal(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if type(a) is not type(b):
        return False
    if isinstance(a, ColumnRef):
        return a.qualifier == b.qualifier and a.name == b.name
    if isinstance(a, AggCall):
        return a.name == b.name and a.star == b.star and _expr_equal(a.arg, b.arg)
    return repr(a) == repr(b)


def render_plan(node, indent: int = 0) -> str:
    """Human-readable logical plan with push-down annotations."""
    pad = "  " * indent
    t = type(node)
    if t is LogicalScan:
        extra = []
        if node.pushed_filter is not None:
            extra.append(f"predicate pushed: {_short(node.pushed_filter)}")
        extra.append(f"projected: {', '.join(node.required_columns) or '<none>'}")
        line = (f"{pad}SeqScan {node.table} AS {node.alias} "
                f"[{'; '.join(extra)}]")
    elif t is LogicalFilter:
        line = f"{pad}Filter {_short(node.cond)}"
        return line + "\n" + render_plan(node.child, indent + 1)
    elif t is LogicalJoin:
        line = f"{pad}{node.join_type} HashJoin {_short(node.cond)}"
        return (line + "\n"
                + render_plan(node.left, indent + 1) + "\n"
                + render_plan(node.right, indent + 1))
    elif t is LogicalAggregate:
        line = (f"{pad}HashAggregate groups=[{len(node.group_exprs)}] "
                f"aggs=[{', '.join(a.name for a in node.aggs)}]")
        return line + "\n" + render_plan(node.child, indent + 1)
    elif t is LogicalProject:
        line = f"{pad}Project [{', '.join(getattr(node, '_output_names', []))}]"
        return line + "\n" + render_plan(node.child, indent + 1)
    elif t is LogicalSort:
        line = f"{pad}Sort [{', '.join(_short(k.expr) for k in node.keys)}]"
        return line + "\n" + render_plan(node.child, indent + 1)
    elif t is LogicalLimit:
        line = f"{pad}Limit n={node.limit} offset={node.offset}"
        return line + "\n" + render_plan(node.child, indent + 1)
    else:  # pragma: no cover
        line = f"{pad}{t.__name__}"
    return line


def _short(node) -> str:
    if isinstance(node, ColumnRef):
        return f"{node.qualifier}.{node.name}" if node.qualifier else node.name
    if isinstance(node, AggCall):
        return f"{node.name.upper()}(*)" if node.star \
            else f"{node.name.upper()}({_short(node.arg)})"
    if isinstance(node, BinOp):
        return f"({_short(node.left)} {node.op} {_short(node.right)})"
    return str(node)
