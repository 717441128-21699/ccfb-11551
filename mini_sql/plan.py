"""Logical planning, name binding and predicate/projection push-down.

Column naming
-------------
Base columns are carried inside batches as ``alias.column`` (the table's FROM
alias, or its table name when unaliased).  Logical planning resolves every
:class:`ColumnRef` against the joined scope and rewrites each one to its fully
qualified form, so later operators never re-resolve ambiguous names.

Push-downs
----------
* WHERE predicates whose columns all belong to one base table are pushed below
  the joins (and all of them below the scan in the form of a required-column
  list + filter).  The remaining predicates stay attached to the join/filter.
* Each table scan is told the exact set of columns its upstream operators need
  (projection push-down); unused CSV columns are never parsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import expr as expr_mod
from .parser import (
    AggCall,
    BinOp,
    CaseExpr,
    ColumnRef,
    FuncExpr,
    InList,
    JoinClause,
    LikeExpr,
    SelectItem,
    SelectStmt,
    Star,
    TableRef,
    UnaryOp,
)
from .types import SQLError


# ---------------------------------------------------------------------------
# Logical plan nodes
# ---------------------------------------------------------------------------

@dataclass
class LogicalScan:
    table: str
    alias: str
    required_columns: list[str] = field(default_factory=list)   # unqualified
    pushed_filter: object = None
    node_type: str = "SeqScan"


@dataclass
class LogicalFilter:
    child: object
    cond: object
    node_type: str = "Filter"
    is_having: bool = False


@dataclass
class LogicalJoin:
    left: object
    right: object
    join_type: str
    cond: object
    right_alias: str
    node_type: str = "Join"


@dataclass
class LogicalAggregate:
    child: object
    group_exprs: list
    aggs: list
    node_type: str = "HashAggregate"


@dataclass
class LogicalProject:
    child: object
    items: list[SelectItem]
    node_type: str = "Project"


@dataclass
class LogicalSort:
    child: object
    keys: list
    node_type: str = "Sort"


@dataclass
class LogicalLimit:
    child: object
    limit: int | None
    offset: int
    node_type: str = "Limit"


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

class Binder:
    def __init__(self, sources: dict[str, str], schemas: dict[str, list[tuple[str, str]]]):
        # sources: alias -> table name ; schemas: alias -> [(unqualified, type)]
        self.sources = sources
        self.schemas = schemas
        self._by_base: dict[str, list[str]] = {}
        for alias, cols in schemas.items():
            for unq, _ in cols:
                self._by_base.setdefault(unq, []).append(alias)

    @property
    def scope(self) -> dict[str, list[str]]:
        return {alias: [f"{alias}.{u}" for u, _ in cols]
                for alias, cols in self.schemas.items()}

    def bind(self, node):
        t = type(node)
        if t is ColumnRef:
            return self._bind_ref(node)
        if t is BinOp:
            return BinOp(node.op, self.bind(node.left), self.bind(node.right))
        if t is UnaryOp:
            return UnaryOp(node.op, self.bind(node.operand))
        if t is LikeExpr:
            return LikeExpr(self.bind(node.expr), self.bind(node.pattern), node.negated)
        if t is InList:
            return InList(self.bind(node.expr), [self.bind(v) for v in node.values],
                          node.negated)
        if t is CaseExpr:
            return CaseExpr([(self.bind(c), self.bind(r)) for c, r in node.whens],
                            self.bind(node.else_expr) if node.else_expr else None)
        if t is FuncExpr:
            return FuncExpr(node.name, [self.bind(a) for a in node.args],
                            dict(node.kwargs))
        if t is AggCall:
            if node.star:
                return AggCall(node.name, None, node.distinct, True)
            return AggCall(node.name, self.bind(node.arg), node.distinct, False)
        return node

    def _bind_ref(self, ref: ColumnRef) -> ColumnRef:
        if ref.qualifier is not None:
            if ref.qualifier not in self.sources:
                raise SQLError(f"unknown table or alias {ref.qualifier!r}")
            cols = {u for u, _ in self.schemas[ref.qualifier]}
            if ref.name not in cols:
                raise SQLError(
                    f"column {ref.qualifier}.{ref.name} not found"
                )
            return ColumnRef(ref.qualifier, ref.name)
        owners = self._by_base.get(ref.name)
        if not owners:
            raise SQLError(f"unknown column {ref.name!r}")
        if len(owners) > 1:
            raise SQLError(f"column {ref.name!r} is ambiguous: {owners}")
        return ColumnRef(owners[0], ref.name)


def collect_refs(node, out: set[str]) -> None:
    """Collect qualified ``alias.column`` names referenced by an expression."""
    if isinstance(node, ColumnRef):
        if node.qualifier is None:
            raise SQLError("internal error: unbound column in push-down")
        out.add(f"{node.qualifier}.{node.name}")
        return
    for child in expr_mod._children(node):
        collect_refs(child, out)


def split_conjuncts(node) -> list:
    if isinstance(node, BinOp) and node.op == "AND":
        return split_conjuncts(node.left) + split_conjuncts(node.right)
    return [node]


def make_and(conds: list):
    if not conds:
        return None
    out = conds[0]
    for c in conds[1:]:
        out = BinOp("AND", out, c)
    return out


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    def __init__(self, stmt: SelectStmt, table_schemas: dict[str, list[tuple[str, str]]]):
        self.stmt = stmt
        # table_schemas: table name -> [(unqualified column, type)]
        self.table_schemas = table_schemas
        self.sources: dict[str, str] = {}          # alias -> table
        self.schemas: dict[str, list] = {}         # alias -> [(unq, type)]
        self._register(stmt.table)
        for j in stmt.joins:
            self._register(j.table)
        self.binder = Binder(self.sources, self.schemas)

    def _register(self, t: TableRef) -> None:
        alias = t.alias or t.name
        if alias in self.sources:
            raise SQLError(f"duplicate table alias {alias!r}")
        if t.name not in self.table_schemas:
            raise SQLError(f"unknown table {t.name!r}")
        self.sources[alias] = t.name
        self.schemas[alias] = list(self.table_schemas[t.name])

    def plan(self):
        stmt = self.stmt
        # 1. bind every expression in the joined scope
        bound_items = [SelectItem(self.binder.bind(it.expr), it.alias) for it in stmt.items]
        bound_where = self.binder.bind(stmt.where) if stmt.where else None
        bound_joins = [
            JoinClause(j.table, j.join_type, self.binder.bind(j.on)) for j in stmt.joins
        ]
        bound_group = [self.binder.bind(g) for g in stmt.group_by]
        bound_having = self.binder.bind(stmt.having) if stmt.having else None
        # ORDER BY: resolve SELECT aliases *before* binding so output-only
        # names are replaced with their underlying (bindable) expressions
        pre_order = self._resolve_order_aliases(stmt.order_by, bound_items)
        bound_order = [self._bind_order(k) for k in pre_order]

        self._validate_stars(bound_items)
        self._validate_aggregates(bound_items, bound_having, bound_group)

        # 2. scans + predicate push-down.  A predicate may only be pushed into
        # a scan when its table is not on the nullable (right) side of a LEFT
        # JOIN, otherwise the filter would remove rows that the outer join is
        # required to preserve (and NULL-padded columns would never appear).
        nullable_aliases = {
            (j.table.alias or j.table.name)
            for j in stmt.joins if j.join_type == "LEFT"
        }
        scans = {alias: LogicalScan(table, alias)
                 for alias, table in self.sources.items()}
        remaining_where = self._push_where(
            bound_where, scans, nullable_aliases)

        # 3. join tree (left-deep, following textual order)
        first_alias = stmt.table.alias or stmt.table.name
        plan: object = scans[first_alias]
        for j in bound_joins:
            right_alias = j.table.alias or j.table.name
            plan = LogicalJoin(plan, scans[right_alias], j.join_type,
                               j.on, right_alias)
        if remaining_where is not None:
            plan = LogicalFilter(plan, remaining_where)

        # 4. aggregation
        aggs: list[AggCall] = []
        for it in bound_items:
            expr_mod.find_aggregates(it.expr, aggs)
        if bound_having is not None:
            expr_mod.find_aggregates(bound_having, aggs)
        if aggs or bound_group:
            plan = LogicalAggregate(plan, bound_group, self._dedup_aggs(aggs))
            if bound_having is not None:
                plan = LogicalFilter(plan, bound_having, is_having=True)

        # 5. order / limit run against the pre-projection row scope so that
        #    ORDER BY can still reference qualified source columns
        if bound_order:
            plan = LogicalSort(plan, bound_order)
        if stmt.limit is not None or stmt.offset:
            plan = LogicalLimit(plan, stmt.limit, stmt.offset)

        # 6. projection is the out-most row operator
        plan = LogicalProject(plan, bound_items)

        # 7. projection push-down: walk the tree and mark required columns
        self._apply_projection(plan, set())
        return plan

    def _bind_order(self, key):
        from .parser import OrderKey

        return OrderKey(self.binder.bind(key.expr), key.descending, key.nulls_first)

    def _resolve_order_aliases(self, order_keys, items: list[SelectItem]):
        """Replace ORDER BY references that match a SELECT output alias with a
        copy of the aliased expression (evaluable in the pre-project scope)."""
        import copy

        from .parser import OrderKey

        alias_to_expr = {}
        for item in items:
            if item.alias is not None and not isinstance(item.expr, Star):
                alias_to_expr[item.alias] = item.expr
        out = []
        for key in order_keys:
            expr = key.expr
            if isinstance(expr, ColumnRef) and expr.qualifier is None \
                    and expr.name in alias_to_expr:
                expr = copy.deepcopy(alias_to_expr[expr.name])
            out.append(OrderKey(expr, key.descending, key.nulls_first))
        return out

    def _validate_stars(self, items: list[SelectItem]) -> None:
        for it in items:
            if isinstance(it.expr, Star):
                if it.expr.qualifier is not None:
                    raise SQLError("qualified * is not supported")
                if len(items) != 1:
                    raise SQLError("* cannot be combined with other select items")
                if self.stmt.group_by or any(
                        expr_mod.is_aggregate(i.expr) for i in items):
                    raise SQLError("* not supported with GROUP BY/aggregates")

    def _validate_aggregates(self, items, having, group_exprs) -> None:
        if not group_exprs:
            return
        group_refs: set[str] = set()
        for g in group_exprs:
            collect_refs(g, group_refs)
        for it in items:
            self._check_agg_scope(it.expr, group_refs)
        if having is not None:
            self._check_agg_scope(having, group_refs)

    def _check_agg_scope(self, node, group_refs: set[str]) -> None:
        if isinstance(node, AggCall):
            return
        if isinstance(node, ColumnRef):
            q = f"{node.qualifier}.{node.name}"
            if q not in group_refs:
                raise SQLError(
                    f"column {q} must appear in GROUP BY or inside an aggregate"
                )
            return
        for child in expr_mod._children(node):
            self._check_agg_scope(child, group_refs)

    @staticmethod
    def _dedup_aggs(aggs: list[AggCall]) -> list[AggCall]:
        seen = {}
        out = []
        for a in aggs:
            key = (a.name, a.star, a.distinct, _expr_key(a.arg))
            if key not in seen:
                seen[key] = len(out)
                out.append(a)
        return out

    def _push_where(self, where, scans: dict[str, LogicalScan],
                    nullable_aliases: set[str] | None = None):
        nullable_aliases = nullable_aliases or set()
        if where is None:
            return None
        kept, pushable = [], {alias: [] for alias in scans}
        for cond in split_conjuncts(where):
            refs: set[str] = set()
            collect_refs(cond, refs)
            owners = {r.split(".", 1)[0] for r in refs}
            if len(owners) == 1:
                owner = next(iter(owners))
                if owner in nullable_aliases:
                    # outer-join nullability makes this unsafe to push down
                    kept.append(cond)
                else:
                    pushable[owner].append(cond)
            else:
                kept.append(cond)
        for alias, conds in pushable.items():
            if conds:
                scans[alias].pushed_filter = make_and(conds)
        return make_and(kept)

    # -- projection push-down --------------------------------------------
    def _apply_projection(self, node, parent_needed: set[str]) -> None:
        """Propagate the union of parent requirements + local references down."""
        t = type(node)
        local: set[str] = set()
        if t is LogicalScan:
            if node.pushed_filter is not None:
                collect_refs(node.pushed_filter, local)
            wanted = sorted(r.split(".", 1)[1]
                            for r in (parent_needed | local)
                            if r.split(".", 1)[0] == node.alias)
            known = {u for u, _ in self.schemas[node.alias]}
            missing = [c for c in wanted if c not in known]
            if missing:
                raise SQLError(f"unknown columns {missing} in {node.table}")
            node.required_columns = wanted
            return
        if t is LogicalFilter:
            collect_refs(node.cond, local)
            self._apply_projection(node.child, parent_needed | local)
            return
        if t is LogicalJoin:
            collect_refs(node.cond, local)
            need = parent_needed | local
            self._apply_projection(node.left, need)
            self._apply_projection(node.right, need)
            return
        if t is LogicalAggregate:
            for g in node.group_exprs:
                collect_refs(g, local)
            for a in node.aggs:
                if not a.star:
                    refs: set[str] = set()
                    collect_refs(a.arg, refs)
                    local.update(refs)
            self._apply_projection(node.child, local)
            return
        if t is LogicalProject:
            for it in node.items:
                if isinstance(it.expr, Star):
                    for alias, cols in self.schemas.items():
                        for u, _ in cols:
                            local.add(f"{alias}.{u}")
                else:
                    collect_refs(it.expr, local)
            local = {r for r in local if not r.startswith("__")}
            self._apply_projection(node.child, parent_needed | local)
            return
        if t is LogicalSort:
            for k in node.keys:
                refs = set()
                collect_refs(k.expr, refs)
                local.update(r for r in refs if not r.startswith("__"))
            self._apply_projection(node.child, parent_needed | local)
            return
        if t is LogicalLimit:
            self._apply_projection(node.child, parent_needed)
            return
        raise SQLError(f"unknown logical node {t.__name__}")  # pragma: no cover


def _expr_key(node) -> str:
    if node is None:
        return "<none>"
    if isinstance(node, ColumnRef):
        return f"col:{node.qualifier}.{node.name}"
    if isinstance(node, BinOp):
        return f"bin:{node.op}({_expr_key(node.left)},{_expr_key(node.right)})"
    if isinstance(node, UnaryOp):
        return f"un:{node.op}({_expr_key(node.operand)})"
    if isinstance(node, FuncExpr):
        return f"fn:{node.name}({','.join(_expr_key(a) for a in node.args)})"
    return repr(node)
