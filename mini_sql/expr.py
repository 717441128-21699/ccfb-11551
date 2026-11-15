"""Vectorised expression evaluation over columnar batches.

Every AST node from :mod:`mini_sql.parser` evaluates to a :class:`Col`.
Predicates do not return bool columns: they return int8 tri-state columns
(TRUE/FALSE/UNKNOWN) so NULL semantics compose correctly through AND/OR/NOT.
"""

from __future__ import annotations

import re

import numpy as np

from .parser import (
    AggCall,
    BinOp,
    CaseExpr,
    ColumnRef,
    FuncExpr,
    InList,
    LikeExpr,
    Literal,
    UnaryOp,
)
from .types import (
    FLOAT64,
    INT64,
    STRING,
    Col,
    SQLError,
    TRI_FALSE,
    TRI_TRUE,
    TRI_UNKNOWN,
    checked_int_binop,
    literal_col,
    tri_and,
    tri_not,
    tri_or,
)

AGGS = {"count", "sum", "avg", "min", "max"}


class Schema:
    """Resolves qualified/unqualified column names against the current batch.

    The planner already binds every :class:`ColumnRef` to its qualified form
    ``alias.column``; unqualified refs only appear after aggregate output
    rewriting, where columns are the internal ``__group_n`` / ``__agg_n``
    names or user aliases.
    """

    def __init__(self, sources=None):
        self.sources = sources or []

    def resolve(self, ref: ColumnRef, batch) -> Col:
        if ref.qualifier is not None:
            full = f"{ref.qualifier}.{ref.name}"
            if full in batch.cols:
                return batch.cols[full].materialised()
            raise SQLError(f"unknown table/alias {ref.qualifier!r}")
        if ref.name in batch.cols:
            return batch.cols[ref.name].materialised()
        matches = [n for n in batch.cols
                   if n.split(".", 1)[-1] == ref.name]
        if not matches:
            raise SQLError(f"unknown column {ref.name!r}")
        if len(matches) > 1:
            raise SQLError(f"column {ref.name!r} is ambiguous")
        return batch.cols[matches[0]].materialised()


# structural alias for documentation only
class BatchLike:  # pragma: no cover - structural type
    cols: dict[str, Col]
    rowcount: int


# ---------------------------------------------------------------------------
# Casting
# ---------------------------------------------------------------------------

def cast_col(col: Col, target: str) -> Col:
    if col.dtype == target:
        return col
    n = len(col)
    if target == FLOAT64:
        if col.dtype == INT64:
            return Col(col.data.astype(np.float64), col.mask, FLOAT64)
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if col.mask[i] == 0:
                continue
            try:
                out[i] = float(col.data[i])
            except (ValueError, TypeError) as exc:
                raise SQLError(f"cannot cast {col.data[i]!r} to DOUBLE") from exc
        return Col(out, col.mask, FLOAT64)
    if target == INT64:
        if col.dtype == FLOAT64:
            out = np.zeros(n, dtype=np.int64)
            for i in range(n):
                if col.mask[i] == 0:
                    continue
                f = col.data[i]
                if not np.isfinite(f) or f != np.trunc(f):
                    raise SQLError(f"cannot cast {f!r} to BIGINT")
                if not (-(2**63) <= f <= 2**63 - 1):
                    raise SQLError(f"cannot cast {f!r} to BIGINT: out of range")
                out[i] = int(f)
            return Col(out, col.mask, INT64)
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if col.mask[i] == 0:
                continue
            try:
                iv = int(str(col.data[i]))
            except ValueError as exc:
                raise SQLError(f"cannot cast {col.data[i]!r} to BIGINT") from exc
            if not (-(2**63) <= iv <= 2**63 - 1):
                raise SQLError(f"cannot cast {iv!r} to BIGINT: out of range")
            out[i] = iv
        return Col(out, col.mask, INT64)
    # target == STRING
    out = np.empty(n, dtype=object)
    for i in range(n):
        if col.mask[i] == 0:
            out[i] = None
        elif col.dtype == FLOAT64:
            out[i] = _format_float(col.data[i])
        else:
            out[i] = str(col.data[i])
    return Col(out, col.mask, STRING)


def _format_float(f: float) -> str:
    if f == int(f) and abs(f) < 1e16:
        return str(int(f))
    return repr(f)


def common_numeric(a: Col, b: Col) -> tuple[Col, Col, str]:
    if a.dtype == STRING or b.dtype == STRING:
        raise SQLError("cannot apply numeric operator to a string")
    if a.dtype == FLOAT64 or b.dtype == FLOAT64:
        return cast_col(a, FLOAT64), cast_col(b, FLOAT64), FLOAT64
    return a, b, INT64


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate(node, batch, schema: Schema) -> Col:
    t = type(node)
    if t is Literal:
        n = batch.rowcount
        return literal_col(node.value, node.dtype, n)
    if t is ColumnRef:
        return schema.resolve(node, batch)
    if t is UnaryOp:
        return _eval_unary(node, batch, schema)
    if t is BinOp:
        return _eval_binop(node, batch, schema)
    if t is LikeExpr:
        return _eval_like(node, batch, schema)
    if t is InList:
        return _eval_in(node, batch, schema)
    if t is CaseExpr:
        return _eval_case(node, batch, schema)
    if t is FuncExpr:
        return _eval_func(node, batch, schema)
    raise SQLError(f"unsupported expression node {t.__name__}")


def _eval_unary(node: UnaryOp, batch, schema) -> Col:
    if node.op == "NOT":
        v = evaluate(node.operand, batch, schema)
        return Col(tri_not(_as_tri(v)), np.ones(len(v), dtype=np.uint8), INT64)
    v = evaluate(node.operand, batch, schema)
    if node.op == "+":
        return v
    # unary minus with SQL overflow semantics for exact integers
    if v.dtype == INT64:
        out = np.empty(len(v), dtype=np.int64)
        for i in range(len(v)):
            if v.mask[i] == 0:
                continue
            x = int(v.data[i])
            if x == -(2**63):
                raise SQLError("data exception - numeric value out of range (-bigint min)")
            out[i] = -x
        return Col(out, v.mask, INT64)
    return Col(-v.data, v.mask, FLOAT64)


def _as_tri(col: Col) -> np.ndarray:
    """Coerce a column to a tri-state predicate array.

    Columns produced by comparisons/AND/OR already carry tri-state int8 codes
    and a fully-valid mask; pass them through.  Other numeric columns follow
    SQL truthiness (non-zero = TRUE), with NULL mapping to UNKNOWN.
    """
    n = len(col)
    codes = set(np.unique(col.data).tolist()) if col.dtype == INT64 else set()
    if col.dtype == INT64 and codes.issubset({TRI_TRUE, TRI_FALSE, TRI_UNKNOWN}):
        out = col.data.astype(np.int8).copy()
        out[col.mask == 0] = TRI_UNKNOWN
        return out
    out = np.full(n, TRI_UNKNOWN, dtype=np.int8)
    if col.dtype == STRING:
        truthy = np.zeros(n, dtype=bool)
    else:
        truthy = col.data != 0
    valid = col.mask == 1
    out[valid & truthy] = TRI_TRUE
    out[valid & ~truthy] = TRI_FALSE
    return out


def _eval_binop(node: BinOp, batch, schema) -> Col:
    op = node.op
    if op == "AND":
        a = _as_tri(evaluate(node.left, batch, schema))
        b = _as_tri(evaluate(node.right, batch, schema))
        return Col(tri_and(a, b), np.ones(len(a), dtype=np.uint8), INT64)
    if op == "OR":
        a = _as_tri(evaluate(node.left, batch, schema))
        b = _as_tri(evaluate(node.right, batch, schema))
        return Col(tri_or(a, b), np.ones(len(a), dtype=np.uint8), INT64)

    left = evaluate(node.left, batch, schema)
    right = evaluate(node.right, batch, schema)

    if op in ("IS NULL", "IS NOT NULL"):
        n = len(left)
        is_null = left.mask == 0
        res = is_null if op == "IS NULL" else ~is_null
        tri = np.where(res, TRI_TRUE, TRI_FALSE).astype(np.int8)
        return Col(tri, np.ones(n, dtype=np.uint8), INT64)

    if op in ("=", "<>", "!=", "<", ">", "<=", ">="):
        return _compare(left, right, op)
    if op == "||":
        return _concat(left, right)
    return _arith(left, right, op)


def _compare(left: Col, right: Col, op: str) -> Col:
    n = len(left)
    if left.dtype == STRING and right.dtype == STRING:
        a, b = left.data, right.data
        with np.errstate(all="ignore"):
            eq = np.array([
                False if (left.mask[i] == 0 or right.mask[i] == 0)
                else a[i] == b[i] for i in range(n)
            ])
            lt = np.array([
                False if (left.mask[i] == 0 or right.mask[i] == 0)
                else a[i] < b[i] for i in range(n)
            ])
        res_dtype = INT64
    else:
        if left.dtype == STRING and right.dtype != STRING:
            left = cast_col(left, right.dtype)
        elif right.dtype == STRING and left.dtype != STRING:
            right = cast_col(right, left.dtype)
        a, b, res_dtype = common_numeric(left, right)
        with np.errstate(all="ignore"):
            eq = a.data == b.data
            lt = a.data < b.data
    valid = (left.mask == 1) & (right.mask == 1)
    if op == "=":
        hit = eq
    elif op in ("<>", "!="):
        hit = ~eq
    elif op == "<":
        hit = lt
    elif op == ">":
        hit = ~lt & ~eq
    elif op == "<=":
        hit = lt | eq
    else:  # >=
        hit = ~lt
    tri = np.full(n, TRI_UNKNOWN, dtype=np.int8)
    tri[valid & hit] = TRI_TRUE
    tri[valid & ~hit] = TRI_FALSE
    return Col(tri, np.ones(n, dtype=np.uint8), INT64)


def _arith(left: Col, right: Col, op: str) -> Col:
    if op == "/":
        if left.dtype == STRING or right.dtype == STRING:
            raise SQLError("cannot divide strings")
        a = cast_col(left, FLOAT64)
        b = cast_col(right, FLOAT64)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = a.data / b.data
        # SQL: division by zero raises rather than producing inf/nan
        zero = (b.mask == 1) & (b.data == 0)
        if zero.any():
            raise SQLError("data exception - division by zero")
        return Col(data, (a.mask & b.mask), FLOAT64)
    a, b, dtype = common_numeric(left, right)
    mask = a.mask & b.mask
    if dtype == FLOAT64:
        with np.errstate(all="ignore"):
            if op == "+":
                data = a.data + b.data
            elif op == "-":
                data = a.data - b.data
            elif op == "*":
                data = a.data * b.data
            else:  # pragma: no cover
                raise SQLError(f"unknown op {op}")
        return Col(data, mask, FLOAT64)
    data = checked_int_binop(a.data, b.data, op)
    return Col(data, mask, INT64)


def _concat(left: Col, right: Col) -> Col:
    a = cast_col(left, STRING)
    b = cast_col(right, STRING)
    n = len(a)
    out = np.empty(n, dtype=object)
    mask = a.mask & b.mask
    for i in range(n):
        out[i] = str(a.data[i]) + str(b.data[i]) if mask[i] else None
    return Col(out, mask, STRING)


def _eval_like(node: LikeExpr, batch, schema) -> Col:
    v = cast_col(evaluate(node.expr, batch, schema), STRING)
    pat = cast_col(evaluate(node.pattern, batch, schema), STRING)
    n = len(v)
    tri = np.full(n, TRI_UNKNOWN, dtype=np.int8)
    for i in range(n):
        if v.mask[i] == 0 or pat.mask[i] == 0:
            continue
        if _like_match(str(v.data[i]), str(pat.data[i])) != node.negated:
            tri[i] = TRI_TRUE
        else:
            tri[i] = TRI_FALSE
    return Col(tri, np.ones(n, dtype=np.uint8), INT64)


def _like_match(value: str, pattern: str) -> bool:
    return re.fullmatch(_like_to_regex(pattern), value, flags=re.DOTALL) is not None


def _like_to_regex(pattern: str) -> str:
    out = ["^"]
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        if c == "%":
            out.append(".*")
        elif c == "_":
            out.append(".")
        else:
            out.append(re.escape(c))
        i += 1
    out.append("$")
    return "".join(out)


def _eval_in(node: InList, batch, schema) -> Col:
    v = evaluate(node.expr, batch, schema)
    n = len(v)
    tri = np.full(n, TRI_UNKNOWN, dtype=np.int8)
    any_unknown = np.zeros(n, dtype=bool)
    found = np.zeros(n, dtype=bool)
    for val_node in node.values:
        c = evaluate(val_node, batch, schema)
        cmp = _compare(v, c, "=")
        found |= cmp.data == TRI_TRUE
        any_unknown |= cmp.data == TRI_UNKNOWN
    tri[found] = TRI_TRUE
    tri[~found & any_unknown] = TRI_UNKNOWN
    tri[~found & ~any_unknown] = TRI_FALSE
    if node.negated:
        # NOT IN: FALSE when matched, UNKNOWN when any UNKNOWN and not matched
        tri[found] = TRI_FALSE
        tri[~found & any_unknown] = TRI_UNKNOWN
        tri[~found & ~any_unknown] = TRI_TRUE
    return Col(tri, np.ones(n, dtype=np.uint8), INT64)


def _eval_case(node: CaseExpr, batch, schema) -> Col:
    n = batch.rowcount
    chosen = np.full(n, -1, dtype=np.int64)
    hit = np.zeros(n, dtype=bool)
    result_cols = []
    for k, (cond, res) in enumerate(node.whens):
        tri = _as_tri(evaluate(cond, batch, schema))
        take = (tri == TRI_TRUE) & ~hit
        chosen[take] = k
        hit |= take
        result_cols.append(evaluate(res, batch, schema))
    if node.else_expr is not None:
        else_col = evaluate(node.else_expr, batch, schema)
    else:
        else_col = literal_col(None, STRING, n)
    result_cols.append(else_col)
    dtype = _unify_types([c.dtype for c in result_cols])
    result_cols = [cast_col(c, dtype) for c in result_cols]
    data = np.empty(n, dtype=object if dtype == STRING else
                    (np.float64 if dtype == FLOAT64 else np.int64))
    mask = np.zeros(n, dtype=np.uint8)
    for k, col in enumerate(result_cols):
        sel = (chosen == k) if k < len(result_cols) - 1 else (chosen == -1)
        if not sel.any():
            continue
        data[sel] = col.data[sel]
        mask[sel] = col.mask[sel]
    return Col(data, mask, dtype)


def _unify_types(dtypes: list[str]) -> str:
    if STRING in dtypes:
        return STRING
    if FLOAT64 in dtypes:
        return FLOAT64
    return INT64


def _eval_func(node: FuncExpr, batch, schema) -> Col:
    name = node.name
    args = [evaluate(a, batch, schema) for a in node.args]
    if name == "coalesce":
        return _func_coalesce(args)
    if name == "substring":
        return _func_substring(args)
    if name == "extract":
        return _func_extract(args, node.kwargs.get("field", "YEAR"))
    if name in ("upper", "lower", "length", "trim", "ltrim", "rtrim"):
        return _func_string(name, args)
    raise SQLError(f"unknown function {name.upper()}")


def _func_coalesce(args: list[Col]) -> Col:
    dtype = _unify_types([c.dtype for c in args])
    args = [cast_col(c, dtype) for c in args]
    n = len(args[0])
    data = np.empty(n, dtype=object if dtype == STRING else
                    (np.float64 if dtype == FLOAT64 else np.int64))
    mask = np.zeros(n, dtype=np.uint8)
    for col in args:
        take = (col.mask == 1) & (mask == 0)
        data[take] = col.data[take]
        mask[take] = 1
    return Col(data, mask, dtype)


def _func_substring(args: list[Col]) -> Col:
    s = cast_col(args[0], STRING)
    start = cast_col(args[1], INT64)
    if len(args) > 2:
        length = cast_col(args[2], INT64)
    else:
        length = literal_col(2**31 - 1, INT64, len(s))
    n = len(s)
    out = np.empty(n, dtype=object)
    mask = s.mask & start.mask & length.mask
    for i in range(n):
        if mask[i] == 0:
            out[i] = None
            continue
        st = int(start.data[i])  # SQL positions are 1-based
        ln = int(length.data[i])
        text = str(s.data[i])
        if st < 1:
            ln += st - 1
            st = 1
        if ln < 0:
            out[i] = ""
        else:
            out[i] = text[st - 1: st - 1 + ln]
    return Col(out, mask, STRING)


def _func_extract(args: list[Col], field: str) -> Col:
    s = cast_col(args[0], STRING)  # dates are stored as 'YYYY-MM-DD'
    n = len(s)
    data = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if s.mask[i] == 0:
            continue
        text = str(s.data[i])
        try:
            if field == "YEAR":
                data[i] = int(text[0:4])
            elif field == "MONTH":
                data[i] = int(text[5:7])
            else:
                data[i] = int(text[8:10])
        except (ValueError, IndexError) as exc:
            raise SQLError(f"cannot extract {field} from {text!r}") from exc
    return Col(data, s.mask, INT64)


def _func_string(name: str, args: list[Col]) -> Col:
    s = cast_col(args[0], STRING)
    n = len(s)
    if name == "length":
        data = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if s.mask[i]:
                data[i] = len(str(s.data[i]))
        return Col(data, s.mask, INT64)
    out = np.empty(n, dtype=object)
    for i in range(n):
        if s.mask[i] == 0:
            out[i] = None
            continue
        v = str(s.data[i])
        if name == "upper":
            out[i] = v.upper()
        elif name == "lower":
            out[i] = v.lower()
        elif name == "trim":
            out[i] = v.strip()
        elif name == "ltrim":
            out[i] = v.lstrip()
        else:
            out[i] = v.rstrip()
    return Col(out, s.mask, STRING)


def is_aggregate(node) -> bool:
    if isinstance(node, AggCall):
        return True
    for child in _children(node):
        if is_aggregate(child):
            return True
    return False


def find_aggregates(node, out: list[AggCall]) -> None:
    if isinstance(node, AggCall):
        out.append(node)
        return
    for child in _children(node):
        find_aggregates(child, out)


def _children(node):
    if isinstance(node, BinOp):
        return [node.left, node.right]
    if isinstance(node, UnaryOp):
        return [node.operand]
    if isinstance(node, CaseExpr):
        return [x for pair in node.whens for x in pair] + (
            [node.else_expr] if node.else_expr is not None else [])
    if isinstance(node, FuncExpr):
        return list(node.args)
    if isinstance(node, LikeExpr):
        return [node.expr, node.pattern]
    if isinstance(node, InList):
        return [node.expr] + list(node.values)
    return []
