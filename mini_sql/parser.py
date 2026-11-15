"""Hand-written SQL tokenizer and recursive-descent parser.

Supported grammar (single SELECT block, no subqueries)::

    SELECT expr [AS alias], ...
    FROM table [alias]
        [ [INNER | LEFT [OUTER]] JOIN table [alias] ON cond ] ...
    [WHERE cond]
    [GROUP BY expr, ...]
    [HAVING cond]
    [ORDER BY expr [ASC|DESC] [NULLS FIRST|LAST], ...]
    [LIMIT n [OFFSET m]]

Expressions include column refs (optionally qualified), integer/float/string/
NULL literals, ``DATE '...'`` typed literals, ``+ - * /``, ``||``, comparisons,
AND/OR/NOT, ``LIKE``, ``IN (...)``, ``CASE WHEN``, ``COALESCE``, ``SUBSTRING``
and ``EXTRACT(field FROM ...)``.  ``--`` line and ``/* ... */`` block comments
are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .types import FLOAT64, INT64, STRING, SQLError

KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "limit",
    "offset", "as", "and", "or", "not", "null", "true", "false", "join",
    "inner", "left", "right", "outer", "on", "asc", "desc", "case", "when",
    "then", "else", "end", "like", "in", "is", "between", "cast", "date",
    "extract", "substring", "substr", "coalesce", "nulls", "first", "last",
    "distinct", "all", "for",
}


class TokKind(Enum):
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()
    KEYWORD = auto()
    PUNCT = auto()
    EOF = auto()


@dataclass
class Token:
    kind: TokKind
    value: str
    pos: int


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass
class Literal:
    value: object          # int, float, str or None
    dtype: str             # INT64 / FLOAT64 / STRING
    sql_type: str | None = None  # e.g. 'DATE' for DATE '1998-01-01'


@dataclass
class ColumnRef:
    qualifier: str | None
    name: str


@dataclass
class UnaryOp:
    op: str                # 'NOT' | '-' | '+'
    operand: object


@dataclass
class BinOp:
    op: str
    left: object
    right: object


@dataclass
class LikeExpr:
    expr: object
    pattern: object
    negated: bool


@dataclass
class InList:
    expr: object
    values: list
    negated: bool


@dataclass
class CaseExpr:
    whens: list  # list[(cond, result)]
    else_expr: object | None


@dataclass
class FuncExpr:
    name: str
    args: list
    kwargs: dict = field(default_factory=dict)  # e.g. {'field': 'YEAR'}


@dataclass
class Star:
    qualifier: str | None = None


@dataclass
class SelectItem:
    expr: object
    alias: str | None


@dataclass
class AggCall:
    name: str               # count/sum/avg/min/max
    arg: object | None      # None => COUNT(*)
    distinct: bool = False
    star: bool = False


@dataclass
class TableRef:
    name: str
    alias: str | None


@dataclass
class JoinClause:
    table: TableRef
    join_type: str          # 'INNER' | 'LEFT'
    on: object


@dataclass
class OrderKey:
    expr: object
    descending: bool
    nulls_first: bool | None  # None => engine default


@dataclass
class SelectStmt:
    items: list[SelectItem]
    table: TableRef
    joins: list[JoinClause]
    where: object | None
    group_by: list
    having: object | None
    order_by: list[OrderKey]
    limit: int | None
    offset: int


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_MULTI_CHAR_PUNCT = ("<=", ">=", "<>", "!=", "||")
_SINGLE_PUNCT = set("(),+-*/<>=.;")


def tokenize(sql: str) -> list[Token]:
    toks: list[Token] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            j = sql.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            j = sql.find("*/", i + 2)
            if j < 0:
                raise SQLError("unterminated block comment")
            i = j + 2
            continue
        if c == "'":
            j = i + 1
            buf = []
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(sql[j])
                j += 1
            else:
                raise SQLError("unterminated string literal")
            toks.append(Token(TokKind.STRING, "".join(buf), i))
            i = j + 1
            continue
        if c == '"':
            j = i + 1
            buf = []
            while j < n and sql[j] != '"':
                buf.append(sql[j])
                j += 1
            if j >= n:
                raise SQLError("unterminated quoted identifier")
            toks.append(Token(TokKind.IDENT, "".join(buf), i))
            i = j + 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and sql[i + 1].isdigit()):
            j = i
            seen_dot = seen_exp = False
            while j < n:
                ch = sql[j]
                if ch.isdigit():
                    j += 1
                elif ch == "." and not seen_dot and not seen_exp:
                    seen_dot = True
                    j += 1
                elif ch in "eE" and not seen_exp:
                    seen_exp = True
                    j += 1
                    if j < n and sql[j] in "+-":
                        j += 1
                else:
                    break
            toks.append(Token(TokKind.NUMBER, sql[i:j], i))
            i = j
            continue
        two = sql[i:i + 2]
        if two in _MULTI_CHAR_PUNCT:
            toks.append(Token(TokKind.PUNCT, two, i))
            i += 2
            continue
        if c in _SINGLE_PUNCT:
            toks.append(Token(TokKind.PUNCT, c, i))
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            word = sql[i:j]
            low = word.lower()
            kind = TokKind.KEYWORD if low in KEYWORDS else TokKind.IDENT
            toks.append(Token(kind, low if kind == TokKind.KEYWORD else word, i))
            i = j
            continue
        raise SQLError(f"unexpected character {c!r} at position {i}")
    toks.append(Token(TokKind.EOF, "", n))
    return toks


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, sql: str):
        self.toks = tokenize(sql)
        self.i = 0

    # -- token helpers ----------------------------------------------------
    def peek(self, k: int = 0) -> Token:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Token:
        t = self.toks[self.i]
        if t.kind != TokKind.EOF:
            self.i += 1
        return t

    def at_kw(self, *words: str) -> bool:
        t = self.peek()
        return t.kind == TokKind.KEYWORD and t.value in words

    def at_punct(self, p: str) -> bool:
        t = self.peek()
        return t.kind == TokKind.PUNCT and t.value == p

    def eat_kw(self, word: str) -> bool:
        if self.at_kw(word):
            self.next()
            return True
        return False

    def expect_kw(self, word: str) -> None:
        if not self.eat_kw(word):
            t = self.peek()
            raise SQLError(f"expected {word.upper()} but got {t.value!r}")

    def eat_punct(self, p: str) -> bool:
        if self.at_punct(p):
            self.next()
            return True
        return False

    def expect_punct(self, p: str) -> None:
        if not self.eat_punct(p):
            t = self.peek()
            raise SQLError(f"expected {p!r} but got {t.value!r}")

    # -- statement --------------------------------------------------------
    def parse_select(self) -> SelectStmt:
        self.expect_kw("select")
        items = [self._parse_select_item()]
        while self.eat_punct(","):
            items.append(self._parse_select_item())

        self.expect_kw("from")
        table = self._parse_table_ref()
        joins: list[JoinClause] = []
        while self.at_kw("inner", "left", "right", "join"):
            joins.append(self._parse_join())

        where = self._parse_where()
        group_by = self._parse_group_by()
        having = self._parse_having()
        order_by = self._parse_order_by()
        limit, offset = self._parse_limit()
        if self.peek().kind != TokKind.EOF:
            raise SQLError(f"trailing tokens after query: {self.peek().value!r}")
        return SelectStmt(items, table, joins, where, group_by, having,
                          order_by, limit, offset)

    def _parse_select_item(self) -> SelectItem:
        if self.at_punct("*"):
            self.next()
            return SelectItem(Star(), None)
        expr = self._parse_expr()
        alias = None
        if self.eat_kw("as"):
            alias = self._parse_alias_name()
        elif self.peek().kind == TokKind.IDENT:
            alias = self.next().value
        elif self.at_kw("date"):  # rare collision; alias named like keyword
            self.next()
            alias = "date"
        return SelectItem(expr, alias)

    def _parse_alias_name(self) -> str:
        t = self.next()
        if t.kind not in (TokKind.IDENT, TokKind.KEYWORD):
            raise SQLError(f"expected alias name but got {t.value!r}")
        return t.value

    def _parse_table_ref(self) -> TableRef:
        name = self._parse_ident()
        alias = None
        if self.eat_kw("as"):
            alias = self._parse_alias_name()
        elif self.peek().kind == TokKind.IDENT:
            alias = self.next().value
        return TableRef(name, alias)

    def _parse_ident(self) -> str:
        t = self.next()
        if t.kind != TokKind.IDENT:
            raise SQLError(f"expected identifier but got {t.value!r}")
        return t.value

    def _parse_join(self) -> JoinClause:
        join_type = "INNER"
        if self.eat_kw("inner"):
            pass
        elif self.eat_kw("left"):
            join_type = "LEFT"
            self.eat_kw("outer")
        elif self.at_kw("right"):
            raise SQLError("RIGHT JOIN is not supported (rewrite as LEFT JOIN)")
        self.expect_kw("join")
        table = self._parse_table_ref()
        self.expect_kw("on")
        cond = self._parse_expr()
        return JoinClause(table, join_type, cond)

    def _parse_where(self):
        if not self.eat_kw("where"):
            return None
        return self._parse_expr()

    def _parse_group_by(self) -> list:
        if not self.eat_kw("group"):
            return []
        self.expect_kw("by")
        exprs = [self._parse_expr()]
        while self.eat_punct(","):
            exprs.append(self._parse_expr())
        return exprs

    def _parse_having(self):
        if not self.eat_kw("having"):
            return None
        return self._parse_expr()

    def _parse_order_by(self) -> list[OrderKey]:
        if not self.eat_kw("order"):
            return []
        self.expect_kw("by")
        keys = [self._parse_order_key()]
        while self.eat_punct(","):
            keys.append(self._parse_order_key())
        return keys

    def _parse_order_key(self) -> OrderKey:
        expr = self._parse_expr()
        desc = False
        if self.eat_kw("asc"):
            desc = False
        elif self.eat_kw("desc"):
            desc = True
        nulls_first = None
        if self.eat_kw("nulls"):
            if self.eat_kw("first"):
                nulls_first = True
            else:
                self.expect_kw("last")
                nulls_first = False
        return OrderKey(expr, desc, nulls_first)

    def _parse_limit(self):
        limit = None
        offset = 0
        if self.eat_kw("limit"):
            limit = self._parse_nonneg_int("LIMIT")
            if self.eat_kw("offset"):
                offset = self._parse_nonneg_int("OFFSET")
        elif self.eat_kw("offset"):
            offset = self._parse_nonneg_int("OFFSET")
        return limit, offset

    def _parse_nonneg_int(self, clause: str) -> int:
        t = self.next()
        if t.kind != TokKind.NUMBER or "." in t.value or "e" in t.value.lower():
            raise SQLError(f"{clause} expects a non-negative integer")
        v = int(t.value)
        if v < 0:
            raise SQLError(f"{clause} expects a non-negative integer")
        return v

    # -- expressions ------------------------------------------------------
    def _parse_expr(self):
        return self._parse_or()

    def _parse_or(self):
        left = self._parse_and()
        while self.at_kw("or"):
            self.next()
            right = self._parse_and()
            left = BinOp("OR", left, right)
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self.at_kw("and"):
            self.next()
            right = self._parse_not()
            left = BinOp("AND", left, right)
        return left

    def _parse_not(self):
        if self.eat_kw("not"):
            return UnaryOp("NOT", self._parse_not())
        return self._parse_predicate()

    def _parse_predicate(self):
        left = self._parse_add()
        while True:
            t = self.peek()
            is_compare = (
                t.kind == TokKind.PUNCT
                and t.value in ("=", "<>", "!=", "<", ">", "<=", ">=")
            )
            if is_compare:
                op_tok = self.next()
                op = "!=" if op_tok.value == "!=" else op_tok.value
                right = self._parse_add()
                left = BinOp(op, left, right)
            elif self.at_kw("is"):
                self.next()
                neg = self.eat_kw("not")
                self.expect_kw("null")
                left = BinOp("IS NOT NULL" if neg else "IS NULL", left, Literal(None, STRING))
            elif self.at_kw("like"):
                self.next()
                pat = self._parse_add()
                left = LikeExpr(left, pat, False)
            elif self.at_kw("not") and self.peek(1).value == "like":
                self.next(); self.next()
                pat = self._parse_add()
                left = LikeExpr(left, pat, True)
            elif self.at_kw("in"):
                self.next()
                left = self._parse_in_tail(left, False)
            elif self.at_kw("not") and self.peek(1).value == "in":
                self.next(); self.next()
                left = self._parse_in_tail(left, True)
            elif self.at_kw("between"):
                self.next()
                lo = self._parse_add()
                self.expect_kw("and")
                hi = self._parse_add()
                left = BinOp("AND", BinOp(">=", left, lo), BinOp("<=", left, hi))
            else:
                return left

    def _parse_in_tail(self, expr, negated: bool) -> InList:
        self.expect_punct("(")
        values = []
        if not self.at_punct(")"):
            values.append(self._parse_expr())
            while self.eat_punct(","):
                values.append(self._parse_expr())
        self.expect_punct(")")
        return InList(expr, values, negated)

    def _parse_add(self):
        left = self._parse_mul()
        while self.at_punct("+") or self.at_punct("-") or self.at_punct("||"):
            op = self.next().value
            right = self._parse_mul()
            left = BinOp(op, left, right)
        return left

    def _parse_mul(self):
        left = self._parse_unary()
        while self.at_punct("*") or self.at_punct("/"):
            op = self.next().value
            right = self._parse_unary()
            left = BinOp(op, left, right)
        return left

    def _parse_unary(self):
        if self.at_punct("-"):
            self.next()
            return UnaryOp("-", self._parse_unary())
        if self.at_punct("+"):
            self.next()
            return UnaryOp("+", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self):
        t = self.peek()
        if t.kind == TokKind.NUMBER:
            self.next()
            if "." in t.value or "e" in t.value.lower():
                return Literal(float(t.value), FLOAT64)
            iv = int(t.value)
            # SQL standard: an exact numeric integer literal outside the
            # bigint range is a data exception, not an implicit float cast
            if not (-(2**63) <= iv <= 2**63 - 1):
                raise SQLError(
                    f"integer literal out of bigint range: {t.value}")
            return Literal(iv, INT64)
        if t.kind == TokKind.STRING:
            self.next()
            return Literal(t.value, STRING)
        if self.at_kw("null"):
            self.next()
            return Literal(None, STRING)
        if self.at_kw("true"):
            self.next()
            return Literal(1, INT64)
        if self.at_kw("false"):
            self.next()
            return Literal(0, INT64)
        if self.at_kw("date"):
            self.next()
            s = self.next()
            if s.kind != TokKind.STRING:
                raise SQLError("DATE requires a string literal")
            return Literal(s.value, STRING, sql_type="DATE")
        if self.at_kw("case"):
            return self._parse_case()
        if self.at_kw("extract"):
            return self._parse_extract()
        if self.at_kw("substring"):
            return self._parse_substring()
        if self.at_kw("coalesce"):
            self.next()
            return self._parse_keyword_func_call("coalesce")
        if self.at_kw("upper") or self.at_kw("lower") or self.at_kw("length") \
                or self.at_kw("trim") or self.at_kw("ltrim") or self.at_kw("rtrim"):
            name = self.next().value
            return self._parse_keyword_func_call(name)
        if t.kind == TokKind.IDENT:
            return self._parse_ident_or_call()
        if self.at_punct("("):
            self.next()
            e = self._parse_expr()
            self.expect_punct(")")
            return e
        raise SQLError(f"unexpected token {t.value!r}")

    def _parse_keyword_func_call(self, fname: str) -> FuncExpr:
        self.expect_punct("(")
        args = []
        if not self.at_punct(")"):
            args.append(self._parse_expr())
            while self.eat_punct(","):
                args.append(self._parse_expr())
        self.expect_punct(")")
        return FuncExpr(fname, args)

    def _parse_ident_or_call(self):
        name = self.next().value
        if self.at_punct("("):
            return self._parse_func_call(name)
        if self.eat_punct("."):
            if self.at_punct("*"):
                self.next()
                return Star(qualifier=name)
            col = self._parse_ident()
            return ColumnRef(name, col)
        return ColumnRef(None, name)

    def _parse_func_call(self, fname: str) -> FuncExpr:
        self.expect_punct("(")
        low = fname.lower()
        distinct = False
        if self.eat_kw("distinct"):
            distinct = True
        if low in ("count",) and self.at_punct("*"):
            self.next()
            self.expect_punct(")")
            return AggCall("count", None, distinct=distinct, star=True)
        args = []
        if not self.at_punct(")"):
            args.append(self._parse_expr())
            while self.eat_punct(","):
                args.append(self._parse_expr())
        self.expect_punct(")")
        if low in ("count", "sum", "avg", "min", "max"):
            if len(args) != 1:
                raise SQLError(f"{fname.upper()} takes exactly one argument")
            return AggCall(low, args[0], distinct=distinct)
        if low == "substr":
            return FuncExpr("substring", args)
        return FuncExpr(low, args)

    def _parse_case(self) -> CaseExpr:
        self.expect_kw("case")
        whens = []
        while self.eat_kw("when"):
            cond = self._parse_expr()
            self.expect_kw("then")
            res = self._parse_expr()
            whens.append((cond, res))
        if not whens:
            raise SQLError("CASE requires at least one WHEN")
        else_expr = None
        if self.eat_kw("else"):
            else_expr = self._parse_expr()
        self.expect_kw("end")
        return CaseExpr(whens, else_expr)

    def _parse_extract(self) -> FuncExpr:
        self.expect_kw("extract")
        self.expect_punct("(")
        field_tok = self.next()
        if field_tok.kind not in (TokKind.IDENT, TokKind.KEYWORD):
            raise SQLError("EXTRACT field expected (YEAR/MONTH/DAY)")
        field = field_tok.value.upper()
        if field not in ("YEAR", "MONTH", "DAY"):
            raise SQLError(f"unsupported EXTRACT field {field}")
        self.expect_kw("from")
        arg = self._parse_expr()
        self.expect_punct(")")
        return FuncExpr("extract", [arg], kwargs={"field": field})

    def _parse_substring(self) -> FuncExpr:
        self.expect_kw("substring")
        self.expect_punct("(")
        s = self._parse_substring_arg()
        args = [s]
        if self.eat_kw("from"):
            args.append(self._parse_substring_arg())
            if self.eat_kw("for"):
                args.append(self._parse_substring_arg())
        else:
            args.append(self._parse_substring_arg())
            if self.eat_punct(","):
                args.append(self._parse_substring_arg())
        self.expect_punct(")")
        return FuncExpr("substring", args)

    def _parse_substring_arg(self):
        """Parse arithmetic only; stop before FROM/FOR/)/, so those tokens
        remain visible to the SUBSTRING grammar."""
        left = self._parse_unary()
        while True:
            if self.at_punct("+") or self.at_punct("-") or self.at_punct("*") \
                    or self.at_punct("/") or self.at_punct("||"):
                op = self.next().value
            else:
                break
            right = self._parse_unary()
            left = BinOp(op, left, right)
        return left


def parse_sql(sql: str) -> SelectStmt:
    return Parser(sql).parse_select()
