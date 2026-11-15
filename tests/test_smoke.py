"""Smoke tests for the mini SQL engine (run with: py -m tests.test_smoke)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_sql import Engine, SQLError  # noqa: E402
from mini_sql.types import CancelledError  # noqa: E402
from mini_sql.spill import CancelToken  # noqa: E402

TMP = tempfile.mkdtemp(prefix="smokecsv_")


def write_file(name: str, text: str) -> str:
    p = os.path.join(TMP, name)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return p


def build_engine(**kw) -> Engine:
    eng = Engine(memory_limit_bytes=kw.pop("memory_limit_bytes", 64 * 1024 * 1024),
                 parallel=kw.pop("parallel", 1), **kw)
    orders = write_file("orders.csv",
        "o_orderkey|o_custkey|o_totalprice|o_orderdate|o_status\n"
        "1|10|100.50|1998-01-01|F\n"
        "2|10|200.00|1998-02-01|O\n"
        "3|20|50.25|1998-03-01|F\n"
        "4|30|999.99|1998-04-01|F\n"          # dirty later? no
        "5|20||1998-05-01|F\n"                # NULL price
        "badrow\n"                              # dirty: wrong field count
        "6|10|75.00|not-a-date|F\n"            # date column is string -> ok
    )
    customers = write_file("customers.csv",
        "c_custkey|c_name|c_nationkey\n"
        "10|Alice|1\n"
        "20|Bob|1\n"
        "40|Carol|2\n"                          # no orders (left join test)
    )
    eng.mount("orders", orders)
    eng.mount("customer", customers)
    return eng


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        raise AssertionError(f"FAILED {name}: {detail}")


def main():
    eng = build_engine()

    print("== basic select + where + order + limit ==")
    r = eng.run("""
        SELECT o_orderkey AS k, o_totalprice AS p
        FROM orders
        WHERE o_status = 'F'
        ORDER BY o_orderkey DESC
        LIMIT 3
    """)
    check("dirty row skipped", eng.sources["orders"].rows_dirty == 1,
          eng.sources["orders"].rows_dirty)
    check("row count", len(r.rows) == 3, r.rows)
    check("order desc", [row[0] for row in r.rows] == [6, 5, 4], r.rows)
    check("null preserved", r.rows[1][1] is None, r.rows)

    print("== three-valued logic ==")
    r = eng.run("SELECT o_orderkey FROM orders WHERE o_totalprice > 60 ORDER BY o_orderkey")
    check("null excluded by >", [x[0] for x in r.rows] == [1, 2, 4, 6], r.rows)
    r = eng.run("SELECT o_orderkey FROM orders WHERE NOT (o_totalprice > 60) ORDER BY o_orderkey")
    # NOT(UNKNOWN)=UNKNOWN: NULL row still excluded
    check("NOT preserves UNKNOWN", [x[0] for x in r.rows] == [3], r.rows)
    r = eng.run("SELECT o_orderkey FROM orders WHERE o_totalprice IS NULL")
    check("IS NULL", [x[0] for x in r.rows] == [5], r.rows)
    r = eng.run("SELECT o_orderkey FROM orders WHERE o_totalprice IS NOT NULL ORDER BY o_orderkey")
    check("IS NOT NULL", len(r.rows) == 5, r.rows)

    print("== aggregates + group by + having ==")
    r = eng.run("""
        SELECT o_custkey AS ck,
               COUNT(*) AS n,
               SUM(o_totalprice) AS s,
               AVG(o_totalprice) AS a,
               MIN(o_totalprice) AS mn,
               MAX(o_totalprice) AS mx
        FROM orders
        GROUP BY o_custkey
        HAVING COUNT(*) >= 2
        ORDER BY o_custkey
    """)
    check("groups", len(r.rows) == 2, r.rows)
    check("count/sum", r.rows[0][:3] == (10, 3, 375.5), r.rows[0])
    check("avg ignores null", abs(r.rows[1][3] - 50.25) < 1e-9, r.rows)
    check("sum all-null group is NULL",
          eng.run("SELECT o_custkey, SUM(o_totalprice) FROM orders "
                  "WHERE o_orderkey = 5 GROUP BY o_custkey").rows[0][1] is None)

    print("== inner join ==")
    r = eng.run("""
        SELECT c.c_name, o.o_orderkey
        FROM customer AS c
        JOIN orders AS o ON c.c_custkey = o.o_custkey
        ORDER BY c.c_name, o.o_orderkey
    """)
    names = [x[0] for x in r.rows]
    check("inner join excludes Carol", "Carol" not in names and names.count("Alice") == 3,
          r.rows)

    print("== left join preserves unmatched left rows ==")
    r = eng.run("""
        SELECT c.c_name, o.o_orderkey
        FROM customer AS c
        LEFT JOIN orders AS o ON c.c_custkey = o.o_custkey
        WHERE c.c_name = 'Carol'
    """)
    check("left join null padding", r.rows == [("Carol", None)], r.rows)

    print("== string functions / case / extract / in / like ==")
    r = eng.run("""
        SELECT o_orderkey,
               CASE WHEN o_totalprice > 100 THEN 'big' ELSE 'small' END AS sz,
               SUBSTRING(o_orderdate FROM 1 FOR 4) AS yr,
               COALESCE(o_totalprice, 0.0) AS pp
        FROM orders WHERE o_orderkey IN (1, 5) ORDER BY o_orderkey
    """)
    check("case/substring/coalesce",
          r.rows == [(1, "big", "1998", 100.5), (5, "small", "1998", 0.0)],
          r.rows)
    r = eng.run("SELECT c_name FROM customer WHERE c_name LIKE 'A%'")
    check("like", r.rows == [("Alice",)], r.rows)
    r = eng.run("SELECT c_name FROM customer WHERE c_name LIKE '_arol'")
    check("like underscore", r.rows == [("Carol",)], r.rows)

    print("== projection / predicate push-down visible in plan ==")
    _, plan = eng.explain("SELECT o_orderkey FROM orders WHERE o_custkey = 10")
    check("plan shows pushdown", "predicate pushed" in plan and "projected: o_custkey, o_orderkey" in plan,
          plan)

    print("== integer overflow raises (SQL standard) ==")
    try:
        eng.run("SELECT o_orderkey + 9223372036854775807 FROM orders WHERE o_orderkey = 1")
        raise AssertionError("overflow should raise")
    except SQLError as e:
        check("overflow error", "out of range" in str(e), str(e))

    print("== division by zero raises ==")
    try:
        eng.run("SELECT o_orderkey / 0 FROM orders")
        raise AssertionError("div0 should raise")
    except SQLError as e:
        check("div0 error", "division by zero" in str(e), str(e))

    print("== cancellation ==")
    token = CancelToken()
    token.cancel()
    try:
        eng.run("SELECT * FROM orders", cancel=token)
        raise AssertionError("cancel should raise")
    except CancelledError:
        check("cancelled", True)

    print("== temp files cleaned ==")
    leftover = [d for d in os.listdir(tempfile.gettempdir())
                if d.startswith("minisql_")]
    check("no spill temp dirs left", not leftover, leftover)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
