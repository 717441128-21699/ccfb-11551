"""Forces spill paths (external sort + grace hash aggregate/join) and checks
results against the in-memory path, plus parallel CSV parsing correctness.

Run: py -I tests/test_spill.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_sql import Engine  # noqa: E402

TMP = tempfile.mkdtemp(prefix="spillcsv_")


def write_file(name: str, text: str) -> str:
    p = os.path.join(TMP, name)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return p


def make_data(n: int):
    orders = os.path.join(TMP, "orders_big.csv")
    with open(orders, "w", encoding="utf-8", newline="") as fh:
        fh.write("o_key|o_cust|o_amount|o_note\n")
        for i in range(n):
            cust = i % 7
            amount = (i * 37) % 1000
            note = f"note_{i % 13}"
            fh.write(f"{i}|{cust}|{amount}.5|{note}\n")
    parts = os.path.join(TMP, "parts.csv")
    with open(parts, "w", encoding="utf-8", newline="") as fh:
        fh.write("p_cust|p_region\n")
        for c in range(7):
            fh.write(f"{c}|region_{c % 3}\n")
    return orders, parts


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        raise AssertionError(f"FAILED {name}: {detail}")


def main():
    n = 20_000
    orders, parts = make_data(n)

    # reference: generous memory, single worker
    ref = Engine(memory_limit_bytes=512 * 1024 * 1024, parallel=1)
    ref.mount("orders", orders)
    ref.mount("parts", parts)

    # stressed: tiny budget so sort/agg/join all spill, 3 parser workers
    tiny = Engine(memory_limit_bytes=96 * 1024, parallel=3,
                  spill_dir=os.path.join(TMP, "spill"))
    tiny.mount("orders", orders)
    tiny.mount("parts", parts)

    print("== external sort spills and matches ==")
    sql_sort = ("SELECT o_key, o_amount FROM orders "
                "ORDER BY o_amount DESC, o_key ASC LIMIT 500")
    r1 = ref.run(sql_sort)
    r2 = tiny.run(sql_sort)
    check("sort rows identical", r1.rows == r2.rows, "mismatch")
    check("sort actually spilled",
          _spilled(r2.stats.plan),
          r2.stats.plan)
    check("sort spill bytes recorded", r2.stats.spill_bytes_written > 0)

    print("== hash aggregate spills and matches ==")
    sql_agg = """
        SELECT o_cust AS ck,
               COUNT(*) AS n,
               SUM(o_amount) AS s,
               AVG(o_amount) AS a,
               MIN(o_amount) AS mn,
               MAX(o_amount) AS mx
        FROM orders GROUP BY o_cust ORDER BY o_cust
    """
    a1 = ref.run(sql_agg)
    a2 = tiny.run(sql_agg)
    check("agg row count", len(a1.rows) == 7 and len(a2.rows) == 7)
    for x, y in zip(a1.rows, a2.rows):
        check(f"group {x[0]} exact", x == y, f"{x} != {y}")

    # high-cardinality aggregation: one group per key forces the Python group
    # table to grow until it crosses the spill threshold
    sql_agg_spill = """
        SELECT o_note AS note, o_key AS k, COUNT(*) AS n,
               SUM(o_amount) AS total
        FROM orders GROUP BY o_note, o_key ORDER BY o_key
    """
    s1 = ref.run(sql_agg_spill)
    s2 = tiny.run(sql_agg_spill)
    check("high-card agg rows", len(s1.rows) == n and len(s2.rows) == n)
    check("high-card agg identical", s1.rows == s2.rows)
    check("agg actually spilled", _spilled(s2.stats.plan), s2.stats.plan)

    print("== aggregate over zero rows returns empty group set ==")
    e = tiny.run("SELECT o_cust, COUNT(*) FROM orders WHERE o_key < 0 GROUP BY o_cust")
    check("empty aggregation", e.rows == [])
    e2 = tiny.run("SELECT COUNT(*) FROM orders WHERE o_key < 0")
    check("count(*) no group -> 0", e2.rows == [(0,)], e2.rows)

    print("== grace hash join spills and matches inner join ==")
    sql_join = """
        SELECT o.o_key, p.p_region
        FROM parts AS p
        JOIN orders AS o ON p.p_cust = o.o_cust
        ORDER BY o.o_key
    """
    # force join spill with an even tinier budget
    micro = Engine(memory_limit_bytes=64 * 1024, parallel=2)
    micro.mount("orders", orders)
    micro.mount("parts", parts)
    j1 = ref.run(sql_join)
    j2 = micro.run(sql_join)
    check("join row count", len(j1.rows) == n and len(j2.rows) == n,
          (len(j1.rows), len(j2.rows)))
    check("join rows identical", j1.rows == j2.rows)
    check("grace join actually spilled", _spilled(j2.stats.plan),
          j2.stats.plan)

    print("== left join with spill ==")
    # add an unmatched probe-side customer via a small extra table
    extra = write_file("extra.csv", "e_cust|e_tag\n99|lonely\n")
    ref.mount("extra", extra)
    micro.mount("extra", extra)
    sql_left = """
        SELECT e.e_tag, o.o_key
        FROM extra AS e
        LEFT JOIN orders AS o ON e.e_cust = o.o_cust
    """
    l1 = ref.run(sql_left)
    l2 = micro.run(sql_left)
    check("left join null pad", l1.rows == l2.rows == [("lonely", None)],
          (l1.rows, l2.rows))

    print("== parallel parse produces identical rows/columns ==")
    seq = Engine(memory_limit_bytes=512 * 1024 * 1024, parallel=1)
    par = Engine(memory_limit_bytes=512 * 1024 * 1024, parallel=4)
    seq.mount("orders", orders)
    par.mount("orders", orders)
    s = seq.run("SELECT o_cust, SUM(o_amount) FROM orders GROUP BY o_cust ORDER BY o_cust")
    p = par.run("SELECT o_cust, SUM(o_amount) FROM orders GROUP BY o_cust ORDER BY o_cust")
    check("parallel result equal", s.rows == p.rows)
    check("parallel dirty count zero",
          par.sources["orders"].rows_dirty == 0
          and par.sources["orders"].rows_ok == n,
          (par.sources["orders"].rows_ok, par.sources["orders"].rows_dirty))

    print("== dirty rows are counted and skipped ==")
    dirty = write_file("dirty.csv",
                       "a|b\n1|x\n2|y\nbad\n3|z\nalso|bad|extra\n")
    d_eng = Engine(memory_limit_bytes=64 * 1024 * 1024)
    d_eng.mount("t", dirty)
    dr = d_eng.run("SELECT a FROM t ORDER BY a")
    check("dirty skipped", [r[0] for r in dr.rows] == [1, 2, 3], dr.rows)
    check("dirty counted", d_eng.sources["t"].rows_dirty == 2
          and d_eng.sources["t"].rows_seen == 5,
          (d_eng.sources["t"].rows_dirty, d_eng.sources["t"].rows_seen))

    print("== all temp spill files removed after run ==")
    leftover = [d for d in os.listdir(tempfile.gettempdir())
                if d.startswith("minisql_")]
    check("no leftover spill dirs", not leftover, leftover)

    print("\nAll spill/parallel tests passed.")


def _spilled(node: dict) -> bool:
    if node.get("spilled"):
        return True
    return any(_spilled(c) for c in node.get("children", []))


if __name__ == "__main__":
    main()
