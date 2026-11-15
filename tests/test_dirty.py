"""Verify dirty-row handling on TPC-H-shaped data.

The generator appends malformed rows *after* the canonical TPC-H rows, so the
clean prefix still produces the reference answer.  The mini engine must skip
and count every injected dirty row while returning identical aggregates over
the clean rows.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_sql import Engine  # noqa: E402

TABLES = ["nation", "region", "part", "supplier", "partsupp",
          "customer", "orders", "lineitem"]


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        raise AssertionError(f"FAILED {name}: {detail}")


def main(data_dir: str | None = None, clean_dir: str | None = None):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = data_dir or os.path.join(base, "data", "tpch_dirty")
    clean_dir = clean_dir or os.path.join(base, "data", "tpch_sf001")
    dirty_eng = Engine(memory_limit_bytes=256 * 1024 * 1024, parallel=2)
    clean_eng = Engine(memory_limit_bytes=256 * 1024 * 1024, parallel=2)
    for t in TABLES:
        p = os.path.join(data_dir, f"{t}.csv")
        if os.path.exists(p):
            dirty_eng.mount(t, p)
        cp = os.path.join(clean_dir, f"{t}.csv")
        if os.path.exists(cp):
            clean_eng.mount(t, cp)

    print("== trigger scans so parse stats populate ==")
    for t in TABLES:
        if t in dirty_eng.sources:
            dirty_eng.run(f"SELECT COUNT(*) FROM {t}")
        if t in clean_eng.sources:
            clean_eng.run(f"SELECT COUNT(*) FROM {t}")

    print("== dirty rows counted per table ==")
    total_dirty = 0
    for t in TABLES:
        if t in dirty_eng.sources:
            src = dirty_eng.sources[t]
            if src.rows_dirty:
                print(f"  {t:10s} seen={src.rows_seen:>7d} "
                      f"ok={src.rows_ok:>7d} dirty={src.rows_dirty:>4d}")
                total_dirty += src.rows_dirty
    check("some dirty rows observed", total_dirty > 0, total_dirty)
    check("seen == ok + dirty", all(
        s.rows_seen == s.rows_ok + s.rows_dirty
        for s in dirty_eng.sources.values()))

    print("== aggregates over clean prefix match clean dataset ==")
    queries = {
        "lineitem_count":
            "SELECT COUNT(*) FROM lineitem",
        "lineitem_sum":
            "SELECT SUM(l_extendedprice), AVG(l_quantity) FROM lineitem",
        "group_q1":
            "SELECT l_returnflag, l_linestatus, COUNT(*) "
            "FROM lineitem GROUP BY l_returnflag, l_linestatus "
            "ORDER BY l_returnflag, l_linestatus",
        "orders_join":
            "SELECT COUNT(*) FROM customer AS c "
            "JOIN orders AS o ON c.c_custkey = o.o_custkey",
    }
    for name, sql in queries.items():
        a = clean_eng.run(sql)
        b = dirty_eng.run(sql)
        check(name, a.rows == b.rows, (a.rows, b.rows))

    print("== dirty counts exposed through engine listing ==")
    listing = dirty_eng.list_sources()
    check("listing carries dirty stats",
          all("rows_dirty" in v for v in listing.values()))

    print(f"\nDirty-row handling verified ({total_dirty} rows skipped).")


if __name__ == "__main__":
    main()
