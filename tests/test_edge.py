"""Edge-case tests: NULL three-valued logic in joins/predicates, string
ordering, empty relations, and SQL-standard integer arithmetic.

Run: py -I tests/test_edge.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_sql import Engine, SQLError  # noqa: E402

TMP = tempfile.mkdtemp(prefix="edgecsv_")


def write_file(name, text):
    p = os.path.join(TMP, name)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return p


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        raise AssertionError(f"FAILED {name}: {detail}")


def main():
    left = write_file("l.csv",
        "id|k|tag\n"
        "1|10|a\n"
        "2||b\n"      # NULL join key
        "3|30|c\n"
        "4|10|d\n")
    right = write_file("r.csv",
        "k|label\n"
        "10|ten\n"
        "20|twenty\n"
        "|nullkey\n")  # NULL key on build side
    eng = Engine(memory_limit_bytes=64 * 1024 * 1024)
    eng.mount("l", left)
    eng.mount("r", right)

    print("== NULL never matches in inner join ==")
    r = eng.run("""
        SELECT l.id, r.label FROM l JOIN r ON l.k = r.k
        ORDER BY l.id, r.label
    """)
    # only id 1 and 4 (k=10) match; NULL keys on either side never join
    check("inner null exclusion",
          r.rows == [(1, "ten"), (4, "ten")], r.rows)

    print("== NULL preserved as unmatched in left join ==")
    r = eng.run("""
        SELECT l.id, r.label FROM l LEFT JOIN r ON l.k = r.k
        ORDER BY l.id
    """)
    check("left null padding",
          r.rows == [(1, "ten"), (2, None), (3, None), (4, "ten")],
          r.rows)

    print("== NULL in NULL-safe IS NULL on join output ==")
    r = eng.run("""
        SELECT l.id FROM l LEFT JOIN r ON l.k = r.k
        WHERE r.label IS NULL ORDER BY l.id
    """)
    check("is null after join", [x[0] for x in r.rows] == [2, 3], r.rows)

    print("== three-valued AND/OR truth tables ==")
    # tag='b' row has NULL k; combine predicates to exercise UNKNOWN
    r = eng.run("SELECT id FROM l WHERE k = 30 OR k IS NULL ORDER BY id")
    check("OR with unknown", [x[0] for x in r.rows] == [2, 3], r.rows)
    r = eng.run("SELECT id FROM l WHERE k > 5 AND tag = 'b'")
    check("AND unknown excluded", r.rows == [], r.rows)
    r = eng.run("SELECT id FROM l WHERE NOT (k > 5) ORDER BY id")
    # NOT(UNKNOWN)=UNKNOWN -> NULL row 2 excluded; only rows falsifying k>5
    # (none here satisfy; k=10,30,10 are true, row2 unknown) -> empty
    check("NOT unknown stays excluded", r.rows == [], r.rows)

    print("== string comparison and ordering ==")
    words = write_file("w.csv", "w\nbanana\napple\nCherry\napple\n")
    eng.mount("words", words)
    r = eng.run("SELECT w FROM words ORDER BY w")
    # SQL string ordering is by code point: uppercase 'C' (67) < lowercase
    check("string code-point order",
          [x[0] for x in r.rows] == ["Cherry", "apple", "apple", "banana"],
          r.rows)
    r = eng.run("SELECT w FROM words WHERE w > 'banana' ORDER BY w")
    check("string gt", [x[0] for x in r.rows] == [], r.rows)
    r = eng.run("SELECT COUNT(*) FROM words WHERE w = 'apple'")
    check("string eq count", r.rows == [(2,)], r.rows)

    print("== empty relations and aggregates ==")
    empty = write_file("e.csv", "a|b\n")
    eng.mount("empty_t", empty)
    r = eng.run("SELECT * FROM empty_t")
    check("empty scan", r.rows == [])
    r = eng.run("SELECT COUNT(*), SUM(a), AVG(a), MIN(a), MAX(a) FROM empty_t")
    check("agg on empty: count 0, rest NULL",
          r.rows == [(0, None, None, None, None)], r.rows)
    r = eng.run("SELECT a FROM empty_t WHERE a = 1")
    check("filter on empty", r.rows == [])

    print("== integer overflow and exact division semantics ==")
    try:
        eng.run("SELECT id + 9223372036854775807 FROM l WHERE id = 1")
        raise AssertionError("expected overflow error")
    except SQLError as ex:
        check("int add overflow raises", "out of range" in str(ex), str(ex))
    try:
        # -9223372036854775808 negated overflows bigint
        eng.run("SELECT -(-9223372036854775807 - 1) FROM l WHERE id = 1")
        raise AssertionError("expected overflow error")
    except SQLError as ex:
        check("unary minus overflow raises", "out of range" in str(ex), str(ex))
    # division always yields float64 here and never truncates integers
    r = eng.run("SELECT id / 2 FROM l WHERE id = 1")
    check("division is float", r.rows == [(0.5,)], r.rows)

    print("== LIMIT / OFFSET boundary ==")
    r = eng.run("SELECT id FROM l ORDER BY id LIMIT 2 OFFSET 1")
    check("limit offset", [x[0] for x in r.rows] == [2, 3], r.rows)
    r = eng.run("SELECT id FROM l ORDER BY id LIMIT 100")
    check("limit beyond count", len(r.rows) == 4)

    print("\nAll edge-case tests passed.")


if __name__ == "__main__":
    main()
