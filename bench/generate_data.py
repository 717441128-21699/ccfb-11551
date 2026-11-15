"""Generate TPC-H data with DuckDB's generator and export pipe-delimited CSV.

DuckDB is used strictly as the *data generator and reference database*; the
mini SQL engine itself never imports it.  Dates are written as ISO text
(``YYYY-MM-DD``) so both engines compare them as strings, and decimals are
written as their canonical decimal text (later parsed as float64).

Usage
-----
py -I bench/generate_data.py --sf 0.01 --out data/tpch_sf001
py -I bench/generate_data.py --sf 10   --out data/tpch_sf10 --dirty 0.001
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import duckdb

TABLES = [
    "nation", "region", "part", "supplier", "partsupp",
    "customer", "orders", "lineitem",
]


def generate(sf: float, out_dir: str, dirty_frac: float = 0.0,
             delimiter: str = "|") -> None:
    os.makedirs(out_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    t0 = time.perf_counter()
    print(f"generating TPC-H sf={sf} …")
    con.execute(f"CALL dbgen(sf={sf});")

    for table in TABLES:
        path = os.path.join(out_dir, f"{table}.csv")
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        # HEADER TRUE, no quoting (TPC-H text contains no pipe/newline),
        # dates rendered as ISO strings
        con.execute(f"""
            COPY {table} TO '{path.replace(os.sep, "/")}'
            (HEADER TRUE, DELIMITER '{delimiter}', QUOTE '',
             DATEFORMAT '%Y-%m-%d', TIMESTAMPFORMAT '%Y-%m-%d')
        """)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  exported {table:9s} {n:>12,d} rows  {size_mb:9.1f} MB")
        if dirty_frac > 0 and n >= 1000:
            injected = inject_dirty(path, dirty_frac, delimiter)
            print(f"             + {injected} dirty rows injected "
                  f"({dirty_frac*100:.2f}%)")
    print(f"done in {time.perf_counter()-t0:.1f}s -> {out_dir}")


def inject_dirty(path: str, frac: float, delimiter: str) -> int:
    """Append malformed rows (wrong field count / bad numeric token) and
    return how many were injected.  Appending keeps canonical rows intact so
    the reference result stays deterministic."""
    import csv

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = next(reader)
        width = len(header)
    target = max(1, int(frac * sum(1 for _ in open(path, "rb"))))
    target = min(target, 100_000)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        for i in range(target):
            kind = i % 3
            if kind == 0:
                # wrong field count -> always rejected atomically
                fh.write("this|is|a|malformed|row\n")
            elif kind == 1:
                # correct width but a non-numeric token in a numeric key col
                vals = ["0"] * width
                vals[0] = "NOT_AN_INT"
                fh.write(delimiter.join(vals) + "\n")
            else:
                # correct width but every numeric-looking position carries a
                # non-numeric sentinel, so no valid numeric cell can leak in
                vals = ["__dirty__"] * width
                fh.write(delimiter.join(vals) + "\n")
    return target


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, required=True, help="scale factor")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dirty", type=float, default=0.0,
                    help="fraction of dirty rows to inject into large tables")
    ap.add_argument("--delimiter", default="|")
    args = ap.parse_args(argv)
    generate(args.sf, args.out, args.dirty, args.delimiter)


if __name__ == "__main__":
    sys.exit(main())
