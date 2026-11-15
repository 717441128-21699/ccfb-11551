"""Run TPC-H queries on both engines and require identical result rows.

For every query file in ``queries/`` that the mini engine can express:

1. DuckDB executes the *same SQL text* against an in-process TPC-H instance.
2. The mini engine executes it against the exported pipe-delimited CSVs.
3. Result columns are aligned by position; rows are compared in ORDER BY order
   (every supported query has a deterministic ORDER BY, or returns one row).
4. Comparison rules:
   * integers / strings / dates / NULL  -> exact equality
   * float / decimal aggregates         -> relative tolerance 1e-6 plus
     absolute tolerance 1e-2 (decimal values are stored as float64 here)

A query that fails to parse/plan is reported as UNSUPPORTED rather than a
correctness failure, so the harness documents the expressible subset honestly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_sql import Engine, SQLError  # noqa: E402
from mini_sql.types import FLOAT64  # noqa: E402

TABLES = [
    "nation", "region", "part", "supplier", "partsupp",
    "customer", "orders", "lineitem",
]
ABS_TOL = 1e-2
REL_TOL = 1e-6


def load_sql(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.readlines()
                 if not ln.strip().startswith("--")]
    return "".join(lines).strip()


def run_duckdb(sql: str, sf: float):
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")
    t0 = time.perf_counter()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    elapsed = (time.perf_counter() - t0) * 1000.0
    con.close()
    return cols, rows, elapsed


def mount_csv(engine: Engine, data_dir: str):
    for table in TABLES:
        path = os.path.join(data_dir, f"{table}.csv")
        if os.path.exists(path):
            engine.mount(table, path)


def values_equal(a, b, is_float: bool) -> tuple[bool, str]:
    if a is None and b is None:
        return True, ""
    if a is None or b is None:
        return False, f"NULL mismatch: {a!r} vs {b!r}"
    if is_float:
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return False, "NaN"
        tol = ABS_TOL + REL_TOL * max(abs(fa), abs(fb))
        if abs(fa - fb) <= tol:
            return True, ""
        return False, f"numeric diff: {fa!r} vs {fb!r} (tol {tol:.4g})"
    if str(a) != str(b):
        return False, f"value diff: {a!r} vs {b!r}"
    return True, ""


def compare(cols_d, rows_d, cols_m, result_m) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if len(rows_d) != len(result_m.rows):
        errors.append(
            f"row count: duckdb={len(rows_d)} mini={len(result_m.rows)}")
    # float status per output column from the mini engine's inferred types
    float_cols = {i for i, t in enumerate(result_m.types) if t == FLOAT64}
    n = min(len(rows_d), len(result_m.rows))
    for ri in range(n):
        rd, rm = rows_d[ri], result_m.rows[ri]
        if len(rd) != len(rm):
            errors.append(f"row {ri}: width {len(rd)} vs {len(rm)}")
            continue
        for ci in range(len(rd)):
            ok, msg = values_equal(rd[ci], rm[ci], ci in float_cols)
            if not ok:
                errors.append(f"row {ri} col {ci} ({cols_m[ci]}): {msg}")
                if len(errors) >= 10:
                    return False, errors
    return not errors, errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, required=True)
    ap.add_argument("--data-dir", required=True,
                    help="directory of exported TPC-H CSVs")
    ap.add_argument("--queries", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "queries"))
    ap.add_argument("--memory-mb", type=int, default=2048)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--query", default=None,
                    help="run only one query id, e.g. q1")
    args = ap.parse_args(argv)

    engine = Engine(
        memory_limit_bytes=args.memory_mb * 1024 * 1024,
        parallel=args.parallel,
    )
    mount_csv(engine, args.data_dir)

    qfiles = sorted(
        f for f in os.listdir(args.queries)
        if f.endswith(".sql") and (args.query is None or f.startswith(args.query + "."))
    )
    report = {"scale_factor": args.sf, "memory_mb": args.memory_mb,
              "parallel": args.parallel, "queries": []}
    n_pass = n_fail = n_skip = 0

    for qf in qfiles:
        qid = qf.split(".")[0]
        sql = load_sql(os.path.join(args.queries, qf))
        entry: dict = {"id": qid, "file": qf}
        print(f"\n=== {qid} ({qf}) ===")
        try:
            cols_d, rows_d, ms_d = run_duckdb(sql, args.sf)
            entry["duckdb_rows"] = len(rows_d)
            entry["duckdb_ms"] = round(ms_d, 2)
        except Exception as exc:  # reference itself failed
            entry["status"] = "reference_error"
            entry["error"] = str(exc)
            n_skip += 1
            print(f"  REFERENCE ERROR: {exc}")
            report["queries"].append(entry)
            continue

        try:
            t0 = time.perf_counter()
            result = engine.run(sql)
            ms_m = (time.perf_counter() - t0) * 1000.0
            entry["mini_rows"] = len(result.rows)
            entry["mini_ms"] = round(ms_m, 2)
            entry["mini_peak_memory_bytes"] = result.stats.memory["peak_bytes"]
            entry["spill_bytes_written"] = result.stats.spill_bytes_written
            entry["spilled"] = _any_spill(result.stats.plan)
        except SQLError as exc:
            entry["status"] = "unsupported"
            entry["error"] = str(exc)
            n_skip += 1
            print(f"  UNSUPPORTED: {exc}")
            report["queries"].append(entry)
            continue

        ok, errors = compare(cols_d, rows_d, result.columns, result)
        if ok:
            entry["status"] = "identical"
            n_pass += 1
            print(f"  IDENTICAL  rows={len(result.rows)}  "
                  f"duckdb={ms_d:.1f}ms  mini={ms_m:.1f}ms"
                  + ("  [spilled]" if entry["spilled"] else ""))
        else:
            entry["status"] = "mismatch"
            entry["errors"] = errors
            n_fail += 1
            for e in errors[:5]:
                print(f"  MISMATCH: {e}")
        report["queries"].append(entry)

    report["summary"] = {
        "identical": n_pass, "mismatch": n_fail,
        "unsupported_or_ref_error": n_skip,
        "total": len(qfiles),
    }
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_pass} identical, {n_fail} mismatch, {n_skip} skipped "
          f"out of {len(qfiles)} queries")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"report written to {args.out}")
    return 0 if n_fail == 0 else 1


def _any_spill(node: dict) -> bool:
    if node.get("spilled"):
        return True
    return any(_any_spill(c) for c in node.get("children", []))


if __name__ == "__main__":
    sys.exit(main())
