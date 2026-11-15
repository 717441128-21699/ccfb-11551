"""Launch the web query console.

Examples
--------
py run_web.py --memory-mb 2048 --parallel 4 --port 8765
py run_web.py --mount orders=D:/data/orders.csv --mount customer=D:/data/customer.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser

# allow running both as ``py run_web.py`` and ``python -m`` from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mini_sql import Engine  # noqa: E402
from web.app import WebServer  # noqa: E402


def parse_mount(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--mount expects name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mini SQL web console")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--memory-mb", type=int, default=2048,
                    help="per-query memory budget in MiB (default 2048 = 2GB)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="CSV parsing worker processes")
    ap.add_argument("--spill-dir", default=None,
                    help="directory for spill temp files (system temp if omitted)")
    ap.add_argument("--mount", action="append", default=[],
                    help="name=path pair, repeatable")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    engine = Engine(
        memory_limit_bytes=args.memory_mb * 1024 * 1024,
        parallel=args.parallel,
        spill_dir=args.spill_dir,
    )
    for spec in args.mount:
        name, path = parse_mount(spec)
        src = engine.mount(name, path)
        print(f"mounted {name}: {path} "
              f"({len(src.column_names)} columns, types inferred)")

    server = WebServer(engine, host=args.host, port=args.port)
    url = server.start()
    print(f"\nMini SQL console running at {url}")
    print(f"memory budget: {args.memory_mb} MiB | parser workers: {args.parallel}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nshutting down…")
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
