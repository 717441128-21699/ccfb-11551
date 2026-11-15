"""Run every mini SQL test suite in order.

Usage: py -I tests/run_all.py
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SUITES = [
    "test_smoke",
    "test_edge",
    "test_spill",
    "test_web",
    "test_dirty",
]


def main():
    failures = []
    for name in SUITES:
        print("\n" + "#" * 70)
        print(f"# suite: {name}")
        print("#" * 70)
        try:
            mod = __import__(f"tests.{name}", fromlist=["main"])
            mod.main()
        except Exception:  # noqa: BLE001 - report and continue
            traceback.print_exc()
            failures.append(name)
    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED suites: {', '.join(failures)}")
        return 1
    print(f"ALL {len(SUITES)} SUITES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
