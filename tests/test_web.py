"""End-to-end test of the web console over real HTTP (no external client)."""

import json
import os
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_sql import Engine  # noqa: E402
from web.app import WebServer  # noqa: E402

TMP = tempfile.mkdtemp(prefix="webcsv_")


def req(method, url, body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=20) as resp:
        raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        if "json" in ctype:
            return json.loads(raw.decode("utf-8"))
        return raw


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        raise AssertionError(f"FAILED {name}: {detail}")


def main():
    csv_path = os.path.join(TMP, "nums.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("id|grp|val\n")
        for i in range(500):
            fh.write(f"{i}|{i % 4}|{(i * 7) % 100}.25\n")
        fh.write("dirty|row\n")  # dirty

    eng = Engine(memory_limit_bytes=8 * 1024 * 1024, parallel=2)
    server = WebServer(eng, port=8799)
    url = server.start()
    base = url.rstrip("/")
    try:
        print("== static page serves ==")
        html = req("GET", base + "/")
        check("html served", b"Mini SQL" in html and b"SQL" in html)

        print("== mount over API ==")
        d = req("POST", base + "/api/mount", {
            "name": "nums", "path": csv_path, "delimiter": "|"})
        check("mount ok", d["ok"] and "nums" in d["sources"])
        check("types inferred",
              d["source"]["schema"][0]["type"] == "int64")

        print("== explain over API ==")
        d = req("POST", base + "/api/explain",
                {"sql": "SELECT grp, SUM(val) FROM nums WHERE id > 10 GROUP BY grp"})
        check("explain has pushdown", "predicate pushed" in d["plan"], d["plan"])

        print("== async query + polling ==")
        d = req("POST", base + "/api/query", {
            "sql": "SELECT grp AS g, COUNT(*) AS n, AVG(val) AS a "
                   "FROM nums GROUP BY grp ORDER BY grp",
            "max_rows": 100})
        job_id = d["job_id"]
        final = None
        for _ in range(60):
            j = req("GET", f"{base}/api/job/{job_id}")
            if j["status"] in ("done", "error", "cancelled"):
                final = j
                break
            time.sleep(0.2)
        check("job finished", final is not None and final["status"] == "done",
              final and final.get("error"))
        check("4 groups", len(final["rows"]) == 4, final["rows"])
        check("counts total 500", sum(r[1] for r in final["rows"]) == 500)
        check("plan present", final["stats"]["plan"]["node"] == "Project")
        check("dirty counted",
              final["stats"]["sources"]["nums"]["rows_dirty"] == 1)

        print("== result download ==")
        csv_bytes = req("GET", f"{base}/api/job/{job_id}/download")
        text = csv_bytes.decode("utf-8")
        check("csv header", text.splitlines()[0] == "g|n|a", text[:40])
        check("csv rows", len(text.strip().splitlines()) == 5)

        print("== cancellation ==")
        d = req("POST", base + "/api/query", {
            "sql": "SELECT a.id FROM nums a "
                   "JOIN nums b ON a.grp = b.grp ORDER BY a.id",
            "max_rows": 10})
        cid = d["job_id"]
        req("POST", base + "/api/cancel", {"job_id": cid})
        cfinal = None
        for _ in range(60):
            j = req("GET", f"{base}/api/job/{cid}")
            if j["status"] in ("done", "error", "cancelled"):
                cfinal = j
                break
            time.sleep(0.1)
        check("cancel observed (or already finished)",
              cfinal is not None and cfinal["status"] in ("cancelled", "done"),
              cfinal and cfinal["status"])

        print("== error handling ==")
        d = req("POST", base + "/api/query",
                {"sql": "SELECT nope FROM nums", "max_rows": 10})
        jid = d["job_id"]
        for _ in range(40):
            j = req("GET", f"{base}/api/job/{jid}")
            if j["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.1)
        check("semantic error reported", j["status"] == "error" and "nope" in j["error"],
              j)

        print("== unmount ==")
        d = req("POST", base + "/api/unmount", {"name": "nums"})
        check("unmount ok", "nums" not in d["sources"])

        print("\nAll web console tests passed.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
