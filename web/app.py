"""Standard-library HTTP query console for the mini SQL engine.

The server keeps one :class:`~mini_sql.Engine` and a registry of running query
jobs (each with its own :class:`~mini_sql.spill.CancelToken`).  No third-party
web framework is used: routing and JSON encoding sit on top of
``http.server.ThreadingHTTPServer``.

Endpoints
---------
GET  /                      static single-page console
GET  /api/sources           mounted tables, schemas and parse stats
POST /api/mount             mount a CSV file {name, path, delimiter, ...}
POST /api/unmount           {name}
POST /api/query             {sql, max_rows} -> job id (runs in a thread)
POST /api/cancel            {job_id}
GET  /api/job/<id>          status, plan tree, per-operator timings, rows
GET  /api/job/<id>/download result set as CSV
GET  /api/explain           {sql} -> logical plan with push-down annotations
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from mini_sql import Engine, SQLError
from mini_sql.spill import CancelToken, TempManager
from mini_sql.types import CancelledError

from . import static_site


class QueryJob:
    def __init__(self, job_id: str, sql: str):
        self.job_id = job_id
        self.sql = sql
        self.token = CancelToken()
        self.status = "queued"          # queued | running | done | error | cancelled
        self.started = time.perf_counter()
        self.finished = None
        self.result = None
        self.error: str | None = None
        self.plan_tree: str | None = None
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            out = {
                "job_id": self.job_id,
                "status": self.status,
                "elapsed_ms": round(
                    ((self.finished or time.perf_counter()) - self.started) * 1000.0,
                    2),
                "error": self.error,
                "plan_tree": self.plan_tree,
            }
            if self.result is not None:
                r = self.result
                out.update({
                    "columns": r.columns,
                    "types": r.types,
                    "row_count": len(r.rows),
                    "rows": [_json_row(row) for row in r.rows],
                    "stats": {
                        "elapsed_ms": round(r.stats.elapsed_ms, 3),
                        "memory": r.stats.memory,
                        "plan": r.stats.plan,
                        "sources": r.stats.sources,
                        "spill_bytes_written": r.stats.spill_bytes_written,
                        "spill_bytes_read": r.stats.spill_bytes_read,
                        "spill_temp_dir": r.stats.spill_temp_dir,
                    },
                })
            return out


def _json_row(row):
    return [v for v in row]


class WebServer:
    def __init__(self, engine: Engine, host: str = "127.0.0.1", port: int = 8765):
        self.engine = engine
        self.host = host
        self.port = port
        self.jobs: dict[str, QueryJob] = {}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def start(self) -> str:
        server = self
        eng = self.engine

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # quiet by default
                pass

            def _send_json(self, payload, code: int = 200):
                body = json.dumps(payload, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", 0))
                if not length:
                    return {}
                raw = self.rfile.read(length)
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise SQLError(f"invalid JSON body: {exc}") from exc

            def do_GET(self):
                path = urlparse(self.path).path
                try:
                    if path == "/" or path == "/index.html":
                        body = static_site.INDEX_HTML.encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    if path == "/api/sources":
                        self._send_json({"sources": eng.list_sources()})
                        return
                    if path.startswith("/api/job/"):
                        rest = path[len("/api/job/"):]
                        if rest.endswith("/download"):
                            self._download(rest[: -len("/download")])
                        else:
                            self._job_status(rest)
                        return
                    self._send_json({"error": "not found"}, 404)
                except SQLError as exc:
                    self._send_json({"error": str(exc)}, 400)
                except Exception as exc:  # pragma: no cover - defensive
                    self._send_json({"error": f"internal error: {exc}"}, 500)

            def do_POST(self):
                path = urlparse(self.path).path
                try:
                    if path == "/api/mount":
                        self._mount()
                    elif path == "/api/unmount":
                        payload = self._read_json()
                        eng.unmount(payload.get("name", ""))
                        self._send_json({"ok": True, "sources": eng.list_sources()})
                    elif path == "/api/query":
                        self._query()
                    elif path == "/api/cancel":
                        self._cancel()
                    elif path == "/api/explain":
                        payload = self._read_json()
                        _, tree = eng.explain(payload["sql"])
                        self._send_json({"plan": tree})
                    else:
                        self._send_json({"error": "not found"}, 404)
                except SQLError as exc:
                    self._send_json({"error": str(exc)}, 400)
                except FileNotFoundError as exc:
                    self._send_json({"error": f"file not found: {exc}"}, 400)
                except Exception as exc:  # pragma: no cover - defensive
                    self._send_json({"error": f"internal error: {exc}"}, 500)

            # -- handlers -------------------------------------------------
            def _mount(self):
                p = self._read_json()
                src = eng.mount(
                    name=p["name"],
                    path=p["path"],
                    delimiter=p.get("delimiter", "|"),
                    has_header=bool(p.get("has_header", True)),
                    encoding=p.get("encoding", "utf-8"),
                    null_tokens=tuple(p.get("null_tokens", [""])),
                )
                self._send_json({
                    "ok": True,
                    "source": {
                        "name": src.name,
                        "schema": [{"name": n, "type": t}
                                   for n, t in src.schema],
                    },
                    "sources": eng.list_sources(),
                })

            def _query(self):
                p = self._read_json()
                job_id = uuid.uuid4().hex[:12]
                job = QueryJob(job_id, p["sql"])
                max_rows = p.get("max_rows")
                server.jobs[job_id] = job

                def runner():
                    job.status = "running"
                    try:
                        result = eng.run(p["sql"], cancel=job.token,
                                         max_rows=max_rows)
                        with job._lock:
                            job.result = result
                            job.plan_tree = result.plan_tree
                            job.status = "done"
                    except CancelledError:
                        with job._lock:
                            job.status = "cancelled"
                            job.error = "query cancelled"
                    except SQLError as exc:
                        with job._lock:
                            job.status = "error"
                            job.error = str(exc)
                    except Exception as exc:  # pragma: no cover - defensive
                        with job._lock:
                            job.status = "error"
                            job.error = f"{type(exc).__name__}: {exc}"
                    finally:
                        job.finished = time.perf_counter()

                threading.Thread(target=runner, daemon=True,
                                 name=f"query-{job_id}").start()
                self._send_json({"job_id": job_id, "status": "running"})

            def _cancel(self):
                p = self._read_json()
                job = server.jobs.get(p.get("job_id", ""))
                if job is None:
                    raise SQLError("unknown job id")
                job.token.cancel()
                self._send_json({"ok": True, "cancelled": job.job_id})

            def _job_status(self, job_id):
                job = server.jobs.get(job_id)
                if job is None:
                    raise SQLError("unknown job id")
                self._send_json(job.snapshot())

            def _download(self, job_id):
                job = server.jobs.get(job_id)
                if job is None or job.result is None:
                    raise SQLError("job has no result yet")
                body = job.result.to_csv_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="query_{job_id}.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="mini-sql-web")
        self._thread.start()
        # best-effort cleanup of spill roots from crashed processes
        TempManager.sweep_stale()
        return f"http://{self.host}:{self.port}/"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
