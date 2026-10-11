"""The studio's local web server: static UI + JSON API, standard library only.

    python -m swarm.studio            # opens the console in an app window (Edge/Chrome) or the default browser
    python -m swarm.studio --no-browser --port 8770

Security model (the API can run code you uploaded, so it must not be reachable by anyone else):

* binds to 127.0.0.1 only;
* every /api call needs the per-launch session token (header ``X-Studio-Token``); the token reaches the
  page through the launch URL and is never served by the server;
* the Host header must be 127.0.0.1:<port> or localhost:<port> (defeats DNS rebinding) and a browser
  Origin, if sent, must be this server (defeats cross-site requests);
* strict Content-Security-Policy, no framing, bounded request bodies, static files from one folder only.

Package and preset code never runs in this process - worker jobs (``jobs.py``) do that.
"""

from __future__ import annotations

import argparse
import gzip
import hmac
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from base64 import b64encode
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from swarm.sdk.export import export_for_device
from swarm.sdk.install import LIBRARY, MAX_UPLOAD_BYTES, install_upload, remove_package
from swarm.sdk.packages import PackageError
from swarm.sdk.templates import KINDS, new_package

from . import store
from .catalog import Catalog
from .jobs import JobManager
from .scenario_io import MARKER, STUDIO_DIR, ScenarioError, normalize, normalize_theatre, save_preset

STATIC = Path(__file__).resolve().parent / "static"
REPO_ROOT = store.REPO_ROOT
MAX_JSON_BYTES = 4 << 20
DOCS = {
    "studio": ("Studio field guide", "docs/STUDIO.md"),
    "sdk": ("SDK - brains & tactics", "docs/SDK.md"),
    "architecture": ("Architecture", "docs/ARCHITECTURE.md"),
    "model-format": (".swm model format", "docs/MODEL_FORMAT.md"),
    "deployment": ("Deployment (USB)", "docs/DEPLOYMENT.md"),
    "runtime": ("Device runtime", "docs/RUNTIME.md"),
    "robot-adapters": ("Robot adapters", "docs/ROBOT_ADAPTERS.md"),
    "security": ("Security model", "docs/SECURITY.md"),
    "readme": ("README", "README.md"),
}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "connect-src 'self'; font-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
       "form-action 'none'")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Studio:
    """Shared state behind the request handlers."""

    def __init__(self, port: int):
        self.token = secrets.token_urlsafe(24)
        self.jobs = JobManager(max_parallel=max(2, min(4, (os.cpu_count() or 2) // 2)))
        self.catalog = Catalog(self.jobs)
        self.frames = store.FrameCache()
        self.httpd = _Server(("127.0.0.1", port), Handler)
        self.httpd.studio = self
        self.port = self.httpd.server_address[1]
        self.hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        self.origins = {f"http://{h}" for h in self.hosts}
        self.started = time.time()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/#t={self.token}"

    def serve(self) -> None:
        self.httpd.serve_forever(poll_interval=0.25)

    def stop(self) -> None:
        self.jobs.shutdown()
        threading.Thread(target=self.httpd.shutdown, daemon=True).start()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    studio: Studio


# ----------------------------------------------------------------------------
# request handling
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "SwarmStudio/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, fmt: str, *args) -> None:  # quiet console; errors are reported to the UI
        if os.environ.get("SWARM_STUDIO_VERBOSE"):
            super().log_message(fmt, *args)

    @property
    def studio(self) -> Studio:
        return self.server.studio

    # -------------------------------------------------------------- plumbing
    def _headers(self, status: int, ctype: str, length: int, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _send_bytes(self, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        extra = {"Cache-Control": cache}
        if len(body) > 32 * 1024 and "gzip" in self.headers.get("Accept-Encoding", "") and \
                not ctype.startswith(("image/", "application/octet-stream")):
            body = gzip.compress(body, compresslevel=5)
            extra["Content-Encoding"] = "gzip"
            extra["Vary"] = "Accept-Encoding"
        self._headers(status, ctype, len(body), extra)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _read_body(self, limit: int) -> bytes:
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length required")
        n = int(length)
        if n > limit:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"request body larger than {limit >> 20} MiB")
        data = self.rfile.read(n)
        if len(data) != n:
            raise ApiError(HTTPStatus.BAD_REQUEST, "truncated request body")
        return data

    def _read_json(self) -> dict:
        try:
            data = json.loads(self._read_body(MAX_JSON_BYTES) or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError(HTTPStatus.BAD_REQUEST, "request body is not JSON") from None
        if not isinstance(data, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
        return data

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self.studio.hosts

    def _api_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.studio.origins:
            return False
        if self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
            return False
        return hmac.compare_digest(self.headers.get("X-Studio-Token", ""), self.studio.token)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_OPTIONS(self) -> None:  # no CORS: cross-origin preflights always fail
        self._error(HTTPStatus.FORBIDDEN, "cross-origin requests are not allowed")

    def _dispatch(self, method: str) -> None:
        try:
            if not self._host_ok():
                return self._error(HTTPStatus.MISDIRECTED_REQUEST, "unexpected Host header")
            url = urlsplit(self.path)
            if url.path.startswith("/api/"):
                if not self._api_allowed():
                    return self._error(HTTPStatus.FORBIDDEN, "missing or wrong session token - open the studio "
                                                             "through the link printed by `python -m swarm.studio`")
                return self._api(method, url.path[5:], parse_qs(url.query))
            if method != "GET":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "static files are read-only")
            return self._static(url.path)
        except ApiError as exc:
            self.close_connection = True  # the body may not have been read
            self._error(exc.status, str(exc))
        except (ScenarioError, PackageError, KeyError, ValueError, TypeError) as exc:
            msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            self._error(HTTPStatus.BAD_REQUEST, str(msg))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:  # noqa: BLE001 - report, do not kill the handler thread
            try:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")
            except OSError:
                pass

    # -------------------------------------------------------------- static
    def _static(self, path: str) -> None:
        if path in ("/", "/index.html"):
            path = "/index.html"
        rel = path.lstrip("/")
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", rel) or ".." in rel.split("/"):
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        target = (STATIC / rel).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        ctype = {".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml",
                 ".json": "application/json", ".ico": "image/x-icon", ".png": "image/png"}.get(
            target.suffix, mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        if ctype.startswith("text/") or ctype.endswith(("json", "svg+xml")):
            ctype += "; charset=utf-8"
        self._send_bytes(200, target.read_bytes(), ctype, cache="no-cache")

    # -------------------------------------------------------------- api
    def _api(self, method: str, route: str, q: dict) -> None:
        st = self.studio
        parts = route.strip("/").split("/")
        head = parts[0]

        if (method, route) == ("GET", "catalog"):
            return self._json(st.catalog.build())
        if (method, route) == ("GET", "ping"):
            return self._json({"ok": True, "uptime": time.time() - st.started})
        if (method, route) == ("POST", "shutdown"):
            self._json({"ok": True})
            st.stop()
            return None

        # ---------------- scenarios
        if (method, route) == ("POST", "scenario/check"):
            body = self._read_json()
            plain = normalize(body.get("scenario"), self._plugin_names())
            return self._json({"scenario": plain, "theatre": normalize_theatre(body.get("theatre"))})
        if (method, route) == ("POST", "scenario/save"):
            body = self._read_json()
            plain = normalize(body.get("scenario"), self._plugin_names())
            theatre = normalize_theatre(body.get("theatre"))
            packages = self._tactic_packages(plain)
            path = save_preset(plain, theatre, packages, where=body.get("where", "studio"),
                               overwrite=bool(body.get("overwrite")))
            st.catalog.invalidate_presets()
            return self._json({"file": path.relative_to(REPO_ROOT).as_posix()})
        if (method, route) == ("POST", "scenario/delete"):
            body = self._read_json()
            target = (REPO_ROOT / str(body.get("file", ""))).resolve()
            if target.parent != STUDIO_DIR.resolve() or target.suffix != ".py" or not target.is_file() or \
                    MARKER not in target.read_text(encoding="utf-8", errors="replace"):
                raise ApiError(HTTPStatus.BAD_REQUEST, "only studio-created presets in scenarios/studio/ can be deleted")
            target.unlink()
            st.catalog.invalidate_presets()
            return self._json({"deleted": body.get("file")})

        # ---------------- runs
        if head == "runs":
            if method == "POST" and len(parts) == 1:
                body = self._read_json()
                plain = normalize(body.get("scenario"), self._plugin_names())
                theatre = normalize_theatre(body.get("theatre"))
                packages = self._tactic_packages(plain)
                strategies = body.get("strategies") or None
                brain_kind = body.get("brain_kind") or None
                if brain_kind not in (None, "auto", "oracle", "numpy", "native"):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "brain_kind must be auto, oracle, numpy or native")
                run_id = store.new_id()
                spec = {"scenario": plain, "theatre": theatre, "packages": packages, "strategies": strategies,
                        "brain_kind": brain_kind, "with_stress": bool(body.get("with_stress", False)),
                        "run_dir": str(store.RUNS / run_id), "preset": body.get("preset")}
                job = st.jobs.submit("run", spec, label=f"simulate {plain['name']}")
                return self._json({"job": job.id, "run": run_id})
            if method == "GET" and len(parts) == 1:
                return self._json({"runs": store.list_runs()})
            run_id = parts[1]
            if method == "GET" and len(parts) == 2:
                d = store.run_dir(run_id)
                return self._send_bytes(200, (d / "run.json").read_bytes(), "application/json; charset=utf-8")
            if method == "DELETE" and len(parts) == 2:
                st.frames.forget(run_id)
                store.delete_run(run_id)
                return self._json({"deleted": run_id})
            if method == "GET" and len(parts) == 3 and parts[2] == "frame":
                img = st.frames.frame(run_id, int(q.get("s", ["0"])[0]), int(q.get("fid", ["-1"])[0]))
                if img is None:
                    raise ApiError(HTTPStatus.NOT_FOUND, "frame not recorded")
                return self._json({"w": int(img.shape[1]), "h": int(img.shape[0]),
                                   "data": b64encode(img.tobytes()).decode("ascii")})

        # ---------------- jobs
        if head == "jobs" and len(parts) >= 2:
            job = st.jobs.get(parts[1])
            if method == "GET" and len(parts) == 2:
                return self._json(job.view())
            if method == "POST" and len(parts) == 3 and parts[2] == "cancel":
                return self._json(st.jobs.cancel(job.id).view())

        # ---------------- packages
        if head == "packages":
            sub = parts[1] if len(parts) > 1 else ""
            if method == "POST" and sub == "upload":
                name = q.get("filename", [""])[0]
                if not re.fullmatch(r"[\w .()-]{1,120}\.(zip|swm|py)", name, flags=re.IGNORECASE):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "filename must end in .zip, .swm or .py")
                data = self._read_body(MAX_UPLOAD_BYTES)
                pkg = install_upload(data, name, LIBRARY, replace=q.get("replace", ["0"])[0] == "1")
                job = st.catalog.validate(pkg)
                return self._json({"package": pkg.summary() | {"root": None}, "job": job.id})
            if method == "POST" and sub == "new":
                body = self._read_json()
                if body.get("type") not in KINDS:
                    raise ApiError(HTTPStatus.BAD_REQUEST, f"type must be one of {KINDS}")
                root = new_package(LIBRARY, str(body.get("name", "")), body["type"], str(body.get("title", ""))[:80])
                pkg = st.catalog.package(root.relative_to(REPO_ROOT).as_posix())
                job = st.catalog.validate(pkg) if pkg.format != "swm" else None
                return self._json({"package": pkg.summary() | {"root": None}, "job": job.id if job else None})
            if method == "POST" and sub in ("validate", "build", "export", "delete"):
                body = self._read_json()
                pkg = st.catalog.package(str(body.get("path", "")))
                if sub == "validate":
                    return self._json({"job": st.catalog.validate(pkg).id})
                if sub == "build":
                    if not pkg.manifest.get("build"):
                        raise ApiError(HTTPStatus.BAD_REQUEST, f"{pkg.name} has no build script")
                    return self._json({"job": st.catalog.validate(pkg, build=True).id})
                if sub == "export":
                    out = store.EXPORTS / f"{pkg.name}-{store.new_id()}"
                    res = export_for_device(pkg.root, out)
                    res["out_rel"] = Path(res["out"]).relative_to(REPO_ROOT).as_posix()
                    return self._json(res)
                if pkg.root.parent.resolve() != LIBRARY.resolve():
                    raise ApiError(HTTPStatus.BAD_REQUEST, "examples are part of the repository; only uploaded or "
                                                           "created packages (brains/) can be deleted")
                remove_package(pkg.name, LIBRARY)
                return self._json({"deleted": pkg.name})
            if method == "GET" and sub == "source":
                pkg = st.catalog.package(q.get("path", [""])[0])
                rel = q.get("file", [""])[0]
                target = (pkg.root / rel).resolve()
                if pkg.root.resolve() not in target.parents or not target.is_file():
                    raise ApiError(HTTPStatus.NOT_FOUND, "no such file in the package")
                if target.suffix.lower() not in (".py", ".json", ".md", ".txt", ".toml", ".cfg"):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "only text sources can be viewed")
                if target.stat().st_size > 512 * 1024:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "file too large to view")
                return self._json({"file": rel, "text": target.read_text(encoding="utf-8", errors="replace")})

        # ---------------- docs
        if head == "docs" and method == "GET":
            if len(parts) == 1:
                return self._json({"docs": [{"id": k, "title": v[0], "path": v[1]} for k, v in DOCS.items()
                                            if (REPO_ROOT / v[1]).is_file()]})
            entry = DOCS.get(parts[1])
            if entry is None or not (REPO_ROOT / entry[1]).is_file():
                raise ApiError(HTTPStatus.NOT_FOUND, "no such document")
            return self._json({"id": parts[1], "title": entry[0], "path": entry[1],
                               "markdown": (REPO_ROOT / entry[1]).read_text(encoding="utf-8")})

        raise ApiError(HTTPStatus.NOT_FOUND, f"no API route {method} /api/{route}")

    # -------------------------------------------------------------- helpers
    def _plugin_names(self) -> set[str]:
        return {s["strategy"] for s in self.studio.catalog.packages() if s.get("type") == "tactic" and s.get("strategy")}

    def _tactic_packages(self, plain: dict) -> list[str]:
        """Packages providing the plugin tactics a scenario names (looked up in the library, never trusted)."""
        by_strategy = {s["strategy"]: s["root_rel"] for s in self.studio.catalog.packages()
                       if s.get("type") == "tactic" and s.get("strategy")}
        needed = sorted({by_strategy[s] for s in plain["strategies"] + [plain["baseline"]] if s in by_strategy})
        if plain["brain"]["kind"] == "package":
            self.studio.catalog.package(plain["brain"]["model"])  # must be a library package
        return needed


# ----------------------------------------------------------------------------
# launching
# ----------------------------------------------------------------------------
def find_app_browser() -> str | None:
    """A Chromium-based browser that supports --app (a chrome-less window), if one is installed."""
    names = ["msedge", "chrome", "chromium", "google-chrome", "chromium-browser", "microsoft-edge", "brave"]
    if sys.platform == "win32":
        bases = [os.environ.get(k, "") for k in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
        cands = [Path(b, *rest) for b in bases if b for rest in (
            ("Microsoft", "Edge", "Application", "msedge.exe"), ("Google", "Chrome", "Application", "chrome.exe"))]
        try:
            import winreg

            for exe in ("msedge.exe", "chrome.exe"):
                for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                    try:
                        with winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as k:
                            cands.append(Path(winreg.QueryValue(k, None)))
                    except OSError:
                        continue
        except ImportError:
            pass
        for c in cands:
            if c.is_file():
                return str(c)
        return None
    if sys.platform == "darwin":
        for app in ("Google Chrome", "Microsoft Edge", "Chromium", "Brave Browser"):
            p = Path(f"/Applications/{app}.app/Contents/MacOS/{app}")
            if p.is_file():
                return str(p)
    for n in names:
        hit = shutil.which(n)
        if hit:
            return hit
    return None


def open_ui(url: str, app: bool = True) -> str:
    exe = find_app_browser() if app else None
    if exe:
        try:
            subprocess.Popen([exe, f"--app={url}", "--window-size=1680,1000", "--new-window"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"app window ({Path(exe).stem})"
        except OSError:
            pass
    webbrowser.open(url)
    return "default browser"


BANNER = r"""
  ____  _      __ ___    ___   __  ___      ____   ___   ____
 / __/ | | /| / // _ |  / _ \ /  |/  /     / __ \ / _ \ / __/
_\ \   | |/ |/ // __ | / , _// /|_/ /     / /_/ // ___/_\ \
/___/  |__/|__//_/ |_|/_/|_|/_/  /_/      \____//_/   /___/   tactical swarm simulation console
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m swarm.studio", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770, help="local port (0 = any free port); default 8770")
    ap.add_argument("--no-browser", action="store_true", help="only print the link")
    ap.add_argument("--browser", action="store_true", help="use the default browser instead of an app window")
    args = ap.parse_args(argv)
    try:
        studio = Studio(args.port)
    except OSError:
        if args.port == 0:
            raise
        studio = Studio(0)  # the preferred port is taken (another studio?) - any free one will do
    print(BANNER)
    print(f"  engine : {studio.catalog.engine['diagnosis'].splitlines()[0]}")
    print(f"  console: {studio.url}")
    print("           (local only; the link carries this session's access token - do not share it)")
    if not args.no_browser:
        print(f"  opened : {open_ui(studio.url, app=not args.browser)}")
    print("  Ctrl+C here (or SHUTDOWN in the console menu) stops the studio and its jobs.\n")
    sys.stdout.flush()
    try:
        studio.serve()
    except KeyboardInterrupt:
        pass
    finally:
        studio.jobs.shutdown()
        studio.httpd.server_close()
    print("studio stopped")
    return 0
