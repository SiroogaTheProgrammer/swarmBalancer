"""The studio's local web server: static UI + JSON API, standard library only.

    python -m swarm.studio            # opens the test bench in an app window (Edge/Chrome) or the browser
    python -m swarm.studio --no-browser --port 8770

Security model (the API runs brains you uploaded, so it must not be reachable by anyone else):

* binds to 127.0.0.1 only;
* every /api call needs the per-launch session token (header ``X-Studio-Token``); the token reaches the
  page through the launch URL and is never served by the server;
* the Host header must be 127.0.0.1:<port> or localhost:<port> (defeats DNS rebinding) and a browser
  Origin, if sent, must be this server (defeats cross-site requests);
* strict Content-Security-Policy, no framing, bounded request bodies, static files from one folder only.

Brain code never runs in this process - worker jobs (``jobs.py``) do that.
"""

from __future__ import annotations

import argparse
import gzip
import hmac
import io
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
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from swarm._arch import native_python, python_is_emulated
from swarm.arena.brains import DEFAULTS, BrainError, a_or_an, list_brains
from swarm.arena.config import ClashConfig, ConfigError, ReconConfig, TrainConfig, from_dict, schema, to_dict
from swarm.arena.run import TESTS
from swarm.devices import DEVICE_PROFILES
from swarm.sdk.install import JUNK, LIBRARY, MAX_UPLOAD_BYTES, install_npz, install_upload, remove_package, slug
from swarm.sdk.packages import NAME_RE, PackageError, discover_packages

from . import store
from .jobs import JobManager

STATIC = Path(__file__).resolve().parent / "static"
REPO_ROOT = store.REPO_ROOT
MAX_JSON_BYTES = 1 << 20
DOCS = {
    "arena": ("Test bench manual", "docs/ARENA.md"),
    "studio": ("Using the studio", "docs/STUDIO.md"),
    "sdk": ("SDK - writing brains", "docs/SDK.md"),
    "readme": ("README", "README.md"),
}
TEST_INFO = {
    "recon": "Survey an area: cover it fast, stay in radio contact, keep the sensing load low.",
    "defence": "A wall of interceptors against an incoming swarm: stop them all, without piling onto one.",
    "offence": "The incoming swarm's side: talk to each other and get past the wall.",
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
# what the UI needs to build its forms (reads manifests and weight headers only - runs no brain code)
# ----------------------------------------------------------------------------
def catalog() -> dict:
    return {
        "tests": {name: {"title": spec["title"], "about": TEST_INFO[name], "sides": list(spec["sides"]),
                         "subject": spec["subject"], "settings": schema(spec["config"])}
                  for name, spec in TESTS.items()},
        "train": schema(TrainConfig),
        "cores": os.cpu_count() or 1,
        # training speed: an emulated interpreter (x64 Python on an ARM64 PC) simulates at about half speed
        "native": not python_is_emulated() or native_python() is not None,
        "brains": list_brains(),
        "defaults": DEFAULTS,
        "devices": {n: p.description for n, p in DEVICE_PROFILES.items() if n != "host"},
    }


def _library_package(name: str):
    if not NAME_RE.match(name or ""):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid brain name {name!r}")
    for _folder, pkg, _err in discover_packages():
        if pkg is not None and pkg.name == name:
            return pkg
    raise ApiError(HTTPStatus.NOT_FOUND, f"no brain package {name!r} in the library")


def _zip_package(pkg) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, _size in pkg.files():
            if any(p in JUNK for p in rel.split("/")) or rel.endswith(".pyc"):
                continue
            zf.write(pkg.root / rel, rel)
    return buf.getvalue()


def _check_brains(test: str, brains: dict) -> dict:
    """Brain ids must exist (checked against the list, nothing is loaded here)."""
    known = {b["id"]: b for b in list_brains() if b.get("kind") != "broken"}
    out = {}
    for role in TESTS[test]["sides"]:
        bid = brains.get(role) or DEFAULTS[role]
        if not isinstance(bid, str) or bid not in known:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"unknown {role} brain {bid!r}")
        if known[bid]["role"] != role:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{bid} is {a_or_an(known[bid]['role'])} brain, not {a_or_an(role)} brain")
        out[role] = bid
    return out


# ----------------------------------------------------------------------------
# request handling
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "SwarmStudio/2.0"
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

    def _send_bytes(self, status: int, body: bytes, ctype: str, cache: str = "no-store",
                    extra: dict | None = None) -> None:
        hdr = {"Cache-Control": cache, **(extra or {})}
        if len(body) > 32 * 1024 and "gzip" in self.headers.get("Accept-Encoding", "") and \
                not ctype.startswith(("image/", "application/zip", "application/octet-stream")):
            body = gzip.compress(body, compresslevel=5)
            hdr["Content-Encoding"] = "gzip"
            hdr["Vary"] = "Accept-Encoding"
        self._headers(status, ctype, len(body), hdr)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _json_file(self, path: Path) -> None:
        self._send_bytes(200, path.read_bytes(), "application/json; charset=utf-8")

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
        except (ConfigError, BrainError, PackageError, KeyError, ValueError, TypeError) as exc:
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
        parts = [p for p in route.strip("/").split("/") if p]
        head = parts[0] if parts else ""
        n = len(parts)

        if (method, route) == ("GET", "catalog"):
            return self._json(catalog())
        if (method, route) == ("GET", "ping"):
            return self._json({"ok": True, "uptime": time.time() - st.started,
                               "jobs": [j.view(log_tail=0) | {"events": []} for j in st.jobs.active()]})
        if (method, route) == ("POST", "shutdown"):
            self._json({"ok": True})
            st.stop()
            return None

        # ---------------- tests
        if (method, route) == ("POST", "test"):
            body = self._read_json()
            test = body.get("test")
            if test not in TESTS:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"test must be one of {list(TESTS)}")
            cfg = from_dict(TESTS[test]["config"], body.get("settings"))
            brains = _check_brains(test, body.get("brains") or {})
            episodes, seed = body.get("episodes", 8), body.get("seed", 0)
            if not isinstance(episodes, int) or not 1 <= episodes <= 64:
                raise ApiError(HTTPStatus.BAD_REQUEST, "episodes must be a whole number from 1 to 64")
            if not isinstance(seed, int) or not 0 <= seed < 2**31:
                raise ApiError(HTTPStatus.BAD_REQUEST, "seed must be a whole number from 0")
            spec = {"test": test, "settings": to_dict(cfg), "brains": brains, "episodes": episodes, "seed": seed,
                    "compare": bool(body.get("compare", True))}
            job = st.jobs.submit("test", spec, label=f"{TESTS[test]['title']} test", meta={"test": test})
            return self._json({"job": job.id})

        # ---------------- saved test reports
        if head == "runs":
            if method == "GET" and n == 1:
                return self._json({"runs": store.list_runs()})
            if method == "GET" and n == 2:
                return self._json_file(store.run_file(parts[1]))
            if method == "DELETE" and n == 2:
                store.delete_run(parts[1])
                return self._json({"deleted": parts[1]})

        # ---------------- training
        if (method, route) == ("POST", "train"):
            return self._start_training(self._read_json())
        if head == "training":
            if method == "GET" and n == 1:
                return self._json({"training": store.list_trainings()})
            tid = parts[1] if n > 1 else ""
            if method == "GET" and n == 2:
                d = store.training_dir(tid)
                hist = d / "history.json"
                history = json.loads(hist.read_text(encoding="utf-8")) if hist.is_file() else {}
                return self._json({"status": store.training_status(tid), "history": history.get("history", [])})
            if method == "GET" and n == 4 and parts[2] == "matches":
                return self._json_file(store.match_file(tid, parts[3]))
            if method == "POST" and n == 3 and parts[2] == "save":
                return self._save_trained(tid, self._read_json())
            if method == "DELETE" and n == 2:
                status = store.training_status(tid)
                if status.get("state") in ("queued", "running"):
                    raise ApiError(HTTPStatus.CONFLICT, "stop the training before deleting it")
                store.delete_training(tid)
                return self._json({"deleted": tid})

        # ---------------- jobs
        if head == "jobs" and n >= 2:
            job = st.jobs.get(parts[1])
            if method == "GET" and n == 2:
                return self._json(job.view(since=int(q.get("since", ["0"])[0] or 0)))
            if method == "POST" and n == 3 and parts[2] == "cancel":
                return self._json(st.jobs.cancel(job.id).view(since=len(job.events)))
            if method == "POST" and n == 3 and parts[2] == "stop":
                return self._json(st.jobs.stop(job.id).view(since=len(job.events)))

        # ---------------- brains (library)
        if head == "brains":
            if method == "GET" and n == 1:
                return self._json({"brains": list_brains()})
            if method == "POST" and n == 2 and parts[1] == "upload":
                return self._upload(q)
            name = parts[1] if n > 1 else ""
            if method == "DELETE" and n == 2:
                pkg = _library_package(name)
                if pkg.root.parent.resolve() != LIBRARY.resolve():
                    raise ApiError(HTTPStatus.BAD_REQUEST, "example brains are part of the repository; only "
                                                           "uploaded or saved brains (brains/) can be deleted")
                remove_package(pkg.name, LIBRARY)
                return self._json({"deleted": pkg.name})
            if method == "GET" and n == 3 and parts[2] == "download":
                pkg = _library_package(name)
                return self._send_bytes(200, _zip_package(pkg), "application/zip",
                                        extra={"Content-Disposition": f'attachment; filename="{pkg.name}.zip"'})
            if method == "POST" and n == 3 and parts[2] == "validate":
                pkg = _library_package(name)
                job = st.jobs.submit("validate", {"path": str(pkg.root)}, label=f"validate {pkg.name}",
                                     meta={"brain": pkg.name})
                return self._json({"job": job.id})
            if method == "GET" and n == 3 and parts[2] == "source":
                pkg = _library_package(name)
                entry = pkg.entry.split(":")[0] if pkg.entry else None
                if not entry:
                    raise ApiError(HTTPStatus.BAD_REQUEST, f"{pkg.name} is a trained network, not source code")
                target = (pkg.root / entry).resolve()
                if pkg.root.resolve() not in target.parents or not target.is_file() or \
                        target.stat().st_size > 512 * 1024:
                    raise ApiError(HTTPStatus.NOT_FOUND, "source not available")
                return self._json({"file": entry, "text": target.read_text(encoding="utf-8", errors="replace")})

        # ---------------- docs
        if head == "docs" and method == "GET":
            if n == 1:
                return self._json({"docs": [{"id": k, "title": v[0], "path": v[1]} for k, v in DOCS.items()
                                            if (REPO_ROOT / v[1]).is_file()]})
            entry = DOCS.get(parts[1])
            if entry is None or not (REPO_ROOT / entry[1]).is_file():
                raise ApiError(HTTPStatus.NOT_FOUND, "no such document")
            return self._json({"id": parts[1], "title": entry[0], "path": entry[1],
                               "markdown": (REPO_ROOT / entry[1]).read_text(encoding="utf-8")})

        raise ApiError(HTTPStatus.NOT_FOUND, f"no API route {method} /api/{route}")

    # -------------------------------------------------------------- actions
    def _start_training(self, body: dict) -> None:
        tc = from_dict(TrainConfig, body.get("train"))
        cfg = from_dict(ReconConfig if tc.mode == "recon" else ClashConfig, body.get("settings"))
        known = {b["id"]: b for b in list_brains() if b.get("kind") != "broken"}
        for role, spec in (("defender", tc.start_defender), ("attacker", tc.start_attacker), ("recon", tc.start_recon)):
            if spec != "fresh" and (spec not in known or known[spec]["role"] != role):
                raise ApiError(HTTPStatus.BAD_REQUEST, f"unknown {role} brain {spec!r} to start from")
        tid, out = store.new_training()
        status = {"id": tid, "created": time.time(), "mode": tc.mode, "state": "queued", "generation": 0,
                  "generations": tc.generations, "train": to_dict(tc), "settings": to_dict(cfg)}
        spec = {"train": to_dict(tc), "settings": to_dict(cfg), "out_dir": str(out), "status": status}

        def done(job) -> None:
            path = out / "status.json"
            try:
                cur = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cur = dict(status)
            if cur.get("state") in ("queued", "running"):
                cur.update(state="cancelled" if job.state == "cancelled" else "error", error=job.error,
                           finished=time.time())
                store.write_json(path, cur)

        label = "train recon swarm" if tc.mode == "recon" else "train both swarms" if (
            tc.learn_defenders and tc.learn_attackers) else "train defenders" if tc.learn_defenders else "train attackers"
        job = self.studio.jobs.submit("train", spec, label=label, on_done=done, meta={"training": tid})
        status["job"] = job.id
        store.write_json(out / "status.json", status)
        return self._json({"job": job.id, "training": tid})

    def _save_trained(self, tid: str, body: dict) -> None:
        role = body.get("role")
        if role not in ("defender", "attacker", "recon"):
            raise ApiError(HTTPStatus.BAD_REQUEST, "role must be defender, attacker or recon")
        which = body.get("which", "latest")
        if which not in ("latest", "best"):
            raise ApiError(HTTPStatus.BAD_REQUEST, "which must be 'latest' or 'best'")
        src = store.training_dir(tid) / (f"{role}.npz" if which == "latest" else f"{role}_best.npz")
        if not src.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, f"this training has no {which} {role} brain (yet)")
        name = slug(str(body.get("name") or f"{role}_{tid[-4:]}"), "brain")
        if not NAME_RE.match(name):
            raise ApiError(HTTPStatus.BAD_REQUEST, "name must start with a letter: letters, digits and _")
        if any(pkg is not None and pkg.name == name for _f, pkg, _e in discover_packages()):
            raise ApiError(HTTPStatus.CONFLICT, f"a brain called {name!r} already exists - pick another name")
        title = str(body.get("title") or "").strip()[:60] or None
        pkg = install_npz(src.read_bytes(), f"{name}.npz", LIBRARY, name=name, title=title)
        return self._json({"brain": pkg.policy_info()})

    def _upload(self, q: dict) -> None:
        name = q.get("filename", [""])[0]
        if not re.fullmatch(r"[\w .()-]{1,120}\.(zip|npz|py)", name, flags=re.IGNORECASE):
            raise ApiError(HTTPStatus.BAD_REQUEST, "upload a brain package (.zip), trained weights (.npz) or one .py")
        data = self._read_body(MAX_UPLOAD_BYTES)
        stem = slug(Path(name).stem, "brain")
        taken = {pkg.name for _f, pkg, _e in discover_packages() if pkg is not None}
        if Path(name).suffix.lower() in (".npz",) and stem in taken:
            raise ApiError(HTTPStatus.CONFLICT, f"a brain called {stem!r} already exists - rename the file")
        pkg = install_upload(data, name, LIBRARY)
        if pkg.type != "policy":
            remove_package(pkg.name, LIBRARY)
            raise ApiError(HTTPStatus.BAD_REQUEST, f"that is {a_or_an(pkg.type)} package; the test bench loads arena brains "
                                                   "(a class with act(obs) and role = 'defender' / 'attacker' / "
                                                   "'recon', or trained .npz weights) - see the SDK guide")
        job = self.studio.jobs.submit("validate", {"path": str(pkg.root)}, label=f"validate {pkg.name}",
                                      meta={"brain": pkg.name})
        return self._json({"brain": pkg.policy_info(), "job": job.id})


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
            subprocess.Popen([exe, f"--app={url}", "--window-size=1600,960", "--new-window"],
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
/___/  |__/|__//_/ |_|/_/|_|/_/  /_/      \____//_/   /___/   drone swarm test bench
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
    print(f"  test bench: {studio.url}")
    print("              (local only; the link carries this session's access token - do not share it)")
    if not args.no_browser:
        print(f"  opened    : {open_ui(studio.url, app=not args.browser)}")
    print("  Ctrl+C here (or the power button in the bench) stops the studio and its jobs.\n")
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
