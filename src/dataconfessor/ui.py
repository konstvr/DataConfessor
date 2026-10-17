"""`dataconfessor ui` — the tool in a browser, served from this machine only.

A privacy tool that phones home would be an odd thing, so this one does not. The server
binds 127.0.0.1, serves three static files, holds no state on disk beyond the run directory
the user can see, and makes no outbound request except the one the user asks for: a chat
turn to the model provider whose key they typed. The page loads no fonts, scripts or
styles from the network.

What the browser is allowed to ask for:

    GET  /                     the page (carries a per-process token)
    GET  /api/env              version, device, workdir -- what the header shows
    POST /api/browse           directory listing, for the folder picker
    POST /api/headers          the COLUMN NAMES of a CSV (never its rows)
    POST /api/exec             start an allow-listed command, returns a job id
    GET  /api/stream?job=      server-sent events: one per output line, then the exit
    POST /api/stop             cancel a running job
    POST /api/chat             one assistant turn, proxied to the user's provider
    POST /api/reveal           open the run directory in the system file manager
    GET  /api/file?path=       serve a generated .html/.json report from the run directory

Two rules are enforced here rather than asked for politely, because the page is driven by
an assistant and assistants are suggestible:

* `/api/exec` runs an allow list (`dataconfessor ...` and a few read-only shell commands),
  parsed with `shlex` and executed without a shell, so no `;`, `|` or `$(...)` can ride in;
* `/api/file` serves only `.html` and `.json` from the run directory, so the per-patient
  CSV -- the one output that ranks individuals by risk -- cannot be fetched into the page
  or into an assistant's context. It can be revealed in the file manager and nothing else.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import secrets
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_ROOT = Path(__file__).parent / "web"
ALLOWED_COMMANDS = {"dataconfessor", "ls", "dir", "find", "head", "wc"}
"""Everything the assistant or the form is allowed to start. `dataconfessor` is re-executed
through this interpreter, so it works even where the console script is not on PATH."""

MAX_OUTPUT_LINES = 4000
TOKEN = secrets.token_urlsafe(18)


# ------------------------------------------------------------------------------------------
# Jobs
# ------------------------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    cmd: list[str]
    lines: list[str] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)
    exit_code: int | None = None
    process: subprocess.Popen | None = None
    started: float = field(default_factory=time.time)

    def emit(self, kind: str, data: str) -> None:
        if kind == "line":
            self.lines.append(data)
            del self.lines[:-MAX_OUTPUT_LINES]
        for q in list(self.subscribers):
            q.put((kind, data))


JOBS: dict[str, Job] = {}


def _resolve(cmd: list[str]) -> list[str]:
    """`dataconfessor ...` runs as this interpreter's module, everything else as itself."""
    if cmd[0] == "dataconfessor":
        return [sys.executable, "-u", "-m", "dataconfessor.cli", *cmd[1:]]
    return cmd


def start_job(cmd_text: str, cwd: Path) -> Job:
    argv = shlex.split(cmd_text, posix=os.name != "nt")
    argv = [a.strip('"') for a in argv]
    if not argv:
        raise ValueError("empty command")
    if argv[0] not in ALLOWED_COMMANDS:
        raise PermissionError(
            f"{argv[0]!r} is not one of the commands this page may run "
            f"({', '.join(sorted(ALLOWED_COMMANDS))})."
        )
    job = Job(id=secrets.token_hex(8), cmd=argv)
    JOBS[job.id] = job

    def run() -> None:
        try:
            job.process = subprocess.Popen(
                _resolve(argv), cwd=str(cwd), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding="utf-8", errors="replace",
            )
            assert job.process.stdout is not None
            for line in job.process.stdout:
                job.emit("line", line.rstrip("\n"))
            job.exit_code = job.process.wait()
        except FileNotFoundError:
            job.emit("line", f"ERROR: {argv[0]} is not installed or not on PATH")
            job.exit_code = 127
        except Exception as exc:  # pragma: no cover - surfaced to the page
            job.emit("line", f"ERROR: {exc}")
            job.exit_code = 1
        finally:
            job.emit("exit", str(job.exit_code))

    threading.Thread(target=run, daemon=True).start()
    return job


# ------------------------------------------------------------------------------------------
# Chat providers
# ------------------------------------------------------------------------------------------


def chat(provider: str, model: str, key: str, system: str, messages: list[dict],
         base_url: str = "") -> str:
    """One assistant turn. The key is used for this request and never stored."""
    if provider == "anthropic":
        url = (base_url or "https://api.anthropic.com") + "/v1/messages"
        payload = {"model": model, "max_tokens": 1500, "system": system,
                   "messages": messages}
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        pick = lambda d: "".join(b.get("text", "") for b in d.get("content", []))
    elif provider == "ollama":
        url = (base_url or "https://ollama.com") + "/api/chat"
        payload = {"model": model, "stream": False, "think": False,
                   "options": {"temperature": 0},
                   "messages": [{"role": "system", "content": system}, *messages]}
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        pick = lambda d: d.get("message", {}).get("content", "")
    else:  # any OpenAI-compatible endpoint: vLLM, LM Studio, Azure, OpenRouter
        url = (base_url or "https://api.openai.com/v1") + "/chat/completions"
        payload = {"model": model, "temperature": 0,
                   "messages": [{"role": "system", "content": system}, *messages]}
        headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
        pick = lambda d: d["choices"][0]["message"]["content"]

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return pick(json.loads(resp.read()))


PROTOCOL = """
You are working in a terminal on this machine. Follow the instructions above exactly.

Reply with EITHER one shell command to run, in a fenced block:

```bash
<one command>
```

OR, when the workflow is finished or you need the user to answer something, plain prose with
no fenced block.

Rules: one command per reply, no commentary around a command, never invent command output,
and never read a file the instructions tell you not to open. The only commands available are
dataconfessor, ls, dir, find, head and wc.
""".strip()
"""How the assistant talks to this page. It lives beside the executor that enforces it rather
than in the browser, so a model can never be handed the skill without the protocol -- which
is what happened once, and the model answered with nothing at all."""


def system_prompt(given: str = "") -> str:
    return given or f"{skill_text()}\n\n---\n\n{PROTOCOL}"


def skill_text() -> str:
    """The SKILL.md that drives the assistant, from the checkout when it is there."""
    for candidate in (
        Path(__file__).parents[2] / "skills" / "dataconfessor" / "SKILL.md",
        Path(__file__).parents[3] / "skills" / "dataconfessor" / "SKILL.md",
        WEB_ROOT / "SKILL.md",
    ):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return ""


# ------------------------------------------------------------------------------------------
# HTTP
# ------------------------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "dataconfessor"
    workdir: Path = Path.cwd()

    # -- plumbing ---------------------------------------------------------------------

    def log_message(self, *args) -> None:  # quiet; the console belongs to the user
        pass

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("content-length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _authorised(self) -> bool:
        """Local-only, plus a per-process token.

        Without this, any page open in the same browser could post to 127.0.0.1 and run
        commands on this machine. The token is minted at startup and handed to the page.
        """
        host = (self.headers.get("host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]"):
            return False
        given = self.headers.get("x-dc-token") or parse_qs(
            urlparse(self.path).query).get("t", [""])[0]
        return secrets.compare_digest(given, TOKEN)

    # -- routes -----------------------------------------------------------------------

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)

        if route in ("/", "/index.html"):
            page = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
            page = page.replace("{{TOKEN}}", TOKEN)
            body = page.encode()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route.startswith("/static/"):
            target = WEB_ROOT / route[len("/static/"):]
            if not target.is_file() or WEB_ROOT not in target.resolve().parents:
                self.send_error(404)
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("content-type", mimetypes.guess_type(target.name)[0] or "text/plain")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if not self._authorised():
            self.send_error(403, "this page is served to this machine only")
            return

        if route == "/api/env":
            self._json(self._env())
            return

        if route == "/api/stream":
            self._stream(query.get("job", [""])[0])
            return

        if route == "/api/file":
            self._serve_artifact(query.get("path", [""])[0])
            return

        self.send_error(404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if not self._authorised():
            self.send_error(403, "this page is served to this machine only")
            return
        try:
            body = self._body()
            if route == "/api/browse":
                self._json(self._browse(body.get("path") or str(self.workdir)))
            elif route == "/api/headers":
                self._json(self._headers(body.get("path", "")))
            elif route == "/api/exec":
                job = start_job(body.get("cmd", ""), self.workdir)
                self._json({"job": job.id, "cmd": job.cmd})
            elif route == "/api/stop":
                job = JOBS.get(body.get("job", ""))
                if job and job.process and job.exit_code is None:
                    job.process.terminate()
                self._json({"ok": True})
            elif route == "/api/chat":
                reply = chat(
                    body.get("provider", "ollama"), body.get("model", ""),
                    body.get("key", ""), system_prompt(body.get("system", "")),
                    body.get("messages", []), body.get("base_url", ""),
                )
                self._json({"reply": reply})
            elif route == "/api/reveal":
                self._reveal(body.get("path") or str(self.workdir))
                self._json({"ok": True})
            else:
                self.send_error(404)
        except PermissionError as exc:
            self._json({"error": str(exc)}, 403)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            self._json({"error": f"the model provider returned {exc.code}: {detail}"}, 502)
        except urllib.error.URLError as exc:
            self._json({"error": f"could not reach the model provider: {exc.reason}"}, 502)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    # -- handlers ---------------------------------------------------------------------

    def _env(self) -> dict:
        from . import __version__

        device, detail = "cpu", "no GPU found: a screen takes tens of minutes"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                detail = torch.cuda.get_device_name(0)
        except Exception:
            detail = "PyTorch is not installed"
        return {"version": __version__, "device": device, "device_detail": detail,
                "workdir": str(self.workdir), "home": str(Path.home()),
                "separator": os.sep}

    def _browse(self, path: str) -> dict:
        here = Path(path).expanduser()
        if not here.exists():
            here = Path.home()
        here = here.resolve()
        if here.is_file():
            here = here.parent
        entries = []
        try:
            for item in sorted(here.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if item.name.startswith("."):
                    continue
                entries.append({"name": item.name, "dir": item.is_dir(),
                                "path": str(item),
                                "size": item.stat().st_size if item.is_file() else None})
                if len(entries) >= 500:
                    break
        except PermissionError:
            pass
        return {"path": str(here), "parent": str(here.parent) if here.parent != here else None,
                "entries": entries}

    def _headers(self, path: str) -> dict:
        """Column names only. The rows are records, and nothing here needs to see them."""
        import csv

        with open(path, newline="", encoding="utf-8-sig") as fh:
            columns = next(csv.reader(fh), [])
        return {"columns": [c for c in columns if c.strip()]}

    def _stream(self, job_id: str) -> None:
        job = JOBS.get(job_id)
        if job is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        q: queue.Queue = queue.Queue()
        backlog = list(job.lines)
        job.subscribers.append(q)
        try:
            for line in backlog:
                self._event("line", line)
            if job.exit_code is not None:
                self._event("exit", str(job.exit_code))
                return
            while True:
                try:
                    kind, data = q.get(timeout=20)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self._event(kind, data)
                if kind == "exit":
                    return
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if q in job.subscribers:
                job.subscribers.remove(q)

    def _event(self, kind: str, data: str) -> None:
        payload = json.dumps({"kind": kind, "data": data})
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()

    def _serve_artifact(self, path: str) -> None:
        """Reports and summaries only.

        The per-patient CSV lives in the same directory and is deliberately not servable:
        it ranks individuals by privacy risk, and a page an assistant can read is exactly
        where it must not end up.
        """
        target = Path(path).expanduser().resolve()
        if target.suffix.lower() not in (".html", ".json"):
            self.send_error(403, "only .html and .json artifacts can be opened here")
            return
        if not target.is_file() or self.workdir.resolve() not in target.parents:
            self.send_error(404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("content-type",
                         "text/html; charset=utf-8" if target.suffix == ".html"
                         else "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _reveal(self, path: str) -> None:
        target = Path(path).expanduser()
        folder = str(target if target.is_dir() else target.parent)
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606 - a local file manager, by user request
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])


def serve(workdir: str | Path | None = None, port: int = 0, open_browser: bool = True) -> int:
    """Start the local server and (by default) open the page."""
    Handler.workdir = Path(workdir or Path.cwd()).expanduser().resolve()
    Handler.workdir.mkdir(parents=True, exist_ok=True)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    chosen = httpd.server_address[1]
    url = f"http://127.0.0.1:{chosen}/?t={TOKEN}"

    print(f"DataConfessor is running at {url}", flush=True)
    print(f"  working directory: {Handler.workdir}", flush=True)
    print("  this server is reachable from this machine only; Ctrl-C to stop", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
