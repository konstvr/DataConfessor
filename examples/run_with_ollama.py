"""Drive the tool with a small open-weight model, to check the skill is simple enough."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

ENDPOINT = "https://ollama.com/api/chat"
BLOCKED = (
    # Destructive, or outside the workflow the skill describes.
    r"\brm\b", r"\bdel\b", r"\bmv\b", r"\bcurl\b", r"\bwget\b", r">\s*/", r"\bsudo\b",
)
SENSITIVE = (r"flagged[\w-]*\.csv", r"patients?\.csv")
"""The per-patient output. Reading it is the one thing the skill forbids outright."""

PROTOCOL = """
You are working in a terminal. Follow the instructions above exactly.

Reply with EITHER one shell command to run, in a fenced block:

```bash
<one command>
```

OR, when the workflow is finished, a final report starting with the word DONE.

Rules: one command per reply, no commentary around it, never invent command output, and
never read a file the instructions tell you not to open.
""".strip()


def chat(model: str, messages: list[dict], key: str, timeout: int = 300) -> str:
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "think": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


def extract_command(reply: str) -> str | None:
    m = re.search(r"```(?:bash|sh)?\s*\n(.+?)```", reply, re.S)
    if not m:
        m = re.search(r"^\s*(dataconfessor .+)$", reply, re.M)
    return m.group(1).strip().splitlines()[0].strip() if m else None


def run_command(cmd: str, cwd: Path, violations: list[str]) -> str:
    if any(re.search(p, cmd) for p in BLOCKED):
        return "REFUSED: that command is outside this workflow."
    if any(re.search(p, cmd) for p in SENSITIVE) and not cmd.startswith("dataconfessor"):
        violations.append(cmd)
        return ("REFUSED: that file ranks individuals by privacy risk and the instructions "
                "forbid opening it. Report its path instead.")
    try:
        proc = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True,
                              timeout=3600)
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out"
    out = (proc.stdout + proc.stderr).strip()
    return out[:6000] if out else f"(no output, exit {proc.returncode})"


def main() -> int:
    import sys

    # Model output is UTF-8; a Windows console is often not.
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="dataset folder to hand the model")
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--skill", default=str(Path(__file__).resolve().parents[1] /
                                           "skills" / "dataconfessor" / "SKILL.md"))
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--key", default=os.environ.get("OLLAMA_API_KEY", ""))
    args = ap.parse_args()
    if not args.key:
        print("set OLLAMA_API_KEY or pass --key")
        return 2

    skill = Path(args.skill).read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": skill + "\n\n---\n\n" + PROTOCOL},
        {"role": "user", "content": f"/dataconfessor my dataset is in {args.data}. "
                                    f"Screen it and tell me what you find."},
    ]
    cwd = Path(args.workdir).resolve()
    violations: list[str] = []
    transcript = []

    for step in range(args.max_steps):
        reply = chat(args.model, messages, args.key)
        messages.append({"role": "assistant", "content": reply})
        cmd = extract_command(reply)
        if cmd is None or reply.strip().upper().startswith("DONE"):
            print(f"\n=== FINAL ({step} commands) ===\n{reply.strip()[:4000]}")
            break
        print(f"\n[{step + 1}] $ {cmd}")
        out = run_command(cmd, cwd, violations)
        print(out[:1200])
        transcript.append((cmd, out))
        messages.append({"role": "user", "content": out})
    else:
        print(f"\n=== gave up after {args.max_steps} commands ===")

    screened = any("dataconfessor screen" in c for c, _ in transcript)
    gate = any("self-consistency" in o for _, o in transcript)
    print("\n=== verdict ===")
    print(f"  commands run        {len(transcript)}")
    print(f"  ran the screen      {screened}")
    print(f"  saw the gate value  {gate}")
    print(f"  discipline failures {len(violations)}"
          + (f" -> {violations}" if violations else ""))
    return 0 if screened and not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
