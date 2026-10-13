"""Gemini CLI hooks for ratings and chat copies, installed into ~/.gemini/settings.json by `lab use` (lab/gemini.py).

    before   BeforeAgent: catches `/rate N` (the custom command expands to "openrate:rate N"), saves the rating to
             metrics/feedback.jsonl and blocks the prompt, so it never reaches the model.
    after    AfterAgent: copies the session's chat file into the repo (see `archive`) and, with --popup, opens the
             rating pop-up (lab/rate_popup.py) in the background if there is a screen.
    end      SessionEnd (exit, /clear): copies the chat file once more.

Gemini CLI passes the event as JSON on stdin and reads an optional JSON decision from stdout. This runs on every
prompt, so it imports only the standard library. The hook's environment includes GEMINI_API_KEY: never print it.
"""

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FEEDBACK_FILE = REPO / "metrics" / "feedback.jsonl"
POPUP = Path(__file__).with_name("rate_popup.py")
POPUP_PID = REPO / "metrics" / "gemini" / "popup.pid"
CHATS_DIR = REPO / "metrics" / "gemini" / "chats"  # <project>/ mirrors ~/.gemini/tmp/<project>/chats/
MARKER = "openrate:rate"
LABELS = {4: "Good as is", 3: "Small fixes needed", 2: "Big fixes needed", 1: "Wrong or unusable"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", choices=["before", "after", "end"])
    parser.add_argument("--profile")
    parser.add_argument("--popup", action="store_true", help="after: ask for a rating in a pop-up")
    args = parser.parse_args()
    event = json.load(sys.stdin)
    if args.event == "before":
        before(event, args.profile)
        return
    try:
        archive(event.get("transcript_path"), event.get("session_id"))
    except OSError as exc:
        print(f"openrate: could not copy the chat file: {exc}", file=sys.stderr)
    if args.event == "after" and args.popup:
        popup(event, args.profile)


def before(event: dict, profile: str | None) -> None:
    prompt = (event.get("prompt") or "").strip()
    if not prompt.startswith(MARKER):
        return
    value = prompt.removeprefix(MARKER).strip()
    if value not in ("1", "2", "3", "4"):
        return block("openrate: rate the last answer with /rate 1-4 (4 = good as is, 1 = wrong or unusable)")
    if not _answered(event.get("transcript_path")):
        return block("openrate: there is no answer to rate yet in this session")
    rating = int(value)
    save({"ts": _now(), "harness": "gemini", "profile": profile, "session_id": event.get("session_id"),
          "rated_at": event.get("timestamp"), "via": "rate", "rating": rating})
    block(f"openrate saved your rating of the last answer: {rating} ({LABELS[rating]})")


def archive(transcript_path: str | None, session_id: str | None) -> None:
    """Copy a session's chat file, and its subagents' files, to metrics/gemini/chats/<project>/.

    Gemini CLI deletes its own chat files after 30 days (general.sessionRetention); lab reads the copy after that.
    The project's .project_root (the folder Gemini CLI ran in) is copied too.
    """
    if not transcript_path:
        return
    source = Path(transcript_path)
    chats = next((p for p in source.parents if p.name == "chats"), None)
    if chats is None or not source.is_file():
        return
    target = CHATS_DIR / chats.parent.name
    _copy(source, target / source.relative_to(chats))
    subagents = chats / session_id if session_id else None  # Gemini CLI keeps them in chats/<session id>/
    if subagents and subagents.is_dir():
        for path in subagents.rglob("*.jsonl"):
            _copy(path, target / path.relative_to(chats))
    root = chats.parent / ".project_root"
    if root.is_file():
        _copy(root, target / ".project_root")


def _copy(source: Path, target: Path) -> None:
    """Copy unless the copy already has the same size and time; through a temp file, so readers never see half."""
    stat = source.stat()
    try:
        copied = target.stat()
        if (copied.st_size, copied.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns):
            return
    except OSError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temp)
    os.replace(temp, target)


def popup(event: dict, profile: str | None) -> None:
    if not _has_screen():
        return
    _close_previous_popup()
    process = subprocess.Popen(
        [sys.executable, "-I", str(POPUP), "--session", event.get("session_id") or "",
         "--answered-at", event.get("timestamp") or _now(), "--profile", profile or "",
         "--prompt", (event.get("prompt") or "")[:300]],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,  # keeps running after this hook returns; Gemini CLI doesn't wait for it
    )
    POPUP_PID.parent.mkdir(parents=True, exist_ok=True)
    POPUP_PID.write_text(str(process.pid))


def block(message: str) -> None:
    print(json.dumps({"decision": "block", "reason": message, "systemMessage": message}))


def save(record: dict) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _answered(transcript_path: str | None) -> bool:
    """Whether the session has a model answer yet (Gemini CLI's chat file has a "gemini" message)."""
    try:
        return '"type":"gemini"' in Path(transcript_path).read_text().replace(" ", "")
    except (OSError, TypeError):
        return True  # can't tell: save it and let `lab` match it to a prompt


def _has_screen() -> bool:
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _close_previous_popup() -> None:
    """One pop-up at a time: an unanswered one for an earlier answer closes without saving anything."""
    try:
        os.kill(int(POPUP_PID.read_text()), signal.SIGTERM)
    except (OSError, ValueError):
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


if __name__ == "__main__":
    main()
