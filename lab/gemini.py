"""Gemini CLI as a harness: its settings, and its own records read into traces.Prompt.

Gemini CLI talks to Google's API directly; nothing of ours sits in between. Everything comes from its own files:

settings   ~/.gemini/settings.json: `lab use` sets the model, local telemetry, usage statistics off, and three hooks
           (lab/gemini_hook.py) for ratings and chat copies, plus the `/rate` command in ~/.gemini/commands/rate.toml.
telemetry  Gemini CLI appends one entry per model call to metrics/gemini/telemetry.log: tokens, duration, model,
           session, prompt id and role (main answer, subagent, or a utility call such as routing or loop checks).
           Each call also writes the whole system prompt and tool list (~100 KB), so `extract()` copies the fields
           we use into metrics/gemini/calls.jsonl and empties the raw file when no Gemini CLI process has it open.
chats      ~/.gemini/tmp/<project>/chats/session-*.jsonl: the conversation, replayed the way Gemini CLI does
           (packages/core/src/services/chatRecordingService.ts). The hooks copy it to metrics/gemini/chats/<project>/
           after each answer and at session end, and lab reads that copy once Gemini CLI deletes its own (30 days).
ratings    metrics/feedback.jsonl lines with "harness": "gemini", from `/rate` and the pop-up.

A call belongs to the latest prompt of its session that started before it. Time to first token isn't recorded.
"""

import bisect
import fcntl
import json
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from lab import gemini_hook, traces
from lab.registry import REPO

GEMINI_DIR = Path.home() / ".gemini"
SETTINGS_FILE = GEMINI_DIR / "settings.json"
RATE_COMMAND = GEMINI_DIR / "commands" / "rate.toml"
DATA_DIR = REPO / "metrics" / "gemini"
TELEMETRY_FILE = DATA_DIR / "telemetry.log"
CHATS_DIR = gemini_hook.CHATS_DIR
CALLS_FILE = DATA_DIR / "calls.jsonl"
OFFSET_FILE = DATA_DIR / "telemetry.offset"
HOOK = REPO / "lab" / "gemini_hook.py"
BACKUP_DIR = REPO / ".lab" / "backups"
HOOK_NAMES = {"openrate-rate", "openrate-after", "openrate-end", "openrate-popup"}  # openrate-popup: an older name
AUTH_TYPES = {"api-key": "gemini-api-key", "google-login": "oauth-personal"}
COMMAND_MARKER = "# Written by openrate"
SIDE = "utility"  # roles starting with this are Gemini CLI's own background calls, not the answer
MATCH_S = 10.0  # a chat file's user message and the telemetry's prompt entry are this close in time


def apply(name: str, profile: dict, previously_managed: list[str]) -> tuple[Path | None, list[str]]:
    """Write this profile into Gemini CLI's settings and install `/rate`. Returns the backup and managed paths.

    Only the paths set here are touched; other settings and other hooks stay. Paths a previous profile's
    `[gemini]` table set but this one doesn't are removed.
    """
    if "model" not in profile:
        raise ValueError(f"profiles/gemini/{name}.toml needs a `model`")
    auth = profile.get("auth", "api-key")
    if auth not in AUTH_TYPES:
        raise ValueError(f"profiles/gemini/{name}.toml: auth must be one of {sorted(AUTH_TYPES)}")
    _write_rate_command()

    backup = None
    current: dict = {}
    if SETTINGS_FILE.exists():
        try:
            current = json.loads(SETTINGS_FILE.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{SETTINGS_FILE} is not plain JSON ({exc}); lab can't merge into it safely") from None
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / f"gemini-settings.{datetime.now():%Y%m%d-%H%M%S}.json"
        shutil.copy2(SETTINGS_FILE, backup)

    extra = _flatten(profile.get("gemini", {}))
    for path in previously_managed:
        if path not in extra:
            _unset(current, path)
    values = {
        "model.name": profile["model"],
        "security.auth.selectedType": AUTH_TYPES[auth],
        "telemetry.enabled": True,
        "telemetry.target": "local",
        "telemetry.outfile": str(TELEMETRY_FILE),
        "telemetry.logPrompts": False,  # the conversation is in Gemini CLI's chat files already
        "privacy.usageStatisticsEnabled": False,
        **extra,
    }
    for path, value in values.items():
        _set(current, path, value)
    hooks = current.setdefault("hooks", {})
    _install_hook(hooks, "BeforeAgent", "openrate-rate", _hook_command("before", name))
    _install_hook(hooks, "AfterAgent", "openrate-after",
                  _hook_command("after", name, *(["--popup"] if profile.get("feedback") else [])))
    _install_hook(hooks, "SessionEnd", "openrate-end", _hook_command("end", name))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
    return backup, list(extra)


def _hook_command(event: str, profile: str, *flags: str) -> str:
    # The venv's Python, by path: `lab` runs from it, and it has tkinter for the pop-up.
    command = (sys.executable, "-I", str(HOOK), event, "--profile", profile, *flags)
    return " ".join(shlex.quote(part) for part in command)


def _install_hook(hooks: dict, event: str, name: str, command: str) -> None:
    definitions = []
    for definition in hooks.get(event, []):
        kept = [h for h in definition.get("hooks", []) if h.get("name") not in HOOK_NAMES]
        if kept:
            definitions.append({**definition, "hooks": kept})
    definitions.append({"hooks": [{"type": "command", "name": name, "command": command, "timeout": 10000}]})
    hooks[event] = definitions


def _write_rate_command() -> None:
    if RATE_COMMAND.exists() and COMMAND_MARKER not in RATE_COMMAND.read_text():
        raise ValueError(f"{RATE_COMMAND} exists and isn't openrate's; rename it so `/rate` can be installed")
    RATE_COMMAND.parent.mkdir(parents=True, exist_ok=True)
    RATE_COMMAND.write_text(
        f"{COMMAND_MARKER} (`lab use`): rates the last answer 1-4 into metrics/feedback.jsonl.\n"
        "# Its BeforeAgent hook catches this prompt, so it never reaches the model.\n"
        'description = "Rate the last answer: /rate 4 (good as is) to /rate 1 (wrong or unusable)"\n'
        'prompt = "openrate:rate {{args}}"\n'
    )


def _flatten(table: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in table.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _set(config: dict, path: str, value) -> None:
    *parents, leaf = path.split(".")
    for key in parents:
        config = config.setdefault(key, {})
    config[leaf] = value


def _unset(config: dict, path: str) -> None:
    *parents, leaf = path.split(".")
    for key in parents:
        config = config.get(key)
        if not isinstance(config, dict):
            return
    config.pop(leaf, None)


# --- telemetry -> calls.jsonl ---

def extract() -> None:
    """Copy new telemetry entries into calls.jsonl; empty the raw file once nothing has it open.

    Also brings the chat copies of those entries' sessions up to date, in case a hook didn't run (Gemini CLI killed).
    """
    if not TELEMETRY_FILE.exists():
        return
    with _lock():
        stat = TELEMETRY_FILE.stat()
        offset = _offset(stat.st_ino)
        if offset > stat.st_size:  # emptied or replaced by someone else
            offset = 0
        with TELEMETRY_FILE.open("rb") as f:
            f.seek(offset)
            data = f.read(stat.st_size - offset)
        records, used = _parse(data)
        compact = [c for c in map(_compact, records) if c]
        if compact:
            with CALLS_FILE.open("a") as f:
                f.writelines(json.dumps(c) + "\n" for c in compact)
        offset += used
        if offset == stat.st_size and offset > 0 and _unused(TELEMETRY_FILE):
            TELEMETRY_FILE.open("w").close()
            offset = 0
        OFFSET_FILE.write_text(json.dumps({"inode": stat.st_ino, "offset": offset}))
    for session_id in {c["session_id"] for c in compact if c.get("session_id")}:
        for path in _originals(session_id):
            try:
                gemini_hook.archive(str(path), session_id)
            except OSError as exc:
                print(f"warning: could not copy {path} into {gemini_hook.CHATS_DIR}: {exc}", file=sys.stderr)


def _parse(data: bytes) -> tuple[list[dict], int]:
    """Whole JSON objects in `data` (Gemini CLI writes them indented, one after another) and the bytes they used.

    A half-written object at the end is left for next time; anything unreadable in the middle is skipped.
    """
    decoder = json.JSONDecoder()
    text = data.decode("utf-8", errors="replace")
    records, pos, used = [], 0, 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            used = len(text)
            break
        try:
            record, pos = decoder.raw_decode(text, pos)
            records.append(record)
            used = pos
        except json.JSONDecodeError:
            nxt = text.find("\n{", pos + 1)
            if nxt == -1:
                break  # incomplete last object
            pos = used = nxt + 1
    return records, len(text[:used].encode("utf-8"))


def _compact(record: dict) -> dict | None:
    a = record.get("attributes")
    if not isinstance(a, dict):
        return None
    name = a.get("event.name")
    base = {"ts": a.get("event.timestamp"), "session_id": a.get("session.id"), "prompt_id": a.get("prompt_id")}
    if name == "gemini_cli.user_prompt":
        return {"event": "prompt", **base}
    if name == "gemini_cli.api_response":
        return {"event": "call", **base, "model": a.get("model"), "role": a.get("role"),
                "duration_ms": a.get("duration_ms"), "input": a.get("input_token_count", 0),
                "cached": a.get("cached_content_token_count", 0), "output": a.get("output_token_count", 0),
                "thoughts": a.get("thoughts_token_count", 0), "tool": a.get("tool_token_count", 0),
                "status": a.get("status_code"), "finish": (a.get("finish_reasons") or [None])[-1],
                "auth_type": a.get("auth_type")}
    if name == "gemini_cli.api_error":
        return {"event": "error", **base, "model": a.get("model"), "role": a.get("role"),
                "duration_ms": a.get("duration_ms"), "status": a.get("status_code"),
                "error": str(a.get("error") or "")[:300], "error_type": a.get("error_type")}
    return None


def _offset(inode: int) -> int:
    try:
        saved = json.loads(OFFSET_FILE.read_text())
    except (OSError, ValueError):
        return 0
    return saved.get("offset", 0) if saved.get("inode") == inode else 0


def _unused(path: Path) -> bool:
    """True only when lsof can tell that no process (a running Gemini CLI) has the file open."""
    if not shutil.which("lsof"):
        return False
    result = subprocess.run(["lsof", "-t", "--", str(path)], capture_output=True, text=True, check=False)
    return result.returncode == 1 and not result.stdout.strip()


@contextmanager
def _lock():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / ".lock").open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


# --- calls.jsonl + chats + ratings -> prompts ---

def load(since: datetime) -> list[traces.Prompt]:
    """Gemini CLI prompts since `since`, oldest first, with their calls, conversation and rating."""
    events = []
    if CALLS_FILE.exists():
        for line in CALLS_FILE.read_text().splitlines():
            try:
                event = json.loads(line)
                event["_t"] = _ts(event["ts"])
            except (ValueError, KeyError, TypeError):
                continue
            events.append(event)
    by_session = defaultdict(list)
    for event in sorted(events, key=lambda e: e["_t"]):
        by_session[event.get("session_id")].append(event)

    prompts = []
    for session_id, rows in by_session.items():
        chat = _chat(session_id)
        session_prompts = [_prompt(session_id, e, chat) for e in rows if e["event"] == "prompt"]
        starts = [p.start for p in session_prompts]
        for e in rows:
            if e["event"] == "prompt":
                continue
            start = e["_t"] - (e.get("duration_ms") or 0) / 1000
            i = bisect.bisect_right(starts, start) - 1
            if i < 0:
                continue  # before the first prompt, e.g. a summary of an earlier session at startup
            step = _step(e, start)
            target = session_prompts[i]
            (target.side if (e.get("role") or "").startswith(SIDE) else target.steps).append(step)
        for prompt in session_prompts:
            _attach_chat(prompt, chat)
        # A prompt with no calls and no message in the chat file is one `/rate` blocked before the model.
        prompts += [p for p in session_prompts if p.steps or p.side or p.text]
    _attach_ratings(prompts)
    return sorted((p for p in prompts if p.start >= since.timestamp()), key=lambda p: p.start)


def _prompt(session_id: str, event: dict, chat: dict) -> traces.Prompt:
    return traces.Prompt(
        id=event.get("prompt_id") or session_id, session_id=session_id, session_title=chat.get("summary") or "",
        directory=chat.get("directory", ""), subagent=False, model="?", agent="-", start=event["_t"], text="",
        harness="gemini",
    )


def _step(event: dict, start: float) -> traces.Step:
    error = None
    if event["event"] == "error":
        error = f"{event.get('error_type') or 'Error'} {event.get('status') or ''}: {event.get('error', '')}".strip()
    cached = event.get("cached") or 0
    return traces.Step(
        id=f"{event.get('prompt_id')}@{event['ts']}", session_id=event.get("session_id"), start=start,
        end=event["_t"], input=max(0, (event.get("input") or 0) - cached), output=event.get("output") or 0,
        reasoning=event.get("thoughts") or 0, cache_read=cached, finish=event.get("finish"), error=error,
        role=event.get("role"), model=event.get("model"),
    )


def _attach_chat(prompt: traces.Prompt, chat: dict) -> None:
    """Give the prompt its text, id, model and conversation from the chat file, matched by time."""
    answers = [s for s in prompt.steps if s.role == "main" and not s.error]
    prompt.model = (answers[-1].model if answers else next((s.model for s in prompt.steps if s.model), "?")) or "?"
    users = [u for u in chat.get("turns", []) if not u.get("_used") and abs(u["t"] - prompt.start) <= MATCH_S]
    if not users:
        prompt.id = f"{prompt.session_id}.{prompt.start:.3f}"
        return
    turn = min(users, key=lambda u: abs(u["t"] - prompt.start))
    turn["_used"] = True
    prompt.id, prompt.text = turn["id"], turn["text"]
    prompt.parts = [{"type": "text", "text": turn["text"]}]
    replies = iter(turn["replies"])
    for step in prompt.steps:
        if step.role == "main" and not step.error:
            step.parts = next(replies, None)


def _attach_ratings(prompts: list[traces.Prompt]) -> None:
    by_session = defaultdict(list)
    for prompt in prompts:
        by_session[prompt.session_id].append(prompt)
    for record in ratings():
        candidates = [p for p in by_session.get(record.get("session_id"), [])
                      if p.start < _ts(record.get("rated_at") or record["ts"])]
        if candidates:
            max(candidates, key=lambda p: p.start).rating = record  # later lines win (rated twice)


def ratings() -> list[dict]:
    rows = []
    if traces.FEEDBACK_FILE.exists():
        for line in traces.FEEDBACK_FILE.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("harness") == "gemini" and record.get("ts"):
                rows.append(record)
    return sorted(rows, key=lambda r: r["ts"])


def _chat(session_id: str) -> dict:
    """The session's chat file replayed: user turns with their model replies, plus summary and directory.

    Gemini CLI's own file while it has one, otherwise openrate's copy.
    """
    candidates = [(path, path.parent.parent / ".project_root") for path in _originals(session_id)]
    candidates += [(path, path.parent / ".project_root") for path in _copies(session_id)]
    for path, root in candidates:
        metadata, messages = _replay(path)
        if metadata.get("sessionId") != session_id:
            continue
        turns = []
        for message in messages:
            if message.get("type") == "user" and _user_text(message):
                turns.append({"id": message["id"], "t": _ts(message["timestamp"]), "text": _user_text(message),
                              "replies": []})
            elif message.get("type") == "gemini" and turns:
                turns[-1]["replies"].append(_parts(message))
        return {"turns": turns, "summary": metadata.get("summary"),
                "directory": root.read_text().strip() if root.exists() else ""}
    return {}


def _originals(session_id: str) -> list[Path]:
    """Gemini CLI's chat files that may be this session's (the name has only the first 8 characters of the id)."""
    return sorted(GEMINI_DIR.glob(f"tmp/*/chats/{_chat_name(session_id)}"))


def _copies(session_id: str) -> list[Path]:
    """openrate's copies of them, in metrics/gemini/chats/<project>/."""
    return sorted(CHATS_DIR.glob(f"*/{_chat_name(session_id)}"))


def _chat_name(session_id: str) -> str:
    return f"session-*-{(session_id or '')[:8]}.jsonl"


def _replay(path: Path) -> tuple[dict, list[dict]]:
    metadata: dict = {}
    messages: dict[str, dict] = {}
    seen: dict[str, dict] = {}  # every message's fullest record so far, including ones a checkpoint dropped
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if "$rewindTo" in record:
            ids = list(messages)
            keep = ids[: ids.index(record["$rewindTo"])] if record["$rewindTo"] in ids else []
            messages = {i: messages[i] for i in keep}
        elif "$set" in record:
            update = dict(record["$set"])
            if isinstance(update.get("messages"), list):
                # A checkpoint (e.g. on --resume) lists the messages without thoughts, tool calls or tokens:
                # rebuild from it, keeping those details from the earlier records of the same messages.
                messages = {m["id"]: {**seen.get(m["id"], {}), **m}
                            for m in update.pop("messages") if isinstance(m, dict) and "id" in m}
            metadata.update(update)
        elif "id" in record and "type" in record:
            messages[record["id"]] = record
            seen[record["id"]] = {**seen.get(record["id"], {}), **record}
        elif "sessionId" in record:
            metadata.update(record)
    return metadata, list(messages.values())


def _user_text(message: dict) -> str:
    """The text a person typed; empty for tool results and Gemini CLI's own session context."""
    content = message.get("displayContent") or message.get("content")
    if isinstance(content, str):
        text = content
    else:
        parts = content if isinstance(content, list) else [content]
        if any(isinstance(p, dict) and "functionResponse" in p for p in parts):
            return ""
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return "" if text.lstrip().startswith("<session_context>") else text.strip()


def _parts(message: dict) -> list[dict]:
    """A Gemini CLI reply as the part list `lab show` prints (same shape as OpenCode's parts)."""
    out = []
    for thought in message.get("thoughts") or []:
        text = " ".join(t for t in (thought.get("subject"), thought.get("description")) if t)
        if text:
            out.append({"type": "reasoning", "text": text})
    content = message.get("content")
    text = content if isinstance(content, str) else "".join(
        p.get("text", "") for p in (content or []) if isinstance(p, dict) and not p.get("thought"))
    if text.strip():
        out.append({"type": "text", "text": text.strip()})
    for call in message.get("toolCalls") or []:
        status = call.get("status")
        output = call.get("resultDisplay") if isinstance(call.get("resultDisplay"), str) else _tool_output(call)
        out.append({"type": "tool", "tool": call.get("name"), "state": {
            "input": call.get("args", {}),
            "status": {"success": "completed"}.get(status, status),
            "output" if status != "error" else "error": output,
        }})
    return out


def _tool_output(call: dict) -> str:
    for part in call.get("result") or []:
        response = (part.get("functionResponse") or {}).get("response") if isinstance(part, dict) else None
        if isinstance(response, dict):
            value = response.get("output", response.get("error", response))
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ""


def _ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
