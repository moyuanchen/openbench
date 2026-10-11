"""The saved traces, joined per prompt: what `lab metrics`, `lab prompts` and `lab show` read.

opencode  OpenCode's session database (read-only): every prompt, its steps (one model request each) with
          tokens and times, and the full conversation (text, tool calls, edits)
server    one line per request, written by the proxy into the `lab-metrics` Volume (downloaded to metrics/server/)
ratings   metrics/feedback.jsonl, written by the rating plugin; one line per prompt, keyed by its message id

A server line is matched to the OpenCode step that sent it: same session (OpenCode sends its session id with
every request and the proxy records it), same prompt token count, and reaching the server while the step ran
(a request can wait minutes in Modal's queue during a cold start). Lines logged before 2026-10-10 21:45 have no
session id; for those the request must have started within a few seconds of the step.
"""

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from lab import cloud
from lab.registry import REPO

SERVER_DIR = REPO / "metrics" / "server"
FEEDBACK_FILE = REPO / "metrics" / "feedback.jsonl"
OPENCODE_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
PROVIDER_ID = "lab"
MATCH_WINDOW_S = 5.0  # a step and a server request belong together if they started this close…
# …and if OpenCode's input + cached tokens equal the server's prompt tokens. Time alone is not enough: at the
# start of a prompt OpenCode also sends a small title-generation request to the same model.
RATING_LABELS = {4: "Good as is", 3: "Small fixes needed", 2: "Big fixes needed", 1: "Wrong or unusable"}


@dataclass
class Step:
    """One assistant message in OpenCode = one request to the model."""

    id: str
    session_id: str
    start: float
    end: float | None
    input: int  # prompt tokens not served from the cache
    output: int
    reasoning: int
    cache_read: int
    finish: str | None
    error: str | None
    request: dict | None = None  # the proxy's line for the same request, when matched

    @property
    def prompt_tokens(self) -> int:
        return self.input + self.cache_read

    @property
    def duration_s(self) -> float | None:
        return None if self.end is None else round(self.end - self.start, 3)  # OpenCode stores milliseconds


@dataclass
class Prompt:
    """One user message and the steps that answered it."""

    id: str  # OpenCode's user message id
    session_id: str
    session_title: str
    directory: str
    subagent: bool
    model: str  # the models.toml name
    agent: str
    start: float
    text: str
    steps: list[Step] = field(default_factory=list)
    rating: dict | None = None  # the rating log line, if the plugin asked

    @property
    def end(self) -> float | None:
        ends = [s.end for s in self.steps if s.end is not None]
        return max(ends) if ends else None

    @property
    def duration_s(self) -> float | None:
        return None if self.end is None else self.end - self.start

    @property
    def tokens_in(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def tokens_out(self) -> int:
        return sum(s.output for s in self.steps)

    @property
    def open_command(self) -> str:
        return f"opencode {self.directory} -s {self.session_id}"


def download() -> None:
    SERVER_DIR.mkdir(parents=True, exist_ok=True)
    result = cloud.modal_cli("volume", "get", "lab-metrics", "/", str(SERVER_DIR), "--force", capture=True)
    if result.returncode != 0 and "not found" not in (result.stderr or "").lower():
        print(f"warning: could not download server metrics: {result.stderr.strip()[:200]}")


def load(since: datetime) -> list[Prompt]:
    """Prompts to `lab/*` models since `since`, oldest first, with their ratings and matched server lines."""
    prompts = _opencode_prompts(since)
    rated = {r.get("user_message_id"): r for rows in ratings(since).values() for r in rows}
    for prompt in prompts:
        prompt.rating = rated.get(prompt.id)
    requests = [r for rows in server_records(since).values() for r in rows if "event" not in r]
    for step, request in match(_steps(prompts), requests):
        step.request = request
    return prompts


def server_records(since: datetime) -> dict[str, list[dict]]:
    by_model = defaultdict(list)
    for path in SERVER_DIR.rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
                ts = datetime.fromisoformat(record["ts"])
            except (ValueError, KeyError):
                continue
            if ts >= since:
                record["_end"] = ts.timestamp()
                by_model[record["model"]].append(record)
    return by_model


def ratings(since: datetime) -> dict[str, list[dict]]:
    by_model = defaultdict(list)
    if not FEEDBACK_FILE.exists():
        return by_model
    for line in FEEDBACK_FILE.read_text().splitlines():
        try:
            record = json.loads(line)
            ts = datetime.fromisoformat(record["ts"].replace("Z", "+00:00"))
        except (ValueError, KeyError):
            continue
        if ts >= since:
            by_model[record["model"].removeprefix(f"{PROVIDER_ID}/")].append(record)
    return by_model


def match(steps: list[Step], requests: list[dict]) -> list[tuple[Step, dict]]:
    """Pairs each finished step with the server line of the same request, using each line at most once."""
    unused = [r for r in requests if r.get("t_done") is not None and r.get("prompt_tokens") is not None]
    pairs = []
    for step in sorted((s for s in steps if s.end is not None), key=lambda s: s.start):
        candidates = [r for r in unused if r["prompt_tokens"] == step.prompt_tokens and (
            step.start - MATCH_WINDOW_S <= _request_start(r) <= step.end if r.get("session_id") == step.session_id
            else r.get("session_id") is None and abs(_request_start(r) - step.start) <= MATCH_WINDOW_S)]
        if candidates:
            best = min(candidates, key=lambda r: abs(_request_start(r) - step.start))
            unused.remove(best)
            pairs.append((step, best))
    return pairs


def find(ident: str, prompts: list[Prompt], requests: list[dict]) -> tuple[list[Prompt], str | None]:
    """The prompt(s) an id points to: a prompt (message) id, a session id or a server request id.

    Any unique part of an id works, e.g. the short ID column of `lab prompts`. Returns the prompts and a note.
    """
    hits = [p for p in prompts if ident in p.id]
    if len(hits) == 1:
        return hits, None
    sessions = {p.session_id for p in prompts if ident in p.session_id}
    if len(sessions) == 1:
        return [p for p in prompts if p.session_id in sessions], None
    found = [r for r in requests if ident in (r.get("request_id") or "")]
    if len(found) == 1:
        request = found[0]
        for prompt in prompts:
            if any(s.request and s.request["request_id"] == request["request_id"] for s in prompt.steps):
                return [prompt], f"server request {request['request_id']}"
        in_session = [p for p in prompts if p.session_id == request.get("session_id")]
        if in_session:
            nearest = min(in_session, key=lambda p: abs(p.start - _request_start(request)))
            return [nearest], (f"server request {request['request_id']} is a side request (e.g. the session title) "
                               "sent while this prompt ran")
        raise KeyError(f"server request {request['request_id']} has no session id (logged before the proxy recorded "
                       "them) and matches no step")
    if len(hits) + len(sessions) + len(found) > 1:
        raise KeyError(f"{ident!r} matches several prompts, sessions or requests; use more of the id")
    raise KeyError(f"no prompt, session or server request matches {ident!r}")


def _steps(prompts: list[Prompt]) -> list[Step]:
    return [s for p in prompts for s in p.steps]


def _request_start(request: dict) -> float:
    return request["_end"] - request["t_done"]


def _opencode_prompts(since: datetime) -> list[Prompt]:
    if not OPENCODE_DB.exists():
        return []
    con = sqlite3.connect(f"file:{OPENCODE_DB}?mode=ro", uri=True)
    try:
        rows = con.execute("select id, session_id, data from message where time_created >= ?",
                           (int(since.timestamp() * 1000),)).fetchall()
        messages = {row[0]: (row[1], json.loads(row[2])) for row in rows}
        steps_by_parent = defaultdict(list)
        for message_id, (session_id, data) in messages.items():
            if data.get("role") == "assistant" and data.get("providerID") == PROVIDER_ID:
                steps_by_parent[data.get("parentID")].append(_step(message_id, session_id, data))
        # A prompt sent just before `since` may have steps after it: fetch those prompts too.
        missing = [m for m in steps_by_parent if m not in messages]
        if missing:
            marks = ",".join("?" * len(missing))
            for row in con.execute(f"select id, session_id, data from message where id in ({marks})", missing):
                messages[row[0]] = (row[1], json.loads(row[2]))
        users = {m: messages[m] for m, (_, data) in messages.items() if data.get("role") == "user"
                 and (m in steps_by_parent or (data.get("model") or {}).get("providerID") == PROVIDER_ID)}
        session_ids = list({session_id for session_id, _ in users.values()})
        sessions = {row[0]: row[1:] for row in con.execute(
            f"select id, title, directory, parent_id from session where id in ({','.join('?' * len(session_ids))})",
            session_ids)}
        texts = defaultdict(list)
        if users:
            marks = ",".join("?" * len(users))
            for message_id, data in con.execute(
                    f"select message_id, data from part where message_id in ({marks}) order by id", list(users)):
                part = json.loads(data)
                if part.get("type") == "text" and not part.get("synthetic"):
                    texts[message_id].append(part.get("text", ""))
    finally:
        con.close()
    prompts = []
    for message_id, (session_id, data) in users.items():
        steps = sorted(steps_by_parent.get(message_id, []), key=lambda s: s.start)
        last = messages[steps[-1].id][1] if steps else {}
        title, directory, parent = sessions.get(session_id, ("", "", None))
        prompts.append(Prompt(
            id=message_id,
            session_id=session_id,
            session_title=title,
            directory=directory,
            subagent=parent is not None,
            model=last.get("modelID") or (data.get("model") or {}).get("modelID", "?"),
            agent=last.get("agent") or data.get("agent", "?"),
            start=data["time"]["created"] / 1000,
            text="\n".join(texts.get(message_id, [])).strip(),
            steps=steps,
        ))
    return sorted(prompts, key=lambda p: p.start)


def _step(message_id: str, session_id: str, data: dict) -> Step:
    time, tokens, error = data.get("time", {}), data.get("tokens", {}), data.get("error")
    return Step(
        id=message_id,
        session_id=session_id,
        start=time["created"] / 1000,
        end=time["completed"] / 1000 if time.get("completed") else None,
        input=tokens.get("input", 0),
        output=tokens.get("output", 0),
        reasoning=tokens.get("reasoning", 0),
        cache_read=(tokens.get("cache") or {}).get("read", 0),
        finish=data.get("finish"),
        error=None if not error else f"{error.get('name')}: {(error.get('data') or {}).get('message', '')}".strip(": "),
    )


def parts(step_ids: list[str]) -> dict[str, list[dict]]:
    """The content of the given messages (text, reasoning, tool calls, edits), in order."""
    by_message = defaultdict(list)
    if not step_ids:
        return by_message
    con = sqlite3.connect(f"file:{OPENCODE_DB}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(step_ids))
        for message_id, data in con.execute(
                f"select message_id, data from part where message_id in ({marks}) order by id", step_ids):
            by_message[message_id].append(json.loads(data))
    finally:
        con.close()
    return by_message


# --- `lab prompts` and `lab show` ---

OUTPUT_LINES = 8  # tool output lines `lab show` prints per call without --full


def rating_value(prompt: Prompt) -> int | str | None:
    """1–4, "skip" (dialog dismissed), or None (never asked)."""
    if prompt.rating is None:
        return None
    return prompt.rating.get("rating") or "skip"


def render_list(prompts: list[Prompt], width: int) -> str:
    rows = [("ID", "WHEN", "MODEL", "AGENT", "RATING", "STEPS", "TOKENS IN", "OUT", "TIME", "PROMPT")]
    for p in prompts:
        rows.append((p.id[-8:], f"{datetime.fromtimestamp(p.start):%m-%d %H:%M}", p.model, p.agent,
                     str(rating_value(p) or "-"), str(len(p.steps)), f"{p.tokens_in:,}", f"{p.tokens_out:,}",
                     _time(p), (p.text.splitlines() or [""])[0]))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    room = max(20, width - sum(widths) - 2 * len(widths))
    lines = ["  ".join([*(cell.ljust(w) for cell, w in zip(row, widths)), _cut(row[-1], room)]).rstrip()
             for row in rows]
    lines.append("\n`lab show <ID>` prints a prompt's conversation and numbers; it also takes a session or server request id.")
    return "\n".join(lines)


def render_show(prompts: list[Prompt], note: str | None, full: bool) -> str:
    first = prompts[0]
    out = [f"({note})"] if note else []
    out.append(f"Session {first.session_id}  “{first.session_title}”" + ("  (subagent)" if first.subagent else ""))
    out.append(f"  open in OpenCode: {first.open_command}")
    content = parts([p.id for p in prompts] + [s.id for p in prompts for s in p.steps])
    for p in prompts:
        rating = rating_value(p)
        cached = sum(s.cache_read for s in p.steps)
        out += ["", f"── Prompt {p.id} · {datetime.fromtimestamp(p.start):%Y-%m-%d %H:%M:%S} · lab/{p.model} · {p.agent}",
                f"   rating {_rating_text(rating)} · {len(p.steps)} steps · tokens in {p.tokens_in:,} "
                f"({cached:,} cached) · out {p.tokens_out:,} · {_time(p)}", "", "You"]
        out += _parts(content.get(p.id, []), full)
        for i, step in enumerate(p.steps, 1):
            out += ["", f"Step {i} · " + _step_numbers(step)]
            out += _parts(content.get(step.id, []), full)
    return "\n".join(out)


def _step_numbers(step: Step) -> str:
    bits = [_s(step.duration_s) if step.end else "unfinished",
            f"in {step.input:,} + {step.cache_read:,} cached", f"out {step.output:,}"]
    if step.reasoning:
        bits.append(f"{step.reasoning:,} reasoning")
    if step.finish not in (None, "stop", "tool-calls"):
        bits.append(f"finished: {step.finish}")
    if step.error:
        bits.append(f"error: {step.error}")
    r = step.request
    if r:
        if r.get("t_first_token") is not None:
            bits.append(f"first token {_s(r['t_first_token'])}")
        if r.get("decode_tok_s"):
            bits.append(f"{r['decode_tok_s']:.1f} tok/s")
        bits.append(f"{r.get('in_flight', '?')} running at once · request {r['request_id']}")
    return " · ".join(bits)


def _parts(items: list[dict], full: bool) -> list[str]:
    out = []
    for part in items:
        kind = part.get("type")
        if kind == "text" and not part.get("synthetic"):
            out += _indent(part.get("text", ""))
        elif kind == "reasoning":
            out += ["  thinking:"] + _indent(part.get("text", ""), 4)
        elif kind == "tool":
            state = part.get("state") or {}
            args = json.dumps(state.get("input", {}), ensure_ascii=False)
            out.append(f"  → {part.get('tool')} {args if full else _cut(args, 300)}")
            if state.get("status") == "error":
                out += _indent(f"error: {state.get('error', '')}", 6)
            elif state.get("output"):
                lines = state["output"].splitlines()
                shown = lines if full else lines[:OUTPUT_LINES]
                out += _indent("\n".join(shown), 6)
                if len(shown) < len(lines):
                    out.append(f"      … {len(lines) - len(shown)} more lines (--full shows everything)")
            elif state.get("status") != "completed":
                out.append(f"      ({state.get('status')})")
            if part.get("tool") == "task" and (state.get("metadata") or {}).get("sessionId"):
                out.append(f"      subagent session: {state['metadata']['sessionId']} (lab show it)")
        elif kind == "patch":
            out.append(f"  (changed: {', '.join(part.get('files', []))})")
        elif kind == "compaction":
            out.append("  (context compacted)")
        elif kind == "file":
            out.append(f"  [file: {part.get('filename') or part.get('url', '')}]")
        elif kind == "subtask":
            out.append(f"  [subtask for {part.get('agent')}: {part.get('description', '')}]")
    return out


def _indent(text: str, spaces: int = 2) -> list[str]:
    return [" " * spaces + line if line else "" for line in text.splitlines()]


def _rating_text(rating) -> str:
    if rating is None:
        return "not asked"
    if rating == "skip":
        return "skipped"
    return f"{rating} ({RATING_LABELS.get(rating, '?')})"


def _time(prompt: Prompt) -> str:
    errors = [s.error for s in prompt.steps if s.error]
    if errors:
        return "aborted" if errors[-1].startswith("MessageAbortedError") else "error"
    return "-" if prompt.duration_s is None else _s(prompt.duration_s)


def _s(seconds: float) -> str:
    return f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.0f}s"


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
