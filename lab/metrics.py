"""`lab metrics`: token usage and completion time per harness and model, from the traces in lab/traces.py.

OpenCode (our models on Modal):
server   the proxy's line per request (all requests, including OpenCode's side requests such as session titles)
client   OpenCode's own record of each prompt and step (one step = one request)
ratings  the 1–4 ratings from the rating plugin

Gemini CLI (Google's API; lab/gemini.py): its own telemetry, one entry per call, and the ratings from `/rate` and
the pop-up. Times are measured on this machine, so they include the network; there is no server side.
"""

from collections import defaultdict
from datetime import datetime, timezone

from lab import gemini, traces

download = traces.download


def summarize(since: datetime, model: str | None = None, harnesses: tuple[str, ...] = ("opencode",)) -> dict:
    """{harness: {model: summary}} for the chosen harnesses."""
    out = {}
    if "opencode" in harnesses:
        server = traces.server_records(since)
        prompts = defaultdict(list)
        for prompt in traces.load(since):
            prompts[prompt.model].append(prompt)
        ratings = traces.ratings(since)
        names = sorted(set(server) | set(prompts) | set(ratings))
        out["opencode"] = {name: _summary(server.get(name, []), prompts.get(name, []), ratings.get(name, []))
                           for name in names if model in (None, name)}
    if "gemini" in harnesses:
        prompts = defaultdict(list)
        for prompt in gemini.load(since):
            prompts[prompt.model].append(prompt)
        out["gemini"] = {name: _gemini_summary(rows) for name, rows in sorted(prompts.items()) if model in (None, name)}
    return out


def render(summaries: dict, since: datetime) -> str:
    out = []
    for harness, models in summaries.items():
        title = {"opencode": "OpenCode (lab models on Modal)", "gemini": "Gemini CLI (Google's API)"}[harness]
        out += [f"== {title}, since {since.astimezone():%Y-%m-%d %H:%M}", ""]
        if not models:
            out += ["No data.", ""]
        out += (_render_gemini if harness == "gemini" else _render_opencode)(models)
    return "\n".join(out).rstrip()


def _render_opencode(summaries: dict[str, dict]) -> list[str]:
    out = []
    for name, s in summaries.items():
        srv, cli, rat = s["server"], s["client"], s["ratings"]
        out.append(name)
        if srv["requests"] or srv["container_starts"]:
            out.append(f"  server   {srv['requests']} requests ({srv['side_requests']} side requests without tools, e.g. "
                       f"session titles: {srv['side_prompt_tokens']:,} tokens in)")
            out.append(f"           {srv['errors']} errors · {srv['aborted']} aborted · "
                       f"{srv['container_starts']} container starts (startup p50 {_s(srv['startup_s']['p50'])})")
            out.append(f"           tokens in {srv['prompt_tokens']:,} ({srv['cached_tokens']:,} cached) · "
                       f"out {srv['completion_tokens']:,}" + (f" ({srv['reasoning_tokens']:,} reasoning)" if srv['reasoning_tokens'] else ""))
            out.append(f"           first token p50 {_s(srv['t_first_token']['p50'])} · p95 {_s(srv['t_first_token']['p95'])}   "
                       f"done p50 {_s(srv['t_done']['p50'])} · p95 {_s(srv['t_done']['p95'])}   "
                       f"output {_n(srv['decode_tok_s']['p50'])} tok/s (p50)")
        if cli["steps"]:
            out.append(f"  client   {cli['prompts']} prompts · {cli['steps']} steps · tokens in {cli['input_tokens']:,} "
                       f"(+{cli['cache_read_tokens']:,} cached) · out {cli['output_tokens']:,}")
            out.append(f"           prompt time p50 {_s(cli['prompt_s']['p50'])} · p95 {_s(cli['prompt_s']['p95'])}   "
                       f"step p50 {_s(cli['step_s']['p50'])}   network+queue p50 {_s(s['overhead_s']['p50'])} "
                       f"({s['matched_steps']} steps matched to server requests)")
        out += _render_ratings(rat)
        out.append("")
    return out


def _render_gemini(summaries: dict[str, dict]) -> list[str]:
    out = []
    for name, s in summaries.items():
        calls, side = s["calls"], s["side"]
        out.append(name)
        out.append(f"  calls    {calls['calls']} calls for {s['prompts']} prompts · {calls['errors']} errors"
                   + (f" ({calls['rate_limited']} rate-limited)" if calls["rate_limited"] else ""))
        out.append(f"           tokens in {calls['prompt_tokens']:,} ({calls['cached_tokens']:,} cached) · "
                   f"out {calls['output_tokens']:,}" + (f" (+{calls['thoughts_tokens']:,} thinking)" if calls['thoughts_tokens'] else ""))
        out.append(f"           call time p50 {_s(calls['call_s']['p50'])} · p95 {_s(calls['call_s']['p95'])}   "
                   f"prompt time p50 {_s(s['prompt_s']['p50'])} · p95 {_s(s['prompt_s']['p95'])}")
        if side["calls"]:
            roles = ", ".join(f"{role} {n}" for role, n in side["roles"].items())
            out.append(f"  side     {side['calls']} calls by Gemini CLI itself ({roles}): tokens in "
                       f"{side['prompt_tokens']:,} · out {side['output_tokens']:,}")
        out += _render_ratings(s["ratings"])
        out.append("")
    return out


def _render_ratings(rat: dict) -> list[str]:
    if not rat["prompts"]:
        return []
    dist = " ".join(f"{k}:{rat['distribution'].get(k, 0)}" for k in (4, 3, 2, 1))
    return [f"  ratings  {rat['rated']} of {rat['prompts']} rated · avg {_n(rat['average'])} · {dist} · "
            f"{rat['prompts'] - rat['rated']} skipped"]


def _summary(server: list[dict], prompts: list[traces.Prompt], ratings: list[dict]) -> dict:
    requests = [r for r in server if "event" not in r]
    side = [r for r in requests if not r.get("tools")]  # OpenCode's title/summary requests carry no tools
    starts = [r for r in server if r.get("event") == "container_start"]
    pick = lambda rows, key: [r[key] for r in rows if r.get(key) is not None]  # noqa: E731
    steps = [s for p in prompts for s in p.steps if s.end is not None]
    done = [p for p in prompts if p.end is not None]
    overhead = [s.duration_s - s.request["t_done"] for s in steps if s.request]  # network, Modal routing, queueing
    return {
        "server": {
            "requests": len(requests),
            "side_requests": len(side),
            "side_prompt_tokens": sum(pick(side, "prompt_tokens")),
            "errors": sum(1 for r in requests if (r.get("status") or 0) >= 400),
            "aborted": sum(1 for r in requests if r.get("outcome") == "aborted"),
            "container_starts": len(starts),
            "startup_s": _stats(pick(starts, "startup_s")),
            "prompt_tokens": sum(pick(requests, "prompt_tokens")),
            "cached_tokens": sum(pick(requests, "cached_tokens")),
            "completion_tokens": sum(pick(requests, "completion_tokens")),
            "reasoning_tokens": sum(pick(requests, "reasoning_tokens")),
            "t_first_token": _stats(pick(requests, "t_first_token")),
            "t_done": _stats(pick(requests, "t_done")),
            "decode_tok_s": _stats(pick(requests, "decode_tok_s")),
        },
        "client": {
            "steps": len(steps),
            "prompts": len(done),
            "input_tokens": sum(s.input for s in steps),
            "output_tokens": sum(s.output for s in steps),
            "reasoning_tokens": sum(s.reasoning for s in steps),
            "cache_read_tokens": sum(s.cache_read for s in steps),
            "step_s": _stats([s.duration_s for s in steps]),
            "prompt_s": _stats([p.duration_s for p in done]),
        },
        "overhead_s": _stats(overhead),
        "matched_steps": len(overhead),
        "ratings": _ratings(ratings),
    }


def _gemini_summary(prompts: list[traces.Prompt]) -> dict:
    calls = [s for p in prompts for s in p.steps]
    side = [s for p in prompts for s in p.side]
    ok = [s for s in calls if not s.error]
    done = [p for p in prompts if p.end is not None]
    roles = defaultdict(int)
    for s in side:
        roles[(s.role or "?").removeprefix("utility_")] += 1
    return {
        "prompts": len(prompts),
        "calls": {
            "calls": len(calls),
            "errors": len(calls) - len(ok),
            "rate_limited": sum(1 for s in calls if s.error and " 429" in s.error),
            "prompt_tokens": sum(s.prompt_tokens for s in ok),
            "cached_tokens": sum(s.cache_read for s in ok),
            "output_tokens": sum(s.output for s in ok),
            "thoughts_tokens": sum(s.reasoning for s in ok),
            "call_s": _stats([s.duration_s for s in ok]),
        },
        "side": {
            "calls": len(side),
            "roles": dict(sorted(roles.items(), key=lambda kv: -kv[1])),
            "prompt_tokens": sum(s.prompt_tokens for s in side),
            "output_tokens": sum(s.output + s.reasoning for s in side),
        },
        "prompt_s": _stats([p.duration_s for p in done]),
        "ratings": _ratings([p.rating for p in prompts if p.rating]),
    }


def _ratings(records: list[dict]) -> dict:
    rated = [r["rating"] for r in records if r.get("rating") is not None]
    return {
        "prompts": len(records),
        "rated": len(rated),
        "average": sum(rated) / len(rated) if rated else None,
        "distribution": {k: rated.count(k) for k in (1, 2, 3, 4)},
    }


def _stats(values: list[float]) -> dict:
    values = sorted(values)
    at = lambda q: values[min(len(values) - 1, int(q * len(values)))] if values else None  # noqa: E731
    return {"n": len(values), "p50": at(0.5), "p95": at(0.95)}


def _s(seconds) -> str:
    return "-" if seconds is None else f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.0f}s"


def _n(value) -> str:
    return "-" if value is None else f"{value:.1f}"


def since_cutoff(duration_s: int) -> datetime:
    return datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - duration_s, timezone.utc)
