"""`lab`: serve models from models.toml on Modal, wire them into OpenCode, and log and rate prompts in OpenCode
and Gemini CLI."""

import argparse
import json
import os
import secrets
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

from lab import cloud, gemini, metrics, profiles, registry, traces
from lab.registry import REPO

MAX_WARM = 23 * 3600  # the keeper runs as one Modal function call, and those are capped at 24h


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    sub.add_parser("init", help="one-time setup: API key and Modal proxy token in .secrets/, and the `lab-api-key` Modal Secret")
    sub.add_parser("ls", help="models in models.toml with deploy state, running containers and URL")
    p = sub.add_parser("pull", help="download a model's weights into the Modal volume (CPU only)")
    p.add_argument("name")
    p = sub.add_parser("deploy", help="deploy (or redeploy) a model's Modal app, then update OpenCode")
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true", help="every model in models.toml")
    p.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    p = sub.add_parser("stop", help="stop a model's Modal app (weights stay in the volume)")
    p.add_argument("name")
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("warm", help="start a container now, or keep one running for a while")
    p.add_argument("name")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--for", dest="duration", metavar="DURATION", help='keep 1 container up, e.g. "90m", "2h"')
    group.add_argument("--off", action="store_true", help="end a --for period now")
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("harness", help="show the harnesses, or switch to one (opencode, gemini)")
    p.add_argument("name", nargs="?", choices=list(profiles.HARNESSES))
    p = sub.add_parser("use", help="choose a profile for the active harness (profiles/<harness>/<name>.toml)")
    p.add_argument("profile")
    sub.add_parser("profiles", help="list the active harness's profiles; * marks the current one")
    sub.add_parser("sync", help="re-apply the active harness's profile (OpenCode: e.g. after a deploy)")
    harness_help = "opencode, gemini or all (default: the active harness)"
    p = sub.add_parser("metrics", help="token usage, response times and ratings per harness and model")
    p.add_argument("--since", default="7d", help='how far back, e.g. "24h", "7d" (default 7d)')
    p.add_argument("--model", help="only this model")
    p.add_argument("--harness", choices=[*profiles.HARNESSES, "all"], help=harness_help)
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    p.add_argument("--offline", action="store_true", help="don't download new server logs first")
    p = sub.add_parser("prompts", help="find prompts by time, tokens or rating (newest first); `lab show` opens one")
    p.add_argument("--since", default="7d", help='how far back, e.g. "24h", "7d" (default 7d)')
    p.add_argument("--model", help="only this model")
    p.add_argument("--harness", choices=[*profiles.HARNESSES, "all"], help=harness_help)
    p.add_argument("--rating", help='only these ratings, e.g. "1,2"; "skip" = dialog dismissed, "none" = not asked')
    p.add_argument("--sort", choices=["time", "tokens", "duration"], default="time",
                   help="newest (default), most tokens, or slowest first")
    p.add_argument("-n", "--limit", type=int, default=20, help="how many to list (default 20)")
    p.add_argument("--offline", action="store_true", help="don't download new server logs first")
    p = sub.add_parser("show", help="a prompt's conversation with its numbers (tokens, times, rating)")
    p.add_argument("id", help="a prompt ID from `lab prompts`, a session id, or a server request id (any unique part); "
                              "any harness")
    p.add_argument("--full", action="store_true", help="don't shorten tool calls and outputs")
    p.add_argument("--offline", action="store_true", help="don't download new server logs first")
    p = sub.add_parser("logs", help="a model's Modal app logs (recent lines, or -f to stream)")
    p.add_argument("name")
    p.add_argument("-f", "--follow", action="store_true", help="keep streaming new lines")

    args = parser.parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"lab: {exc.args[0] if exc.args else exc}")


def cmd_init(args) -> None:
    key_file = profiles.KEY_FILE
    if key_file.exists():
        key = key_file.read_text().strip()
        print(f"API key: using existing {key_file.relative_to(REPO)}")
    else:
        key = secrets.token_urlsafe(32)
        cloud.private_write(key_file, key)
        print(f"API key: created {key_file.relative_to(REPO)}")
    secret_file = key_file.with_name("lab-api-key.json")  # a file, so the key never appears in `ps`
    cloud.private_write(secret_file, json.dumps({"API_KEY": key}))
    try:
        result = cloud.modal_cli("secret", "create", "lab-api-key", "--from-json", str(secret_file), "--force")
    finally:
        secret_file.unlink()
    if result.returncode != 0:
        raise RuntimeError("could not create the `lab-api-key` Modal Secret (is `modal` logged in?)")
    print("Modal Secret `lab-api-key`: up to date. Volumes are created on first pull/deploy.")

    token_files = profiles.PROXY_TOKEN_FILES
    shown = " and ".join(str(path.relative_to(REPO)) for path in token_files.values())
    if all(path.exists() for path in token_files.values()):
        print(f"Modal proxy token: using existing {shown}")
        return
    result = cloud.modal_cli("workspace", "proxy-tokens", "create", "--json", capture=True)
    if result.returncode != 0:
        raise RuntimeError("could not create a Modal proxy token (is `modal` logged in?)")
    token = json.loads(result.stdout)  # {"Modal-Key": ..., "Modal-Secret": ..., "Authorization": ...}
    for name, path in token_files.items():
        cloud.private_write(path, token[name])
    print(f"Modal proxy token: created {shown}")


def cmd_ls(args) -> None:
    specs = registry.load()
    apps = cloud.deployed_apps()
    state = cloud.load_state()
    rows = [("MODEL", "BACKEND", "GPU", "STATE", "RUNNING", "WARM UNTIL", "URL")]
    for name, spec in specs.items():
        app = apps.get(spec.app_name, {})
        until = state["warm"].get(name, {}).get("until")
        rows.append((
            name,
            spec.backend,
            spec.modal_gpu,
            app.get("state", "not deployed"),
            str(app.get("tasks", "-")),
            until[11:16] if until and until > datetime.now().isoformat() else "-",
            state["urls"].get(name, "-"),
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def cmd_pull(args) -> None:
    spec = registry.get(args.name)
    print(f"Pulling {spec.source}@{spec.revision or 'main'} into the lab-hf-cache volume (CPU only)...", flush=True)
    # --detach: the download keeps going on Modal even if this terminal closes.
    if cloud.modal_cli("run", "--detach", "-m", "serving.app::pull", model=spec.name).returncode != 0:
        raise RuntimeError(f"pull failed for {spec.name}")


def cmd_deploy(args) -> None:
    specs = registry.load()
    names = list(specs) if args.all else args.names
    if not names:
        raise ValueError("name a model to deploy, or pass --all")
    state = cloud.load_state()
    for name in names:
        spec = registry.get(name)
        if spec.min_containers > 0:
            cost = cloud.hourly_cost(spec, spec.min_containers)
            _confirm(
                f"{name} has min_containers = {spec.min_containers}: GPUs stay on around the clock"
                + (f" (≈ ${cost:.2f}/h, ${cost * 24 * 30:,.0f}/month)" if cost else "") + ". Deploy?",
                args.yes,
            )
        if cloud.modal_cli("deploy", "-m", "serving.app", model=name).returncode != 0:
            raise RuntimeError(f"deploy failed for {name}")
        state["urls"][name] = cloud.web_url(spec)
        # Make models.toml authoritative after every deploy, but let an active `lab warm --for` period continue
        # (its keeper still ends it on time). Changed code replaces running containers; unchanged code keeps them.
        warm = state["warm"].get(name)
        warm_active = bool(warm) and warm["until"] > datetime.now().isoformat()
        cloud.apply_autoscaler(spec, max(1, spec.min_containers) if warm_active else None)
        if not warm_active:
            state["warm"].pop(name, None)
        cloud.save_state(state)
        print(f"{name}: {state['urls'][name]}" + (f" (still kept warm until {warm['until'][11:16]})" if warm_active else ""))
    _resync(state)


def cmd_stop(args) -> None:
    spec = registry.get(args.name)
    _confirm(f"Stop the Modal app {spec.app_name}? (Weights stay; `lab deploy {spec.name}` brings it back.)", args.yes)
    if cloud.modal_cli("app", "stop", spec.app_name, "--yes").returncode != 0:
        raise RuntimeError(f"could not stop {spec.app_name}")
    state = cloud.load_state()
    state["urls"].pop(spec.name, None)
    state["warm"].pop(spec.name, None)
    cloud.save_state(state)
    _resync(state)


def cmd_warm(args) -> None:
    spec = registry.get(args.name)
    state = cloud.load_state()
    url = state["urls"].get(spec.name)
    if not url:
        raise ValueError(f"{spec.name} is not deployed (`lab deploy {spec.name}`)")

    if args.off or args.duration:  # a new period replaces the old one; a plain ping leaves it alone
        previous = state["warm"].pop(spec.name, None)
        if previous:
            cloud.cancel_call(previous["call_id"])
    if args.off:
        cloud.set_min_containers(spec, spec.min_containers)
        cloud.save_state(state)
        print(f"{spec.name}: warm period ended; it will shut down after {spec.scaledown_window // 60} idle minutes")
        return

    if args.duration:
        seconds = registry.parse_duration(args.duration)
        if not 60 <= seconds <= MAX_WARM:
            raise ValueError("--for must be between 1m and 23h")
        cost = cloud.hourly_cost(spec)
        _confirm(
            f"Keep 1 container of {spec.name} ({spec.modal_gpu}) running for {_human(seconds)}"
            + (f", ≈ ${cost * seconds / 3600:.2f} of GPU time" if cost else "") + "?",
            args.yes,
        )
        cloud.set_min_containers(spec, max(1, spec.min_containers))
        until = datetime.now() + timedelta(seconds=seconds)
        state["warm"][spec.name] = {"until": until.isoformat(timespec="seconds"), "call_id": cloud.spawn_keeper(spec, seconds)}
        cloud.save_state(state)
        print(f"{spec.name}: kept warm until {until:%H:%M}; `lab warm {spec.name} --off` ends it early")

    print(f"Waiting for {spec.name} to answer (a cold start takes a few minutes)...", flush=True)
    t0 = time.monotonic()
    request = urllib.request.Request(f"{url}/health", headers=profiles.proxy_auth_headers())
    with urllib.request.urlopen(request, timeout=30 * 60) as response:
        response.read()
    print(f"{spec.name}: ready ({time.monotonic() - t0:.0f}s)")


def cmd_harness(args) -> None:
    state = cloud.load_state()
    if args.name:
        state["harness"] = args.name
        name = state["profiles"].get(args.name)
        if not name:
            choices = profiles.names(args.name)
            if not choices:
                raise FileNotFoundError(f"no profiles in profiles/{args.name}/")
            name = choices[0]
            print(f"{profiles.HARNESSES[args.name]} has no profile yet: using {name!r} "
                  f"(others: {', '.join(choices[1:]) or 'none'}; `lab use <profile>` changes it)")
        _apply(state, args.name, name)
        return
    rows = [("", "HARNESS", "PROFILE", "RUN")]
    for harness in profiles.HARNESSES:
        name = state["profiles"].get(harness)
        rows.append(("*" if harness == state["harness"] else "", harness, name or "-",
                     harness if name else f"(not set up: `lab harness {harness}`)"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    print("\n* = active: `lab use`, `lab profiles`, `lab metrics` and `lab prompts` work on it. "
          "Both stay set up and logged.")


def cmd_use(args) -> None:
    state = cloud.load_state()
    _apply(state, state["harness"], args.profile)


def cmd_profiles(args) -> None:
    state = cloud.load_state()
    harness = state["harness"]
    print(f"{profiles.HARNESSES[harness]} profiles (`lab harness` switches harness):")
    for name in profiles.names(harness):
        print(("* " if name == state["profiles"].get(harness) else "  ") + name)


def cmd_sync(args) -> None:
    state = cloud.load_state()
    name = state["profiles"].get(state["harness"])
    if not name:
        raise ValueError(f"no {state['harness']} profile yet; run `lab use <profile>`")
    _apply(state, state["harness"], name)


def cmd_metrics(args) -> None:
    since = metrics.since_cutoff(registry.parse_duration(args.since))
    harnesses = _harnesses(args.harness)
    _refresh(harnesses, args.offline)
    summaries = metrics.summarize(since, args.model, harnesses)
    print(json.dumps(summaries, indent=2) if args.json else metrics.render(summaries, since))


def cmd_prompts(args) -> None:
    harnesses = _harnesses(args.harness)
    _refresh(harnesses, args.offline)
    prompts = _prompts(harnesses, metrics.since_cutoff(registry.parse_duration(args.since)))
    if args.model:
        prompts = [p for p in prompts if p.model == args.model]
    if args.rating:
        wanted = {None if v == "none" else v if v == "skip" else int(v)
                  for v in (x.strip() for x in args.rating.split(",")) if v}
        prompts = [p for p in prompts if traces.rating_value(p) in wanted]
    key = {"time": lambda p: p.start, "tokens": lambda p: p.tokens_in + p.tokens_out,
           "duration": lambda p: p.duration_s or 0}[args.sort]
    prompts = sorted(prompts, key=key, reverse=True)[: args.limit]
    if not prompts:
        print(f"No {' or '.join(profiles.HARNESSES[h] for h in harnesses)} prompts since {args.since} ago match.")
        return
    print(traces.render_list(prompts, shutil.get_terminal_size().columns))


def cmd_show(args) -> None:
    harnesses = tuple(profiles.HARNESSES)
    _refresh(harnesses, args.offline)
    since = datetime.fromtimestamp(0, timezone.utc)
    requests = [r for rows in traces.server_records(since).values() for r in rows if "event" not in r]
    prompts, note = traces.find(args.id, _prompts(harnesses, since), requests)
    print(traces.render_show(prompts, note, args.full))


def _harnesses(choice: str | None) -> tuple[str, ...]:
    if choice == "all":
        return tuple(profiles.HARNESSES)
    return (choice or cloud.load_state()["harness"],)


def _refresh(harnesses: tuple[str, ...], offline: bool) -> None:
    """Bring the local copies up to date: OpenCode's server lines from Modal, Gemini CLI's telemetry."""
    if "opencode" in harnesses and not offline:
        traces.download()
    if "gemini" in harnesses:
        gemini.extract()


def _prompts(harnesses: tuple[str, ...], since: datetime) -> list[traces.Prompt]:
    prompts = []
    if "opencode" in harnesses:
        prompts += traces.load(since)
    if "gemini" in harnesses:
        prompts += gemini.load(since)
    return sorted(prompts, key=lambda p: p.start)


def cmd_logs(args) -> None:
    cloud.modal_cli("app", "logs", registry.get(args.name).app_name, *(["-f"] if args.follow else []))


def _apply(state: dict, harness: str, name: str) -> None:
    """Write a harness's profile into its config and remember it as that harness's profile."""
    if harness == "opencode":
        profiles.load("opencode", name)  # fail before changing the state
        state["profiles"]["opencode"] = name
        _resync(state, required=True)
        print("Run `opencode` (restart it if it's open).")
        return
    profile = profiles.load("gemini", name)
    backup, state["gemini_managed"] = gemini.apply(name, profile, state["gemini_managed"])
    state["profiles"]["gemini"] = name
    cloud.save_state(state)
    print(f"Gemini CLI profile {name!r}: {profile['model']}. Wrote {gemini.SETTINGS_FILE}"
          + (f" (backup: {backup.relative_to(REPO)})" if backup else "") + f" and {gemini.RATE_COMMAND}")
    print(f"Ratings: `/rate 1-4` in Gemini CLI; pop-up after each answer {'on' if profile.get('feedback') else 'off'}"
          " (profile key `feedback`)")
    print(f"Chats: copied to {gemini.CHATS_DIR.relative_to(REPO)}/ after each answer and when a session ends")
    if profile.get("auth", "api-key") == "api-key" and not os.environ.get("GEMINI_API_KEY"):
        print("warning: GEMINI_API_KEY isn't set in this shell; add it to your shell profile (Google AI Studio key)")
    if not shutil.which("gemini"):
        print("warning: `gemini` isn't on PATH; install it with `npm install -g @google/gemini-cli@0.61.0`")
    print("Run `gemini` (restart it if it's open).")


def _resync(state: dict, required: bool = False) -> None:
    """Render OpenCode's profile to build/opencode.json and copy it into the global OpenCode config.

    Runs after deploys and stops whatever the active harness is: OpenCode's config lists the deployed models.
    """
    name = state["profiles"].get("opencode")
    if not name:
        if required:
            raise ValueError("no OpenCode profile; run `lab harness opencode`, then `lab use <profile>`")
        cloud.save_state(state)
        print("OpenCode not updated: no profile yet (`lab harness opencode`, then `lab use <profile>`)")
        return
    profile = profiles.load("opencode", name)
    config, warnings = profiles.render(profile, registry.load(), state["urls"])
    profiles.write_build(config)
    backup, state["managed_keys"] = profiles.apply_global(config, state.get("managed_keys", []))
    cloud.save_state(state)
    for warning in warnings:
        print(f"warning: {warning}")
    listed = ", ".join(f"lab/{m}" for m in config["provider"]["lab"]["models"]) or "no models (nothing deployed)"
    print(f"OpenCode profile {name!r}: {listed}. Wrote {profiles.BUILD_FILE.relative_to(REPO)} and "
          f"{profiles.GLOBAL_CONFIG}" + (f" (backup: {backup.relative_to(REPO)})" if backup else ""))
    print(f"Chats: every session is copied to {profiles.CHATS_DIR.relative_to(REPO)}/ after each answer "
          f"(plugin {profiles.ARCHIVE_PLUGIN.relative_to(REPO)})")
    tui_backup = profiles.apply_tui(name, profile)
    print(f"Rating prompts: {'on' if profile.get('feedback') else 'off'} by default (toggle in OpenCode with /feedback); "
          f"plugin set in {profiles.GLOBAL_TUI_CONFIG}" + (f" (backup: {tui_backup.relative_to(REPO)})" if tui_backup else ""))


def _confirm(question: str, yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError(f"{question} — pass --yes to confirm non-interactively")
    if input(f"{question} [y/N] ").strip().lower() not in ("y", "yes"):
        raise RuntimeError("cancelled")


def _human(seconds: int) -> str:
    hours, minutes = divmod(seconds // 60, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


COMMANDS = {
    "init": cmd_init, "ls": cmd_ls, "pull": cmd_pull, "deploy": cmd_deploy, "stop": cmd_stop,
    "warm": cmd_warm, "harness": cmd_harness, "use": cmd_use, "profiles": cmd_profiles, "sync": cmd_sync,
    "logs": cmd_logs,
    "metrics": cmd_metrics, "prompts": cmd_prompts, "show": cmd_show,
}

if __name__ == "__main__":
    main()
