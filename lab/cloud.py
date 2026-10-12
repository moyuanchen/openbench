"""Talking to Modal (CLI subprocesses and the Python client), plus the local state file .lab/state.json."""

import json
import os
import subprocess
import sys
from pathlib import Path

from lab.registry import REPO, Spec

STATE_FILE = REPO / ".lab" / "state.json"

# USD per GPU-second, from modal.com/pricing on 2026-10-10. CPU and memory are billed on top (small).
GPU_PRICE = {
    "T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542,
    "A100": 0.000583, "A100-40GB": 0.000583, "A100-80GB": 0.000694,
    "H100": 0.001097, "H100!": 0.001097, "H200": 0.001261, "B200": 0.001736,
}


def hourly_cost(spec: Spec, containers: int = 1) -> float | None:
    price = GPU_PRICE.get(spec.gpu_type.upper()) or GPU_PRICE.get(spec.gpu_type)
    return None if price is None else price * spec.gpus * containers * 3600


def modal_cli(*args: str, model: str | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    """Run the `modal` CLI from this venv, in the repo root, optionally for one models.toml entry."""
    env = {**os.environ, **({"LAB_MODEL": model} if model else {})}
    env.pop("LAB_SPEC", None)
    return subprocess.run(
        [sys.executable, "-m", "modal", *args], cwd=REPO, env=env, text=True,
        capture_output=capture, check=False,
    )


def deployed_apps() -> dict[str, dict]:
    """App name -> latest record from `modal app list` (state, running containers)."""
    result = modal_cli("app", "list", "--json", capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"`modal app list` failed: {result.stderr.strip()}")
    apps: dict[str, dict] = {}
    for record in sorted(json.loads(result.stdout), key=lambda r: r.get("created_at") or ""):
        apps[record["description"]] = record
    return apps


def web_url(spec: Spec) -> str:
    import modal

    return modal.Function.from_name(spec.app_name, "serve").get_web_url()


def set_min_containers(spec: Spec, count: int) -> None:
    import modal

    modal.Function.from_name(spec.app_name, "serve").update_autoscaler(min_containers=count)


def apply_autoscaler(spec: Spec, min_containers: int | None = None) -> None:
    """Set the deployed app's scaling to models.toml's values (optionally overriding min_containers).

    Needed after `modal deploy` because a redeploy with unchanged code keeps any runtime overrides
    (a `lab warm` period, a shortened idle window), while a changed one resets them. Tested 2026-10-10.
    """
    import modal

    settings = {
        "min_containers": spec.min_containers if min_containers is None else min_containers,
        "max_containers": spec.max_containers,
        "scaledown_window": spec.scaledown_window,
    }
    if spec.buffer_containers is not None:
        settings["buffer_containers"] = spec.buffer_containers
    modal.Function.from_name(spec.app_name, "serve").update_autoscaler(**settings)


def spawn_keeper(spec: Spec, seconds: int) -> str:
    import modal

    return modal.Function.from_name(spec.app_name, "keeper").spawn(seconds).object_id


def cancel_call(call_id: str) -> None:
    import modal

    modal.FunctionCall.from_id(call_id).cancel()


def load_state() -> dict:
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"urls": {}, "warm": {}}
    # `harness` is the one `lab use` and the views work on; `profiles` holds each harness's current profile.
    # Before harnesses, the OpenCode profile was `active_profile`.
    state.setdefault("profiles", {"opencode": state.pop("active_profile", None)})
    state.setdefault("harness", "opencode")
    state.setdefault("managed_keys", [])  # top-level OpenCode config keys the profile set
    state.setdefault("gemini_managed", [])  # Gemini CLI settings paths the profile's [gemini] table set
    return state


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def private_write(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
