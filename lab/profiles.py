"""Profiles: versions of each harness's setup, in profiles/<harness>/<name>.toml.

OpenCode profiles are rendered to build/opencode.json, then copied into the global config (this module).
Gemini CLI profiles are written into ~/.gemini/settings.json (lab/gemini.py).
"""

import json
import shutil
import tomllib
from datetime import datetime
from pathlib import Path

from lab.registry import REPO, Spec

PROFILES_DIR = REPO / "profiles"
BUILD_FILE = REPO / "build" / "opencode.json"
KEY_FILE = REPO / ".secrets" / "api-key"
# Modal proxy token (`lab init`): header name -> file holding its value.
PROXY_TOKEN_FILES = {"Modal-Key": REPO / ".secrets" / "modal-key", "Modal-Secret": REPO / ".secrets" / "modal-secret"}
BACKUP_DIR = REPO / ".lab" / "backups"
GLOBAL_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"
GLOBAL_TUI_CONFIG = Path.home() / ".config" / "opencode" / "tui.json"
FEEDBACK_PLUGIN = REPO / "plugins" / "feedback.ts"
FEEDBACK_LOG = REPO / "metrics" / "feedback.jsonl"
ARCHIVE_PLUGIN = REPO / "plugins" / "archive.ts"
CHATS_DIR = REPO / "metrics" / "opencode" / "chats"  # one JSON file per session, written by plugins/archive.ts

PROVIDER_ID = "lab"  # not "modal": models.dev already has a `modal` provider, and OpenCode merges the two
HARNESSES = {"opencode": "OpenCode", "gemini": "Gemini CLI"}  # id -> name; the id is also the command to run
PROFILE_KEYS = {
    "opencode": {"model", "models", "feedback", "feedback_models", "opencode"},
    "gemini": {"model", "feedback", "auth", "gemini"},
}


def names(harness: str) -> list[str]:
    return sorted(p.stem for p in (PROFILES_DIR / harness).glob("*.toml"))


def load(harness: str, name: str) -> dict:
    path = PROFILES_DIR / harness / f"{name}.toml"
    if not path.exists():
        others = [h for h in HARNESSES if h != harness and (PROFILES_DIR / h / f"{name}.toml").exists()]
        hint = f"; it's in profiles/{others[0]}/: `lab harness {others[0]}` switches" if others else ""
        raise FileNotFoundError(f"no {harness} profile {name!r} (have: {', '.join(names(harness)) or 'none'}){hint}")
    profile = tomllib.loads(path.read_text())
    unknown = set(profile) - PROFILE_KEYS[harness]
    if unknown:
        raise ValueError(f"profiles/{harness}/{name}.toml: unknown key(s) {sorted(unknown)}")
    return profile


def render(profile: dict, specs: dict[str, Spec], urls: dict[str, str]) -> tuple[dict, list[str]]:
    """The OpenCode config for this profile, plus warnings. Only deployed models are listed."""
    warnings = []
    wanted = profile.get("models", list(specs))
    models = {}
    for name in wanted:
        if name not in specs:
            warnings.append(f"profile lists {name!r}, which is not in models.toml")
        elif name not in urls:
            if "models" in profile:
                warnings.append(f"{name} is not deployed yet (`lab deploy {name}`); left out")
        else:
            models[name] = _model_entry(specs[name], urls[name])

    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Lab (Modal)",
                "options": {
                    "apiKey": "{file:" + str(KEY_FILE) + "}",
                    # Modal's proxy auth, checked before a request reaches a container (serving/app.py).
                    "headers": {name: "{file:" + str(path) + "}" for name, path in PROXY_TOKEN_FILES.items()},
                    # Cold starts load the weights before the first byte comes back.
                    "timeout": 900000,
                    "headerTimeout": 900000,
                },
                "models": models,
            }
        },
    }
    default = profile.get("model")
    if default:
        if default.startswith(f"{PROVIDER_ID}/") and default.split("/", 1)[1] not in models:
            warnings.append(f"default model {default} is not deployed; OpenCode's default left unchanged")
        else:
            config["model"] = default
    config.update(profile.get("opencode", {}))
    return config, warnings


def proxy_auth_headers() -> dict[str, str]:
    """The Modal proxy token as request headers, for lab's own requests to the endpoints."""
    for path in PROXY_TOKEN_FILES.values():
        if not path.exists():
            raise FileNotFoundError(f"{path.relative_to(REPO)} is missing: run `lab init`")
    return {name: path.read_text().strip() for name, path in PROXY_TOKEN_FILES.items()}


def write_build(config: dict) -> None:
    BUILD_FILE.parent.mkdir(exist_ok=True)
    BUILD_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


def apply_global(config: dict, previously_managed: list[str]) -> tuple[Path | None, list[str]]:
    """Copy the `lab` provider, the session-copying plugin and this profile's top-level keys into the global config.

    Everything else in that file, other plugins included, is left alone. Keys a previous profile set but this one
    doesn't are removed. Returns the backup path and the keys now managed by lab.
    """
    if GLOBAL_CONFIG.exists():
        try:
            current = json.loads(GLOBAL_CONFIG.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{GLOBAL_CONFIG} is not plain JSON ({exc}); lab can't merge into it safely") from None
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / f"opencode.{datetime.now():%Y%m%d-%H%M%S}.json"
        shutil.copy2(GLOBAL_CONFIG, backup)
    else:
        current, backup = {"$schema": "https://opencode.ai/config.json"}, None

    managed = [k for k in config if k not in ("$schema", "provider")]
    for key in previously_managed:
        if key not in managed:
            current.pop(key, None)
    current.setdefault("provider", {})[PROVIDER_ID] = config["provider"][PROVIDER_ID]
    for key in managed:
        current[key] = config[key]
    ours = str(ARCHIVE_PLUGIN)
    plugins = [p for p in current.get("plugin", []) if (p[0] if isinstance(p, list) else p) != ours]
    current["plugin"] = plugins + [[ours, {"dir": str(CHATS_DIR)}]]

    GLOBAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_CONFIG.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
    return backup, managed


def apply_tui(profile_name: str, profile: dict) -> Path | None:
    """Install the rating plugin in OpenCode's TUI config with this profile's settings. Returns the backup path.

    The plugin is always installed so `/feedback` works; the profile only decides whether it asks by default.
    Other plugins and settings in tui.json are left alone.
    """
    backup = None
    current = {"$schema": "https://opencode.ai/tui.json"}
    if GLOBAL_TUI_CONFIG.exists():
        try:
            current = json.loads(GLOBAL_TUI_CONFIG.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{GLOBAL_TUI_CONFIG} is not plain JSON ({exc}); lab can't merge into it safely") from None
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / f"tui.{datetime.now():%Y%m%d-%H%M%S}.json"
        shutil.copy2(GLOBAL_TUI_CONFIG, backup)

    ours = str(FEEDBACK_PLUGIN)
    plugins = [p for p in current.get("plugin", []) if (p[0] if isinstance(p, list) else p) != ours]
    plugins.append([ours, {
        "enabled": bool(profile.get("feedback", False)),
        "models": profile.get("feedback_models", [f"{PROVIDER_ID}/*"]),
        "log": str(FEEDBACK_LOG),
        "profile": profile_name,
    }])
    current["plugin"] = plugins
    GLOBAL_TUI_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_TUI_CONFIG.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
    return backup


def _model_entry(spec: Spec, url: str) -> dict:
    hardware = f"{spec.gpus}×{spec.gpu}" if spec.gpus > 1 else spec.gpu
    return {
        "name": f"{spec.display_name or spec.name} · {spec.backend} · {hardware}",
        # Per-model URL: with no provider-level baseURL, OpenCode uses this (provider.ts, model.api.url).
        "provider": {"api": url.rstrip("/") + "/v1"},
        "tool_call": True,
        "temperature": True,
        "attachment": False,
        "limit": {"context": spec.context, "output": spec.output},
    }
