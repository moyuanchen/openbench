"""Load models.toml: apply [defaults] and `extends`, validate, and derive Modal settings."""

import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_FILE = REPO / "models.toml"

BACKENDS = {"vllm", "ollama"}


@dataclass(frozen=True)
class Spec:
    name: str
    source: str
    backend: str = "vllm"
    display_name: str = ""
    revision: str | None = None
    exclude: list[str] = field(default_factory=list)
    gpu: str = "L40S"
    gpus: int = 1
    allow_upgrade: bool = False
    cpu: float | None = None
    memory: int | None = None
    region: str | None = None
    max_concurrent: int = 4
    min_containers: int = 0
    max_containers: int = 1
    buffer_containers: int | None = None
    scaledown_window: int = 1200
    context: int = 65536
    output: int = 16384
    tool_parser: str | None = None
    extra_args: list[str] = field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    reasoning_effort: str | None = None  # models with a thinking switch, e.g. "none" or "high"

    @property
    def app_name(self) -> str:
        return "lab-" + re.sub(r"[^a-z0-9-]+", "-", self.name.lower()).strip("-")

    @property
    def gpu_type(self) -> str:
        # Modal may run "H100" on an H200 at the same price; "!" forbids that, which benchmarks need.
        if self.gpu.upper() == "H100" and not self.allow_upgrade:
            return "H100!"
        return self.gpu

    @property
    def modal_gpu(self) -> str:
        return f"{self.gpu_type}:{self.gpus}" if self.gpus > 1 else self.gpu_type

    @property
    def sampling_defaults(self) -> dict:
        values = {"temperature": self.temperature, "top_p": self.top_p, "presence_penalty": self.presence_penalty,
                  "reasoning_effort": self.reasoning_effort}
        return {k: v for k, v in values.items() if v is not None}

    def to_dict(self) -> dict:
        return asdict(self)


def load(path: Path = MODELS_FILE) -> dict[str, Spec]:
    data = tomllib.loads(path.read_text())
    unknown = set(data) - {"defaults", "models"}
    if unknown:
        raise ValueError(f"{path.name}: unknown top-level table(s) {sorted(unknown)}")
    defaults = data.get("defaults", {})
    entries = data.get("models", {})
    return {name: _resolve(name, entries, defaults, ()) for name in entries}


def get(name: str) -> Spec:
    specs = load()
    if name not in specs:
        raise KeyError(f"no model {name!r} in models.toml (have: {', '.join(specs) or 'none'})")
    return specs[name]


def _resolve(name: str, entries: dict, defaults: dict, seen: tuple) -> Spec:
    if name in seen:
        raise ValueError(f"models.toml: `extends` loop: {' -> '.join(seen + (name,))}")
    entry = dict(entries[name])
    base = {}
    if "extends" in entry:
        parent = entry.pop("extends")
        if parent not in entries:
            raise ValueError(f"models.toml: {name} extends unknown model {parent!r}")
        base = {k: v for k, v in _resolve(parent, entries, defaults, seen + (name,)).to_dict().items()
                if k not in ("name", "display_name")}
    merged = {**defaults, **base, **entry}
    if "name" in merged:  # in TOML, `name` is the display name; the table key is the id
        merged["display_name"] = merged.pop("name")
    return _build(name, merged)


def _build(name: str, raw: dict) -> Spec:
    fields = set(Spec.__dataclass_fields__) - {"name"}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"models.toml: {name}: unknown setting(s) {sorted(unknown)}")
    if "source" not in raw:
        raise ValueError(f"models.toml: {name}: `source` is required")
    try:
        raw["scaledown_window"] = parse_duration(raw.get("scaledown_window", 1200))
    except ValueError as exc:
        raise ValueError(f"models.toml: {name}: {exc}") from None
    spec = Spec(name=name, **raw)
    if spec.backend not in BACKENDS:
        raise ValueError(f"models.toml: {name}: backend {spec.backend!r} not supported yet (have: {sorted(BACKENDS)})")
    if not 2 <= spec.scaledown_window <= 1200:
        raise ValueError(f"models.toml: {name}: scaledown_window must be between 2s and 20m (Modal's limits)")
    if spec.min_containers > spec.max_containers:
        raise ValueError(f"models.toml: {name}: min_containers > max_containers")
    if len(spec.app_name) > 50:
        raise ValueError(f"models.toml: {name}: name too long for a Modal app/URL")
    return spec


def parse_duration(value) -> int:
    """Seconds from 90, "90s", "20m", "2h" or "7d"."""
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"\s*(\d+)\s*([smhd]?)\s*", str(value))
    if not match:
        raise ValueError(f"bad duration {value!r} (use e.g. 90, \"90s\", \"20m\", \"2h\")")
    return int(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
