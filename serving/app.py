"""One Modal app per models.toml entry, named `lab-<name>`.

Don't deploy this file by hand: `lab pull/deploy <name>` set LAB_MODEL so it knows which entry to serve.
The resolved entry is baked into the image as LAB_SPEC (base64 JSON), because models.toml is not uploaded to Modal.

Functions:
  serve   GPU container: the model server on 127.0.0.1 behind an authenticating, metrics-recording proxy,
          at https://<workspace>--lab-<name>-serve.modal.run
  pull    CPU-only: download the weights into the backend's Volume (no GPU charge)
  keeper  CPU-only timer behind `lab warm --for`: restores the normal autoscaler settings when the time is up
"""

import base64
import json
import os
import time

import modal

from serving import metrics, ollama_server, proxy, vllm_server

MINUTES = 60
METRICS_DIR = "/metrics"
BACKENDS = {"vllm": vllm_server, "ollama": ollama_server}


def _load_spec() -> dict:
    if "LAB_SPEC" in os.environ:  # inside a container
        return json.loads(base64.b64decode(os.environ["LAB_SPEC"]))
    from lab import registry

    name = os.environ.get("LAB_MODEL")
    if not name:
        raise RuntimeError("LAB_MODEL is not set; use `lab pull <name>` / `lab deploy <name>`")
    spec = registry.get(name)
    return {
        **spec.to_dict(),
        "app_name": spec.app_name,
        "modal_gpu": spec.modal_gpu,
        "sampling_defaults": spec.sampling_defaults,
    }


SPEC = _load_spec()
# base64: the image builder's ENV line mangles quotes and backslashes inside JSON (e.g. extra_args).
SPEC_ENV = {"LAB_SPEC": base64.b64encode(json.dumps(SPEC, sort_keys=True).encode()).decode()}
backend = BACKENDS[SPEC["backend"]]

metrics_volume = modal.Volume.from_name("lab-metrics", create_if_missing=True)
api_key = modal.Secret.from_name("lab-api-key", required_keys=["API_KEY"])

server_image = backend.image(SPEC_ENV)
keeper_image = modal.Image.debian_slim(python_version="3.12").env(SPEC_ENV).add_local_python_source("serving")

app = modal.App(SPEC["app_name"])


@app.function(image=server_image, volumes=backend.PULL_VOLUMES, cpu=4, timeout=120 * MINUTES)
def pull():
    backend.pull(SPEC)


serve_options = {key: SPEC[key] for key in ("cpu", "memory", "region", "buffer_containers") if SPEC.get(key) is not None}


@app.function(
    image=server_image,
    gpu=SPEC["modal_gpu"],
    volumes={**backend.SERVE_VOLUMES, METRICS_DIR: metrics_volume},
    secrets=[api_key],
    min_containers=SPEC["min_containers"],
    max_containers=SPEC["max_containers"],
    scaledown_window=SPEC["scaledown_window"],
    timeout=60 * MINUTES,  # per request
    startup_timeout=30 * MINUTES,  # cold start: load weights, compile, capture CUDA graphs
    **serve_options,
)
@modal.concurrent(max_inputs=SPEC["max_concurrent"])
# Modal checks the proxy token (Modal-Key/Modal-Secret headers, from `lab init`) before routing to a container, so
# a request without it can't start a GPU. The proxy's own API key check stays behind it.
@modal.asgi_app(requires_proxy_auth=True)
def serve():
    t0 = time.monotonic()
    recorder = metrics.Recorder(METRICS_DIR, SPEC, metrics_volume.commit)
    upstream = backend.start(SPEC)
    recorder.write({"event": "container_start", "startup_s": round(time.monotonic() - t0, 1)})
    return proxy.build(
        api_key=os.environ["API_KEY"],
        upstream=upstream,
        model=SPEC["name"],
        sampling_defaults=SPEC["sampling_defaults"],
        recorder=recorder,
        upstream_model=backend.upstream_model(SPEC),
    )


@app.function(image=keeper_image, timeout=24 * 60 * MINUTES)
def keeper(seconds: int):
    time.sleep(seconds)
    modal.Function.from_name(SPEC["app_name"], "serve").update_autoscaler(min_containers=SPEC["min_containers"])
    print(f"warm period over: min_containers back to {SPEC['min_containers']}")
