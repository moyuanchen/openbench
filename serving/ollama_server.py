"""Ollama backend (GGUF models, e.g. the abliterated Qwens): image, volume, pull, and starting the daemon.

Carried over from the old qwen_modal.py, including its hard-won fixes (blob preload, pull retries).
Each lab app serves exactly one Ollama model, so the client-facing name is rewritten to the Ollama tag.
"""

import json
import os
import subprocess
import time
import urllib.request

import modal

VERSION = "0.34.4"
PORT = 11434
MODEL_DIR = "/root/.ollama"

models_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)  # shared with the old app
SERVE_VOLUMES = {MODEL_DIR: models_volume}
PULL_VOLUMES = SERVE_VOLUMES

BASE_ENV = {
    "OLLAMA_HOST": f"127.0.0.1:{PORT}",
    "OLLAMA_MODELS": f"{MODEL_DIR}/models",
    "OLLAMA_KEEP_ALIVE": "-1",  # stay resident for the container's life; Modal's scaledown controls cost
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_NOPRUNE": "1",  # don't garbage-collect blobs cached in the Volume
}


def image(spec_env: dict) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("curl", "zstd")  # the installer needs zstd to extract
        .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={VERSION} sh")
        .uv_pip_install("fastapi[standard]==0.115.6", "httpx==0.28.1")
        .env(BASE_ENV)
        .env(spec_env)
        .add_local_python_source("serving")
    )


def upstream_model(spec: dict) -> str:
    return spec["source"]


def pull(spec: dict, attempts: int = 10) -> None:
    """`ollama pull` resumes from cached blobs, so retry (ollama.com often 524s mid-pull) and commit in between."""
    _start_daemon(spec)
    for attempt in range(1, attempts + 1):
        result = subprocess.run(["ollama", "pull", spec["source"]])
        models_volume.commit()
        if result.returncode == 0:
            break
        print(f"attempt {attempt}/{attempts} failed (exit {result.returncode}); retrying...")
        time.sleep(5)
    else:
        raise RuntimeError(f"failed to pull {spec['source']} after {attempts} attempts")
    subprocess.run(["ollama", "list"], check=True)


def start(spec: dict) -> str:
    _preload_blobs(spec["source"])
    url = _start_daemon(spec)
    print(f"loading {spec['source']} into VRAM...", flush=True)
    subprocess.run(["ollama", "run", spec["source"], "--keepalive", "-1m", "hi"], check=False, capture_output=True)
    print("model resident, ready to serve", flush=True)
    return url


def _start_daemon(spec: dict, wait: float = 300) -> str:
    env = {
        **os.environ,
        "OLLAMA_CONTEXT_LENGTH": str(spec["context"]),
        "OLLAMA_NUM_PARALLEL": str(spec["max_concurrent"]),
    }
    subprocess.Popen(["ollama", "serve"], env=env)
    url = f"http://127.0.0.1:{PORT}"
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/version", timeout=2) as response:
                if response.status == 200:
                    return url
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"ollama daemon did not start within {wait:.0f}s")


def _preload_blobs(model: str) -> None:
    """Read the model's blobs sequentially before ollama mmaps them.

    Against a cold Volume, ollama's mmap demand-pages the weights in random order at a few hundred MB/s —
    a multi-minute load that can outlive the client's header timeout. One sequential sweep warms the
    Volume's local cache at full speed first.
    """
    name, _, tag = model.partition(":")
    manifest = os.path.join(MODEL_DIR, "models", "manifests", "registry.ollama.ai", name, tag or "latest")
    try:
        with open(manifest) as f:
            layers = json.load(f).get("layers", [])
        t0, total = time.monotonic(), 0
        for layer in layers:
            path = os.path.join(MODEL_DIR, "models", "blobs", layer.get("digest", "").replace(":", "-"))
            if os.path.exists(path):
                with open(path, "rb", buffering=0) as blob:
                    while chunk := blob.read(64 * 1024 * 1024):
                        total += len(chunk)
        print(f"preloaded {total / 1e9:.1f}GB of model blobs in {time.monotonic() - t0:.0f}s", flush=True)
    except Exception as exc:
        print(f"blob preload skipped ({exc}); model load will demand-page instead", flush=True)
