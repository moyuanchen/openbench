"""vLLM backend: image, volumes, weight download, and starting the server inside the serve container."""

import os
import subprocess
import time
import urllib.request

import modal

VERSION = "0.30.0"
PORT = 8000
HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"

hf_cache = modal.Volume.from_name("lab-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("lab-vllm-cache", create_if_missing=True)

# Volumes the serve container needs; pull only needs the weights.
SERVE_VOLUMES = {HF_CACHE_DIR: hf_cache, VLLM_CACHE_DIR: vllm_cache}
PULL_VOLUMES = {HF_CACHE_DIR: hf_cache}


def image(spec_env: dict) -> modal.Image:
    return (
        # torch 2.13 (pinned by vLLM 0.30) is built for CUDA 13.0; the devel image lets flashinfer JIT-compile.
        modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        # vLLM 0.30 says transformers>=5.10.4 but imports PixtralRotaryEmbedding, renamed in transformers 5.17.
        .uv_pip_install(f"vllm=={VERSION}", "transformers==5.16.1")
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        .env(spec_env)
        .add_local_python_source("serving")
    )


def upstream_model(spec: dict) -> None:
    return None  # vLLM serves the model under our name (--served-model-name)


def pull(spec: dict, attempts: int = 5) -> None:
    """Download the weights at the pinned revision. Resumable: committed after every attempt."""
    from huggingface_hub import snapshot_download

    for attempt in range(1, attempts + 1):
        try:
            path = snapshot_download(spec["source"], revision=spec["revision"], ignore_patterns=spec["exclude"] or None)
            break
        except Exception as exc:
            print(f"attempt {attempt}/{attempts} failed: {exc!r}")
            hf_cache.commit()
            time.sleep(10)
    else:
        raise RuntimeError(f"could not download {spec['source']} after {attempts} attempts")
    hf_cache.commit()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files)
    print(f"{spec['source']}@{spec['revision'] or 'main'}: {size / 1e9:.1f} GB in the lab-hf-cache Volume")


def command(spec: dict) -> list[str]:
    cmd = [
        "vllm", "serve", spec["source"],
        "--served-model-name", spec["name"],
        "--host", "127.0.0.1",  # only the proxy is reachable from outside
        "--port", str(PORT),
        "--max-model-len", str(spec["context"]),
        "--max-num-seqs", str(spec["max_concurrent"]),
        "--tensor-parallel-size", str(spec["gpus"]),
        "--enable-prompt-tokens-details",  # report prefix-cache hits (cached_tokens) in usage
    ]
    if spec["revision"]:
        cmd += ["--revision", spec["revision"], "--tokenizer-revision", spec["revision"]]
    if spec["tool_parser"]:  # OpenCode is a tool-calling agent; without these, tool calls come back as plain text
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", spec["tool_parser"]]
    return cmd + list(spec["extra_args"])


def start(spec: dict, timeout: float = 25 * 60) -> str:
    """Launch vLLM and block until /health answers. Returns its base URL."""
    cmd = command(spec)
    print("starting:", " ".join(cmd), flush=True)
    # Offline: weights must already be in the Volume (`lab pull`), so a cold start never re-downloads.
    proc = subprocess.Popen(cmd, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    url = f"http://127.0.0.1:{PORT}"
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {proc.returncode} during startup; the reason is above in "
                f"`lab logs {spec['name']}` (missing weights show up as a file-not-found: `lab pull {spec['name']}`)"
            )
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    print(f"vLLM ready after {time.monotonic() - t0:.0f}s", flush=True)
                    return url
        except OSError:
            pass
        time.sleep(2)
    proc.terminate()
    raise TimeoutError(f"vLLM did not become ready within {timeout:.0f}s")
