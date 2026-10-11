"""
Serve abliterated Qwen models on Modal as an OpenAI-compatible endpoint.

Usage:
    modal run   qwen_modal.py::download      # one-time: pull a model into the Volume
    modal deploy qwen_modal.py               # deploy the endpoint

    # add another model to the Volume (does not disturb what's already there):
    modal run qwen_modal.py::download --model "huihui_ai/Qwen3.6-abliterated:35b-a3b-q4_K"

A container that is already running will NOT see a model pulled after it started
— a mounted Volume is not re-read live. After a download, `modal app stop
qwen3-coder-abliterated --yes && modal deploy qwen_modal.py`, or just wait out
the scaledown_window so the next request gets a fresh container.

One endpoint serves *every* model cached in the Volume: the `model` field of the
request selects it, so switching models is a client-side change (see the two
entries under the `modal` provider in ~/.config/opencode/opencode.json) rather
than a redeploy. Only one is resident at a time — see OLLAMA_MAX_LOADED_MODELS.

The deployed URL exposes /v1/* (OpenAI-compatible) and /api/* (Ollama-native),
guarded by a bearer token stored in the `qwen-api-key` Modal Secret.
"""

import os
import subprocess
import time

import modal

# Which model is preloaded into VRAM at container start. This only decides which
# one is warm after a cold start; the others in the Volume are still servable,
# they just pay a ~1min load. Qwen3.6 publishes no `latest` tag, so pin one.
MODEL = os.environ.get("QWEN_MODEL", "huihui_ai/Qwen3.6-abliterated:35b-a3b-q4_K")
GPU = "L40S"  # 48GB — fits 24GB weights + a large KV cache, ~$1.95/hr

OLLAMA_PORT = 11434
OLLAMA_URL = f"http://127.0.0.1:{OLLAMA_PORT}"

# Weights live in a Volume so containers don't re-download 19GB on every cold start.
model_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)
MODEL_DIR = "/root/.ollama"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd")  # the ollama installer needs zstd to extract
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("fastapi[standard]==0.115.6", "httpx==0.28.1")
    .env(
        {
            "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}",
            "OLLAMA_MODELS": f"{MODEL_DIR}/models",
            # Keep the model resident in VRAM for the container's whole life.
            # Modal's scaledown_window is what actually controls cost.
            "OLLAMA_KEEP_ALIVE": "-1",
            # 19GB + 24GB of weights will not co-reside on a 48GB card alongside
            # a 64K KV cache. Cap residency at one so switching models in the
            # client evicts the incumbent instead of OOMing the GPU.
            "OLLAMA_MAX_LOADED_MODELS": "1",
            # Ollama defaults to a tiny context; a coding agent needs far more.
            "OLLAMA_CONTEXT_LENGTH": "65536",
            "OLLAMA_FLASH_ATTENTION": "1",
            # Don't garbage-collect blobs we've cached in the Volume.
            "OLLAMA_NOPRUNE": "1",
        }
    )
)

app = modal.App("qwen3-coder-abliterated")


def _preload_model_blobs(model: str) -> None:
    """Sequentially read the model's blobs before ollama touches them.

    Against a cold Volume, ollama's mmap load demand-pages the weights in
    random order at a few hundred MB/s of small network reads — a 3-4 minute
    model load, long enough to trip the client's ~300s header timeout on the
    first request. One sequential sweep pulls the blobs into the Volume's
    local cache at full throughput, and the real load then hits warm cache.
    """
    import json

    name, _, tag = model.partition(":")
    manifest = os.path.join(
        MODEL_DIR, "models", "manifests", "registry.ollama.ai", name, tag or "latest"
    )
    try:
        with open(manifest) as f:
            layers = json.load(f).get("layers", [])
        t0 = time.monotonic()
        total = 0
        for layer in layers:
            path = os.path.join(
                MODEL_DIR, "models", "blobs", layer.get("digest", "").replace(":", "-")
            )
            if not os.path.exists(path):
                continue
            with open(path, "rb", buffering=0) as blob:
                while chunk := blob.read(64 * 1024 * 1024):
                    total += len(chunk)
        print(f"preloaded {total / 1e9:.1f}GB of model blobs in {time.monotonic() - t0:.0f}s")
    except Exception as exc:
        print(f"blob preload skipped ({exc}); model load will demand-page instead")


def _start_ollama(wait_seconds: int = 300) -> None:
    """Start the ollama daemon and block until it answers."""
    subprocess.Popen(["ollama", "serve"])

    import httpx

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{OLLAMA_URL}/api/version", timeout=2.0).status_code == 200:
                print("ollama daemon is up")
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"ollama daemon did not start within {wait_seconds}s")


@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=60 * 60,  # a 19GB pull can be slow
)
def download(model: str = MODEL, attempts: int = 10):
    """One-time: pull the model into the Volume. Run with `modal run`.

    ollama.com sits behind Cloudflare and regularly 524s partway through a
    19GB pull. `ollama pull` resumes from cached blobs, so we retry — and
    commit between attempts so partial blobs survive into the next one.
    """
    _start_ollama()
    print(f"pulling {model} (~19GB, this takes a few minutes)...")

    for attempt in range(1, attempts + 1):
        result = subprocess.run(["ollama", "pull", model])
        model_volume.commit()  # persist whatever blobs landed
        if result.returncode == 0:
            break
        print(f"attempt {attempt}/{attempts} failed (exit {result.returncode}); retrying...")
        time.sleep(5)
    else:
        raise RuntimeError(f"failed to pull {model} after {attempts} attempts")

    subprocess.run(["ollama", "list"], check=True)
    print(f"done — {model} cached in the 'ollama-models' Volume")


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("qwen-api-key", required_keys=["API_KEY"])],
    # Container stays warm 20 min after the last request, then scales to zero.
    # 5 min proved too aggressive for interactive use: any pause in a coding
    # session tore the container down, and the next message ate a multi-minute
    # cold start that could outlive the client's header timeout entirely.
    # Idle cost at $1.95/hr is ~$0.65 per wind-down; lower to save credits.
    scaledown_window=1200,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def serve():
    """Authenticated streaming reverse-proxy in front of the Ollama daemon."""
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask

    _preload_model_blobs(MODEL)
    _start_ollama()

    # Load weights into VRAM now so the first real request isn't slow.
    print(f"loading {MODEL} into VRAM...")
    subprocess.run(
        ["ollama", "run", MODEL, "--keepalive", "-1m", "hi"],
        check=False,
        capture_output=True,
    )
    print("model resident, ready to serve")

    API_KEY = os.environ["API_KEY"]
    client = httpx.AsyncClient(
        base_url=OLLAMA_URL,
        timeout=httpx.Timeout(connect=10.0, read=900.0, write=900.0, pool=900.0),
    )

    web_app = FastAPI(title="qwen3-coder-abliterated on Modal")

    # Hop-by-hop and length headers must not be forwarded verbatim.
    DROP_REQUEST = {"host", "content-length", "authorization", "connection"}
    DROP_RESPONSE = {"content-length", "content-encoding", "transfer-encoding", "connection"}

    # Ollama's /v1 compat layer stamps OpenAI defaults (temperature=1.0,
    # top_p=1.0) over Modelfile PARAMETERs whenever the client omits them,
    # while the Modelfile's presence_penalty 1.5 still applies — a hot,
    # unfiltered sampler that makes agent turns derail and end early. Inject
    # sane defaults here; anything the client sends explicitly still wins.
    SAMPLING_DEFAULTS = {
        "temperature": 0.7,
        "top_p": 0.95,
        "presence_penalty": 1.0,
    }

    def with_sampling_defaults(body: bytes) -> bytes:
        import json

        try:
            payload = json.loads(body)
            for key, value in SAMPLING_DEFAULTS.items():
                payload.setdefault(key, value)
            return json.dumps(payload).encode()
        except Exception:
            return body  # not JSON we understand — forward untouched

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL}

    @web_app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    )
    async def proxy(path: str, request: Request):
        if request.method == "OPTIONS":
            return Response(status_code=204)

        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if token != API_KEY:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST}
        body = await request.body()
        if request.method == "POST" and path.endswith("chat/completions"):
            body = with_sampling_defaults(body)
        upstream = client.build_request(
            request.method,
            f"/{path}",
            headers=headers,
            params=request.query_params,
            content=body,
        )
        response = await client.send(upstream, stream=True)

        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers={
                k: v for k, v in response.headers.items() if k.lower() not in DROP_RESPONSE
            },
            background=BackgroundTask(response.aclose),
        )

    return web_app
