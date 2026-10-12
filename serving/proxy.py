"""Authenticating, streaming reverse proxy in front of the model server (one per container).

Besides forwarding, it fills in the model's sampling defaults and records one metrics line per completion
request (see serving/metrics.py). The response bytes are passed through unchanged.

OpenCode sends its session id with every request (packages/opencode/src/session/llm/request.ts), plus the parent
session's for subagents. They are recorded so each line points to its conversation (`lab show`).
"""

import contextlib
import hmac
import json
import time
import uuid

# Hop-by-hop and length headers must not be forwarded verbatim; credentials stop here.
DROP_REQUEST = {"host", "content-length", "authorization", "connection", "modal-key", "modal-secret"}
DROP_RESPONSE = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def build(*, api_key: str, upstream: str, model: str, sampling_defaults: dict, recorder=None, upstream_model=None):
    """`upstream_model`: the name the model server knows the model by, if it differs from ours (Ollama tags)."""
    # Imported here: these exist in the serving image, not in the keeper's.
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask

    from serving.metrics import StreamStats

    client = httpx.AsyncClient(
        base_url=upstream, timeout=httpx.Timeout(connect=10.0, read=900.0, write=900.0, pool=900.0)
    )
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        if recorder is not None:  # container shutting down: save the last metrics lines
            recorder.flush()

    web_app = FastAPI(title=f"lab: {model}", lifespan=lifespan)

    def rewrite(body: bytes) -> tuple[bytes, dict]:
        # The model's sampling settings from models.toml, used when the client sends none.
        # Anything the client sends explicitly still wins.
        try:
            payload = json.loads(body)
        except ValueError:
            return body, {}
        if not isinstance(payload, dict):
            return body, {}
        for key, value in sampling_defaults.items():
            payload.setdefault(key, value)
        if upstream_model and payload.get("model") == model:
            payload["model"] = upstream_model
        return json.dumps(payload).encode(), payload

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": model}

    @web_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def forward(path: str, request: Request):
        if request.method == "OPTIONS":
            return Response(status_code=204)

        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token.encode(), api_key.encode()):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        t_received = time.monotonic()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST}
        body = await request.body()
        payload: dict = {}
        is_completion = request.method == "POST" and path.endswith("completions")
        if is_completion:
            body, payload = rewrite(body)
        upstream_request = client.build_request(
            request.method, f"/{path}", headers=headers, params=request.query_params, content=body
        )
        response = await client.send(upstream_request, stream=True)
        response_headers = {k: v for k, v in response.headers.items() if k.lower() not in DROP_RESPONSE}

        if not (is_completion and recorder is not None):
            return StreamingResponse(
                response.aiter_raw(), status_code=response.status_code,
                headers=response_headers, background=BackgroundTask(response.aclose),
            )

        request_id = uuid.uuid4().hex[:16]
        response_headers["x-lab-request-id"] = request_id
        session = {
            "session_id": request.headers.get("x-opencode-session-id") or request.headers.get("x-session-id"),
            "parent_session_id": request.headers.get("x-parent-session-id"),
        }
        stats = StreamStats(streaming=bool(payload.get("stream")))
        first_request = recorder.served == 0
        recorder.served += 1
        recorder.in_flight += 1
        in_flight = recorder.in_flight

        async def relay():
            outcome = "ok"
            try:
                async for chunk in response.aiter_raw():
                    stats.feed(chunk)
                    yield chunk
            except BaseException:  # client went away (e.g. Esc in OpenCode) or upstream broke
                outcome = "aborted"
                raise
            finally:
                recorder.in_flight -= 1
                await response.aclose()
                stats.finish()
                recorder.write(_record(request_id, session, path, payload, response.status_code, outcome,
                                       first_request, in_flight, t_received, stats))

        return StreamingResponse(relay(), status_code=response.status_code, headers=response_headers)

    return web_app


def _record(request_id, session, path, payload, status, outcome, first_request, in_flight, t_received, stats) -> dict:
    t_done = time.monotonic() - t_received
    usage = stats.usage or {}
    completion = usage.get("completion_tokens")
    t_first = None if stats.first_byte is None else stats.first_byte - t_received
    decode = None
    if completion and completion > 1 and t_first is not None and t_done > t_first:
        decode = round((completion - 1) / (t_done - t_first), 2)
    return {
        "request_id": request_id,
        **session,
        "path": "/" + path,
        "stream": bool(payload.get("stream")),
        "status": status,
        "outcome": outcome,
        "finish_reason": stats.finish_reason,
        "first_request_in_container": first_request,
        "in_flight": in_flight,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": completion,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "max_tokens": payload.get("max_tokens") or payload.get("max_completion_tokens"),
        "temperature": payload.get("temperature"),
        "tools": len(payload.get("tools") or []),
        "t_first_token": None if t_first is None else round(t_first, 3),
        "t_done": round(t_done, 3),
        "decode_tok_s": decode,
    }
