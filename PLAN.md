# Plan: `lab`, a model-serving pipeline for OpenCode on Modal

Status (2026-10-10): **steps 1–2 done.** Step 3 dropped: the Qwen models were removed from `lab` at your request. The Ollama backend code stays in `serving/ollama_server.py`, unused and untested on a GPU, and the old `qwen3-coder-abliterated` app is untouched. Step 4: `git init` done, no commits yet (your call). Hardware presets: not wanted, profiles are enough. Added after step 2: `lab prompts` / `lab show` to find a conversation from its numbers (see "Finding a conversation"), and rating prompts on by default. 2026-10-11: profiles are `ministral`, `devstral` and `mistral` (one per model, replacing `default`); Devstral Small 2 (L40S) and Mistral Medium 3.5 (B200) added to `models.toml` and pulled, not deployed yet.

## Step 2 results (2026-10-10)

- **Server metrics:** the proxy writes one line per request to the `lab-metrics` volume, with prompt, cached and output tokens, time to first token, total time, output speed and finish reason. Tested end to end in the real image on a CPU container, then live.
- **Cached tokens:** vLLM now runs with `--enable-prompt-tokens-details`. In an OpenCode run, step 2 had 10,768 of 10,996 input tokens served from the prefix cache, so only 228 new tokens were processed.
- **`lab metrics`:** joins server lines to OpenCode's own step records by start time **and** token count (OpenCode input + cached = server prompt tokens). Measured network-plus-routing overhead: **~0.26 s** per step.
- **Side requests:** OpenCode sends a small request without tools (~550 tokens) at the start of each new session to generate its title. These are counted separately so they don't skew the agent-step numbers.
- **Rating plugin:** `plugins/feedback.ts`, installed in `~/.config/opencode/tui.json` by `lab use`. Tested in a real OpenCode window: `/feedback` on, a prompt with a tool call, the dialog, rating 3 saved, Esc saved as skipped, `/feedback` off means no dialog.
- **Fix:** FastAPI no longer has `add_event_handler`, so the shutdown flush uses `lifespan`.

## Step 1 results (2026-10-10)

What's live: `lab-ministral-14b` on 1×L40S (vLLM 0.30.0), listed in OpenCode as `lab/ministral-14b` through the `default` profile.

Measured on the first cold start:

| | Value |
|---|---|
| Weights in GPU memory | 25.23 GiB (the image encoder is skipped because of `image: 0`) |
| Memory left for conversations | 14.17 GiB, which holds **92,864 tokens**: 1.42 requests at the full 65k context, or ~2.5 OpenCode sessions of ~37k |
| Cold start | 215 s total. vLLM takes 208 s of it, including 128 s of engine warmup (35 s of that is compilation; the compiled kernels are now cached in `lab-vllm-cache`). |
| Output speed | ~27 tokens/s for one request, measured including the network (the L40S maximum is ~31) |
| Tool calls | Work, using vLLM's `mistral` parser |
| End-to-end OpenCode run | 2 steps (`read` the file, then answer), 5.6 s, ~11k tokens of input per step |

Things learned while building:
- **`transformers` has to be pinned to 5.16.1.** vLLM 0.30 allows any `transformers` ≥ 5.10.4, but it imports `PixtralRotaryEmbedding`, which 5.17 renamed.
- **The image builder's `ENV` mangles JSON.** It strips backslashes, which broke the `{"image": 0}` argument, so the baked settings are stored base64-encoded (`LAB_SPEC`).
- **A container that crashes on startup keeps getting restarted, on a GPU.** Modal retries it as long as a request is waiting. If `lab logs` shows a crash loop, `modal app stop lab-<name>` stops it.
- **Ministral can garble long paths.** It copied a long scratch path with a typo. That's a model weakness, worth watching in the ratings.
- **Cached tokens aren't reported.** OpenCode shows `cache.read = 0` because vLLM doesn't report prefix-cache hits by default. Step 2 should add `--enable-prompt-tokens-details` so the metrics include cached tokens.


## Checks (2026-10-10)

| Check | Result |
|---|---|
| Scale to zero | ✅ Ministral went to 0 running containers by itself after the default 20 idle minutes. |
| Parallelism (`max_concurrent = 4`, 1 container) | ✅ 1 request: 23.9 tok/s. 4 at once: ~22 tok/s each, **86.6 tok/s total** (the server saw `in_flight` 1→4 at ~27 tok/s each). 8 at once: 4 ran together and the other 4 queued at Modal until the first batch finished. |
| A timed warm running out by itself | ✅ `lab warm --for 1m`: the timer fired on time ("warm period over: min_containers back to 0"), and with a 60 s idle window set at runtime the container was gone about 1 min later. |
| Redeploy during a warm period | Tested on a throwaway CPU-only app. A redeploy with **unchanged** code keeps the running container and any runtime overrides (a warm period, a shortened idle window). A redeploy with **changed** settings (e.g. another GPU) shuts the old container down at once and resets the overrides to the code's values. `--strategy recreate` made no difference for unchanged code. **Fix:** `lab deploy` now re-applies the `models.toml` scaling settings after every deploy, and keeps an active `lab warm --for` period going. |

**Switching GPUs therefore needs only:** edit `gpu` in `models.toml`, then `lab deploy <name>`. The old container is replaced right away.

## Why

Right now the setup lives in three places and only fits one case:

- `qwen_modal.py` hard-codes one backend (Ollama), one GPU (`L40S`), one app, and Qwen-specific defaults.
- The OpenCode side is a hand-edited `modal` provider block in `~/.config/opencode/opencode.json`, with the API key pasted inline.
- Nothing in the repo says which models exist, how they are served, or how to add one.

The goal is that **this repo becomes the single place where models are declared, served, and wired into OpenCode**. You edit files here and run `lab` commands. You keep using OpenCode exactly as now (`opencode`, then `/models`). Token usage and response time are recorded automatically on both ends, since we run all of the compute ourselves.

## Day to day

```
lab pull ministral-14b          # fetch weights into the Modal volume (CPU only, cheap)
lab deploy ministral-14b        # start serving it on Modal
lab use ministral               # switch OpenCode to the "ministral" profile (also: devstral, mistral)
opencode                        # pick lab/ministral-14b in /models
lab warm ministral-14b --for 2h # keep it running for a long session
lab metrics --since 7d          # token usage, response times, your 1–4 ratings
lab prompts --rating 1,2        # find prompts by time, tokens or rating…
lab show <ID>                   # …and read one: the conversation with its numbers
```

## Files in the repo

```
models.toml              # what can be served: each entry = a model + the hardware it runs on
profiles/*.toml          # versions of the OpenCode setup, switched with `lab use`
build/opencode.json      # generated from the active profile (in repo, readable)
pyproject.toml           # project Python environment, modal pinned, managed with uv
lab/                     # the command (Python, standard-library argparse)
  cli.py                 # subcommands
  registry.py            # load + validate models.toml
  profiles.py            # render profiles, merge into the global OpenCode config
  cloud.py               # Modal CLI/client calls, GPU prices, .lab/state.json
  traces.py              # reads OpenCode's database, server lines and ratings, and joins them per prompt
  metrics.py             # `lab metrics` summary
plugins/feedback.ts      # step 2: OpenCode plugin that asks for a 1–4 rating after each prompt
serving/                 # everything that runs on Modal
  app.py                 # ONE app definition (functions serve, pull, keeper), configured from the chosen entry
  proxy.py               # API key check + sampling defaults (+ per-request logging in step 2)
  vllm_server.py         # vLLM launch command + readiness wait
  # step 3: Ollama backend (qwen_modal.py's logic)
.secrets/api-key         # gitignored; read by OpenCode through {file:…}
.lab/                    # gitignored: active profile name, backups of the global OpenCode config
ref/opencode/            # OpenCode source, reference only
```

## `models.toml`: what can be served

Each entry is one **deployment**: a set of weights plus the hardware to run them on. It becomes one Modal app, `lab-<name>`, and one OpenCode model, `lab/<name>`. Settings left out come from `[defaults]`. `extends` copies another entry, so you only write what differs. That makes it easy to put the same model on two kinds of hardware and compare them.

```toml
[defaults]
backend          = "vllm"
gpu              = "L40S"
gpus             = 1
max_concurrent   = 4        # requests one container works on at once
min_containers   = 0        # 0 = scale to zero when idle (no cost)
max_containers   = 1
scaledown_window = "20m"    # Modal's maximum
context          = 65536
output           = 16384

# First model: Ministral 3 14B Instruct, full precision (BF16, no quantization)
[models.ministral-14b]
source      = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
revision    = "3cea74c1ebaf5ce5f5a2553de470e2ceab825142"   # pinned, so results stay comparable
exclude     = ["model-*.safetensors", "model.safetensors.index.json"]  # same weights in a 2nd format; skip 28 GB
gpu         = "L40S"
tool_parser = "mistral"
extra_args  = ["--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral"]
temperature = 0.05          # Mistral recommends < 0.1

# Same weights on a faster GPU, for comparison
[models.ministral-14b-h100]
extends        = "ministral-14b"
gpu            = "H100"
context        = 131072
max_concurrent = 8

# Later: today's Qwen models, moved over unchanged (existing volume, no re-download)
[models."qwen3.6-35b"]      # quoted: a bare dot would nest tables
backend = "ollama"
source  = "huihui_ai/Qwen3.6-abliterated:35b-a3b-q4_K"
```

### The two GPUs

| | L40S | H100 (SXM, as Modal provides it) |
|---|---|---|
| Architecture | Ada Lovelace | Hopper |
| Memory | 48 GB GDDR6 (~45 GiB usable) | 80 GB HBM3 (~79.6 GiB usable) |
| Memory bandwidth (limits output speed) | 864 GB/s | 3,350 GB/s (3.9×) |
| BF16 compute, dense (limits prompt processing) | 362 TFLOPS | 989 TFLOPS (2.7×) |
| FP8 compute, dense | 733 TFLOPS | 1,979 TFLOPS |
| GPU-to-GPU link (matters for `gpus > 1`) | PCIe Gen4, ~64 GB/s, no NVLink | NVLink, 900 GB/s |
| Power | 350 W | 700 W |
| Modal price | $0.000542/s ≈ **$1.95/h** | $0.001097/s ≈ **$3.95/h** |

The two specs that matter most:
- **Output speed** depends mainly on memory bandwidth: every generated token reads all the weights.
- **Time to first token** on a long prompt depends mainly on compute.

Two Modal details that matter for benchmarking:
- **`H100` can silently run on an H200** (141 GB, 4.8 TB/s) at the same price. `gpu = "H100!"` prevents that, so `lab` adds the `!` by default for H100. Likewise, a plain `A100` request can be upgraded to the 80 GB model, so use `A100-80GB`.
- **L40S has no fast link between GPUs.** Splitting a model across several L40S works, but communication between the GPUs is slow, so a multi-GPU L40S setup gains much less than a multi-GPU H100 one.

Ministral 3 also includes a small image encoder (~0.4B parameters). OpenCode only sends text, so vLLM can be told to skip images (`--limit-mm-per-prompt`) to free some memory.

### Compute settings

| Field | Meaning | Modal setting it controls |
|---|---|---|
| `gpu` | GPU type: `T4`, `L4`, `A10`, `L40S`, `A100-80GB`, `H100`, `H200`, `B200`, or a fallback list like `["H100", "A100-80GB"]`. `H100` is pinned to `H100!` unless `allow_upgrade = true` (see "The two GPUs"). | `gpu="H100!:2"` |
| `gpus` | Number of GPUs in one container | count part of `gpu=` |
| `cpu`, `memory` | Optional CPU cores and RAM | `cpu=`, `memory=` |
| `region` | Optional, to pin where it runs | `region=` |
| `scaledown_window` | How long a container stays up after the last request (Modal allows 2 s – 20 min) | `scaledown_window=` |

### Parallelism: three separate settings

1. **Splitting one model across GPUs** (`gpus`, optionally `tensor_parallel` / `pipeline_parallel`). By default this splits the model into one slice per GPU in the container (vLLM `--tensor-parallel-size`). Use it when a model doesn't fit on one GPU, or to answer faster. Ollama can only split layers across GPUs, which is slower.
2. **Requests at once inside one container** (`max_concurrent`). Requests that arrive together are processed in one batch. This sets both Modal's `@modal.concurrent(max_inputs=…)` and vLLM's `--max-num-seqs` (Ollama: `OLLAMA_NUM_PARALLEL`), so the two limits stay in sync.
3. **Copies of the server** (`min_containers`, `max_containers`, `buffer_containers`). Modal starts more containers when traffic exceeds what one can handle. `min_containers > 0` keeps GPUs running around the clock: one L40S is about $1,400 a month. `lab deploy` asks for confirmation before deploying with that.

## Profiles: versions of the OpenCode setup

A profile is a small file saying what OpenCode should see. You can keep as many as you like and switch between them with one command.

```toml
# profiles/default.toml
model  = "lab/ministral-14b"                       # OpenCode's default model
models = ["ministral-14b", "ministral-14b-h100"]   # which entries OpenCode lists (omit = all deployed)
feedback = true                                    # ask for a 1–4 rating after each prompt (see "Rating responses")

# [sampling."ministral-14b"]                       # NOT BUILT: per-profile sampling. For now sampling is set per
# temperature = 0.05                               # model in models.toml and applied by the proxy as the default.

[opencode]                                         # copied as-is into the OpenCode config,
# small_model = "lab/ministral-14b"               # for any other OpenCode setting
```

How switching works:

- **`lab use <profile>`** turns the profile and `models.toml` into `build/opencode.json`, a file in the repo. That file holds one provider, `lab`, plus the default `model`.
- **The same command copies those two keys into `~/.config/opencode/opencode.json`.** It first saves a backup to `.lab/backups/`, and nothing else in the global config is changed, so your local `ollama` provider, theme and other settings stay.
- **The repo file is the source of truth.** The global config is just a copy. Copying is used instead of `OPENCODE_CONFIG=…`, because OpenCode started from the desktop app or an IDE doesn't see shell variables.
- **`lab sync`** rebuilds the active profile, for example after a deploy adds a model.
- **`lab profiles`** lists the profiles and marks the active one.
- **Versions you can go back to:** profiles are named versions you can switch between. Once the repo is in git, every edit to them is in the history too.

What the generated provider looks like (everything below was checked against the OpenCode source in `ref/opencode`):

```jsonc
"lab": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "Lab (Modal)",
  "options": { "apiKey": "{file:~/project/opencode/.secrets/api-key}", "timeout": 900000, "headerTimeout": 600000 },
  "models": {
    "ministral-14b": {
      "name": "Ministral 3 14B BF16 (vLLM, L40S)",
      "provider": { "api": "https://moyuanchen--lab-ministral-14b-serve.modal.run/v1" },
      "tool_call": true, "temperature": true,
      "limit": { "context": 65536, "output": 16384 }
    }
  }
}
```

- **One URL per model inside one provider.** When the provider sets no `baseURL`, OpenCode uses each model's own `provider.api` URL (`packages/opencode/src/provider/provider.ts:1810`).
- **The key stays out of the file.** OpenCode replaces `{file:…}` in its config with the referenced value (`config/variable.ts`).
- **The provider is named `lab`, not `modal`.** OpenCode's built-in model list already has a provider called `modal`, and the two get merged.
- **`"temperature": true` makes OpenCode actually send the temperature.** Without it, sampling silently depends on the server.

## What an "app" is, and where a prompt goes

### Modal's terms

| Term | What it is | In our setup |
|---|---|---|
| **Workspace** | Your Modal account | `moyuanchen` |
| **App** | A named deployment: one version of our code, its container image and its settings, created by `modal deploy`. Costs nothing by itself. | `lab-ministral-14b`, one per `models.toml` entry. Today's is `qwen3-coder-abliterated`. |
| **Function** | A Python function inside an app that Modal can run. A web function gets a public URL: `https://<workspace>--<app>-<function>.modal.run`. | `serve` (the model server), plus `pull` and `keeper` |
| **Container** | One running copy of a function on a machine with its GPU attached. This is what you pay for, per second. | 0 when idle, up to `max_containers` |
| **Input** | One request being handled by a container | up to `max_concurrent` per container |

So today's URL `moyuanchen--qwen3-coder-abliterated-serve.modal.run` reads as: workspace `moyuanchen`, app `qwen3-coder-abliterated`, function `serve`.

### One prompt, end to end

```
 YOUR MAC                                          MODAL
┌────────────────────────────┐   ③ HTTPS POST    ┌──────────────────────────────────────────────┐
│ ① opencode (terminal UI)   │  /v1/chat/        │ ④ Modal's web gateway (Modal's, not ours)    │
│        │                   │  completions      │     picks a container of app                 │
│ ② opencode server ─────────┼──────────────────►│     lab-ministral-14b, function serve        │
│    builds request          │                   │     (starts one if none running = cold start)│
│                            │                   │        ▼                                     │
│ ⑧ runs tool calls HERE     │ ◄─────────────────┼── ⑤ container (1×L40S)                       │
│    (read, edit, bash...)   │  ⑦ streamed       │      proxy: key check, defaults, metrics     │
│ ⑨ next step → back to ③    │     tokens        │        └► ⑥ vLLM on 127.0.0.1 → GPU          │
│ opencode.db: tokens, times │                   │      volumes: lab-hf-cache, lab-vllm-cache   │
└────────────────────────────┘                   └──────────────────────────────────────────────┘
```

1. **You type a prompt.** The terminal UI is only a front end: it hands the prompt to the OpenCode server, a local process on your Mac.
2. **OpenCode builds the request.** It combines its system prompt, your project's instructions (`AGENTS.md`), the whole conversation so far, and the list of tools the model may call (read, edit, bash, grep…). It sends this through `@ai-sdk/openai-compatible` (`streamText`, `packages/opencode/src/session/llm.ts:280`).
3. **It sends an HTTPS request** to the model's own URL: `POST https://moyuanchen--lab-ministral-14b-serve.modal.run/v1/chat/completions`, with the API key, `stream: true`, the model name, temperature and tools.
4. **Modal's gateway routes it.** If a container has room (fewer than `max_concurrent` requests in progress), the request goes there. If none is running and fewer than `max_containers` exist, Modal starts one. That's the **cold start**: get a GPU machine, start the image, attach the volumes, load 28 GB of weights, warm up vLLM. It takes minutes, and your request waits. If every container is full, the request waits in Modal's queue.
5. **Our proxy in the container** checks the key, fills in sampling defaults, starts the clock, and forwards the request to vLLM. vLLM itself is never reachable from outside.
6. **vLLM processes the conversation and generates the reply.** It adds the request to the batch it's already running. It processes the prompt, reusing the cached beginning from the previous step when it can, then generates the reply on the GPU one token at a time.
7. **Tokens stream back** through the proxy, which records time to first token, the final token counts and the end time, then through Modal's gateway to OpenCode.
8. **If the model asked for tools, OpenCode runs them on your Mac.** The model server never sees your files except through what OpenCode sends.
9. **OpenCode loops.** It adds the tool results to the conversation and sends a **new request** (back to ③), until the model replies without asking for tools (`packages/opencode/src/session/prompt.ts:1088`). So **one prompt from you = several requests**, one per step. Each step is stored in `opencode.db`.
10. **Shutdown:** after the last request, the container waits `scaledown_window` (20 min). If nothing arrives, Modal stops it and the GPU charge stops.

Where the completion time goes, and which side can see it:

| Part | Seen by |
|---|---|
| Network Mac ↔ Modal, and Modal's gateway | the client only (the gap between client and server times) |
| Cold start or queue wait | server (`cold_start`, receive time) |
| Prompt processing → first token | server (`t_first_token`) |
| Generating the reply | server (`t_done - t_first_token`, `decode_tok_s`) |
| Running tools on your Mac between steps | the client only (gap between steps in `opencode.db`) |

## Serving on Modal

### How one `app.py` serves any model

`lab deploy <name>` runs `modal deploy serving/app.py --name lab-<name>` with the chosen model in an environment variable. When `app.py` is loaded, it reads that model's entry and applies its settings to the server code: GPU, number of containers, concurrency, volumes and image. The resolved settings are also written into the image (`LAB_SPEC`). That way the code running on Modal sees exactly what was deployed, even though `models.toml` is not uploaded.

### Inside each container

A small proxy runs in front of vLLM or Ollama, extending the one in `qwen_modal.py`:

- **API key:** checks the key against one shared Modal Secret, `lab-api-key`.
- **Sampling defaults:** fills in the model's sampling settings when the client leaves them out.
- **Streaming:** streams the response through to OpenCode unchanged.
- **Logging:** records each request (see Metrics).

For vLLM, the proxy also starts the server with:
- `--enable-auto-tool-choice --tool-call-parser <tool_parser>`, so OpenCode's tool calls work;
- `--max-model-len <context>`, `--max-num-seqs <max_concurrent>` and `--tensor-parallel-size <gpus>`;
- whatever is listed in `extra_args`.

Things from `qwen_modal.py` that are carried over: reading the weights into the cache before load, retrying downloads and saving after each attempt, the long timeouts, and the Ollama sampling-default fix.

Storage: `lab-hf-cache` for Hugging Face weights (Ministral is about 28 GB), `lab-vllm-cache` for vLLM's compiled kernels (prefixed `lab-` so they don't mix with other projects' volumes), and the existing `ollama-models` (about 40 GiB). Storage is free up to 1 TiB a month.

### One app per model (decided)

We considered putting one gateway in front of all models and chose against it. In terms of the request flow above, the only difference is step ③:

- **A. One app per model (chosen).** Step ③ goes straight to the model's own URL (`…--lab-ministral-14b-serve.modal.run`). That's the flow drawn above.
- **B. One gateway.** Step ③ goes to an extra CPU-only app, `…--lab-gateway-serve.modal.run`. It reads the `model` field and makes a second HTTPS request to the model's app, so steps ④ and ⑦ happen twice.

What B would give you is one URL and one key for everything. That only matters if tools other than OpenCode will call these models. It would also give you one central place for logs, but each model's proxy can log just as well.

What B costs:
- **An extra network hop on every request.** In an agent session that's every step.
- **Its own cold start.** The gateway also scales to zero, so it needs a few seconds to start after an idle period.
- **Both delays land inside the measured times.** They'd be counted in the completion times we're trying to measure.
- **One more app** to deploy and keep running.

Under A, the multiple URLs are invisible to you, because `lab use` / `lab sync` writes them into OpenCode. B can be added later without changing `models.toml` or the profiles.

## Keeping a model warm

Modal shuts a container down after `scaledown_window` without requests. We set that value, but Modal caps it at 20 minutes, and `qwen_modal.py` already uses the full 20. To keep a model up longer, `lab warm` temporarily tells Modal to always keep one container running:

| Command | Effect |
|---|---|
| `lab warm <name>` | Sends one small request: starts a container now so your first message doesn't wait. It stays up for the normal 20-minute idle window. |
| `lab warm <name> --for 2h` | Tells the deployed app to keep 1 container running (`update_autoscaler(min_containers=1)`) and starts the timer `keeper` on Modal. After 2 h the timer restores the normal setting. Before starting, it shows the cost (e.g. "1×L40S for 2h ≈ $3.90") and asks to confirm. |
| `lab warm <name> --off` | Ends it now and cancels the timer. |
| `lab ls` | Shows "warm until 16:40" for models being kept warm. |

The timer runs on Modal, not on your laptop, so it still turns the GPU off if the laptop sleeps or the terminal closes. It's a tiny CPU-only function costing well under a cent an hour. It can run for at most 24 h, Modal's limit on function run time.

A `lab deploy` of that model during a warm period probably restores the settings from `models.toml`, ending the warm period early. We'll test this; if so, `lab deploy` will warn.

## Estimating memory and compute

*Left empty on purpose; to be designed after you've read Hugging Face's parallelism playbook.*

## Rating responses (feedback mode)

When feedback mode is on, OpenCode asks you to rate each response after it finishes. The rating is saved next to that response's token counts and completion time, so `lab metrics` can relate quality to speed and cost.

```
┌ How good was this response? ─────────────────┐
│ > 4  Good as is                              │
│   3  Small fixes needed                      │
│   2  Big fixes needed                        │
│   1  Wrong or unusable                       │
│                                 esc  skip    │
└──────────────────────────────────────────────┘
```

### What counts as "after each prompt"

You type one prompt and OpenCode may make several requests to the model (one per agent step, see "One prompt, end to end"). The dialog appears **once**, when the whole run is finished and the session goes idle, not after every step.

It is **not** shown when:
- the run ended in an error, or you stopped it with Esc;
- the session is a subagent (OpenCode's helper sessions);
- you're using `opencode run` (no terminal UI to show it in);
- the model isn't in scope. By default only `lab/*` models are rated; the profile can add others, such as API models used as a baseline.

### How it's built

- **A small OpenCode plugin** for the terminal UI, `plugins/feedback.ts` in this repo, written in plain TypeScript, with no build step or dependencies.
- **Same pattern as OpenCode's own "session done" notification.** It listens for the session going idle (`api.event.on("session.status")`) and skips sessions that errored (`session.error`). See `packages/tui/src/feature-plugins/system/notifications.ts` in the reference clone.
- **The dialog** is OpenCode's own selection dialog (`api.ui.dialog.replace(() => api.ui.DialogSelect(…))`), so it looks and works like the rest of OpenCode: arrow keys or 1–4, Enter, Esc to skip.
- **Finding the rated response:** the plugin looks up the last assistant message of the run through OpenCode's API (`api.client`), to get its message ID, model, tokens and times.

These APIs were checked against the plugin types at the installed version (`v1.18.35:packages/plugin/src/tui.ts`).

### Where ratings go

One line per prompt in `metrics/feedback.jsonl` (gitignored):

```json
{"ts": "...", "profile": "default", "session_id": "ses_…", "user_message_id": "msg_…",
 "message_id": "msg_…", "model": "lab/ministral-14b", "rating": 3,
 "steps": 6, "input_tokens": 41230, "output_tokens": 1874, "duration_s": 48.2}
```

- **Skipped prompts** are saved with `"rating": null`, so "skipped often" is visible too.
- **Matching with server logs:** `user_message_id` and `session_id` are the same IDs as in `opencode.db`, so a rating points to its prompt there, and through the session id to the server-side request lines (see "Finding a conversation").

### Turning it on and off

- **Per profile (the lasting setting):** `feedback = true` or `false` in `profiles/<name>.toml`.
  - `lab use` adds the plugin to OpenCode's terminal-UI config `~/.config/opencode/tui.json`. That file doesn't exist yet: `lab use` creates it and backs up any existing one first.
  - It passes the plugin its settings: on/off, which models to rate, and where to write.
- **Quick toggle inside OpenCode:** a `/feedback` command turns it off or on for the current OpenCode window, with a short confirmation message. Restarting OpenCode goes back to the profile's setting, so what the profile says is always what you get by default.

## Commands

| Command | What it does |
|---|---|
| `lab init` | One-time setup: create the API key and store it in a Modal Secret and `.secrets/`, create the volumes |
| `lab ls` | Every model in `models.toml`: backend, GPU, deployed or not, URL, running containers, warm until |
| `lab pull <name>` | Fetch the weights into the volume on a CPU-only Modal function (no GPU charge) |
| `lab deploy <name> \| --all` | Deploy or redeploy the model's app, then `lab sync` |
| `lab stop <name>` | Stop the app; weights stay on the volume |
| `lab warm <name> [--for 2h \| --off]` | See above |
| `lab use <profile>` / `lab profiles` / `lab sync` | Switch, list, or rebuild OpenCode profiles (including feedback mode) |
| `/feedback` (inside OpenCode) | Turn rating prompts off or on for this OpenCode window |
| `lab logs <name>` | Show the model app's Modal logs |
| `lab metrics [--model M] [--since 7d]` | Token usage and response-time summary (below) |
| `lab prompts [--since 7d] [--rating 1,2] [--sort tokens\|duration]` | List prompts with their numbers, to find one |
| `lab show <id> [--full]` | One prompt's conversation with its numbers; takes a prompt, session or server request id |

## Metrics: token usage and completion time

These are collected on both ends without any extra effort from you.

**Client side (already recorded by OpenCode).** `~/.local/share/opencode/opencode.db` (SQLite) stores every assistant message with `modelID`, `tokens {input, output, reasoning, cache}`, and `time {created, completed}`. It also stores per-step `step-start` / `step-finish` parts. `lab metrics` reads it without modifying it. This measures what you actually experienced, including network and cold starts.

**Server side (new, in the proxy).** One JSON line per request:

```json
{"ts": "...", "model": "ministral-14b", "backend": "vllm", "gpu": "L40S", "container": "ta-…",
 "request_id": "f63be833…", "session_id": "ses_…", "parent_session_id": null,
 "status": 200, "outcome": "ok", "finish_reason": "tool_calls", "first_request_in_container": true,
 "in_flight": 1, "prompt_tokens": 10824, "cached_tokens": 0, "completion_tokens": 65,
 "tools": 10, "t_first_token": 3.4, "t_done": 5.85, "decode_tok_s": 26.2}
```

- **Session ids:** OpenCode sends its session id with every request (`x-opencode-session-id`, plus `x-parent-session-id` for subagents; `packages/opencode/src/session/llm/request.ts:187`). The proxy records them, so every line points to its conversation. Lines logged before 2026-10-10 21:45 don't have them.

- **Token counts:** OpenCode already asks for a usage summary at the end of each streamed reply (`stream_options.include_usage`, `provider.ts:1806`). The proxy reads the counts from that final message, so nothing is estimated.
- **Where it's stored:** lines are appended to a Modal Volume `lab-metrics` (`metrics/<model>/<date>.jsonl`). `lab metrics` downloads new files into `metrics/` here, which is gitignored.

**What `lab metrics` reports, per model:**
- Requests, input and output tokens, and their totals.
- Completion time p50/p95, time to first token, and output speed in tokens per second.
- Cold-start count and errors.
- The gap between client-measured and server-measured time, which shows network and queueing overhead.
- With feedback mode: average rating, how many prompts got each rating 1–4, how many were skipped, and rating against completion time and tokens.

## Finding a conversation

The conversation itself is saved by OpenCode, not by `lab`: every session, message, tool call and edit is in `~/.local/share/opencode/opencode.db`. `lab` saves only numbers (server lines, ratings). They are linked by ids:

```
rating line ──user_message_id──▶ prompt in opencode.db ──steps──▶ one request each
server line ──session_id + prompt tokens + time──▶ the step that sent it
```

- **`lab prompts`** lists prompts to `lab/*` models, one row per prompt: short ID, time, model, agent, rating (`skip` = dismissed, `-` = not asked), steps, tokens in/out, total time, and the first line of the prompt. Filter with `--since`, `--model`, `--rating 1,2` (`skip`, `none`); order with `--sort tokens` or `--sort duration`.
- **`lab show <id>`** prints the prompt, then each step with its numbers (time, tokens, and from the server line: time to first token, tok/s, how many requests ran at once, request id) and its content (text, thinking, tool calls with shortened output; `--full` for everything). It also prints the command to reopen the session in OpenCode: `opencode <dir> -s <session id>`.
- **Any id works:** the short ID, a full message id, a session id (shows the whole session) or a server `request_id` from `metrics/server/` (shows the prompt that sent it; a title request shows the prompt it ran beside). Any unique part of the id is enough.
- **Matching a server line to a step:** same session, OpenCode's input + cached tokens = the server's prompt tokens, and the request reached the server while the step ran. A request can wait minutes in Modal's queue during a cold start; that wait shows up as network+queue time in `lab metrics`.
- Code: `lab/traces.py` reads and joins the three sources; `lab metrics` uses the same reader.

## Build order

1. **Basic structure + vLLM + Ministral.** `models.toml`, profiles, the registry loader, `app.py` + `vllm_server.py` + `proxy.py`, and the commands `init / ls / pull / deploy / stop / use / sync / warm`. This only adds new things: the old Qwen app and the `modal` provider keep working the whole time.
2. **Metrics + feedback mode:** proxy logging, the `lab-metrics` volume, the `plugins/feedback.ts` rating plugin, and `lab metrics`, reading the server logs, `opencode.db` and `feedback.jsonl`.
3. **Ollama backend and moving the Qwen models over:**
   - add both Qwen models to `models.toml`, pointing at the existing `ollama-models` volume;
   - reuse today's API key in `lab-api-key`;
   - check they work, then remove the old `modal` provider from the global config;
   - stop the `qwen3-coder-abliterated` app and delete `qwen_modal.py`.
4. **Cleanup:** `git init`, `.gitignore` (`.secrets/`, `.lab/`, `metrics/`, `ref/`, `.venv/`).

## How we'll check it works

- **Ministral responds in OpenCode:** run `lab pull ministral-14b` and `lab deploy ministral-14b`, then `opencode run -m lab/ministral-14b "list the files here and summarize them" --format json`. The reply should include working tool calls.
- **Profiles switch cleanly:** `lab use default`, then `opencode models lab` lists exactly the profile's models. The rest of the global config is unchanged (compare it with the backup in `.lab/backups/`).
- **Feedback mode:**
  - With `feedback = true`, a prompt that uses tools shows the dialog once, after the last step. Choosing 3 writes one line to `metrics/feedback.jsonl` with the right model and message ID.
  - Esc writes `"rating": null`.
  - Stopping a run mid-way shows no dialog.
  - `/feedback` off: no dialog until OpenCode is restarted.
  - `lab use` with a `feedback = false` profile: no dialog at all.
- **Token counts agree:** for the same request, the server-side token counts match what `opencode.db` stores.
- **Timed warm works:** `lab warm ministral-14b --for 10m`, then `lab ls` shows 1 running container. About 10 minutes after the timer ends plus the 20-minute idle window, it shows 0.
- **Parallelism works:** send 8 requests at once to `ministral-14b-h100` (`max_concurrent = 8`). The log should show overlapping requests (`in_flight` > 1).
- **Scales to zero:** once idle, `lab ls` shows 0 running containers.

## Decided

- The command is called `lab`.
- OpenCode settings live in this repo as profiles; `lab use` copies the active one into the global config.
- The first model is Ministral 3 14B Instruct, full precision (BF16).
- `lab warm --for <duration>` keeps a model up for a set time.
- Benchmark entries pin their GPU type (`H100!`).
- One app per model, no gateway.
- Feedback mode: a 1–4 rating after each prompt, set per profile and toggled with `/feedback`.

## Still open

1. **Estimating memory and compute:** left empty until you've read the parallelism playbook.
2. **Rating labels:** the four labels in "Rating responses" are a suggestion; change them as you like.
3. **Which GPU first for Ministral?** Recommend deploying both `ministral-14b` (L40S, $1.95/h) and `ministral-14b-h100` ($3.95/h): comparing them is the first real benchmark. A100-80GB ($2.50/h) is the middle option.
