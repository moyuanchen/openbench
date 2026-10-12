# openrate

Benchmark open-weight models by doing real work with them.

openrate serves open-weight models on [Modal](https://modal.com) GPUs and connects them to [OpenCode](https://opencode.ai), the terminal coding agent. You code as usual. In the background, openrate records every token, every response time and your own rating of each answer, and links them all to the conversation they came from.

## Why

Leaderboards score models on fixed tasks. openrate scores them on the work you actually do. Each prompt you send gets three kinds of measurement:

- **Tokens:** prompt, cached and output tokens for every model request, including each step of an agent loop.
- **Time:** time to first token, total time, tokens per second, and how many requests the GPU was handling at once.
- **Your feedback:** after each prompt, a 1–4 rating of the result, from "good as is" to "wrong or unusable".

All three are tied to the OpenCode session and message, so `lab metrics` can compare models side by side and `lab show` can replay any prompt step by step with its numbers.

To keep the measurements fair:

- **Every model gets its own deployment.** There is no shared gateway in front of the models, so no extra network hop ends up in the timings.
- **Weights are served as published.** They are pinned to an exact Hugging Face commit and downloaded once into a Modal Volume, and every container loads that same copy.
- **The hardware is exactly what you asked for.** The GPU type and count come from `models.toml`, and an H100 is pinned so Modal can't quietly swap in a faster H200.
- **Sampling settings are written down.** Each model's default temperature and other sampling settings are declared in `models.toml`. The proxy applies them only when the client doesn't send its own.

## How it works

```
                 ┌──────────── Modal app per model ────────────┐
  OpenCode ────► │  proxy (API key, sampling defaults) ──► vLLM │
     │           │     └─► one log line per request ──► Volume  │
     │           └─────────────────────────────────────────────┘
     ├─► ~/.local/share/opencode/opencode.db   full conversations
     └─► rating plugin ──► metrics/feedback.jsonl

  lab metrics / prompts / show   join the three by session and message id
```

You list models in `models.toml`. `lab` deploys each one as its own Modal app, with vLLM behind a small logging proxy, and registers it in OpenCode as `lab/<name>`. A model scales to zero when idle, so you pay only while a GPU container is running.

## Installation

You need macOS or Linux, `git`, and the three accounts and tools below. The CLI is called `lab`.

### 1. Create a Modal account

Sign up at [modal.com](https://modal.com) with GitHub or Google. The free **Starter plan includes $30 of compute credit every month**, which is enough to try openrate properly. Credit that you don't use doesn't carry over. See [modal.com/pricing](https://modal.com/pricing) for current terms.

Roughly what $30 buys (GPU prices from Modal on 2026-10-10, billed per second only while a container runs):

| GPU | ≈ per hour | ≈ hours for $30 |
|---|---|---|
| L40S (48 GB) | $1.95 | 15 |
| H100 (80 GB) | $3.95 | 7.5 |
| B200 (180 GB) | $6.25 | 4.8 |

Downloading weights runs on a CPU-only function and costs very little. Storing them in Modal Volumes is free up to 1 TiB.

### 2. Install uv and OpenCode

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh      # or: brew install uv
curl -fsSL https://opencode.ai/install | bash         # or: brew install anomalyco/tap/opencode
```

Run `opencode` once to make sure it starts, then quit.

### 3. Get openrate

```sh
git clone https://github.com/moyuanchen/openrate.git
cd openrate
uv sync                        # creates .venv with Python 3.12, Modal's client and the `lab` command
source .venv/bin/activate      # puts `lab` and `modal` on your PATH (or prefix commands with `uv run`)
```

### 4. Connect to Modal

```sh
modal setup                    # opens a browser to log in, then saves a token to ~/.modal.toml
```

If you've used Modal on this machine before and `~/.modal.toml` exists, skip this step.

### 5. Initialize

```sh
lab init
```

This generates a random API key in `.secrets/api-key` and stores it in your Modal workspace as the secret `lab-api-key`. Your model endpoints reject any request that doesn't carry this key.

### 6. Serve your first model

Every model is an entry in `models.toml`, and each entry has two identifiers. For example:

```toml
[models.ministral-14b]                                   # the name: a short label, used as `lab pull ministral-14b`
source = "mistralai/Ministral-3-14B-Instruct-2512-BF16"  # the Hugging Face repo id: where the weights come from
```

- **The name** (`ministral-14b`) is just a label. Nothing looks it up anywhere, so it could have been any short name. It becomes the Modal app `lab-ministral-14b` and the OpenCode model `lab/ministral-14b`.
- **The `source`** must match Hugging Face exactly. It's the part of the model page's URL after `huggingface.co/`. Here that's [huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16).

The models already in `models.toml` come with their hardware configured. Run `lab ls` to see their names and the GPU each one uses. To serve a model that isn't listed, find it on Hugging Face and add an entry; see [Adding a model from Hugging Face](#adding-a-model-from-hugging-face).

```sh
lab ls                         # model names, their GPU, and whether they're deployed
lab pull <model>               # download the weights into a Modal Volume (CPU only; waits until done)
lab deploy <model>             # deploy the model's Modal app and print its URL
lab profiles                   # OpenCode setups in profiles/
lab use <profile>              # add the `lab` provider and the rating plugin to OpenCode
opencode
```

In OpenCode, choose the model with `/models`. It's listed under **Lab (Modal)**. The first prompt waits for a cold start, which can take a few minutes for a large model. After that, the container stays up until it has been idle for 20 minutes.

If you add your own entry, check that its `gpu` has enough memory for the weights plus working room before you deploy (see [Models](#models-modelstoml)).

## Daily use

```sh
opencode                         # the active profile's model is the default
lab use <profile>                # switch the default model (restart OpenCode afterwards)
lab warm <model>                 # optional: start a container before you begin, to skip the cold start
lab warm <model> --for 2h        # keep one running for a long session (shows the cost and asks first)
lab metrics --since 24h          # tokens, response times and ratings per model
lab prompts --rating 1,2         # find prompts by time, tokens or rating…
lab show <id>                    # …and read one, with the numbers for every step
```

## Reading the results

- **`lab metrics`** gives one summary per model, from three points of view:
  - **Server:** requests, errors and cold starts; tokens in, cached and out; time to first token and total time (p50 and p95); output tokens per second.
  - **Client:** prompts and agent steps, time per prompt, and the network and queueing overhead between OpenCode and the GPU.
  - **Ratings:** the average and the 4/3/2/1 counts.

  Narrow it with `--since 7d` or `--model <name>`, or add `--json` to process the numbers yourself.
- **`lab prompts`** lists individual prompts with their numbers. Filter by `--rating 1,2` (or `skip`, `none`) and sort by `--sort tokens` or `--sort duration` to find the expensive, slow or bad ones.
- **`lab show <id>`** replays one prompt: the conversation, tool calls and edits, with the tokens and times of each model request and your rating. It takes a short ID from `lab prompts`, an OpenCode session id or a server `request_id`. It also prints the command to reopen that session in OpenCode.

The `metrics`, `prompts` and `show` commands download new server log lines first. Add `--offline` to skip that.

## Ratings

With `feedback = true` in the active profile, OpenCode asks after each finished prompt:

| Rating | Meaning |
|---|---|
| **4** | Good as is |
| **3** | Small fixes needed |
| **2** | Big fixes needed |
| **1** | Wrong or unusable |

Use the arrow keys or type the digit, then press Enter. Esc skips the rating, which is recorded as skipped. `/feedback` in OpenCode turns ratings off or on for that window. Ratings aren't requested for subagent sessions, for prompts that errored or were interrupted, or for models outside `lab/`.

## Commands

| Command | What it does |
|---|---|
| `lab init` | One-time setup: API key and Modal Secret |
| `lab ls` | Models in `models.toml`: GPU, deployed or not, running containers, warm-until time, URL |
| `lab pull <name>` | Download weights into the `lab-hf-cache` volume on a CPU-only function |
| `lab deploy <name> \| --all` | Deploy or redeploy, reapply the scaling in `models.toml`, update OpenCode |
| `lab stop <name>` | Stop the app (the weights stay in the volume) |
| `lab warm <name> [--for 2h \| --off]` | Start a container now, or keep one up for a while |
| `lab use <profile>` · `lab profiles` · `lab sync` | Switch, list or reapply OpenCode profiles |
| `lab metrics [--since 7d] [--model M] [--json]` | Summary per model |
| `lab prompts [--since 7d] [--rating 1,2] [--sort tokens\|duration] [-n 20]` | List prompts with their numbers |
| `lab show <id> [--full]` | One prompt's conversation and numbers |
| `lab logs <name> [-f]` | The model app's Modal logs (vLLM output, errors) |

## Models: `models.toml`

Each `[models.<name>]` entry is one deployment: a set of weights plus the hardware to run them on. It becomes the Modal app `lab-<name>` and the OpenCode model `lab/<name>`. Any setting an entry leaves out comes from `[defaults]`. `extends = "<other>"` copies another entry first, which makes it easy to benchmark the same weights on different hardware.

```toml
[models.my-model]
name        = "My Model (BF16)"          # display name in OpenCode
source      = "<org>/<repo>"             # Hugging Face repo
revision    = "<commit sha>"             # pin the exact weights
gpu         = "H100"
tool_parser = "<parser>"                 # vLLM's tool-call parser for this model family
temperature = 0.2                        # used when the client sends none

[models.my-model-2gpu]
extends = "my-model"
gpus    = 2                              # same weights, split across two GPUs
```

| Setting | Default | Meaning |
|---|---|---|
| `gpu`, `gpus` | `"L40S"`, `1` | GPU type and count per container (`gpus > 1` means tensor parallel). `"H100"` is pinned so Modal can't swap in an H200; set `allow_upgrade = true` to allow the swap |
| `max_concurrent` | `4` | Requests one container handles at once (Modal `max_inputs` = vLLM `--max-num-seqs`) |
| `min_containers`, `max_containers` | `0`, `1` | `0` means scale to zero; more than 1 lets Modal add GPUs under load |
| `scaledown_window` | `"20m"` | Idle time before shutdown (Modal allows 2s–20m) |
| `context`, `output` | `65536`, `16384` | vLLM `--max-model-len`, and OpenCode's limits |
| `revision`, `exclude` | | Pin the Hugging Face commit; skip files when downloading |
| `tool_parser`, `extra_args` | | vLLM tool-call parser and any extra vLLM flags |
| `temperature`, `top_p`, `presence_penalty`, `reasoning_effort` | | Defaults the proxy fills in when the request has none (`reasoning_effort` is for models with a thinking switch, e.g. `"none"` or `"high"`) |

- **Change hardware:** edit `gpu` or `gpus`, then run `lab deploy <name>`. The running container is replaced.
- **Size the GPU:** the weights take about 2 bytes per parameter in BF16 and 1 byte in FP8. What's left of GPU memory becomes the KV cache that holds conversations. OpenCode's prompts are long, so leave plenty of room.

### Adding a model from Hugging Face

1. **Find the model** on [huggingface.co](https://huggingface.co/models). Check that vLLM supports its architecture ([supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)) and that the model isn't gated. A gated model asks you to accept its terms before downloading, and openrate doesn't pass a Hugging Face token yet.
2. **Copy the repo id** from the URL into `source`. To pin the exact weights, copy a commit hash from the repo's *Files and versions* → *History* page into `revision`.
3. **Choose a GPU** with room for the weights plus the KV cache. *Files and versions* lists the file sizes.
4. **Copy the serving flags from the model card.** Many model cards include a `vllm serve` command. OpenCode relies on tool calls, so its `--tool-call-parser` value goes into `tool_parser`. The other flags go into `extra_args`, and any recommended sampling settings go into `temperature`, `top_p` and so on.
5. **Choose a name**, then run `lab pull <name>` and `lab deploy <name>`.

Every profile lists all deployed models, so the new model appears in OpenCode's `/models` list after `lab deploy`. A profile only chooses which model is the default.

## Profiles: `profiles/*.toml`

A profile is one version of the OpenCode setup. `lab use <profile>` writes it to `build/opencode.json` and merges it into `~/.config/opencode/opencode.json`. Restart OpenCode after switching.

```toml
model    = "lab/<name>"          # OpenCode's default model
# models = ["<name>"]            # which models to list; omitted = every deployed one
feedback = true                  # ask for a 1–4 rating after each prompt
# [opencode]                     # copied as-is into the OpenCode config
# small_model = "lab/<name>"
```

If a profile's default model isn't deployed, OpenCode's default model is left unchanged and you get a warning.

## MCP servers

OpenCode can give the model extra tools through MCP servers, for example [Context7](https://github.com/upstash/context7) for library docs. openrate doesn't set any up. Add them under `mcp` in `~/.config/opencode/opencode.json`, as described in [OpenCode's MCP docs](https://opencode.ai/docs/mcp-servers/):

```json
"mcp": {
  "context7": { "type": "remote", "url": "https://mcp.context7.com/mcp" }
}
```

`lab use` leaves this key alone. A profile can carry its own servers instead, under `[opencode.mcp.<server>]`. In that case `lab use` replaces the whole `mcp` block, and switching to a profile without one removes it.

MCP servers change the measurements:

- **Tokens:** every server's tool descriptions are sent with every request, so prompt tokens go up. The `tools` count in the server records goes up too. For runs you want to compare, use the same servers throughout.
- **Traces:** an MCP call is stored like any other tool call, as `<server>_<tool>` with its input, output and start and end times. `lab show` prints it. The time the model waits for a call falls between requests, so it counts toward the prompt's total time but not toward server response times.

## What openrate changes on your machine

- **`~/.config/opencode/opencode.json`:** adds the `lab` provider and the keys the active profile sets. Everything else in the file is left alone.
- **`~/.config/opencode/tui.json`:** adds the rating plugin (`plugins/feedback.ts`).
- **Backups:** before every write, the previous version of each file is copied to `.lab/backups/`.
- **Comments in your config:** if either file contains comments (JSONC), `lab` refuses to merge rather than risk losing them. Remove the comments or move them elsewhere.

On Modal, openrate creates one app per deployed model (`lab-<name>`), the volumes `lab-hf-cache`, `lab-vllm-cache` and `lab-metrics`, and the secret `lab-api-key`. To remove everything, `lab stop` each model, delete those volumes and the secret in the Modal dashboard, and remove the `lab` provider and the plugin entry from the two OpenCode files.

## Where the data is

| What | Where | Written by |
|---|---|---|
| Full conversations: prompts, replies, tool calls, edits, per-step tokens and times | `~/.local/share/opencode/opencode.db` | OpenCode itself (`lab` only reads it) |
| One line per model request: tokens (prompt, cached, output), time to first token, total time, tok/s, requests running at once, session id | Modal volume `lab-metrics`, downloaded to `metrics/server/` | The proxy in front of vLLM |
| One line per rated prompt | `metrics/feedback.jsonl` | The rating plugin (`plugins/feedback.ts`) |

The three sources are linked by ids. A rating carries the prompt's message id, and OpenCode sends its session id with every request, which the proxy records. All collected data stays in your Modal workspace and on your machine.

## Costs and timing

- **You pay while a container is up, and that includes the idle window.** After the last request, a container stays up for `scaledown_window` (20 minutes by default) before it shuts down. Even one quick prompt costs about 20 minutes of GPU time. Shorten the window in `models.toml` if you'd rather trade that for more cold starts.
- **Cold starts take minutes.** Loading weights, compiling and capturing CUDA graphs takes a few minutes and grows with model size. `lab warm` starts a container ahead of time.
- **Long sessions:** `lab warm <name> --for 2h` keeps a container up and shows the estimated cost before it starts. `min_containers > 0` keeps GPUs on around the clock, so `lab deploy` asks before deploying such a model.
- **Interpreting token counts:** every OpenCode request carries about 11k tokens of OpenCode's own system prompt and tool descriptions. vLLM's prefix cache serves most of that after the first step, which shows up as cached tokens in `lab metrics`.

## Troubleshooting

- **`lab init` says Modal isn't logged in:** run `modal setup` with the venv active.
- **A prompt hangs for minutes:** this is probably a cold start. `lab ls` shows running containers, and `lab logs <name> -f` shows vLLM starting up.
- **A container keeps crashing at startup:** Modal keeps retrying it on a GPU while a request waits, and that GPU time is billed. Check `lab logs <name>`, then run `lab stop <name>`. Common causes are a GPU too small for the weights or a missing vLLM flag for the model.
- **vLLM fails with an import error:** vLLM and `transformers` are pinned together in `serving/vllm_server.py`. If you change one, change the other to a compatible version.
- **A model is missing from OpenCode's `/models` list:** run `lab sync` and restart OpenCode. The picker moves recently used models into a "Recent" group, so search by name to see every model.
- **The rating dialog doesn't appear:** check that the active profile has `feedback = true`, run `lab use <profile>` and restart OpenCode.

## Repo layout

```
models.toml            what can be served
profiles/              OpenCode setups (`lab use <name>`)
lab/                   the `lab` command
  cli.py               subcommands
  registry.py          load and validate models.toml
  profiles.py          render profiles, merge into the global OpenCode config
  cloud.py             Modal calls, GPU prices, .lab/state.json
  traces.py            read and join OpenCode's database, server lines and ratings
  metrics.py           `lab metrics` summary
serving/               runs on Modal
  app.py               one app definition (serve, pull, keeper), configured per model
  proxy.py             API key check, sampling defaults, one log line per request
  vllm_server.py       vLLM image, download and launch
  metrics.py           the per-request log writer
  ollama_server.py     Ollama backend (experimental, not tested on a GPU)
plugins/feedback.ts    OpenCode plugin for the 1–4 rating
build/opencode.json    generated from the active profile
.secrets/ .lab/ metrics/   gitignored: API key, local state, collected data
```
