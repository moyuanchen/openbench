# lab

Serve open-weight models on [Modal](https://modal.com) GPUs and use them from [OpenCode](https://opencode.ai), with token usage, response times and your own 1–4 ratings recorded along the way.

You declare models in `models.toml`, `lab` deploys each one as its own Modal app (vLLM behind a small logging proxy), and wires it into OpenCode as `lab/<name>`. Then you just run `opencode`. The benchmark data is collected in the background; `lab metrics`, `lab prompts` and `lab show` read it.

## Setup (once)

Needs [uv](https://docs.astral.sh/uv/), a Modal account, and OpenCode (`brew install anomalyco/tap/opencode`).

```sh
uv sync                          # .venv with modal and the `lab` command
.venv/bin/modal token new        # only if ~/.modal.toml doesn't exist yet
.venv/bin/lab init               # API key in .secrets/ + the `lab-api-key` Modal Secret
.venv/bin/lab pull ministral-14b # download the weights into a Modal volume (CPU only, cheap)
.venv/bin/lab deploy ministral-14b
.venv/bin/lab use ministral      # put the `lab` provider and settings into ~/.config/opencode/
```

The examples below write `lab` for `.venv/bin/lab` (or activate the venv, or use `uv run lab`).

## Day to day

```sh
opencode                         # the active profile's model is the default
lab use devstral                 # switch the default model: ministral, devstral or mistral (restart OpenCode)
lab warm ministral-14b           # optional: start a container before you begin (cold start ≈ 3 min)
lab warm ministral-14b --for 2h  # keep one running for a long session (asks to confirm the cost)
lab metrics --since 24h          # tokens, response times, ratings per model
lab prompts --rating 1,2         # find prompts by time, tokens or rating…
lab show <ID>                    # …and read one, with the numbers on every step
```

A model scales to zero 20 minutes after its last request; the first prompt after that waits for a cold start.

## Commands

| Command | What it does |
|---|---|
| `lab init` | One-time setup: API key and Modal Secret |
| `lab ls` | Models in `models.toml`: GPU, deployed or not, running containers, warm-until, URL |
| `lab pull <name>` | Download weights into the `lab-hf-cache` volume on a CPU-only function |
| `lab deploy <name> \| --all` | Deploy or redeploy, re-apply `models.toml` scaling, update OpenCode |
| `lab stop <name>` | Stop the app (weights stay in the volume) |
| `lab warm <name> [--for 2h \| --off]` | Start a container now, or keep one up for a while |
| `lab use <profile>` · `lab profiles` · `lab sync` | Switch, list or re-apply OpenCode profiles |
| `lab metrics [--since 7d] [--model M] [--json]` | Summary per model |
| `lab prompts [--since 7d] [--rating 1,2] [--sort tokens\|duration] [-n 20]` | List prompts with their numbers (downloads new server lines first, like `metrics` and `show`) |
| `lab show <id> [--full]` | One prompt's conversation and numbers |
| `lab logs <name> [-f]` | The model app's Modal logs (vLLM output, errors) |

## Models: `models.toml`

Each `[models.<name>]` entry is one deployment: weights plus the hardware to run them on. It becomes the Modal app `lab-<name>` and the OpenCode model `lab/<name>`. Settings not given come from `[defaults]`; `extends = "<other>"` copies another entry first.

```toml
[models.ministral-14b]
name        = "Ministral 3 14B Instruct (BF16)"     # display name in OpenCode
source      = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
revision    = "3cea74c1ebaf5ce5f5a2553de470e2ceab825142"
tool_parser = "mistral"
temperature = 0.05                                   # used when the client sends none
```

| Setting | Default | Meaning |
|---|---|---|
| `gpu`, `gpus` | `"L40S"`, `1` | GPU type and count per container (`gpus > 1` = tensor parallel). `"H100"` is pinned so Modal can't swap in an H200; `allow_upgrade = true` allows it |
| `max_concurrent` | `4` | Requests one container works on at once (Modal `max_inputs` = vLLM `--max-num-seqs`) |
| `min_containers`, `max_containers` | `0`, `1` | `0` = scale to zero; more than 1 lets Modal add GPUs under load |
| `scaledown_window` | `"20m"` | Idle time before shutdown (Modal allows 2s–20m) |
| `context`, `output` | `65536`, `16384` | vLLM `--max-model-len`, and OpenCode's limits |
| `revision`, `exclude` | | Pin the Hugging Face commit; skip files when downloading |
| `tool_parser`, `extra_args` | | vLLM tool-call parser and any extra vLLM flags |
| `temperature`, `top_p`, `presence_penalty`, `reasoning_effort` | | Defaults the proxy fills in when the request has none (`reasoning_effort`: `"none"` or `"high"` for models that can think first, e.g. Mistral Medium 3.5) |

- **Add a model:** add an entry, then `lab pull <name>` and `lab deploy <name>`. Every profile lists all deployed models; the profile only picks the default one.
- **Change hardware:** edit `gpu` / `gpus`, then `lab deploy <name>`. The running container is replaced.
- Weights are full precision (no quantization). Gated Hugging Face models aren't supported yet (no HF token is passed).

## Profiles: `profiles/*.toml`

A profile is one version of the OpenCode setup; `lab use <profile>` writes it to `build/opencode.json` and merges it into `~/.config/opencode/opencode.json` (only the `lab` provider and the keys the profile sets; the previous file is backed up in `.lab/backups/`). Restart OpenCode after switching.

| Profile | Default model | GPU |
|---|---|---|
| `ministral` | `lab/ministral-14b`: Ministral 3 14B Instruct (BF16) | 1×L40S |
| `devstral` | `lab/devstral-small-2`: Devstral Small 2 24B (FP8) | 1×L40S |
| `mistral` | `lab/mistral-medium-3.5`: Mistral Medium 3.5 128B (FP8), thinking on | 1×B200 |

A profile whose model isn't deployed leaves OpenCode's default model unchanged (with a warning). `profiles/ministral.toml`:

```toml
model    = "lab/ministral-14b"   # OpenCode's default model
# models = ["ministral-14b"]     # which models to list; omitted = every deployed one
feedback = true                  # ask for a 1–4 rating after each prompt
# [opencode]                     # copied as-is into the OpenCode config
# small_model = "lab/ministral-14b"
```

## Ratings

With `feedback = true`, OpenCode asks after each finished prompt: **4** good as is, **3** small fixes needed, **2** big fixes needed, **1** wrong or unusable. Use the arrows or type the digit and press Enter; Esc skips (saved as skipped). `/feedback` in OpenCode turns it off or on for that window. It isn't asked for subagent sessions, errored or interrupted prompts, or models outside `lab/`.

Restart OpenCode after `lab use` for a change to take effect.

## Where the data is

| What | Where | Written by |
|---|---|---|
| Full conversations: prompts, replies, tool calls, edits, per-step tokens and times | `~/.local/share/opencode/opencode.db` | OpenCode itself (`lab` only reads it) |
| One line per model request: tokens (prompt, cached, output), time to first token, total time, tok/s, requests running at once, session id | Modal volume `lab-metrics`, downloaded to `metrics/server/` | The proxy in front of vLLM |
| One line per rated prompt | `metrics/feedback.jsonl` | The rating plugin (`plugins/feedback.ts`) |

They are linked by ids: a rating carries the prompt's message id, and OpenCode sends its session id with every request, which the proxy records. `lab show` accepts any of them (a short ID from `lab prompts`, a session id, or a server `request_id`) and prints the conversation with its numbers, plus the command to reopen it in OpenCode (`opencode <dir> -s <session id>`). Server lines logged before 2026-10-10 21:45 have no session id and are matched by time and token count instead.

## What to expect (Ministral 3 14B on 1×L40S)

- Cold start ≈ 2.5–3.5 min; ≈ $1.95/h while a container runs, nothing while scaled to zero. Volumes are free up to 1 TiB.
- ≈ 26 tokens/s per reply; ≈ 87 tokens/s in total with 4 requests at once.
- Every OpenCode request carries ≈ 11k tokens of its own system prompt and tool descriptions; vLLM's prefix cache serves most of it after the first step.
- KV cache room ≈ 99k tokens, shared by the requests running at once.

## Troubleshooting

- **A prompt hangs for minutes:** probably a cold start. `lab ls` shows running containers; `lab logs <name> -f` shows vLLM starting.
- **A container keeps crashing at startup:** Modal retries it on a GPU while a request waits. Check `lab logs <name>`, then `lab stop <name>`.
- **vLLM fails with an import error about `PixtralRotaryEmbedding`:** vLLM 0.30.0 needs `transformers==5.16.1` (pinned in `serving/vllm_server.py`).
- **The rating dialog doesn't appear:** check `feedback = true` in the active profile, run `lab use <profile>`, and restart OpenCode.

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
  ollama_server.py     Ollama backend (unused, untested on a GPU)
plugins/feedback.ts    OpenCode plugin for the 1–4 rating
build/opencode.json    generated from the active profile
PLAN.md                design and decisions; QUESTIONS.md: open questions
.secrets/ .lab/ metrics/ ref/   gitignored: API key, local state, collected data, OpenCode source for reference
```
