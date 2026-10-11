# Open questions

Parked while steps 2–4 of [PLAN.md](PLAN.md) are built. Each one lists what we know so far.

## 1. What does OpenCode add to every request?

Every step of a session sends about **11k input tokens** even when your prompt is one sentence (measured: 10,791 tokens for a one-line question to `lab/ministral-14b`). That is about 17% of a 65k context before any conversation. What we know about where it comes from:

- **System prompt.** OpenCode chooses it by model name (`session/system.ts:29-50`): `claude` gets `anthropic.txt`, `gpt` gets `gpt.txt` or `beast.txt`, `gemini-` gets `gemini.txt`, `kimi` gets `kimi.txt`, and every other name, including ours, gets `default.txt` (about 2.1k tokens).
- **Tool descriptions.** About 3.8k tokens of text (`tool/*.txt`) plus each tool's JSON parameter schema, for roughly 15 tools: read, edit, write, apply_patch, glob, grep, shell, task, todo, webfetch, websearch, lsp, skill, question, and plan enter/exit.
- **Environment block:** working directory, platform, date, and so on, plus any `AGENTS.md` instructions.

To do: capture one real request and break it down exactly. One way is an opt-in setting in the proxy that saves request bodies.

## 2. For coding/building, Instruct or Reasoning?

- **Instruct** (what's deployed): answers directly, with no thinking tokens. It's fast, and each step's output is short.
- **Reasoning** (`mistralai/Ministral-3-14B-Reasoning-2512`): also full precision BF16 and the same size. It thinks before answering or calling a tool, which should help with planning and debugging. Thinking is generated at the same rate as visible text, so each step takes longer, and many steps make a session much longer.

To do: deploy both and compare them on real tasks using `lab metrics` and the 1–4 ratings.

## 3. Adding the Reasoning model

- **What it needs:** vLLM's `--reasoning-parser mistral`, temperature 1 and top_p 0.95 (from the model card), and Mistral's `SYSTEM_PROMPT.txt` (which contains a `[THINK]` part) appended to OpenCode's system prompt. The proxy could append it.
- **OpenCode can display the thinking:** its AI SDK reads `reasoning_content` or `reasoning`, and vLLM 0.30 sends `reasoning`.
- **Speed:** the L40S makes about 27 tok/s, so 2,000 thinking tokens take about 75 s per step. A faster GPU is advisable (see 6).

## 4. Automatic model switching

- **By mode (built into OpenCode, no new code):** give each agent its own model, for example Plan uses Reasoning and Build uses Instruct, and switch with Tab. It would be a profile setting.
- **Per prompt (small plugin):** OpenCode's `chat.message` plugin hook can change the model a prompt goes to before it runs (`session/prompt.ts:999`, then `:1141` uses that message's model for every step). The router could use rules or a quick yes/no from the Instruct model. Each message records which model answered.
- **Catch:** these are two separate apps, so a switch to a sleeping model waits about 3.5 minutes for its cold start. Either keep both warm, or run both on one H100 (about 25 GiB of weights each, which fits).

## 5. A longer context window

| Option | Max context | Cost |
|---|---|---|
| Stay on L40S, raise `context` to ~90k | GPU memory holds 92,864 tokens in total, so one long session at a time | free |
| A100 80GB or H100 | room for ~295k tokens, so the full 262k context fits | +28% (A100) / +103% (H100) per hour |
| Turn off OpenCode tools you don't use | saves a few thousand tokens per request | free |

## 6. Which GPU for Ministral?

| GPU | $/h | Output speed (rough, scaled from the L40S measurement) | Max context |
|---|---|---|---|
| L40S (now) | 1.95 | 27 tok/s measured | ~90k |
| A100 80GB | 2.50 | ~65 tok/s | 262k |
| H100 | 3.95 | ~105 tok/s | 262k |

Suggestion so far: A100 80GB for both Instruct and Reasoning.

## 7. The same system prompt for every model?

Because OpenCode picks the system prompt by model name, a later comparison against Claude or GPT would mix model differences with prompt differences. An option is a profile setting that gives every model the same prompt (OpenCode lets an agent's prompt be overridden in config).

## Carried over from PLAN.md

- The estimation section stays empty until the Hugging Face parallelism playbook is read.
- The wording of the four rating labels.
