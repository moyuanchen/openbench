// OpenCode TUI plugin: after each prompt finishes, ask for a 1–4 rating of the response.
//
// Installed into ~/.config/opencode/tui.json by `lab use <profile>`; the profile's `feedback` key sets whether it
// asks by default, and `/feedback` toggles it for the current OpenCode window. Ratings are appended, one JSON
// line per prompt, to the `log` file (metrics/feedback.jsonl in the lab repo), where `lab metrics` reads them.
//
// Built like OpenCode's own "session done" notification (packages/tui/src/feature-plugins/system/notifications.ts):
// a session going busy -> idle without a session.error is one finished prompt.

import { appendFileSync, mkdirSync } from "node:fs"
import { dirname } from "node:path"
import type { TuiPlugin, TuiPluginApi, TuiPluginModule } from "@opencode-ai/plugin/tui"

type Options = { enabled?: boolean; models?: string[]; log?: string; profile?: string }

const RATINGS = [
  { value: 4, title: "4  Good as is" },
  { value: 3, title: "3  Small fixes needed" },
  { value: 2, title: "2  Big fixes needed" },
  { value: 1, title: "1  Wrong or unusable" },
]

const tui: TuiPlugin = async (api, options) => {
  const opts = (options ?? {}) as Options
  const scope = opts.models ?? ["lab/*"]
  let enabled = opts.enabled ?? false
  const active = new Set<string>()
  const errored = new Set<string>()

  api.keymap.registerLayer({
    commands: [
      {
        name: "lab.feedback.toggle",
        title: "Toggle response ratings",
        category: "Lab",
        namespace: "palette",
        slashName: "feedback",
        run() {
          enabled = !enabled
          api.ui.toast({ variant: "info", message: `Response ratings ${enabled ? "on" : "off"} for this window` })
        },
      },
    ],
  })

  api.event.on("session.status", (event) => {
    const sessionID = event.properties.sessionID
    const type = event.properties.status.type
    if (type === "busy" || type === "retry") {
      active.add(sessionID)
      errored.delete(sessionID)
      return
    }
    if (type !== "idle" || !active.delete(sessionID)) return
    if (errored.delete(sessionID) || !enabled) return
    if (api.state.session.get(sessionID)?.parentID) return // subagent
    const run = lastRun(api, sessionID)
    if (!run || !inScope(run.model, scope) || api.ui.dialog.open) return
    ask(api, opts, run)
  })

  api.event.on("session.error", (event) => {
    const sessionID = event.properties.sessionID
    if (sessionID && active.has(sessionID)) errored.add(sessionID)
  })
}

// The last user prompt of the session and the assistant steps that answered it.
function lastRun(api: TuiPluginApi, sessionID: string) {
  const messages = api.state.session.messages(sessionID)
  const user = [...messages].reverse().find((m) => m.role === "user")
  if (!user) return
  const steps = messages.filter((m) => m.role === "assistant" && m.parentID === user.id)
  const last = steps.at(-1)
  if (!last || last.role !== "assistant" || last.error || !last.time.completed) return
  const sum = (pick: (m: (typeof steps)[number]) => number) => steps.reduce((total, m) => total + pick(m), 0)
  return {
    session_id: sessionID,
    user_message_id: user.id,
    message_id: last.id,
    model: `${last.providerID}/${last.modelID}`,
    agent: last.agent,
    steps: steps.length,
    input_tokens: sum((m) => (m.role === "assistant" ? m.tokens.input : 0)),
    output_tokens: sum((m) => (m.role === "assistant" ? m.tokens.output : 0)),
    reasoning_tokens: sum((m) => (m.role === "assistant" ? m.tokens.reasoning : 0)),
    cache_read_tokens: sum((m) => (m.role === "assistant" ? m.tokens.cache.read : 0)),
    duration_s: (last.time.completed - user.time.created) / 1000,
  }
}

function inScope(model: string, patterns: string[]) {
  return patterns.some((p) => p === "*" || p === model || (p.endsWith("/*") && model.startsWith(p.slice(0, -1))))
}

function ask(api: TuiPluginApi, opts: Options, run: NonNullable<ReturnType<typeof lastRun>>) {
  let answered = false
  const record = (rating: number | null) => {
    if (answered) return
    answered = true
    if (!opts.log) return
    try {
      mkdirSync(dirname(opts.log), { recursive: true })
      const entry = { ts: new Date().toISOString(), profile: opts.profile ?? null, ...run, rating }
      appendFileSync(opts.log, JSON.stringify(entry) + "\n")
    } catch (error) {
      api.ui.toast({ variant: "error", message: `Could not save rating: ${error}` })
    }
  }
  api.ui.dialog.replace(
    () =>
      api.ui.DialogSelect({
        title: "How good was this response?  (Esc to skip)",
        options: RATINGS,
        onSelect(option) {
          record(option.value)
          api.ui.dialog.clear()
        },
      }),
    () => record(null), // closed without choosing: saved as skipped
  )
}

const plugin: TuiPluginModule & { id: string } = { id: "lab.feedback", tui }

export default plugin
