// OpenCode server plugin: keeps a copy of every session in the lab repo, saved again after each answer.
//
// Installed into ~/.config/opencode/opencode.json by `lab use <profile>` (OpenCode harness). Server plugins load in
// every OpenCode process (the TUI, `opencode run`, the desktop app), so every session is kept, whatever the model.
// OpenCode has no "session ended" event: each save is the whole session so far, so the last one is the full log.
//
// Each file is metrics/opencode/chats/<session id>.json in `opencode export`'s format ({info, messages: [{info,
// parts}]}), so `opencode import <file>` brings a session back. A subagent's session gets its own file (its
// info.parentID names the parent). Deleting a session in OpenCode leaves its copy here.

import { mkdirSync, renameSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import type { Plugin, PluginModule } from "@opencode-ai/plugin"

type Options = { dir?: string }

const server: Plugin = async ({ client }, options) => {
  const dir = (options as Options | undefined)?.dir
  if (!dir) return {}
  const saving = new Map<string, Promise<void>>() // per session, so two saves of one session never interleave

  async function save(sessionID: string) {
    const [session, messages] = await Promise.all([
      client.session.get({ path: { id: sessionID }, throwOnError: true }),
      client.session.messages({ path: { id: sessionID }, throwOnError: true }),
    ])
    mkdirSync(dir!, { recursive: true })
    const file = join(dir!, `${sessionID}.json`)
    const temp = `${file}.${process.pid}.tmp`
    writeFileSync(temp, JSON.stringify({ info: session.data, messages: messages.data }, null, 2) + "\n")
    renameSync(temp, file) // readers never see half a file
  }

  return {
    async event({ event }) {
      if (event.type !== "session.idle") return
      const sessionID = event.properties.sessionID
      const next = (saving.get(sessionID) ?? Promise.resolve())
        .then(() => save(sessionID))
        .catch((error) =>
          client.app
            .log({ body: { service: "lab.archive", level: "error", message: `could not save ${sessionID}: ${error}` } })
            .then(
              () => {},
              () => {},
            ),
        )
        .finally(() => {
          if (saving.get(sessionID) === next) saving.delete(sessionID)
        })
      saving.set(sessionID, next)
    },
    async dispose() {
      await Promise.allSettled(saving.values()) // `opencode run` exits right after the answer
    },
  }
}

const plugin: PluginModule & { id: string } = { id: "lab.archive", server }

export default plugin
