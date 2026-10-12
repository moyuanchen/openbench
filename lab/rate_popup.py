"""The rating pop-up for Gemini CLI: a small window asking for a 1–4 rating of the answer that just finished.

Started in the background by the AfterAgent hook (lab/gemini_hook.py). Choosing saves one line to
metrics/feedback.jsonl; Skip, Esc or closing the window saves it as skipped, like OpenCode's dialog. A newer
pop-up replaces an unanswered one, and an ignored one closes by itself after a while; neither saves anything.

Standard library only (tkinter ships with lab's uv-managed Python on macOS, Linux and Windows).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run as a script with -I: find gemini_hook next to it
import gemini_hook  # noqa: E402

TIMEOUT_MS = 15 * 60 * 1000
RATINGS = [(4, "4  Good as is"), (3, "3  Small fixes needed"), (2, "2  Big fixes needed"), (1, "1  Wrong or unusable")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="")
    parser.add_argument("--answered-at", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()
    try:
        ask(args)
    finally:
        _forget_pid()


def ask(args) -> None:
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError:
        return  # no screen after all
    root.title("openrate")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    done = False

    def record(rating: int | None) -> None:
        nonlocal done
        if not done:
            done = True
            gemini_hook.save({"ts": gemini_hook._now(), "harness": "gemini", "profile": args.profile or None,
                              "session_id": args.session, "rated_at": args.answered_at, "via": "popup",
                              "prompt": args.prompt, "rating": rating})
        root.destroy()

    frame = tk.Frame(root, padx=16, pady=12)
    frame.pack()
    tk.Label(frame, text="How good was this response?", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
    if args.prompt:
        excerpt = " ".join(args.prompt.split())
        tk.Label(frame, text=excerpt[:120] + ("…" if len(excerpt) > 120 else ""), fg="gray", wraplength=320,
                 justify="left").pack(anchor="w", pady=(2, 8))
    for value, title in RATINGS:
        tk.Button(frame, text=title, anchor="w", width=24, command=lambda v=value: record(v)).pack(fill="x", pady=1)
    tk.Button(frame, text="Skip  (Esc)", command=lambda: record(None)).pack(fill="x", pady=(8, 0))
    for value, _ in RATINGS:
        root.bind(str(value), lambda _e, v=value: record(v))
    root.bind("<Escape>", lambda _e: record(None))
    root.protocol("WM_DELETE_WINDOW", lambda: record(None))
    root.after(TIMEOUT_MS, root.destroy)  # ignored: close without saving
    if os.environ.get("OPENRATE_POPUP_TEST_CHOICE"):  # tests: pick a rating without a person
        choice = os.environ["OPENRATE_POPUP_TEST_CHOICE"]
        root.after(300, lambda: record(None if choice == "skip" else int(choice)))
    root.lift()
    root.mainloop()


def _forget_pid() -> None:
    try:
        if gemini_hook.POPUP_PID.read_text().strip() == str(os.getpid()):
            gemini_hook.POPUP_PID.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
