# External Libraries & Current APIs (Context7)

Applies to: any task that imports, adds, upgrades, or integrates a third-party
library or framework, or uses an external API you are not certain is current.

Your training data lags real libraries. Guessing at an API you half-remember
produces code that fails to compile, fails tests, or silently uses a deprecated
pattern — and then you burn a re-read / retry loop fixing it. When you touch an
external dependency, get the *current* facts instead of guessing.

## The tool: Context7

Mission Control wires the **Context7** MCP server into this session (both agent
backends). It serves up-to-date, version-specific docs and code examples. Two
tools:

- `resolve-library-id` — turn a library name (e.g. "react-router", "fastapi")
  into a Context7 id.
- `get-library-docs` — fetch docs for that id, optionally scoped to a topic and
  a version.

You do not need a slash command or a magic phrase; call the tools directly when
the situation below applies.

## When to consult it (be selective — this is the token-critical part)

Consult Context7 when **all** of these are true:

- You are using a **third-party** library/framework (not the language standard
  library and not this repo's own code), **and**
- you are **adding or upgrading** it, integrating an unfamiliar part of it, or
  you are **not confident** the API you are about to write is current, **and**
- the answer is **not already cached** in this repo (see below).

Do **not** fetch docs for well-known, stable APIs you are sure of, for the
standard library, or "just in case." Every fetch costs tokens; an unnecessary
one cuts against keeping the session small.

## Keep the docs OUT of your main context

`get-library-docs` can return thousands of tokens. Do not dump that into the
working session you are trying to keep small:

- **Prefer a subagent.** When your toolset has a subagent/Task facility, spawn a
  subagent to run `resolve-library-id` + `get-library-docs`, read the payload,
  and return only the distilled answer — the specific current API signature,
  the correct import, the 5-line usage snippet, gotchas. The bulky docs die with
  the subagent; only the conclusion reaches you.
- **Scope the fetch.** Pass a specific topic and the version you actually use
  (read it from `package.json` / `pyproject.toml` / lockfile) so you get the
  relevant slice, not the whole manual.

## Cache what you learned (fetch once per project, not once per session)

After you distill a library's current usage, write it to
`mission-control-notes/libs/<library>.md`: the version in use, the confirmed
API/imports, a short working snippet, and any gotchas. Commit it (additive — new
file, direct to the default branch).

**Before fetching, check `mission-control-notes/libs/` first.** If a current
note for this library and version already exists, use it and skip the fetch.
This turns Context7 into a one-time cost per library per project instead of a
per-session tax. If the cached note is for an older version than the repo now
uses, refresh it and update the note.
