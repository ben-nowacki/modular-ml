# Working Efficiently — Context & Subagents

Applies to: every task. A smaller working context is faster, cheaper, and
sharper — a session bloated with raw file dumps and dead-end exploration reasons
worse, not better. Treat your context window as a scarce resource you spend
deliberately.

## Delegate bulky work to a subagent, keep only the conclusion

The single biggest lever: when a step will pull a lot of text into the session
but you only need its *conclusion*, run it in a subagent (Task facility) and let
the subagent's large context be discarded. Bring back a short, distilled answer.

Delegate to a subagent when the work is:

- **Codebase research / broad search** — "where is X handled", "list every call
  site of Y", "how does subsystem Z fit together". The subagent greps and reads
  dozens of files; you get a paragraph and the 3 file:line references that matter.
- **Enumerating capabilities** — e.g. building a spec's feature-exposure matrix
  by grepping routes/schemas/tools and scanning prior specs/decisions. That is a
  lot of reading for a compact list — ideal to delegate.
- **External library docs** — see the library-docs guide: fetch and distill in a
  subagent so the raw docs never land in your session.
- **Verification passes** — running a broad review/checklist over a diff.

Give the subagent a precise question and tell it exactly what shape to return
(a list, a table, specific fields). Its final message is data for you, not prose
for a human.

Note on backends: subagents are strongest on the Claude backend. If your toolset
has no subagent facility, do the work inline — but still apply the economy rules
below rather than reading everything.

## Economy rules for your own context

- **Search before you read.** Grep to the exact file and line; open the slice you
  need, not the whole file. Do not read a large file twice — remember what you saw.
- **Don't paste to think.** Avoid dumping large files, long command output, or
  whole docs into the conversation just to reason over them. Extract the relevant
  lines.
- **Reuse the notes.** `mission-control-notes/` already holds the repo summary,
  prior specs, decisions, and cached library notes — read those instead of
  re-deriving what the project already figured out.
- **Quote tests and errors tightly.** When a check fails, focus on the failing
  assertion and the relevant traceback frames, not the entire log.
- **Stop when you have the answer.** Do not keep exploring "to be thorough" once
  you can act. Breadth that does not change what you do is wasted context.
