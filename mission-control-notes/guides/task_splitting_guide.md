# Task splitting guide

How to split a Final spec into executable tasks. Task generation chats in
Mission Control are primed with this guide; follow it exactly. Edit this
repo's copy to customize the process.

---

## What a task is

One unit of autonomous work an execution agent completes on its own git
branch, ending in a pull request. Every task carries:

| Field | Rule |
|---|---|
| name | Short imperative phrase ("CRUD endpoints for protocols") |
| criteria | Testable acceptance criteria; cite the spec requirements (R#) |
| priority | P0 (blocking), P1 (normal), P2 (nice-to-have) |
| estimated_hours | 1 to 4 hours; split anything larger |
| touchpoints | Read/write paths the task will access |
| refs | Read-only paths the agent needs for context |
| depends_on | Tasks whose merged output this task builds on |
| task_type | "code" (default), or "checkpoint" for a manual review pause |

## Checkpoint tasks

A checkpoint is a deliberate pause for human review and steering. Insert
one with `task_type: "checkpoint"` when the plan reaches a point where a
human should inspect the work before later tasks build on it - after risky
or foundational work (schema migrations, public API shapes, core
abstractions many tasks consume).

How it runs in a sequential batch: on reaching the checkpoint, a review
agent inspects the shared branch, posts a plain-language summary, and the
batch pauses. The reviewer replies in that chat with feedback ("going
forward do X, Y, Z"); the review agent discusses over as many turns as
needed and applies the feedback to the remaining tasks - updating their
criteria and dependencies, replacing tasks, or inserting new ones. The
batch resumes only when the reviewer explicitly approves continuing.

- Its `criteria` are the review focus: what the reviewer (and the review
  agent summarizing for them) should inspect and confirm.
- Give it `depends_on` covering everything that must land before the
  review, and make later tasks depend on the checkpoint so they stay
  blocked until it completes.
- In auto-run (non-batch) mode the checkpoint sits Ready gating its
  dependents until the user marks it reviewed from the task table.
- Checkpoints need no `estimated_hours`, `touchpoints`, or `refs`.
- Use them sparingly - every checkpoint stops autonomous progress until a
  human responds.

## Splitting rules

1. **1-4 hours per task.** Bigger means split further; smaller means merge
   with a neighbor.
2. **Independently testable.** Each task's criteria can be verified without
   the unfinished work of others.
3. **Derive touchpoints from the repository**, not from guesswork: read the
   code and name the real paths.
4. **Declare touchpoint access.** Use
   `{"path": "backend/app/example.py", "access": "write"}` for a path the
   task creates or modifies, and use `access: "read"` for a path it must read
   while it runs. A bare path string remains shorthand for write access.
   Touchpoints overlap by directory prefix and glob matching. Write/write and
   read/write pairs conflict; read/read pairs do not.
5. **Merge tasks that share write paths.** If two tasks write the same file
   or overlapping directory, they are one unit of work - merge them before
   proposing the split. A task may share a read path with another reader.
   Use `refs` for contextual paths handed to the task but not accessed as
   part of its execution.
6. **Minimize dependency chains.** Prefer many independent tasks over long
   chains; a dependency exists only when a task consumes another's merged
   output.
7. **No cycles.** The dependency graph must be acyclic; the insert tool
   rejects cycles.
8. **Cover the whole spec.** Every requirement lands in exactly one task's
   criteria; note anything deliberately deferred.

## Process

1. Read the spec named by the user and the parts of the repository the
   work touches.
2. Draft the tasks, then check every pair of touchpoints. Merge tasks that
   share write paths so the set you present is already conflict-free.
3. Propose the split in chat as a table: id, name, hours, deps,
   touchpoints, criteria summary. Call out the critical path and any task
   that many others depend on.
4. Iterate with the user until they approve.
5. On approval, insert the final task set with the
   `mcp__mission-control__insert_tasks` tool. Reference batch-internal
   dependencies by list index (0-based) and existing tasks by short id.
   Tasks land as Backlog; the user marks them Ready.
6. Write a decision summary to
   `mission-control-notes/decisions/<YYYY-MM-DD>_<slug>.md` and commit it
   (additive, direct to the default branch).

## Targeted re-split

When asked to split one existing task, use
`mcp__mission-control__replace_task` with that task's short id. Divide the
criteria per the user's guidance, re-derive touchpoints from the repo, and
set dependencies between the replacement tasks.
