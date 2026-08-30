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
| touchpoints | Files and directories the task will create or modify |
| refs | Read-only paths the agent needs for context |
| depends_on | Tasks whose merged output this task builds on |
| task_type | "code" (default), or "checkpoint" for a manual review pause |

## Checkpoint tasks

A checkpoint is a deliberate pause for human review and steering. Insert
one with `task_type: "checkpoint"` when the plan reaches a point where a
human should inspect the work before later tasks build on it — after risky
or foundational work (schema migrations, public API shapes, core
abstractions many tasks consume).

How it runs in a sequential batch: on reaching the checkpoint, a review
agent inspects the shared branch, posts a plain-language summary, and the
batch pauses. The reviewer replies in that chat with feedback ("going
forward do X, Y, Z"); the review agent discusses over as many turns as
needed and applies the feedback to the remaining tasks — updating their
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
- Use them sparingly — every checkpoint stops autonomous progress until a
  human responds.

## Splitting rules

1. **1-4 hours per task.** Bigger means split further; smaller means merge
   with a neighbor.
2. **Independently testable.** Each task's criteria can be verified without
   the unfinished work of others.
3. **Derive touchpoints from the repository**, not from guesswork: read the
   code and name the real paths.
4. **No two tasks may share a touchpoint.** Touchpoints gate parallel
   dispatch: two tasks whose touchpoints overlap by path prefix cannot run
   in parallel, and two tasks that edit the same file will conflict. If you
   find yourself giving two tasks the same file or directory (or one a
   directory that contains the other's file), that is a single unit of work
   — **merge them into one task** before proposing the split. Do this
   yourself; do not present overlapping tasks for the user to reconcile.
   The only exception is a task that inherits a file solely to read it: put
   that path in `refs`, not `touchpoints`.
5. **Minimize dependency chains.** Prefer many independent tasks over long
   chains; a dependency exists only when a task consumes another's merged
   output.
6. **No cycles.** The dependency graph must be acyclic; the insert tool
   rejects cycles.
7. **Cover the whole spec.** Every requirement lands in exactly one task's
   criteria; note anything deliberately deferred.

## Process

1. Read the spec named by the user and the parts of the repository the
   work touches.
2. Draft the tasks, then check every pair of touchpoints: merge any tasks
   that share a file or directory (rule 4) so the set you present is
   already conflict-free.
3. Propose the split in chat as a table: id, name, hours, deps,
   touchpoints, criteria summary. Call out the critical path and any task
   that many others depend on.
5. Iterate with the user until they approve.
6. On approval, insert the final task set with the
   `mcp__mission-control__insert_tasks` tool. Reference batch-internal
   dependencies by list index (0-based) and existing tasks by short id.
   Tasks land as Backlog; the user marks them Ready.
7. Write a decision summary to
   `mission-control-notes/decisions/<YYYY-MM-DD>_<slug>.md` and commit it
   (additive, direct to the default branch).

## Targeted re-split

When asked to split one existing task, use
`mcp__mission-control__replace_task` with that task's short id. Divide the
criteria per the user's guidance, re-derive touchpoints from the repo, and
set dependencies between the replacement tasks.
