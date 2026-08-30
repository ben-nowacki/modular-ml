# Spec writing guide

How to write a feature specification in this repository. Spec chats in
Mission Control are primed with this guide; follow it exactly. Edit this
repo's copy to customize the process.

---

## Process

1. Read the repository before writing. Ground every claim in real files;
   reference them by path.
2. Interview the user: clarify objective, scope, and constraints before
   drafting. Ask one focused question at a time.
3. Draft incrementally and confirm each section with the user before
   moving on.
4. A spec is **Draft** until the user says it is **Final**. Record the
   status in the header table.

## Required structure

Every spec is one markdown file with these sections, in order:

| Section | Contents |
|---|---|
| Header table | Status (Draft/Final), Date, Related specs |
| Objective | What the feature does and why, in one or two paragraphs |
| Scope | Explicit in-scope and out-of-scope bullet lists |
| Requirements | Numbered requirements (R1, R2, ...) that are testable |
| Interfaces and data | Data model, API shapes, file formats touched |
| Behavior details | Edge cases, state transitions, failure handling |
| Validation | How to verify: automated tests plus acceptance steps |
| References | Repo files, prior specs, external docs consulted |

## Conventions

- File name: `mission-control-notes/specs/spec_<slug>.md`, lowercase
  snake-case slug.
- Requirements are numbered (R1, R2, ...) so task splits can cite them.
- Prefer tables over prose for enumerable facts.
- Never invent repository details; read the files.

## On finalize

When the user declares the spec Final:

1. Set Status to Final in the header table.
2. Write the spec to `mission-control-notes/specs/spec_<slug>.md`.
3. Write a decision summary to
   `mission-control-notes/decisions/<YYYY-MM-DD>_<slug>.md` capturing the
   key decisions made in this chat and the alternatives rejected.
4. Verify additivity with `git status`: new files only. Commit them
   directly to the default branch. If you would modify an existing file,
   use a PR branch instead.
