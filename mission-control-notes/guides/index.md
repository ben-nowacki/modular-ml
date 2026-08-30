# AmpWell Developer Guides — Index

Read this file at the start of every task. For each section below, if your task
Global repository-history rule: never attribute work to Claude, Codex, AI, or
an agent. Do not add co-author trailers, generated-by notices, signatures, or
similar attribution to commits, pull requests, comments, or documentation.

involves that domain, read the linked guide before writing any code.

---

## Working Efficiently (context & subagents)

Applies to: every task. Keep your working context small — delegate bulky research,
capability enumeration, doc lookups, and verification to subagents and bring back
only the distilled result; search before you read; reuse the notes.

- `mission-control-notes/guides/context_efficiency_guide.md`

---

## Test-Driven Development & Debugging

Applies to: every task that writes or changes code, and every bug fix. Drive each
change with a failing test first (RED → GREEN → REFACTOR); fix bugs at the root
cause with a regression test, not by patching symptoms.

- `mission-control-notes/guides/tdd_and_debugging_guide.md`

---

## External Libraries & Current APIs

Applies to: any task that imports, adds, upgrades, or integrates a third-party
library/framework, or uses an external API you are not certain is current. Consult
current docs (Context7) instead of relying on model memory — selectively, via a
subagent, cached to `mission-control-notes/libs/`.

- `mission-control-notes/guides/library_docs_guide.md`

---

## Docstrings & Comments

Applies to: any file where you are creating or editing functions, classes, methods,
or inline comments.

- Python files → `mission-control-notes/guides/python_docstring_style.md`
- JavaScript or TypeScript files → `mission-control-notes/guides/typescript_docstring_style.md`

---

## Documentation Files

Applies to: any task that creates or modifies a `.md` or `.html` documentation file,
including user guides, developer references, and management or IT documents.

- `mission-control-notes/guides/documentation_writing_guide.md`

---

## Error Handling

Applies to: any Python file where you are raising, catching, or propagating exceptions,
or adding error paths to existing logic.

- `mission-control-notes/guides/python_error_handling.md`

---

## Database Migrations

Applies to: any task that creates or modifies a file in `backend/alembic/versions/`,
or changes `backend/app/models.py` in a way that requires a migration.

- `mission-control-notes/guides/alembic_migration_guide.md`

---

## API Design

Applies to: any task that adds or changes a FastAPI route, Pydantic request or response
schema, or WebSocket message structure.

- `mission-control-notes/guides/api_response_patterns.md`

---

## Git & Pull Requests

Applies to: every task, before making any commit or opening a PR.

- `mission-control-notes/guides/git_branch_and_pr_guide.md`

---

## Testing (Python)

Applies to: any task that creates or modifies a file in `backend/tests/`, or adds a
`conftest.py`, fixture, or test utility.

- `mission-control-notes/guides/python_testing_guide.md`

---

## Design System (read FIRST for any UI work)

Applies to: any task that creates or modifies UI. This project's frontend is built
on **Radix UI primitives + Tailwind CSS**, against the theme recorded in
`mission-control-notes/design_system.md`. Read the manifest before writing UI and
conform to it; if it is missing or `not-established`, do not invent a one-off
style — bootstrap the design system first (and, as an execution agent, surface
`NEED_INPUT:` rather than guessing a theme).

- `mission-control-notes/guides/design_system_manifest_guide.md` (read + conform)
- `mission-control-notes/guides/design_system_bootstrap_guide.md` (when establishing/redesigning the system)

---

## Frontend Feature Coverage & UI Design Process

Applies to: any task that builds or changes user-facing UI, and any spec that
includes UI. Every user-facing backend capability in scope must be exposed in the
frontend (no silent gaps), and UI surfaces are designed via the guide's four-step
process: feature inventory, interaction analysis, grouping/presentation, layout.

- `mission-control-notes/guides/frontend_feature_coverage_guide.md`

---

## Frontend Testing

Applies to: any task that creates or modifies a frontend test (`frontend/src/**/
__tests__/`), or that adds or changes frontend code — new components, hooks, or
API-layer functions need tests, and changed code must keep existing tests and the
typecheck/lint gates green.

- `mission-control-notes/guides/frontend_testing_guide.md`

---

## React Components & Hooks

Applies to: any task that creates or modifies a file in `frontend/src/`, including
components, pages, hooks, and the API service layer.

- `mission-control-notes/guides/react_component_patterns.md`

---

## Frontend Design Quality

Applies to: any task that creates new UI or visually modifies existing UI — new
components or pages, layout changes, styling, and interaction states. Design against
this project's Radix + Tailwind design system (see Design System above).

- `mission-control-notes/guides/frontend_design_guide.md`

---

## Async Python

Applies to: any task involving `async def` functions, `asyncio` primitives, background
tasks, or the long-poll endpoint.

- `mission-control-notes/guides/python_async_patterns.md`

---

## Security

Applies to: every task, before opening a PR. Confirm each checklist item is satisfied
or explicitly not applicable to your changes.

- `mission-control-notes/guides/security_checklist.md`

---

## Tailwind & UI Styling

Applies to: any task that writes or modifies Tailwind classes, shared component
variants, or status color conventions. Use semantic tokens from the design-system
manifest and Radix primitives for interactive elements; never raw hex/palette
classes, shadcn dumps, CSS modules, styled-components, or ad-hoc CSS.

- `mission-control-notes/guides/tailwind_ui_patterns.md`

---

## Database Queries

Applies to: any task that writes a SQLAlchemy `select()`, join, eager load, or
pagination query.

- `mission-control-notes/guides/database_query_patterns.md`

---

## Logging

Applies to: any task that adds, removes, or modifies a log statement in any file.

- `mission-control-notes/guides/logging_and_observability.md`

---

## Planning documents (not for execution tasks)

`mission-control-notes/guides/spec_writing_guide.md` and
`mission-control-notes/guides/task_splitting_guide.md` prime the spec and split
chats. Execution tasks do not need to read them.
