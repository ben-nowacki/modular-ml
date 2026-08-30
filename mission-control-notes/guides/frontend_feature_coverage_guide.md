# Frontend Feature Coverage and UI Design Guide

A frontend must expose EVERY user-facing capability the backend provides for its
scope, and expose it in a way that matches how users actually work. This guide
defines a four-step UI design process for spec writing (feature inventory,
interaction analysis, grouping and presentation, layout) plus an execution-time
reachability check.

Applies at two points: **spec writing** (work the four design steps below into the
spec) and **execution/verify** (prove every mapped feature is reachable).

Two failures this guide prevents:

- A UI that answers the prompt literally while leaving real backend features
  (endpoints, actions, states, filters, permissions) unreachable because nobody
  enumerated them
- A UI that exposes everything as an undifferentiated pile of controls because
  nobody asked how each feature is actually used

---

## The four-step design process (spec writing)

Work the steps in order; each consumes the previous step's output. For a small
surface (for example, adding one control to an existing page) steps 2 and 3 can
each be a sentence or two, but the questions they ask must still be answered
explicitly. What is never optional: the Feature Exposure Matrix (step 1) and a
concrete frontend element for every user-facing row (filled in during step 3).

### Step 1: Feature inventory (the Feature Exposure Matrix)

Enumerate the backend capabilities in scope. Build the list by SEARCHING, not from
memory:

1. **Backend surface** - grep the API layer for every route/endpoint, request and
   response field, action, and state relevant to this feature (FastAPI routes,
   Pydantic models, WebSocket messages, MCP tools, enum/status values, permission
   checks). Note query params and filters - each is often a distinct UI control.
2. **Prior decisions** - read `mission-control-notes/specs/` and
   `mission-control-notes/decisions/` for features already agreed on but perhaps
   not yet built, and for explicit out-of-scope calls. Don't re-expose something a
   decision deliberately deferred; don't drop something a prior spec promised.
3. **Data model** - user-facing fields on the relevant models usually each need a
   display and, if editable, an input.

Then start the matrix - one row per capability. Leave the "Frontend element"
column empty for now: choosing the element is the OUTPUT of steps 2 and 3, not a
step-1 guess. Naming a widget here turns the remaining steps into rationalization
of a choice already made.

```markdown
| Backend capability (source) | User-facing? | Frontend element (step 3) | Status |
|---|---|---|---|
| POST /tasks (create) | yes | | to build |
| GET /tasks?status= filter | yes | | to build |
| task.priority field | yes | | to build |
| DELETE /tasks/{id} | yes | | to build |
| POST /tasks/{id}/internal-reindex | no (internal) | - | out of scope: not user-facing |
```

"User-facing?" forces an explicit call. Internal/admin/system endpoints can be
`no`, but say so - an unexplained omission is the bug this guide prevents.

### Step 2: Interaction analysis

For each `yes` row, work out how the user will actually use the feature. Answer
four dimensions per feature (extra matrix columns or a second small table - the
format matters less than the answers being explicit):

- **Frequency** - constant, per-session, occasional, or rare
- **Context** - what is the user in the middle of when they need this?
- **Mode** - glance at it, act on one item, act in bulk, or configure once
- **Weight** - routine, or destructive/irreversible (drives confirmation and
  placement)

Then name the **primary workflows** (2 to 4) the surface serves: the chains of
features a user strings together to get something done (for example: triage the
failed run --> read its logs --> retry it). Users do tasks, not features;
workflows are what step 3 must optimize for.

### Step 3: Grouping, ordering, and presentation

Decide the information architecture from step 2's answers:

- **Group by workflow**, not merely by backend resource. The next thing a user
  needs should sit next to the thing they just used.
- **Order and weight by frequency and context**: constant-use features get
  primary placement; rare or configure-once features go behind progressive
  disclosure (menus, settings panels, expanders) rather than crowding the
  surface.
- **Derive each control from its mode**: glance --> display; act on one -->
  row/inline action; bulk --> selection plus toolbar action; configure once -->
  settings form. Destructive weight adds confirmation and visual separation from
  routine actions.
- **Walk each step-2 workflow** through the proposed arrangement and check that
  nothing forces a detour across the surface.

Now fill in the matrix's "Frontend element" column: every `yes` row must name a
concrete frontend element (component, control, route, state), or be explicitly
deferred with a reason. No silent gaps.

```markdown
| Backend capability (source) | User-facing? | Frontend element (step 3) | Status |
|---|---|---|---|
| POST /tasks (create) | yes | "New task" button --> NewTaskModal | to build |
| GET /tasks?status= filter | yes | Status filter dropdown in TaskTable | to build |
| task.priority field | yes | Priority column + editable Select | to build |
| DELETE /tasks/{id} | yes | Row action menu --> Delete (confirm dialog) | to build |
```

### Step 4: Page and surface layout

Design the full layout of each page or surface: which regions exist, what goes in
each, and how the step-3 groups map onto them. Record the result in the spec - a
region-by-region description or an ASCII sketch is enough.

Do the visual and component-level design against this project's design system:
`mission-control-notes/design_system.md` (tokens plus Radix primitive kit) and the
frontend guides named by `mission-control-notes/guides/index.md`. This guide
governs what appears where and why; those govern how it looks and how it is
built.

### Splitting into tasks

When splitting the spec into tasks, ensure every unbuilt `yes` row is covered by
some task's touchpoints/acceptance criteria - coverage rows become work.

---

## Execution / verify: prove reachability

Before finishing a frontend task, confirm every capability the task's exposure
matrix rows assign to it is actually **reachable in the running UI** - not merely
present in a file:

- The element renders on a real path a user can navigate to (not dead code, not an
  unrendered component, not behind a route that nothing links to)
- It is wired to the backend (calls the real endpoint; handles loading, error,
  empty, and success states plus the permission/disabled states the backend can
  return)
- Nothing in scope is stranded: re-grep the backend surface for this feature and
  check each user-facing item against what the UI now exposes

If you find an in-scope user-facing capability with no UI, build it or, if it is
genuinely out of scope, record that explicitly - never leave it silently
unexposed.
