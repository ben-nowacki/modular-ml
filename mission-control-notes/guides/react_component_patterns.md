# React Component Patterns

How to structure React components, hooks, and pages so new UI stays consistent with
the codebase. The conventions below are stack-current; the concrete file paths,
token names, and primitive kit are **this project's** — read
`mission-control-notes/design_system.md` (the design-system manifest) and match the
files already in the repo rather than the illustrative names here.

**Stack:** React 18+ (function components + hooks) with TypeScript, a bundler
(Vite/Next), a router, and **TanStack Query v5** for server state. Styling is
**Radix UI + Tailwind** per the manifest. Interactive primitives (dialog, menu,
select, tooltip, popover, tabs, …) come from the project's shared Radix kit — never
hand-rolled. Related guides under `mission-control-notes/guides/`:
`tailwind_ui_patterns.md` (styling mechanics), `frontend_design_guide.md` (visual
judgment), `frontend_testing_guide.md` (tests + gates), and any API-response guide
the project ships.

---

## 1. File & folder conventions

Match the layout already in `src/`. A typical shape:

```
src/
  api/          The service layer — the ONE place fetch() lives (see §3)
  components/
    ui/         Low-level Radix + Tailwind primitives (Button, Dialog, Select…)
    shared/     Composed cross-feature widgets (DataTable, EmptyState, Modal…)
    layout/     App chrome (Sidebar, PageHeading)
  contexts/     React Context providers (e.g. AuthContext)
  hooks/        Reusable query/mutation and UI hooks
  lib/          Framework-agnostic helpers (utils.ts → cn())
  pages/        Route-level screens, grouped by area, tests colocated in __tests__/
  styles/       Tailwind theme + tokens (the manifest's token file)
```

- **Components are `PascalCase.tsx`**; non-component modules are `camelCase.ts`.
  Match the neighbouring files in a folder rather than renaming to a different
  convention.
- **`ui/` vs `shared/`:** `ui/` holds dumb, styling-only Radix primitives with no
  app knowledge; `shared/` holds composed widgets that know about app concerns
  (loading/empty/permissions). Page-specific pieces live under that page's folder,
  not in `shared/` — promote to `shared/` only on the second consumer.
- **Import via the path alias** (`@/…`), never long relative chains.
- **Never hand-edit generated API clients/types** if the project generates them;
  add hand-written clients alongside as `api/<feature>.ts`.
- **Colocate tests** in `__tests__/` next to what they cover.

---

## 2. Custom hooks (server state via TanStack Query)

Wrap TanStack Query so components never touch fetching mechanics. Keep each hook a
thin, single-purpose query/mutation wrapper — not a monolithic god-hook.

```ts
export function useAgents(
  options?: Partial<UseQueryOptions<{ agents: AgentSummary[] }>>,
) {
  return useQuery({
    queryKey: ['agents'],   // stable, serializable; first element = cache namespace
    queryFn: fetchAgents,   // a plain async fn from the same module
    staleTime: STALE.SHORT, // pick a tier by data volatility
    ...options,             // let callers override (enabled, refetch, …)
  })
}
```

Conventions:

- **Set `staleTime` in the hook, not the call site**, from the project's shared
  tiers. **Spread `...options` last** so callers can pass `enabled`, `onSuccess`,
  etc.; use `enabled: !!id` to defer a query until its input exists.
- **Query keys are arrays** whose first element namespaces the cache; include every
  input that changes the result so caching and invalidation are correct.
- **Mutations return `useMutation`** wrappers typed `<Data, Error, Vars>`, and on
  success **invalidate the affected query keys** via
  `useQueryClient().invalidateQueries(...)` so the UI re-reads fresh data. Don't
  hand-mutate the cache without a measured reason.
- **Keep the fetch function separate from the hook** (`fetchAgents` + `useAgents`)
  so it can also be called imperatively.
- **Prefer a generated hook** where the project generates one; hand-write a hook
  only for endpoints not yet covered.

---

## 3. The API service layer (never inline `fetch`)

**`fetch()` lives in exactly one place** — a shared client module (e.g.
`api/client.ts`). Components and hooks call generated hooks or hand-written `api/*`
functions, never `fetch` directly.

- **One mutator** handles auth (send the session cookie / token), JSON parsing,
  empty-body (`204`) handling, and normalizing errors to a thrown `Error` carrying
  `status` + `data`. Hand-written clients call it too.
- **The one legitimate raw-`fetch` exception is multipart upload** (a `FormData`
  body must let the browser set its own `Content-Type` boundary) — and even then it
  replicates the shared cookie + error-shape contract.
- **Components import hooks, not URLs.** A component should never contain a path
  string or HTTP method — that belongs in `api/`.
- **Types flow from the backend.** Prefer generated types; hand-written clients
  mirror the backend response field-for-field and are updated when it changes.

---

## 4. Loading, error, and empty states

Handle all three explicitly and consistently — a component that only renders the
happy path is incomplete. Use the project's shared surfaces so every page matches.

- **A shared empty/placeholder surface** (e.g. `<EmptyState message description
  action />`) is the canonical empty state and doubles as a lightweight loading /
  no-access surface.
- **A shared data-table wrapper** bakes in the loading and empty branches for
  tabular data — pass `loading` and an `emptyMessage` rather than re-implementing
  table states.
- **Route-level auth/loading screens** live in the route guard (§6), not duplicated
  per page.

```tsx
const { data = [], isLoading, isError } = useThing()

if (!canView) return <EmptyState message="You don't have access to this view." />
if (isError)  return <EmptyState message="Couldn't load…" description="Try again." />
// For tables, hand isLoading/empty to the shared DataTable instead of branching:
return <DataTable rows={data} loading={isLoading} emptyMessage="No items yet." />
```

- **Distinguish empty from loading from error** — never show "No data" while a
  request is in flight, never a blank screen on error. Branch on `isLoading` /
  `isError` / `data`.
- **Provide a default for list data** (`const { data = [] } = useThing()`) so the
  first render is safe.
- **Surface mutation errors inline** near the action (an inline banner on the
  `destructive` token), not via `alert()`.
- **Style with the manifest's tokens and primitives** — semantic tokens, `cn()`,
  the shared kit — never raw hex/palette (`tailwind_ui_patterns.md`).

---

## 5. State: Context vs. local vs. server cache

Pick the narrowest scope that works. Three tiers, in order of preference:

1. **Server state → TanStack Query** (default for anything from the API). Cache,
   loading, and refetch are the query's job — **do not copy fetched data into
   `useState`.** Read it from the hook; derive view-model shapes with `useMemo`.
2. **Local UI state → `useState` / `useReducer`** for ephemeral, component-scoped
   values: dialog open/closed, form fields, selected tab, filter toggles. Keep it
   in the lowest component that needs it; lift only when a sibling needs it too.
3. **React Context → only for truly global, low-churn, cross-cutting state** (e.g.
   `AuthContext`). Reach for it when data is needed by many unrelated subtrees *and*
   changes rarely. Don't put fast-changing or server-fetched data in Context — it
   re-renders the whole tree and duplicates the query cache.

Anti-patterns: a "global store" mirroring API data; Context for a value two
adjacent components share (lift state or pass a prop); `useEffect` + `useState` to
fetch (use a query hook). Consume context through its dedicated hook (e.g.
`useAuth()`), which should throw if used outside its provider — never
`useContext(...)` directly.

---

## 6. Authentication & route protection

Centralize auth in a context + a route guard; individual pages assume they only
render for an allowed user and don't re-check the session.

- **An `AuthProvider`** bootstraps the session once and exposes `user`,
  `isLoading`, the derived role, and helper predicates (`hasPermission`, `isAdmin`,
  …).
- **Wrap every authenticated route in a `<ProtectedRoute>`** that runs the gate
  chain in the same order the backend lifts restrictions: loading → not-initialized
  → redirect-to-login → required setup steps (email/password/MFA) → access checks.
  Props select the requirement (`adminOnly`, `permission="…"`, etc.).
- **Public auth pages** (login/register) are wrapped in a gate that bounces
  already-authenticated users home.
- **Failed checks render a shared screen** — a page never renders its content for
  an unauthorized user, so page code can assume `user` exists.

---

## 7. Permission-based component visibility

Route protection decides whether a *page* renders; within an allowed page,
show/hide individual controls with the auth-context helpers. The permission source
of truth is the backend.

- **Gate an action on its own permission key**, matching the backend key the
  endpoint enforces (`hasPermission('protocol.run')`). Don't infer capability from
  role when a specific permission exists.
- **Conditional rendering** is the norm — `{canEdit && <EditButton/>}`, or build
  nav lists by spreading in items conditionally.
- **Compute permission booleans once and memoize** when they feed a hot render path
  (a stable `perms` object handed to many memoized rows — a fresh object every poll
  re-renders them all).
- **Gate the query too, not just the button.** Pass `enabled: canView` to a hook so
  a user who can't see a resource never fires its request.
- **Hiding a control is UX, not security.** The backend enforces every permission
  independently; client gating just avoids showing actions that would 403.

---

## 8. Checklist for a new component / page

- [ ] File in the right place (`pages/<area>/`, `shared` vs `ui`), `PascalCase.tsx`,
      imports via the alias.
- [ ] Data comes from a query/mutation hook — **no inline `fetch`**, no path
      strings or HTTP methods in the component.
- [ ] Server data read from the query cache (not copied into `useState`); local
      state kept low; Context only for global low-churn state.
- [ ] Interactive primitives from the shared Radix kit; styling via manifest tokens
      and `cn()` — no raw hex/palette, no hand-rolled dialogs/menus/selects.
- [ ] Loading, error, and empty states all handled via the shared surfaces;
      mutation errors shown inline.
- [ ] Route wrapped in the guard with the correct permission/role; no per-page
      session re-checks; in-page controls gated on the matching backend key and
      queries `enabled`-gated.
- [ ] New/changed API shape → client + types regenerated/updated.
- [ ] Tests added per `frontend_testing_guide.md` (all gates green); visual and
      interaction states per `frontend_design_guide.md`.
- [ ] No AI attribution in any commit, PR, comment, or file.
