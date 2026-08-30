# Frontend Testing Guide

Reference material for writing and maintaining tests in the frontend (`src/`). It
documents the conventions the suite follows and the checks every frontend change
must pass before it is finished. Match the exact runner and scripts the project
already uses; the commands below are the common case.

**Stack:** Vitest + React Testing Library + user-event, jsdom environment,
TypeScript strict. The UI is **Radix UI + Tailwind** (per
`mission-control-notes/design_system.md`) — Radix primitives render real roles and
accessible names, which is exactly what these tests query by (§3). Tests are
colocated in `__tests__/` folders next to the code they cover (see
`react_component_patterns.md` §1).

---

## 1. The three checks — run all of them, every change

A frontend change is not done until all three are green (run from `frontend/`):

```bash
npx vitest run          # the test suite (or the package.json "test" script)
npx tsc --noEmit        # typecheck — the compiler is part of the test suite here
npm run lint            # ESLint
```

Prefer the repo's `package.json` scripts when they exist (`npm test`,
`npm run typecheck`, `npm run lint`) — they carry the project's exact flags.
Run the checks that cover what you touched at minimum; run all three before
claiming a task complete. A change that "works in the browser" but fails
`tsc --noEmit` is a failing change.

---

## 2. Test placement & naming

- **Colocate**: `pages/equipment/__tests__/EquipmentOverview.test.tsx` sits next
  to `pages/equipment/EquipmentOverview.tsx`. Never a parallel top-level
  `tests/` tree.
- **Name for the file under test**: `<Name>.test.tsx` for components,
  `<name>.test.ts` for plain modules (API clients, helpers).
- **One behavior per test; name it as an assertion**:
  `test('disables submit while the mutation is pending')` — not
  `test('submit button')`.
- Shared render helpers/fixtures live in a non-test module (e.g.
  `src/test/utils.tsx`), imported by suites — never copy-pasted between files.

---

## 3. React Testing Library conventions

Test what the user sees and does, not the component's internals.

- **Query priority** (accessibility-first, in order): `getByRole` (with `name`)
  → `getByLabelText` → `getByPlaceholderText` → `getByText`. Reach for
  `getByTestId` only when no accessible query can address the element — and
  treat that as a hint the markup is missing a role or label.
- **Radix makes `getByRole` reliable.** A Radix Dialog, Menu, Select, Tab, or
  Switch exposes the correct role and accessible name, so query the trigger and
  its content by role/name (`getByRole('dialog')`, `getByRole('menuitem', { name
  })`, `getByRole('button', { name: 'Delete' })`) rather than by test id or class.
  Overlay content is portalled — assert against `screen` (the whole document), not
  the render container. If a Radix control needs `aria-label` to have a name (an
  icon-only trigger), the test that queries by name also guards that it is set.
- **`userEvent`, not `fireEvent`**: `await userEvent.click(button)` /
  `await userEvent.type(input, 'x')` simulate real interaction sequences
  (focus, keydown, etc.). `fireEvent` is a last resort for events user-event
  cannot produce.
- **Async assertions**: use `await screen.findByRole(...)` for elements that
  appear after data resolves, and `waitFor` for non-element conditions. Never
  assert immediately after an interaction that triggers async work, and never
  sprinkle arbitrary timeouts.
- **No implementation-detail assertions**: don't assert on state variables,
  hook internals, or that a specific child component rendered — assert on the
  rendered output and observable behavior. Tests coupled to internals break on
  every refactor without catching bugs.

---

## 4. Testing components that fetch (TanStack Query)

Components read server state through query hooks (`react_component_patterns.md`
§2-3), so tests must provide a QueryClient and control the API layer.

- **Fresh `QueryClient` per test**, retries off, wrapped around the render:

```tsx
function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
  )
}
```

  A shared client between tests leaks cache across them; retries turn a clean
  error-branch test into a multi-second timeout.
- **Mock the `api/` layer, never global `fetch`**: `vi.mock('@/api/registry')`
  (or mock `apiFetch` in `@/api/client` for broad coverage). The api module is
  the app's seam — mocking `fetch` re-tests `apiFetch`'s parsing in every suite
  and breaks when headers/cookies change.
- **Cover all three branches**: loading (mock returns a pending promise), error
  (mock rejects — assert the `EmptyState`/inline error renders, not a blank
  screen), and data (including `[]` → the empty message). A component test that
  only covers the happy path is incomplete, same rule as the component itself
  (`react_component_patterns.md` §4).
- **Mutations**: assert the visible result of success (dialog closes, row
  appears, invalidated list refetches) and of failure (inline error shown,
  form still editable). Pending state — a disabled button/spinner — is part of
  the contract; test it.

---

## 5. What to test for a new component or page

Minimum coverage for new UI:

1. **Renders its data** given a successful query (the happy path).
2. **Loading and error branches** render the shared surfaces, not blanks.
3. **Empty data** renders the empty message (empty ≠ loading ≠ error).
4. **Permission gating**: with the permission absent, the gated control is not
   in the document; with it present, it is (mock `useAuth`/`hasPermission`).
5. **Primary interaction**: the main action calls the right mutation with the
   right payload and the UI reflects success and failure.

Pure logic (formatting, sorting, view-model derivation) belongs in plain
functions with direct unit tests — don't test logic only through component
renders.

---

## 6. ESLint & TypeScript discipline

The compiler and linter are quality gates, not suggestions:

- **Zero new warnings.** A change that adds ESLint warnings is not done.
- **No `any`, no `@ts-ignore`/`@ts-expect-error`** to silence a type error you
  don't understand — fix the types. If a suppression is genuinely unavoidable
  (a third-party gap), it needs a comment stating why.
- **No `eslint-disable`** without a same-line comment justifying it.
- **Don't weaken types to make a test compile** — fix the test (or the type).
  Casting through `as unknown as X` in tests hides exactly the breakage the
  typecheck exists to catch.

---

## 7. Checklist for a frontend change

- [ ] Tests colocated in `__tests__/`, named `<Name>.test.tsx`, one behavior
      per test.
- [ ] Queries follow the RTL priority; interactions via `userEvent`; async via
      `findBy*`/`waitFor`.
- [ ] Fetching components tested with a fresh retry-off `QueryClient`; the
      `api/` layer mocked, not `fetch`.
- [ ] Loading, error, empty, and permission branches all covered.
- [ ] Logic-heavy helpers unit-tested directly as pure functions.
- [ ] Radix components queried by role/accessible name; portalled overlays
      asserted against `screen`.
- [ ] `npx vitest run`, `npx tsc --noEmit`, and `npm run lint` all green — run
      locally before opening the PR.
- [ ] No AI attribution in any commit, PR, comment, or file.
