# Test-Driven Development & Debugging

Applies to: every task that writes or changes code, and every bug fix. This is a
*process* guide — it governs the order you work in, not just what you deliver.

## Test-driven development (RED → GREEN → REFACTOR)

Write the test before the implementation. The discipline is not "tests exist at
the end"; it is that a failing test drives each change.

1. **RED** — Write (or extend) a test that expresses the new behavior or pins the
   bug. Run it and *watch it fail for the right reason*. A test that passes
   before you have written any code is testing nothing — fix the test.
2. **GREEN** — Write the minimum code needed to make that test pass. Run it and
   see it pass. Do not add behavior no test demands yet.
3. **REFACTOR** — With the test green, clean up names, duplication, and shape.
   Re-run the test; it must stay green.

Rules:

- **Never write the assertion to match code you already wrote.** Tests authored
  after the fact tend to encode whatever the code happens to do, including its
  bugs. If you must add code first (spike), delete the spike and re-drive it
  test-first, or at minimum change the code and confirm the test then fails.
- One behavior at a time. Small RED→GREEN loops beat one giant test at the end.
- Test behavior and contracts, not private internals — so a refactor that keeps
  behavior keeps the tests green.
- A bug fix starts with a **failing regression test** that reproduces the bug.
  The fix is done when that test — and the rest of the suite — is green.
- Match the repo's existing test framework, layout, and conventions. Read the
  project's testing guide (Python/frontend) before writing tests.

## Debugging (root cause before fix)

Do not pattern-match a fix onto a symptom. Work in phases:

1. **Reproduce** — Get a reliable, ideally automated reproduction (a failing
   test is best). If you cannot reproduce it, you cannot know you fixed it.
2. **Investigate the root cause** — Read the actual code path, inspect real
   state (logs, values, types), and find *why* it happens. Form one hypothesis
   the evidence supports. Do not guess-and-check edits.
3. **Fix at the root** — Change the underlying cause, not the surface symptom.
   Add the regression test from the Reproduce step if you have not already.
4. **Verify** — The regression test passes, the full relevant suite passes, and
   you have not introduced new failures.

**Three-strikes rule:** if three genuine fix attempts have not worked, stop
editing. Your model of the problem is wrong. Step back, re-investigate from
phase 2, and reconsider the design/architecture rather than trying a fourth
variation. For an autonomous execution agent, if the design itself looks wrong,
surface `NEED_INPUT:` with what you found instead of thrashing.
