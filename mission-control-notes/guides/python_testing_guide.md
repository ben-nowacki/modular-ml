# Python Testing Guide

Reference material for writing pytest tests in the AmpWell backend
(`backend/tests/`). It documents the conventions the suite already follows and
the practices to keep it consistent as it grows.

Run the suite with `uv run pytest` (from `backend/`). Deselect the DB/warehouse
integration tests with `-m 'not integration'`. Config lives in
`backend/pyproject.toml` under `[tool.pytest.ini_options]`
(`testpaths = ["tests"]`, `asyncio_mode = "auto"`, the `integration` marker).

> **Note on scope.** The backend uses **synchronous SQLAlchemy** (`Session`), so
> route-handler tests drive a real session rather than mocking one (§2).
> Coverage is **not yet measured** in this repo (no `pytest-cov`, no gate) — §4 is
> a plan for adopting it, and is written as best-practice guidance to implement,
> not a description of existing config. Everything else describes the suite as it
> stands today.

---

## 1. Fixture structure & `conftest.py` layout

There is **one** `conftest.py`, at `backend/tests/conftest.py`. It holds every
shared fixture; there are no nested per-directory conftests (the suite is flat).
Its module docstring is the canonical fixture reference — keep it updated when you
add a fixture.

### The core fixtures and their scopes

| Fixture | Scope | What it gives you |
|---|---|---|
| `db_engine` | session | One SQLAlchemy engine for the whole run, from `settings.database_url` |
| `session_factory` | session | `sessionmaker` bound to that engine (`expire_on_commit=False`) |
| `db_session` | function | A fresh `Session` per test; **closed, not rolled back**, so in-test commits persist for assertions |
| `test_org` | module | An isolated `Organization` (unique slug), CASCADE-deleted after the module |
| `test_user` | module | A `User` with an `org_admin` role inside `test_org` (so permission checks pass) |
| `client` | module | `TestClient` with `get_db` + `get_auth_with_org` overridden |
| `_disable_model_builds` | session, autouse | Globally switches off real Docker model builds |
| `fake_email` | function | Installs an in-memory email backend; read `backend.sent` to assert on links |
| `server_config` | function | `set(**keys)` callable for `server_config` flags, auto-restored after the test |

Key design points to preserve:

- **Tests run against a real, migrated PostgreSQL — not SQLite, not a mock.**
  `client` overrides `get_db` with the real `session_factory`, and
  `get_auth_with_org` to return `(test_user, test_org)` without a browser session.
  Permission checks still hit the real DB via `test_user.id`. So the DB must be up
  and at `alembic upgrade head` before running (see `developer_guide.md`).
- **Cleanup is by CASCADE.** `test_org` deletes via raw
  `DELETE FROM organization WHERE id = :id` so PostgreSQL's FK cascade removes all
  child rows in one shot. An ORM delete would try to NULL org-scoped
  `role.organization_id` first and collide with the partial unique index on
  `role_key`. If you add a fixture that creates org-scoped rows, let the org
  cascade clean them up rather than deleting them yourself.
- **`expire_on_commit=False`** is deliberate: it keeps ORM attributes readable
  after the session closes, which is why fixtures can `yield` a loaded object.

### Where shared *non-fixture* helpers live

Reusable helper code that isn't a fixture goes in a module **without** a `test_`
prefix, so pytest never collects it as a test:

- `tests/agent_signing.py` — `sign_agent_request()`, `build_multipart()` (§3).
- `tests/registration_helpers.py` — `make_org_user()`, `make_sysadmin()`,
  `delete_users()`, `login()`.

Import these into test modules; never copy-paste the logic. If you write a helper
used by two suites, add it to one of these (or a new non-`test_` module), not to a
test file.

### Conventions

- **Group with banner comments** (`# ==== Upload digest enforcement ====`) — the
  suite uses these instead of test classes. Prefer plain functions over classes.
- **One behavior per test; name it as an assertion**:
  `test_upload_with_mismatched_digest_is_rejected`.
- **Always pass `resp.text` as the assert message** on status-code checks:
  `assert resp.status_code == 200, resp.text` — it surfaces the server error body
  when it fails.
- **Local fixtures stay local.** A fixture only one module needs (e.g.
  `upload_agent` in `test_upload_integrity.py`) is defined in that module, not
  `conftest.py`. Promote to `conftest.py` only on the second consumer.

---

## 2. Testing route handlers (the real session pattern)

Route handlers depend on a synchronous session — `db: Session = Depends(get_db)` —
and the house style is **not to mock the session**. Instead, drive the real
endpoint through the `client` fixture against the real database, and assert on
both the HTTP response and the resulting rows:

```python
def test_create_data_source(client, db_session):
    resp = client.post("/api/v1/data-sources", json={"name": "NMC round-robin"})
    assert resp.status_code == 201, resp.text

    ds_id = resp.json()["id"]
    row = db_session.get(DataSource, ds_id)      # verify persistence directly
    assert row is not None and row.name == "NMC round-robin"
```

Why this over mocking a session:

- Route handlers are thin; the behavior worth testing is the SQL, the constraints,
  the cascades, and the permission checks — all of which a mock would stub away.
- The `client` fixture already wires real auth and a real DB, so an integration-
  style test is the *cheapest* correct test here, not the expensive one.

**Overriding dependencies** is how you vary auth or inject state — via
`app.dependency_overrides`, exactly as `conftest.py` does for `get_db` and
`get_auth_with_org`. To test as a different user/permission set, override
`get_auth_with_org` inside the test and pop it in a `finally`/fixture teardown.

**When a fake session is genuinely warranted** — a pure unit test of a helper that
takes a `Session` but whose logic you want to isolate from the DB — use a
lightweight stub or `unittest.mock.MagicMock(spec=Session)` and assert on the
calls. Keep this rare: if the function does anything non-trivial with the query
results, a real `db_session` is more faithful and less brittle.

---

## 3. Testing HMAC-signed (Bridge Agent) requests

Agent endpoints authenticate with an HMAC signature, not a session cookie. Use the
shared helpers in `tests/agent_signing.py` to sign requests exactly as the agent
does — don't reconstruct the scheme inline.

### The signing scheme

`sign_agent_request()` reproduces `app.auth.dependencies.compute_agent_signature`:

- **Key:** `sha256(bytes.fromhex(secret_hex))` — the HMAC key is the SHA-256 of the
  agent's raw secret, not the secret itself.
- **Signed message:**
  `f"{agent_id}:{METHOD}:{path}:{timestamp}:{nonce}:{signed_digest}"`.
- **`signed_digest`** is `"-"` for requests with no declared body digest, or the
  `content_sha256` for multipart uploads (also emitted as the
  `X-AmpWell-Content-SHA256` header — MEDIUM-6).
- **Headers returned:** `X-AmpWell-Equipment-ID`, `X-AmpWell-Timestamp`,
  `X-AmpWell-Nonce`, `X-AmpWell-Signature` (+ `X-AmpWell-Content-SHA256` for
  uploads).

### The pattern

1. **Create an agent with a known secret.** Persist the encrypted secret with
   `crypto.encrypt_agent_secret(bytes.fromhex(secret_hex), agent.id)` and keep
   `secret_hex` in the test so you can sign. Build the `agent → device → channel`
   graph it needs (see the `upload_agent` fixture in `test_upload_integrity.py`).
2. **Sign and send** through the `client` fixture:

```python
path = "/api/v1/files/upload"
content = b"cycle,voltage\n1,3.70\n"
content_sha = hashlib.sha256(content).hexdigest()

body, content_type = build_multipart(fields, "run.csv", content)
headers = sign_agent_request(
    secret_hex, str(agent.id), "POST", path, content_sha256=content_sha
)
headers["Content-Type"] = content_type

resp = client.post(path, content=body, headers=headers)
assert resp.status_code == 200, resp.text
```

Testing guidance:

- **Cover the negative paths, not just the happy one.** The upload suite proves a
  *mismatched* digest is rejected 400 and never persisted, and that omitting the
  digest header is rejected — assert both the status **and** that nothing landed on
  disk or in `DataFile`. Security endpoints are where "and nothing was written" is
  the important half of the test.
- **Sign the path you actually POST to** (including `/api/v1` prefix) — the path is
  part of the signed message; a mismatch fails signature verification, not routing.
- **Each call gets a fresh `nonce`/`timestamp`** (the helper does this) so anti-
  replay doesn't reject the second request in a test.

---

## 4. Coverage & unit-testing best practices (to implement)

Coverage is not measured in this repo yet — there is no `pytest-cov` dependency
and no gate. This section is the plan for adopting it and the unit-testing
discipline that makes the number meaningful. Treat it as the target to build
toward, and follow it for new code so coverage rises as the suite grows.

### Step 1 — turn coverage measurement on

Add `pytest-cov` to the dev dependencies and configure `coverage.py` in
`backend/pyproject.toml`:

```toml
[tool.coverage.run]
branch = true                     # count branch coverage, not just line hits
source = ["app"]
omit = ["app/cli/*", "*/__init__.py", "app/main.py"]

[tool.coverage.report]
show_missing = true               # print the exact unhit lines
skip_covered = true               # keep the report focused on the gaps
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "\\.\\.\\.",                  # protocol/abstract stubs
]
```

Run it locally with:

```bash
uv run pytest --cov=app --cov-report=term-missing -m 'not integration'
uv run pytest --cov=app --cov-report=html            # browsable htmlcov/ report
```

Turn on **branch coverage** from day one (`branch = true`): line coverage alone
counts an `if` as covered even if the `else` never runs, which hides exactly the
negative paths that matter most here.

### Step 2 — adopt thresholds by module tier

Set the bar by blast radius, not uniformly. Security- and data-integrity code
earns a high bar; glue and admin scaffolding a lower one.

| Tier | Modules | Target (line + branch) |
|---|---|---|
| **Security-critical** | `app/auth/*` (HMAC, sessions, crypto, password, MFA), `app/callback_sandbox.py`, upload integrity/containment paths | **95%+**, with explicit negative-path tests |
| **Core domain** | `app/api/routers/*`, `app/services/*`, `app/ingestion/*`, model registry | **85%+** |
| **Supporting** | schema/query services, notifications, config | **75%+** |
| **Low-value / hard-to-reach** | `app/cli/*`, one-off scripts, `__init__` re-exports | best-effort; may be `omit`-ed |

Enforce with a **global floor** in CI — `--cov-fail-under=85` — plus reviewer
discipline on the security tier. `coverage.py` has no clean per-file `fail_under`,
so track the tiers in review rather than trying to encode them in tooling. Adopt
the floor at roughly the current measured number and **ratchet it upward** as
coverage improves; never let a PR lower it.

### Step 3 — write unit tests that make the number mean something

Coverage measures execution, not correctness — a line can be "covered" by a test
that asserts nothing. The number is a *floor on what was exercised*, useful for
finding untested code, not a proof of quality. Make each covered line earn it:

- **Test behavior, not implementation.** Assert on outputs, persisted rows, and
  raised exceptions — not on which private helper got called. Tests coupled to
  internals break on every refactor and discourage cleanup.
- **Cover branches and negative paths first.** For every happy path, test the
  rejections: bad input → the right 4xx, *and* that nothing was written (§3). This
  is where branch coverage and correctness align — it's the highest-value
  coverage you can add.
- **One reason to fail per test.** A focused test names the exact broken behavior
  when it goes red. Split "arrange one thing, assert many unrelated things."
- **Cover the boundaries.** Empty, zero, one, many; null/optional fields; limits
  and off-by-one edges (page sizes, `le=200` caps); duplicate/conflict cases that
  hit unique constraints. Bugs cluster at edges, and so should tests.
- **Parametrize instead of copy-pasting.** Use `@pytest.mark.parametrize` to sweep
  input variants through one test body — it raises meaningful coverage cheaply and
  keeps the intent readable.
- **Push logic-heavy code into pure functions and unit-test them directly.**
  Validators, signature computation, permission resolution, parsing, and pricing/
  limit math are fastest and most thoroughly tested in isolation, with no DB or
  `client`. Reserve the `client`+DB integration style (§2) for the wiring; unit-
  test the logic underneath it.
- **Make tests deterministic and isolated.** No wall-clock or network
  dependencies; inject time/randomness (`rng = np.random.default_rng(42)` as the
  iceberg test does); unique slugs/IDs per test; never depend on execution order.
- **A bug fix starts with a failing test.** Reproduce the bug as a red test first,
  then fix it — that test is the regression guard and it targets coverage exactly
  where a gap let the bug through.
- **Assert on error *type and message*, not just that it raised**
  (`pytest.raises(ValueError, match="...")`) so a test can't pass on the wrong
  failure.

### What not to chase

- **Don't test framework or generated code** — Pydantic validation, SQLAlchemy
  itself, ORM column definitions. Test *your* logic, not the library's.
- **Don't write assertion-free tests to hit a number.** A test that calls code and
  checks nothing inflates coverage while proving nothing; it's worse than no test
  because it looks like protection.
- **Don't chase the last few percent through boilerplate.** 95% on `app/auth`
  with real negative tests beats 100% everywhere padded with empty calls. Mark
  genuinely unreachable defensive lines `# pragma: no cover` with a reason.

---

## 5. Integration tests — the equipment-to-database flow

Integration tests that need the live PostgreSQL **and** the Iceberg warehouse are
marked `@pytest.mark.integration` and deselected with `-m 'not integration'` (they
require `AMPWELL_DATA_DIR` and a migrated DB). Mark any test that touches the
warehouse or otherwise needs external state beyond the base DB.

The full equipment → database path is exercised in **two halves** that meet at the
`DataFile` + on-disk artifact:

1. **Agent → server → relational DB + disk** (`test_upload_integrity.py`,
   `test_upload_confirm.py`): build the `agent → device → channel` graph, sign an
   upload with the real HMAC scheme (§3), POST it, then assert the artifact landed
   on disk **and** a `DataFile` row exists with the right size/digest.
2. **Server → Iceberg warehouse → read-back** (`test_iceberg.py`): use
   `SchemaService.get_or_create_iceberg_table("timeseries")`, write synthetic rows
   via PyArrow, and read them back through DuckDB
   (`app._iceberg.catalog.get_duckdb_connection`), asserting row count and a
   computed statistic (e.g. voltage mean). Use a **negative sentinel `file_id`**
   (`-9999`) so test rows never collide with real serial IDs and are trivially
   identifiable for cleanup.

Patterns to follow for a new end-to-end test:

- **Build the object graph as fixtures, drive real endpoints.** Assemble
  `Organization → BridgeAgent → Device → Channel` (and a locked `Protocol`/run when
  needed), then hit the actual signed endpoints rather than calling service
  functions directly — that's what proves the wiring.
- **Assert at every layer the data crosses:** HTTP status, the on-disk file, the
  relational row, and (for warehouse tests) the read-back query. A green status
  code alone doesn't prove the bytes persisted.
- **Isolate and clean up:** unique slugs/run-ids per test, sentinel IDs for
  warehouse rows, and `landed.unlink(missing_ok=True)` for files the test wrote.
  Let the `test_org` cascade handle relational cleanup.
- **Keep warehouse/Docker cost behind the marker** so the default `uv run pytest`
  stays fast for the common inner loop and CI can run `-m integration` separately.

---

## 6. Checklist for a new test

- [ ] Lives in `backend/tests/`, `test_*.py`, plain functions grouped by banner
      comments.
- [ ] Reuses `conftest.py` fixtures (`client`, `db_session`, `test_org`,
      `test_user`) instead of re-building auth/DB setup.
- [ ] Route tests drive the real endpoint via `client` against the real DB — no
      session mocking.
- [ ] Logic-heavy helpers are unit-tested as pure functions, not only through the
      endpoint.
- [ ] Agent-endpoint tests sign via `tests/agent_signing.py`; shared helpers go in
      non-`test_` modules.
- [ ] Security/upload tests assert the **negative** case *and* that nothing was
      persisted.
- [ ] Tests the branches/edges, not just the happy path; asserts on outputs and
      error type/message, never assertion-free.
- [ ] `assert resp.status_code == ..., resp.text` on every request.
- [ ] Warehouse/external-dependency tests marked `@pytest.mark.integration`, with
      isolated IDs and cleanup.
- [ ] Ran `uv run pytest -m 'not integration'` clean locally before pushing.
