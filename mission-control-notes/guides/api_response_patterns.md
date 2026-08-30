# API Response Patterns

Reference material for keeping FastAPI request/response shapes consistent across
the AmpWell backend, and for how the generated TypeScript client on the frontend
consumes them. It combines the conventions already in the codebase (read out of
`backend/app/api/*` and `frontend/src/api/*`) with the practices that keep a
growing REST surface predictable.

Related guides: `notes/python_error_handling.md` (how exceptions become HTTP
responses in depth), `notes/developer_guide.md` (regenerating the TS client),
`notes/docstring_styling.md` (endpoint docstrings).

---

## 1. The big picture

- Backend routers live in `backend/app/api/routers/*.py`, are assembled in
  `backend/app/api/router.py`, and are mounted under **`/api/v1/*`**. Auth is
  enforced *inside* each router, never at the mount.
- The API is the **contract**: the frontend never hand-writes fetch calls
  against it. `orval` reads the live OpenAPI schema
  (`http://localhost:8000/api/openapi.json`) and generates react-query hooks
  into `frontend/src/api/generated/` plus TypeScript models into
  `frontend/src/api/model/`. **Regenerate after any schema/route change**
  (`npm run generate:api`, per `developer_guide.md`).
- Because the generated client is derived from your Pydantic models, **response
  shape discipline on the backend is what keeps the frontend types correct.**
  A sloppy `-> dict` return with no `response_model` produces an untyped blob on
  the client.

---

## 2. Success and error response structure

### Success bodies

- **Return a Pydantic model and declare `response_model=`** wherever practical.
  This is the shape that flows into the OpenAPI schema and therefore into the
  generated TS types. Example (`organizations.py`):
  ```python
  @router.get("/{organization_id}", response_model=OrganizationRead)
  def get_organization(...) -> OrganizationRead:
      ...
      return OrganizationRead.model_validate(org)
  ```
- Some endpoints return a bare `dict` (e.g. the Protocol Registry list). That's
  tolerated for hand-shaped aggregate rows, but it produces a weakly-typed
  client. Prefer a declared `response_model`; reach for `dict` only when the row
  is a genuinely dynamic aggregate that no static model captures.
- **Status codes:** default `200`; use `status_code=status.HTTP_201_CREATED` for
  resource creation (`POST /protocols`, most `POST` create endpoints); `202` for
  accepted-but-async (org self-registration, forgot-password); `204` for empty
  bodies. Always use the `fastapi.status` constants, not integer literals.
- There is **no envelope** around single-resource success bodies — the resource
  is the body. Do not wrap in `{"data": ...}`. (List endpoints are the one
  deliberate exception; see §4.)

### Error bodies

- Errors are raised as `HTTPException(status_code=..., detail=...)`, which
  FastAPI renders as **`{"detail": <string>}`**. This is the single error shape
  the whole app relies on — the frontend mutator reads exactly `data.detail`
  (see §5). Keep `detail` a human-readable string.
- **Validation errors (422)** are produced by FastAPI itself from Pydantic and
  use its standard structured form:
  `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}`. You get this for
  free by typing your request models — don't hand-roll field validation that
  duplicates it.
- **There are no custom global exception handlers.** Conversion happens at the
  route boundary. Every route is responsible for turning failure into the right
  `HTTPException`. See `notes/python_error_handling.md` for the full rationale.
- **Database errors go through the shared translators** in
  `app/api/common.py` — never let a raw `IntegrityError` escape:
  - `commit_or_translate(db)` — commit and map failures: unique/FK violation →
    **409**, CHECK/invalid-data → **422**, with `db.rollback()` first.
  - `get_or_404(db, Model, id, "Resource")` — fetch-or-404 helper.
  - `ensure_exists(...)` — FK-target existence check → **422** if missing.
- **Status-code conventions** (match these):

  | Code | When |
  |------|------|
  | 400 | Malformed but syntactically valid request the schema can't catch (e.g. corrupt uploaded file) |
  | 401 | No/invalid session or agent credential |
  | 403 | Authenticated but lacks permission, or no org context |
  | 404 | Resource absent — **also** used instead of 403 to avoid ID enumeration (see `get_organization`) |
  | 409 | Unique/FK conflict (from the DB translators) |
  | 422 | Request/entity validation failure (Pydantic, or business-rule invalid data) |
  | 503 | Dependency unavailable (e.g. agent offline) |

- **Don't leak internals in `detail`.** The translators return generic strings
  (`"Resource conflict"`, `"Invalid request payload"`) rather than raw driver
  text. Follow that: user-actionable message in, stack traces to the log.

---

## 3. Pydantic schema conventions

### Where schemas live

- **Shared / cross-router schemas → `backend/app/api/schemas.py`.** This holds
  the models used by more than one router (`UserRead`, `OrganizationRead`), the
  auth/MFA/registration request+response models, and the generic
  `ListResponse[T]` wrapper.
- **Router-local schemas → defined inline** at the top of the owning router file,
  under a `# ==== Pydantic schemas ====` banner (see `protocols.py`,
  `callbacks.py`, `models.py`, `bridge_agent.py`). Keep a schema local until a
  second router needs it, then promote it to `schemas.py`. Don't pre-emptively
  centralize.

### Base classes

- **Any model read from a SQLAlchemy row must subclass `ORMModel`** (from
  `schemas.py`), which sets `model_config = ConfigDict(from_attributes=True)`.
  Then serialize with `Model.model_validate(orm_obj)`. Request bodies (not read
  from the ORM) subclass plain `BaseModel`.

### Request vs. response naming

Two naming families coexist. Match whichever the surrounding router already uses;
prefer the first for new CRUD resources:

- **CRUD triad (majority):** `XxxCreate`, `XxxUpdate`, `XxxRead`
  (+ occasional `XxxSummary` for a compact nested form, e.g. `TestRunSummary`).
  - `Create` — required fields for creation.
  - `Update` — **every field `| None = None`** so it doubles as a PATCH body;
    only provided fields are applied.
  - `Read` — the full serialized row; nested relations are optional and default
    to `None` (`organization: OrganizationRead | None = None`) so they're only
    populated when explicitly loaded.
- **Request/Out family (some routers):** `XxxRequest` for bodies,
  `XxxOut` / `XxxPayload` for responses (`callbacks.py`, `models.py`,
  `bridge_agent.py`).

### Conventions to keep

- **Optional-with-default** for nullable/omittable fields:
  `email: str | None = None`. Reserve required (no default) for genuinely
  mandatory input.
- **`metadata_` trailing underscore** — the field is named `metadata_` in Python
  to avoid shadowing SQLAlchemy/BaseModel internals; it serializes as the JSON
  key the client expects. Follow the existing pattern rather than renaming.
- **One class = one purpose.** Don't reuse a `Read` model as a request body or
  vice-versa; the asymmetry (server-set fields like `id`, `created_at`) is the
  whole point of separating them.
- **Docstring every schema and endpoint** per `docstring_styling.md` — the
  endpoint docstring's `Args`/`Returns`/`Raises` is the human half of the
  contract the OpenAPI schema describes.

---

## 4. Pagination for list endpoints

List endpoints return a **paginated envelope**, never a bare array — this leaves
room for `total` and keeps the client stable when filtering is added.

### Canonical pattern: `limit` / `offset` + `ListResponse[T]`

This is the preferred shape for new list endpoints. Use the generic wrapper and
the shared builder:

```python
from app.api.schemas import ListResponse, OrganizationRead
from app.api.common import build_list_response

@router.get("", response_model=ListResponse[OrganizationRead])
def list_organizations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    ...
) -> dict:
    total = query.count()
    items = query.order_by(...).offset(offset).limit(limit).all()
    rows = [OrganizationRead.model_validate(o) for o in items]
    return build_list_response(rows, total, limit, offset)
```

Envelope shape → `{"items": [...], "total": int, "limit": int, "offset": int}`.

Rules:
- `limit` defaults to **50**, bounded `ge=1, le=200` (cap so a client can't ask
  for unbounded rows). `offset` defaults to **0**, `ge=0`.
- Always compute `total` from the *filtered* query (before `offset`/`limit`) so
  the client can render page counts.
- Apply a deterministic `order_by` before slicing — pagination over an unordered
  query is undefined.

**Per-router variants that mirror the same fields** (`CallbackListResponse`,
`ModelListResponse`) exist because they name the item list after the domain
(`callbacks`, `models`) instead of `items`. That's acceptable for readability,
but keep the other three keys identical (`total`, `limit`, `offset`). Prefer
`ListResponse[T]` for anything new unless there's a reason to name the list.

### Registry variant: `page` / `page_size`

The Protocol Registry / run-history endpoints (`protocols.py`, `runs.py`) use a
1-based page model instead:

```python
page: int = Query(1, ge=1),
page_size: int = Query(_DEFAULT_PAGE_SIZE, ge=1, le=200),   # _DEFAULT_PAGE_SIZE = 50
...
rows = q.offset((page - 1) * page_size).limit(page_size).all()
return {"protocols": items, "total": total, "page": page, "page_size": page_size}
```

Both patterns are live; **pick one per endpoint and don't mix them.** Use
`limit`/`offset` + `ListResponse[T]` for new work; the `page`/`page_size` form is
appropriate when the UI is genuinely page-numbered (Registry tables). Whichever
you choose, always return `total`.

---

## 5. How the frontend consumes responses

The generated hooks all route through the single mutator
`frontend/src/api/client.ts` (`apiFetch`). Understanding it tells you what the
backend contract must hold to:

- **Auth is the `ampwell_session` httpOnly cookie**, sent automatically via
  `credentials: 'include'`. No `Authorization` header management for browser
  clients (non-browser clients may send `Authorization: Bearer <token>`).
- **`204` → `undefined` body**; every other status is parsed as JSON.
- **Error handling normalizes the two `detail` shapes:** on `!response.ok` the
  mutator builds a message from `data.detail` and throws an `Error` with `status`
  and the raw `data` attached. A string `detail` (deliberate `HTTPException`) is
  used as-is; a list `detail` (FastAPI's automatic 422 validation errors) is
  joined from each entry's `msg`; anything else falls back to
  `"<status> <statusText>"`. Because the raw `data` is attached, components that
  want per-field validation info can still read `error.data.detail`. Keep
  deliberate, user-facing errors as a string `detail` so they surface verbatim.
- **Hand-written clients** (`controlCenter.ts`, etc.) declare `interface`s that
  must mirror the backend response field-for-field (see `ChannelSummary`,
  `ChannelLiveData`). When you change a response model, update these too — they
  are not generated.

### Polling & staleness (there is no WebSocket — see §6)

"Live" data is react-query polling, not push. Set `staleTime` from the tiers in
`frontend/src/api/queryConfig.ts` (`STALE.STATIC/LONG/MEDIUM/SHORT`) in the
**hook definition**, not the call site, so every consumer inherits the right
cadence. Real-time channel/agent status uses `STALE.SHORT` (15 s).

---

## 6. WebSocket / real-time message structure

**AmpWell has no WebSocket (or SSE) endpoints today.** This is deliberate and
documented in `controlCenter.ts`:

> There is no Server-Sent-Events endpoint yet, so "live" data is obtained by
> polling `GET /channels` on a short interval.

How real-time data actually moves:

- **Agent → server is HTTP push, typed by Pydantic payloads.** The Bridge Agent
  periodically `POST`s `StatusPayload` to `/equipment/{agent_id}/status`
  (HMAC-authenticated). Its shape (`bridge_agent.py`):
  ```python
  class StatusPayload(BaseModel):
      timestamp: float
      agent_version: str | None = None
      channel_states: dict[str, str]        # full_label -> "idle"|"running"|...
      live_data: dict[str, Any] = {}        # full_label -> measurement snapshot
      adapter_health: list[dict] = []
  ```
  The server persists the latest snapshot onto the channel row; the browser
  polls `GET /channels` (`STALE.SHORT`) to read it back as `ChannelLiveData`.
- **Run lifecycle events** arrive as discrete `RunEventPayload` posts, typed by a
  string `event_type` field (`"run_complete" | "run_cancelled" | "run_error" |
  ...`) plus a free-form `data: dict`.

### If you add a WebSocket/SSE channel later

Keep the same discipline the HTTP surface already has — a real-time message is
still a contract:

- **Type every message with a Pydantic model**, exactly like `StatusPayload`.
  Don't emit ad-hoc dicts over the socket.
- **Use a discriminated union for multi-type streams.** Where a stream carries
  several message kinds, give each a literal `type`/`event_type` tag and model
  the union with a Pydantic discriminated union (`Field(discriminator="type")`),
  so both server and generated client can switch on the tag exhaustively. This
  mirrors the msgspec tagged-union pattern already used in the `ampwell-protocol`
  package.
- **Reuse the existing envelope vocabulary** — `timestamp`, string status enums,
  `full_label` channel keys — so socket payloads and the polled REST shapes stay
  interchangeable and the frontend can share types.
- **Errors over the socket still carry a string `detail`** so client handling
  matches the REST path.

---

## 7. Checklist for a new or changed endpoint

- [ ] Route mounted under a domain router in `app/api/routers/`, auth enforced in
      the handler (`get_auth_with_org` / `require_device` + `check_permission`).
- [ ] Declares `response_model=` with a Pydantic model (not a bare `dict`) unless
      the row is a genuinely dynamic aggregate.
- [ ] Read models subclass `ORMModel`; request bodies subclass `BaseModel`;
      `Update` bodies are all-optional.
- [ ] Correct `status_code` (201 create / 202 async / 204 empty), from
      `fastapi.status`.
- [ ] Errors raised as `HTTPException(detail="<string>")`; DB writes go through
      `commit_or_translate` / `get_or_404` / `ensure_exists`.
- [ ] List endpoints return a paginated envelope with `total` (`limit`/`offset`
      + `ListResponse[T]` preferred), bounded page size (`le=200`), deterministic
      ordering.
- [ ] Shared schema promoted to `app/api/schemas.py` only once a second router
      needs it.
- [ ] Ran `npm run generate:api` and updated any hand-written client interfaces
      (`controlCenter.ts`, ...) that mirror the changed shape.
