# Database query patterns

Reference material for Claude Code agents and human developers writing
SQLAlchemy queries in the AmpWell backend. This guide documents the idioms the
codebase actually uses, corrects a few premises that don't match reality, and
sets the conventions for new query code.

> **Reality check — read before writing any query.**
> 1. **The backend is synchronous.** `app/database.py` builds a plain
>    `create_engine` + `sessionmaker`; there is no `AsyncSession` and no
>    `await db.execute(...)`. (See `python_async_patterns.md` for how sync DB
>    work coexists with the event loop.)
> 2. **The dominant idiom is the ORM `Query` API** — `db.query(Model)` (~170
>    uses) and `db.get(Model, pk)` (~95) — not 2.0-style
>    `select()`/`scalars()` (~8 uses, concentrated in `services/audit.py` and
>    `_services/schema_service.py`). Both are documented below.
> 3. **There is no soft-delete convention.** Deletes are hard deletes with FK
>    `ON DELETE CASCADE`; "keep it but hide it" is expressed as a **status
>    lifecycle** (`archived`, `locked`, `retired`, `suspended`), not a
>    `deleted_at` column. §3 covers what to filter on instead.

---

## 1. Query idioms: `db.query()` (house style) and `select()` + `scalars()`

### The house style

```python
# Primary-key fetch — always db.get(), never query().filter_by(id=...).first()
run = db.get(ProtocolRun, rid)

# Filtered list
pending = (
    db.query(PendingCommand)
    .filter_by(bridge_agent_id=agent_pk, status="pending")
    .order_by(PendingCommand.issued_at)
    .all()
)

# Single-or-none
org = db.query(Organization).filter_by(slug=slug).first()
```

Conventions visible throughout the routers:

- `db.get(Model, pk)` for primary-key lookups (it can hit the identity map and
  skip a round-trip). `app/api/common.py` wraps it: `get_or_404` (fetch or 404),
  `ensure_exists` (validate an FK reference or 422).
- `filter_by(col=value)` for simple equality; `filter(Model.col == value,
  Model.other.in_(ids))` when you need operators, `or_`, `ilike`, etc.
- Build queries **incrementally** — start with the base joins + org scope, then
  conditionally add `.filter(...)` per optional query param (see the runs list
  in §5).
- Terminal call matches intent: `.all()`, `.first()`, `.one_or_none()`,
  `.count()`, `.delete()` (bulk).

### The 2.0 `select()` + `scalars()` idiom

Used where the code was written against the modern API, e.g. `services/audit.py`:

```python
stmt = (
    select(AuditLog.row_hash)
    .order_by(AuditLog.occurred_at.desc(), AuditLog.id.desc())
    .limit(1)
)
latest = db.execute(stmt).scalar_one_or_none()

stmt = select(AuditLog).where(AuditLog.row_hash.is_not(None))
for entry in db.execute(stmt).scalars():   # .scalars() unwraps Row -> AuditLog
    ...
```

Key points when using it:

- `db.execute(select(Model))` returns **`Row` tuples**; call `.scalars()` to get
  model instances, or `.scalar_one_or_none()` / `.scalar_one()` for a single
  value. Forgetting `.scalars()` is the classic bug — you'll iterate 1-tuples.
- `select(Model.col)` for single-column queries avoids loading whole rows.
- This is the API SQLAlchemy is converging on, and the only one that survives a
  future async migration unchanged.

### Which to use

**Match the file you're editing.** Router code overwhelmingly uses
`db.query()`; keep new endpoints consistent with their neighbors. `select()` is
fine (and preferred) for new **service-layer** modules, column-only queries, and
anything you expect to reuse in both scalar and row form. Don't mix both styles
inside one function, and don't refactor working `db.query()` code to `select()`
as a drive-by.

---

## 2. Eager loading — kill N+1 with `joinedload` chains or `selectinload`

The serializers read across relationships (a run row displays its protocol name,
channel label, and the **two-hop** bridge-agent name). Without eager loading, a
page of 50 runs fires 150+ lazy-load SELECTs. The two real patterns:

### `joinedload` for to-one chains (the runs list, `api/routers/runs.py`)

```python
runs = (
    q.options(
        joinedload(ProtocolRun.channel)
        .joinedload(Channel.device)
        .joinedload(Device.bridge_agent),   # chained two-hop to-one path
        joinedload(ProtocolRun.protocol),
        joinedload(ProtocolRun.dut),
    )
    .offset((page - 1) * page_size)
    .limit(page_size)
    .all()
)
```

`joinedload` folds the related rows into the same SELECT via LEFT OUTER JOIN —
ideal for **many-to-one / one-to-one** paths, where the join can't multiply rows.

### `selectinload` for collections (`_services/schema_service.py`)

```python
schema = (
    db.query(SchemaRegistry)
    .options(selectinload(SchemaRegistry.columns))   # one extra SELECT ... IN (...)
    ...
)
```

`selectinload` issues a second query with `WHERE parent_id IN (...)` — ideal for
**one-to-many collections**, because a `joinedload` on a collection multiplies
parent rows (and breaks LIMIT/OFFSET pagination).

### Rules

- **Any list endpoint whose serializer touches a relationship must declare
  eager loads** for every relationship the serializer reads. Trace the
  serializer, then mirror its access paths in `.options(...)`.
- To-one path → `joinedload` (chain with `.joinedload(...)` for multi-hop).
  Collection → `selectinload`. Never `joinedload` a collection on a paginated
  query.
- Detail endpoints fetching one row can rely on lazy loading — N+1 on N=1 is
  fine; don't add options noise there.
- When you only need one or two scalar fields from a big set of parents (e.g.
  project names for tag chips), a **bulk `IN` query into a dict** beats eager
  loading — the runs list does exactly this for project names:
  ```python
  name_map = _project_names(db, org.id, all_project_ids)  # one SELECT id,name IN (...)
  ```

---

## 3. Lifecycle filtering (the convention that replaces soft-delete)

AmpWell **does not soft-delete**. There is no `deleted_at` / `is_deleted`
anywhere; deleting an org or agent is a hard `DELETE` with `ondelete="CASCADE"`
FKs, and the act is captured in the append-only audit log — which is the
tamper-evident record of "what existed," so hidden rows aren't needed for that.

What exists instead is **status lifecycles**, enforced by CHECK constraints:

| Model | States | "Hidden" state |
|---|---|---|
| `Project` | `active`, `archived` | `archived` |
| Callback/model artifacts | `draft`, `locked`, `archived`, `retired`, `superseded` | `retired` / `superseded` |
| `User` / `Organization` | `active`, `suspended`, ... | `suspended` |
| `Protocol` | `draft`, `locked`, ... | (locked = immutable, still visible) |

The rules that follow:

- Listing endpoints filter to live states explicitly
  (`.filter(Project.status == "active")`); "include archived" is an explicit
  query parameter, not the default.
- Auth-adjacent code must treat non-active states as fenced —
  `get_current_user` rejects suspended users; suspended orgs fail login. Never
  add a lookup path that resolves a suspended/retired row into an active flow.
- **Don't introduce a `deleted_at` column** for a new feature. Use a status
  enum + CHECK constraint (matching the table above), or a real delete +
  cascade + audit entry. If genuine soft-delete ever becomes a requirement,
  that's a design discussion, not a one-table pattern to sneak in.
- Rows that must never be deleted or mutated are protected at the DB layer
  (audit trigger), not by filter discipline.

---

## 4. Timestamp ordering and pagination

### Timestamps

- All timestamp columns are **`DateTime(timezone=True)`** with
  `server_default=sa_text("now()")` for creation stamps; Python-side stamps use
  `datetime.now(UTC)`. Keep both conventions — never naive datetimes.
- Newest-first is the default listing order:
  `.order_by(AuditLog.occurred_at.desc())`,
  `.order_by(Dut.created_at.desc())`, etc.
- **Nullable timestamps need `.nullslast()`** — a queued run has no
  `started_at` yet, and Postgres sorts NULLs first under `DESC`:
  ```python
  q = q.order_by(
      ProtocolRun.started_at.desc().nullslast(),
      ProtocolRun.created_at.desc(),      # deterministic tie-break
  )
  ```
- **Always add a tie-break column** (`created_at`, `id`) after a non-unique
  sort key, so pagination is stable when timestamps collide. The audit chain
  does this deliberately: `.order_by(occurred_at.desc(), id.desc())`.
- Range filters take ISO-8601 params parsed to aware datetimes, applied as
  `>=` / `<=` bounds (`started_after`/`started_before` in the runs list).

### Pagination — two sanctioned shapes

**Canonical `limit`/`offset`** (most list endpoints) via
`build_list_response` in `app/api/common.py`, returning
`{"items", "total", "limit", "offset"}`.

**`page`/`page_size`** (runs list, registry) returning
`{"runs"/"items", "total", "page", "page_size"}` with
`.offset((page - 1) * page_size).limit(page_size)`.

The invariant sequence, regardless of shape:

```python
total = q.count()          # 1. count BEFORE applying order/offset/limit
q = q.order_by(...)        # 2. deterministic order (with tie-break)
rows = q.offset(...).limit(...).all()   # 3. then the page
```

- Count first: `.count()` on the filtered query, so `total` reflects the full
  match set, not the page.
- Clamp `page_size`/`limit` with validation (runs allows 1–200); never accept an
  unbounded page size.
- Apply `.options(joinedload(...))` **after** you've taken `total` — eager-load
  options don't affect the count but keep the counted query cheaper.
- Offset pagination is fine at AmpWell's scale. If a table grows to where deep
  offsets hurt (audit log is the candidate), the upgrade path is keyset
  pagination on `(occurred_at, id)` — the tie-break ordering above is exactly
  what makes that possible later.

---

## 5. AmpWell-specific: querying across equipment, channels, and runs

### The hierarchy

```
BridgeAgent ──< Device ──< Channel ──< ProtocolRun
     (org)      (org)       (org)        (org)      ── every level carries organization_id
                              └──< DataFile / current_dut / live_data
```

Display fields for a run (agent name, channel label) are **not** columns on
`protocol_run` — they resolve through the join chain. The canonical list query
(`api/routers/runs.py`) shows every convention at once:

```python
q = (
    db.query(ProtocolRun)
    .join(Channel, ProtocolRun.channel_id == Channel.id)
    .join(Device, Channel.device_id == Device.id)
    .join(BridgeAgent, Device.bridge_agent_id == BridgeAgent.id)
    .join(Protocol, ProtocolRun.protocol_id == Protocol.id)   # inner: RESTRICT-protected
    .outerjoin(Dut, ProtocolRun.dut_id == Dut.id)             # outer: dut_id nullable
    .filter(ProtocolRun.organization_id == org.id)            # tenant scope, always
)
# then conditional filters: BridgeAgent.id.in_(...), Channel ids, status.in_(...),
# ProtocolRun.project_ids.overlap(project_ids)  # native UUID[] array overlap
```

The rules embedded there:

1. **Tenant-scope every query.** Every level of the hierarchy denormalizes
   `organization_id` precisely so any query can scope in one filter without
   joining up to the agent. This is a security invariant (see
   `security_checklist.md` §5), not an optimization.
2. **Inner vs outer join follows nullability + FK policy.** `protocol_id` is
   NOT NULL and RESTRICT-protected → inner join. `dut_id` is nullable → outer
   join, or filtered runs silently vanish.
3. **Filter on the joined table when the param targets it** — "runs for agent
   X" filters `BridgeAgent.id.in_(...)` through the join, not by first
   collecting channel ids in Python.
4. **Free-text search** is a single `or_` of `ilike` needles across the joined
   display columns (protocol name, DUT label/barcode, channel labels).
5. **Project tags** live as a native `UUID[]` on the run — query with
   `.overlap(ids)` ("any of"), not a join table.

### Agent-facing channel resolution

Agents identify channels by **`full_label`** (e.g. `"NW-1::12-1-3"`), never by
internal UUID. Resolution is always scoped to the authenticated agent's org —
both tenant isolation and correctness:

```python
channel = (
    db.query(Channel)
    .filter_by(organization_id=agent.organization_id, full_label=channel_id)
    .first()
)
if not channel:
    raise HTTPException(404, f"Channel '{channel_id}' not found in this org")
```

Status dashboards read `Channel.last_status` / `live_data` (denormalized by the
agent's status broadcasts), served by the composite index
`idx_channel_org_status (organization_id, last_status)` — filter on those two
columns together to use it.

### Detail fetch: `db.get` + explicit org check

Detail endpoints fetch by PK then verify tenant, returning 404 (not 403 — don't
leak existence) on mismatch:

```python
run = db.get(ProtocolRun, rid)
if not run or run.organization_id != org_id:
    raise HTTPException(404, "Run not found")
```

### Point-in-time lookups ("as of" queries)

"What was the DUT's capacity going into this run" = the newest history row at or
before the run's start — the standard *latest-before* idiom:

```python
capacity_row = (
    db.query(DutCapacityHistory)
    .filter(
        DutCapacityHistory.dut_id == dut.id,
        DutCapacityHistory.recorded_at <= run.started_at,
    )
    .order_by(DutCapacityHistory.recorded_at.desc())
    .first()
)
```

Use this shape for any history/audit-style table: filter to the entity, bound
the timestamp, order desc, `.first()`.

---

## Pre-merge checklist

- [ ] PK lookups use `db.get` (or `get_or_404`); style matches the surrounding
      file (`db.query()` in routers; `select()` OK in new service modules —
      with `.scalars()` when you want instances).
- [ ] Every org-scoped query filters `organization_id`; detail fetches verify
      tenant and 404 on mismatch.
- [ ] List endpoints eager-load every relationship the serializer reads —
      `joinedload` chains for to-one paths, `selectinload` for collections;
      never `joinedload` a collection on a paginated query.
- [ ] No `deleted_at` columns introduced; hidden states use the status-lifecycle
      convention and listings filter to live states by default.
- [ ] Ordering is deterministic: timestamp key + tie-break, `.nullslast()` on
      nullable timestamps; `total` counted before offset/limit; page size
      clamped.
- [ ] Joins follow nullability (outer for nullable FKs) and filters target the
      joined table rather than pre-collecting ids in Python.
- [ ] Timestamps are timezone-aware end to end (`DateTime(timezone=True)`,
      `datetime.now(UTC)`, ISO-8601 params).
