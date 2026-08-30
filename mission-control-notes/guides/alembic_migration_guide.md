# Alembic Migration Guide

Reference material for writing, reviewing, and applying Alembic migrations in the
AmpWell backend. It combines the conventions this repository already follows
(read out of `backend/alembic/versions/001`–`020`) with the industry practices
that keep a migration history safe to run against production data.

Operational commands (generate, upgrade dev, promote to prod) live in
`notes/developer_guide.md` → **Migrations**. This guide is about *how to write a
migration correctly*, not how to run one.

---

## 1. How this project is wired

- `backend/alembic.ini` — `script_location = alembic`; `sqlalchemy.url` is left
  **empty on purpose**. Never hardcode a URL here.
- `backend/alembic/env.py` — overrides the URL at runtime with
  `str(settings.database_url)`, so every command targets whatever `DATABASE_URL`
  the environment resolves (dev symlink vs. `DATABASE_URL=...prod`). Online mode
  runs with `compare_type=True` and a `NullPool`; offline mode uses
  `literal_binds`. `target_metadata = Base.metadata`.
- `backend/alembic/script.py.mako` — the template new revisions are rendered
  from. It is already modern and type-annotated; don't fight it.
- `backend/alembic/versions/NNN_*.py` — the linear chain of migrations.

The database is **synchronous** PostgreSQL. Migrations use `op.*` and raw SQL via
`op.execute(...)`; there is no async in the migration layer.

---

## 2. File & revision naming conventions

AmpWell does **not** use Alembic's default random hash revision IDs. The whole
history is a hand-curated, zero-padded, sequential chain. Match it exactly.

**Filename:** `NNN_short_snake_case_description.py`

- `NNN` — zero-padded 3-digit sequence number, one greater than the current head
  (`020` → next is `021`).
- Description — lowercase snake_case, terse but specific:
  `device_limits`, `model_build_status`, `purge_legacy_agent_hash`.

**Inside the file** the four module-level identifiers are string literals:

```python
revision = "021"          # matches the NNN in the filename — a STRING, not a hash
down_revision = "020"     # the current head; None only for 001_initial_schema
branch_labels = None      # AmpWell history is strictly linear — keep this None
depends_on = None
```

Rules:

- `revision` **must** equal the filename's `NNN`. They are kept in lockstep so the
  chain is readable at a glance.
- `down_revision` points at the previous head. Confirm the head first
  (`uv run alembic heads` / `alembic current`) — never guess.
- **No branches.** `branch_labels` and `depends_on` stay `None`. If
  `alembic heads` ever shows more than one head, two people numbered a migration
  off the same parent; renumber yours onto the true head rather than merging.
- Do **not** create a migration with `--autogenerate` and keep its default hash
  name. If you autogenerate, rename the file and rewrite the four identifiers to
  the `NNN` scheme before committing.

### Docstring header

Every migration opens with a structured docstring. Copy this shape — reviewers
rely on it:

```python
"""
One-line summary of what this migration does.

Revision ID: 021
Revises: 020
Create Date: 2026-07-05

Changes:
  - Bullet each schema/data change with the table and column touched, and a
    pointer to the spec or prompt that motivated it.

Why: (include this for anything non-obvious — data migrations, security fixes,
  irreversible downgrades). Explain the reasoning, not just the mechanics.

Design refs: 08_ampwell_model_registry.md, 99_pending_todos.md
"""
```

`Create Date` is a plain calendar date (`2026-07-05`), not a timestamp.

---

## 3. Reversibility: every `upgrade()` needs a working `downgrade()`

**Hard rule:** every migration defines both `upgrade()` and `downgrade()`, and the
`downgrade()` must actually reverse the change. This is what makes
`alembic downgrade -1` a real rollback path when a deploy goes wrong.

Both functions get a one-line PEP 257 docstring describing what they do
(see `notes/docstring_styling.md`).

### Order matters — downgrade is the mirror image

Tear down in the reverse order you built up: drop indexes before the table they
sit on, drop dependent objects before their parents. Compare
`008_device_limits.py`:

```python
def upgrade() -> None:
    op.create_table("device_limits", ...)                 # 1. table
    op.create_index("uq_device_limits_default", ...)      # 2. indexes
    op.create_index("idx_device_limits_org", ...)

def downgrade() -> None:
    op.drop_index("idx_device_limits_org", table_name="device_limits")  # reverse
    op.drop_index("uq_device_limits_default", table_name="device_limits")
    op.drop_table("device_limits")
```

### Reverse *everything*, not just the table

If `upgrade()` seeds rows, creates a trigger/function, or adds a CHECK
constraint, `downgrade()` must undo those too:

- `009_model_build_status.py` seeds a `model.upload` permission → downgrade
  `DELETE`s it, drops the CHECK constraint, then drops the column.
- `013_audit_append_only.py` installs a trigger + `plpgsql` function → downgrade
  `DROP TRIGGER IF EXISTS ... / DROP FUNCTION IF EXISTS ...` before dropping the
  columns. Use `IF EXISTS` on raw drops so a partially-applied downgrade is
  re-runnable.

### Constraint changes need a data step in the downgrade

You cannot narrow a CHECK constraint while rows violate it. When `upgrade()`
*widens* an enumeration, the `downgrade()` must first migrate the now-illegal
values back into the old domain, then re-apply the narrow constraint. This is the
canonical pattern from `011_protocol_registry.py`:

```python
def downgrade() -> None:
    ...
    # Collapse the new statuses back so the narrower CHECK applies cleanly
    op.execute(
        "UPDATE protocol SET status = 'locked' "
        "WHERE status IN ('archived', 'retired', 'superseded')"
    )
    op.drop_constraint("ck_protocol_status", "protocol", type_="check")
    op.create_check_constraint(
        "ck_protocol_status", "protocol", "status IN ('draft', 'locked')"
    )
```

### The rare, deliberate one-way migration

A downgrade is allowed to be a **documented no-op** only when reversing it would
be impossible or actively harmful — never out of laziness. The sole current
example is `019_purge_legacy_agent_hash.py`, which NULLs out a
plaintext-equivalent signing key: the migration cannot restore secret material it
destroyed, and re-deriving it would recreate the vulnerability it fixed. When you
do this:

- Make the `downgrade()` body an explicit no-op with a comment explaining *why*
  it cannot be reversed.
- State the irreversibility loudly in the module docstring's `Why:` section.
- Ensure the schema is left in a consistent state (019 relies on the column
  already being nullable from migration 010, so no schema change is needed).

If you're reaching for a no-op downgrade for any other reason, redesign the
migration instead.

---

## 4. Schema-only vs. data migrations

**Schema-only** migration: creates/alters/drops tables, columns, indexes,
constraints. The large majority of AmpWell migrations. Fully reversible by
construction.

**Data migration:** moves or rewrites row *contents* with `op.execute(...)`
running INSERT/UPDATE/DELETE. Present in AmpWell whenever a schema change would
otherwise leave existing rows inconsistent or a new feature needs seed rows:

| Migration | Data step | Purpose |
|---|---|---|
| `009_model_build_status` | `INSERT ... ON CONFLICT DO NOTHING` | seed `model.upload` permission + grant to existing roles |
| `011_protocol_registry` (downgrade) | `UPDATE ... SET status='locked'` | make rows legal before narrowing a CHECK |
| `019_purge_legacy_agent_hash` | `UPDATE ... SET hmac_secret_hash = NULL` | destroy a plaintext-equivalent credential |

Guidance for data migrations:

- **Make seeds idempotent.** Use `INSERT ... ON CONFLICT (key) DO NOTHING`
  (see 009) so re-running or applying to an already-provisioned database is safe.
  Never assume the target table is empty.
- **Set-based SQL, not row-by-row.** Prefer a single `INSERT ... SELECT` /
  `UPDATE ... WHERE` over a Python loop. If you must read rows in Python, go
  through `op.get_bind()` and a lightweight `sa.table(...)` / `sa.column(...)`
  definition — do **not** import the ORM models. Application models drift over
  time; a migration must keep meaning the same thing forever, so it should depend
  only on the column shapes that exist *at that revision*.
- **Sequence schema and data correctly.** When adding a NOT NULL column to a
  populated table, add it nullable (or with a `server_default`), backfill in the
  same migration, then tighten. Adding a column with a `server_default` (as most
  AmpWell tables do for `created_at`, `source`, `build_status`) fills existing
  rows automatically and sidesteps the problem.
- **Autogenerate does not see data.** It also misses CHECK constraints, partial
  indexes, triggers, and functions (per `developer_guide.md` and the note in
  `001_initial_schema.py`). Always hand-write data steps and those object types,
  and review autogenerated diffs before trusting them.

---

## 5. Migrations against tables with existing production data

Dev is disposable; prod is not. `ampwell_prod` holds real data and the guidance in
`developer_guide.md` is explicit: **run every migration on dev first, promote to
prod only after verifying.** When a migration touches a table that has live
production rows, add these considerations:

- **Additive changes are safe to run before the code deploy.** New tables, new
  nullable columns, new columns with a `server_default`, new indexes — apply the
  migration, then restart the service (`developer_guide.md` → *Deploy with a
  migration*). Existing rows get the default; old code ignores the new column.

- **A NOT NULL column on a populated table is a three-step move**, never a bare
  `add_column(nullable=False)` with no default:
  1. add the column nullable / with a `server_default`,
  2. backfill existing rows,
  3. alter to `NOT NULL` (drop the default afterward if it was only for backfill).

- **Destructive or tightening changes need the service stopped.** Dropping a
  column/table, narrowing a CHECK, or anything needing an exclusive lock: stop
  the API first so no request races the migration
  (`developer_guide.md` → *Rollback* / *Common Issues*):
  ```bash
  sudo systemctl stop ampwell-api
  DATABASE_URL=postgresql://admin:password@localhost:5432/ampwell_prod \
      uv run alembic upgrade head
  sudo systemctl start ampwell-api
  ```

- **Respect FK cascades.** AmpWell leans on `ondelete` semantics
  (`CASCADE`, `RESTRICT`, `SET NULL`) — e.g. `protocol_run` uses `RESTRICT` on
  `channel_id`/`protocol_id` but `SET NULL` on optional refs. A data migration
  that deletes parent rows can cascade far more widely than intended; know the
  cascade behavior of every FK you touch before issuing a `DELETE`.

- **Never mutate prod data outside a migration.** `developer_guide.md` warns:
  connect to `ampwell_prod` for **read-only inspection only — never
  DELETE/DROP/UPDATE directly.** Any data change to prod goes through a reviewed,
  reversible migration in the versions chain, so it is captured in history and
  reproducible on every environment.

- **Back up before irreversible or large data changes.** For anything you can't
  cleanly `downgrade` (e.g. a 019-style purge) or any bulk rewrite, snapshot the
  affected table first (`pg_dump -t <table>`), verify on dev, and only then run
  prod.

---

## 6. Never edit an already-applied migration — the hard rule

Once a migration has been applied anywhere it can't reach back into (prod, a
teammate's dev DB, CI), it is **immutable**. Alembic tracks only the revision ID
in `alembic_version`; it does not re-diff the file. So editing the body of an
applied migration changes what a *fresh* database gets while every
already-migrated database keeps the old effect — the two silently diverge and
`alembic current` still reports "up to date" on both.

Therefore:

- **Do not edit `upgrade()`/`downgrade()` of any migration that has been applied
  outside your own throwaway dev DB.** Fixing a docstring typo is fine; changing
  what the migration *does* is not.
- **To change already-shipped schema, write a new migration** (`NNN+1`) that
  makes the correction going forward. This is exactly how the chain already
  works: 003 replaced 001's global unique constraint with a partial index; 011
  widened a CHECK that 001 first defined; 019 finished cleanup that 010 left
  incomplete. Each is a new, forward step — none reached back to edit its
  predecessor.
- **Do not renumber or delete a migration that others may have applied.** The
  `down_revision` chain and everyone's `alembic_version` row must stay valid.
- **Only exception:** a migration still purely local to your machine — never
  pushed, never applied to any shared DB — may be reworked freely. The moment it
  merges or lands on prod, it's frozen. When in doubt, treat it as frozen and add
  a new one.

---

## 7. Pre-merge checklist

- [ ] Filename is `NNN_snake_case.py`; `revision = "NNN"` matches it.
- [ ] `down_revision` is the true current head (`alembic heads` shows one head).
- [ ] `branch_labels` / `depends_on` are `None`.
- [ ] Docstring has the summary / Revision ID / Revises / Create Date / Changes
      block, plus a `Why:` for anything non-obvious.
- [ ] `downgrade()` exists and genuinely reverses `upgrade()` (objects dropped in
      reverse order; seeds, triggers, functions, and constraints all undone) —
      or is a documented, justified no-op.
- [ ] Any constraint-narrowing has a data step that makes rows legal first.
- [ ] Data steps are idempotent (`ON CONFLICT DO NOTHING`) and use core SQL /
      `sa.table`, not ORM models.
- [ ] New NOT NULL columns on populated tables use the nullable→backfill→tighten
      (or `server_default`) pattern.
- [ ] CHECK constraints, partial indexes, triggers, and functions are hand-written
      (autogenerate misses them); autogenerated diffs were reviewed.
- [ ] Ran `alembic upgrade head` **and** `alembic downgrade -1` clean on dev
      before promoting to prod.
- [ ] Not editing any migration already applied outside your own dev DB.
