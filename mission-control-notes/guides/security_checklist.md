# Security checklist (pre-PR)

Reference material for Claude Code agents and human developers. This is a
**threat-model checklist, not a style guide**. It exists so that no PR silently
removes or bypasses one of AmpWell's security invariants. Work through the
relevant sections before opening any PR that touches auth, file access, the
Bridge Agent surface, database queries, or logging.

Each section is: **the mechanism that exists → what to check → the rule to
uphold**. File references are load-bearing; when you change one of these files,
re-read the corresponding section.

---

## 1. No secrets or tokens in committed code

### Mechanism
Every secret is an **environment variable with a Pydantic alias** in
`app/config.py` — `SECRET_KEY`, `AMPWELL_SECRET_KEY`, `AMPWELL_AGENT_KEK`,
`AMPWELL_SMTP_PASSWORD`, `DATABASE_URL`. The built-in defaults are *deliberately
insecure sentinels* (`"dev-secret-change-in-production-please"`), and a
**fail-closed startup gate** refuses to boot in production if they survive:

```python
# config.py — assert_secure_configuration(), called from the app lifespan
if cfg.secret_key == DEFAULT_SECRET_KEY:
    problems.append("SECRET_KEY is still the built-in development default")
elif len(cfg.secret_key) < _MIN_SECRET_LEN:
    problems.append(f"SECRET_KEY must be at least {_MIN_SECRET_LEN} characters")
# ...KEK must parse to a valid 32-byte key; CORS must be an explicit allow-list...
# non-debug + any problem  -> raise RuntimeError (process will not start)
```

The KEK that unwraps at-rest secrets **lives only in the process environment**,
never in the database (§3), so a DB dump alone unlocks nothing.

### Check before PR
- [ ] No literal secret, token, password, API key, private key, or connection
      string with credentials appears in any committed file (including tests,
      fixtures, migrations, comments, and `notes/`).
- [ ] New secret-bearing config is a `Field(default=<insecure-sentinel-or-"">,
      alias="AMPWELL_...")`, and if it must be strong in prod, it's added to
      `security_problems()` so the boot gate enforces it.
- [ ] `.env`, `certs/`, `*.pem`, KEK material, and the agent `config.json` stay
      untracked (confirm `git status`; grep the diff for `SECRET`, `KEK`,
      `PASSWORD`, `BEGIN PRIVATE KEY`).
- [ ] No secret is pasted into a PR description, commit message, or issue — PRs
      are published to GitHub and indexed.

### Rule to uphold
Secrets come from the environment or a secrets manager at runtime, are validated
by the boot gate, and never touch version control. Rotating a secret must be a
config change, never a code change.

---

## 2. HMAC headers required on all Bridge Agent endpoints

### Mechanism
Every Bridge Agent route depends on `require_device` (`app/auth/dependencies.py`),
which verifies an HMAC-SHA256 signature over the canonical request and enforces
anti-replay. It is the *only* auth on those routes — agents never hold a session
cookie. The signed message binds method, path, timestamp, nonce, and body digest:

```
{agent_id}:{METHOD}:{path}:{timestamp}:{nonce}:{body_sha256}
```

The verification key is **re-derived from a KEK-encrypted secret** each request
(`crypto.derive_agent_hmac_key`), so a DB-only compromise cannot forge a
signature. Protections layered in `require_device`:

- **Timestamp freshness** — rejects requests outside `hmac_replay_window_seconds`.
- **Nonce anti-replay** — a `{agent_id}:{nonce}` seen again within the window is
  rejected; the nonce is burned **only after** the signature verifies, so an
  attacker can't pre-poison the cache.
- **Timing-safe comparison** — `hmac.compare_digest(expected, signature)`, never
  `==`.
- **Body binding** — JSON bodies are digested and signed; multipart uploads bind
  the client-declared `X-AmpWell-Content-SHA256` (re-verified against the
  received bytes at the upload endpoint, §3).

### Check before PR
- [ ] Every new agent-facing route has `agent = Depends(require_device)` — no
      exceptions, including health/probe endpoints that echo any org data.
- [ ] The route asserts `agent_id`/tenant match where a path/body carries an id
      (`_assert_agent_id_matches`, `str(agent.id) != agent_id → 403`).
- [ ] If the signing scheme changes, the agent's `sign_request`
      (`packages/bridge-agent/`) and `compute_agent_signature` change **together**
      — the canonical string must match byte-for-byte.
- [ ] New comparisons of secrets/signatures/digests use `hmac.compare_digest`.

### Rule to uphold
Agent authentication is HMAC-only, replay-protected, and body-bound. Never add a
bypass "for testing," never widen the replay window casually, and never store or
compare the raw secret — always the KEK-derived key with a constant-time compare.

---

## 3. Signed URLs (and the KEK) for file access

### Mechanism — two distinct file paths, each fully gated
**Agent downloads (staging protocol files)** use **double auth**: `require_device`
HMAC headers **plus** a short-lived signed-URL token
(`files.generate_signed_url_token`), HMAC-SHA256 of `"{filename}:{expires}"`
under `AMPWELL_SECRET_KEY`, default TTL 300 s, verified constant-time:

```python
if not _verify_signed_url_token(filename, token, expires):
    raise HTTPException(403, "Invalid or expired download token")
# + path-traversal guard: reject "/", "\", ".." in filename
```

**Agent uploads** stream to a **tenant- and run-scoped** path
(`uploads_dir/{org_id}/{run_uuid}/`), sanitize the filename to a single
component (`safe_dest_path`, rejecting traversal), and **gate on content
integrity**: the received bytes are re-digested and `compare_digest`'d against
the signed `X-AmpWell-Content-SHA256`; a mismatch deletes the partial file and
returns 400 *before* any DB row or usage accounting.

**At-rest secrets** (agent HMAC secrets, TOTP seeds) are **AES-256-GCM under the
KEK** (`crypto.encrypt_secret`), stored as `nonce || ciphertext || tag`, with the
**agent id bound as AEAD associated data** so a blob copied between rows fails
authentication.

**User-facing downloads** (exports, artifacts) are gated by **session +
permission + org scoping** (§4/§6), not signed URLs — the correct model for a
browser client carrying a cookie.

### Check before PR
- [ ] Any new agent file endpoint enforces signed-URL token **and** HMAC, with a
      short TTL and a constant-time token check.
- [ ] Any user-supplied filename or id that becomes a path component is validated
      (UUID-parse ids; `safe_dest_path`/traversal-reject names) — never trust a
      name from an authenticated-but-remote agent.
- [ ] Uploaded/streamed content is integrity-checked (`compare_digest` on
      SHA-256) before it is persisted or accounted.
- [ ] New recoverable secret columns use `encrypt_secret`/`decrypt_secret` with
      an AAD that binds the row identity; the KEK is never logged or persisted.
- [ ] User-facing file reads are org-scoped and permission-gated (no IDOR: filter
      by `organization_id`, don't fetch by bare id).

### Rule to uphold
Agent file access = signed URL + HMAC + traversal-safe path + content-digest gate.
User file access = session + permission + tenant scope. Recoverable secrets at
rest are always AEAD-encrypted under the KEK with row-binding AAD.

---

## 4. Session-token validation on all user-facing endpoints (there is no JWT)

### Mechanism
User auth is a **server-side opaque session token**, not a JWT:

- The token is `secrets.token_hex(32)` (256 bits of entropy) generated in
  `auth/session.py`. **Only its SHA-256 hash is stored** (`token_hash`); the raw
  value never hits the database.
- It rides in an **httpOnly, `SameSite=strict`, `Secure`** cookie
  (`Secure` is dropped only under `AMPWELL_DEBUG`):
  ```python
  response.set_cookie(_COOKIE_NAME, token, max_age=max_age,
                      httponly=True, samesite="strict",
                      secure=not settings.ampwell_debug, path="/")
  ```
- `get_current_session` validates on every request: token → hash → row lookup,
  **absolute expiry**, **idle timeout**, and rejects sessions still `mfa_pending`.
  `get_current_user` additionally rejects suspended accounts.
- **Stateful states** a JWT can't cleanly express and this design gets for free:
  `is_restricted` (password-reset-only), `requires_email_setup`, `mfa_pending`,
  in-place token rotation after password change (`rotate_session_token`), and
  **instant server-side revocation** (`delete_session` /
  `delete_all_user_sessions`).

**Why this beats JWT here:** revocation is immediate (no token-still-valid
window), the credential is never readable by JS (httpOnly), and step-up/restricted
states are enforced server-side per request. A stateless JWT would trade all of
that away for scaling this app doesn't need.

Supporting controls: **bcrypt over a SHA-256 pre-hash** for passwords (avoids
bcrypt's 72-byte truncation, `auth/password.py`); **TOTP MFA** with seeds
KEK-encrypted; **login throttle** (`auth/login_throttle.py`) — sliding window
over **both** account and IP keys, lockout after `login_max_attempts`, with
non-existent accounts counted identically to real ones (no user enumeration).

### Check before PR
- [ ] Every user-facing route depends on `get_current_user` (or a
      `require_*`/`require_permission` wrapper) — never reads identity from an
      unauthenticated header or query param.
- [ ] Privileged actions use the right gate: `require_permission("...")`,
      `require_org_admin`, `require_system_admin`, and honor `is_restricted`.
- [ ] No new endpoint trusts a client-supplied `user_id`/`org_id` for authz;
      identity comes from the validated session only.
- [ ] Session cookies keep `httponly=True`, `samesite="strict"`, and `Secure` in
      non-debug; new tokens are stored **hashed**, never raw.
- [ ] Auth-state changes (password change, reset, role change) invalidate or
      rotate sessions as appropriate.

### Rule to uphold
Identity is the validated server-side session, checked every request, revocable
instantly, delivered in an httpOnly+SameSite+Secure cookie. Do not introduce
JWTs or any client-trusted identity claim.

---

## 5. SQL injection prevention via the ORM

### Mechanism
All data access goes through **SQLAlchemy ORM / Core with bound parameters** —
`db.query(Model).filter(...)`, `select(...)`, `update(...)`, `delete(...)`.
There is **no string-interpolated SQL from user input** anywhere. The two things
that look like raw SQL are both safe:

- `command_events.py`: `text("SELECT pg_notify(:channel, :payload)")` — **bind
  parameters**, values passed in a dict.
- `command_events.py`: `cur.execute(f"LISTEN {_PG_CHANNEL}")` — the channel is a
  **module constant**, not user input (Postgres `LISTEN` can't be parameterized).

### Check before PR
- [ ] New queries use ORM/Core constructs. No f-strings, `%`, `.format()`, or
      `+` build a SQL string from any request-derived value.
- [ ] If raw `text()` is unavoidable, **every** dynamic value is a bind parameter
      (`:name` + params dict) — identifiers that genuinely can't be bound (table
      names, `LISTEN` channels) must come from a fixed allow-list/constant, never
      from input.
- [ ] `ORDER BY` / column choice driven by client input maps through an explicit
      allow-list, not the raw string.
- [ ] Multi-tenant queries filter on `organization_id` so a valid id from one org
      can't read another's rows.

### Rule to uphold
Parameterize everything; interpolate nothing from input. The ORM is the default
and the raw-SQL exceptions are constants-only and bound-params-only.

---

## 6. Audit-log entries for all state-changing operations

### Mechanism
`services/audit.record_audit` writes an `audit_log` row **inside the caller's
transaction** (added + flushed, not committed) so the action and its audit entry
**commit or roll back together** — you cannot have one without the other. The log
is defense-in-depth hardened:

- **Append-only at the database level** — a BEFORE UPDATE/DELETE trigger rejects
  mutation of existing rows.
- **Tamper-evident at the application level** — each row stores a SHA-256
  `row_hash` over its canonical fields **chained to the previous row's hash**;
  `verify_audit_chain` re-walks the chain and reports the first broken row, so
  any edit, deletion, or reordering is detectable.

```python
record_audit(db, action="data_file.uploaded",
             organization_id=agent.organization_id,
             resource_type="data_file", resource_id=data_file.id,
             context={"run_id": ..., "channel": ..., "bytes": ...})
db.commit()   # action + audit row commit atomically
```

### Check before PR
- [ ] Every state-changing action (create/update/delete, auth events, admin
      mutations, command dispatch, run lifecycle, file uploads) calls
      `record_audit` **in the same transaction** as the change.
- [ ] The `action` is dot-namespaced (`resource.verb`), and `organization_id` +
      `resource_type`/`resource_id` are set; reversible changes capture
      `before_state`/`after_state`.
- [ ] Browser-originated actions pass `ip_address`; system/agent actions leave
      `user_id` null rather than faking an actor.
- [ ] The audit write is **not** separately committed or wrapped in its own
      try/except that could let the action commit while the audit fails.

### Rule to uphold
If it changes state, it emits an audit row in the same transaction. Never write
around the append-only trigger or the hash chain; never "best-effort" the audit
entry separately from its action.

---

## 7. No sensitive data in log output

### Mechanism
Logs record **identifiers and outcomes, never secrets**. The pattern is visible
in the codebase — e.g. secret rotation logs the agent id and org slug, not the
secret:

```python
logger.info("Rotated Bridge Agent secret id=%s org=%s", agent.id, org.slug)
```

Passwords, raw session/reset tokens, HMAC secrets, the KEK, TOTP seeds, and full
`Authorization`/`Cookie` headers are never logged. Errors log the *fact* of a
failure (e.g. "secret could not be decrypted"), not the material involved.

### Check before PR
- [ ] No `logger.*` (or `print`) emits a password, raw token, session cookie,
      HMAC secret, KEK, TOTP seed, or full auth header — grep the diff for
      `password`, `token`, `secret`, `cookie`, `authorization`, `kek`.
- [ ] Exceptions/tracebacks that could carry a secret (e.g. a request body, a
      decrypted blob) are summarized, not dumped verbatim.
- [ ] New log lines that reference a credentialed object log its **id**, not its
      secret field; PII in logs is minimized to what's needed to operate.
- [ ] Error responses to clients don't leak internals beyond the standard
      `{"detail": ...}` message (see `api_response_patterns.md`).

### Rule to uphold
Log identifiers, actions, and outcomes — never the secret itself. When in doubt,
log the id and the verb, not the value.

---

## Cross-cutting invariants (don't regress these)

- **Constant-time comparison** (`hmac.compare_digest`) for every
  secret/signature/digest/token check — never `==`.
- **Fail closed.** Missing KEK, insecure secret, or wildcard CORS refuses to boot
  in production; missing/invalid auth is a `401`/`403`, never a soft pass.
- **Multi-tenant isolation.** Filter by `organization_id` on every org-scoped
  query; resolve external ids to internal rows within the actor's org (no IDOR).
- **Defense in depth.** Agent file access carries *both* HMAC and a signed URL;
  uploads are *both* traversal-safe *and* content-digest-gated. Don't collapse a
  layered control to a single check.
- **Secrets recoverable at rest are AEAD-encrypted under the KEK with
  row-binding AAD** — never stored plaintext, never keyed from a DB column.

## Final pre-PR gate
- [ ] Ran the diff against §1–§7 checklists above; each touched area passes.
- [ ] `assert_secure_configuration` still passes with production-shaped config.
- [ ] No new endpoint is unauthenticated, no query is string-built from input, no
      state change skips its audit row, no secret reaches a log or the repo.
- [ ] If auth/crypto/signing changed, server and Bridge Agent were updated in
      lockstep and the change is called out in the PR description.
