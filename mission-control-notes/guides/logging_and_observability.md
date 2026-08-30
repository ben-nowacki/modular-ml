# Logging & Observability Guide

Reference for working on logging in the AmpWell backend (`backend/app/`) and the
Bridge Agent (`packages/bridge-agent/`). Covers where logs go, what belongs at each
level, message conventions, what the agent logs vs. suppresses, and how to log
without leaking credentials.

> **Reality check (read first).** Three premises worth correcting against the
> actual codebase before you write or review logging code:
>
> 1. **There is no structured / JSON logging.** Both sides use stdlib `logging`
>    with a plain-text format (`%(asctime)s %(levelname)s %(name)s: %(message)s`).
>    No `structlog`, no `python-json-logger`, no required JSON keys. §3 documents
>    the plain-text conventions that fill that role, and the JSON upgrade path if
>    it is ever wanted.
> 2. **Backend `logger.info()` calls are currently silently dropped.** The backend
>    never configures the root logger (`basicConfig`/`dictConfig` appear nowhere in
>    `backend/`), and uvicorn's default config only wires up its own `uvicorn.*`
>    loggers. App loggers therefore inherit the root default of WARNING — verified
>    by test: all ~38 `logger.info(...)` call sites (session-cleanup counts, agent
>    registration, run events, ...) never emit, in dev or prod. WARNING and above
>    escape only via Python's last-resort stderr handler, unformatted. §1.3 has the
>    fix recipe; it has deliberately **not** been applied — flag it in your PR if
>    you take it on.
> 3. **The Bridge Agent log file does not rotate.** `run.py` uses a plain
>    `logging.FileHandler`, so `logs/bridge_agent.log` grows without bound on the
>    lab PC. §1.2 has the recommended `RotatingFileHandler` upgrade.

---

## 1. Where logs live and how they rotate

### 1.1 Backend — stdout/stderr → journald

The backend process writes to stdout/stderr and owns no log files. In production
(`backend/scripts/setup_prod_service.sh`) uvicorn runs under systemd:

```ini
ExecStart=... uvicorn app.main:app --host 127.0.0.1 --port 8001 \
    --workers 2 --log-level info --access-log
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ampwell-api
```

Everything — uvicorn startup, access log, app log records — lands in **journald**
under the identifier `ampwell-api`:

```bash
journalctl -u ampwell-api -f                 # follow live
journalctl -u ampwell-api --since "1h ago"   # recent window
journalctl -u ampwell-api -p warning         # warnings and errors only
```

**Rotation is journald's job.** Size caps, compression, and retention come from
`/etc/systemd/journald.conf` (`SystemMaxUse=`, `MaxRetentionSec=`). Never add a
`FileHandler` to backend code — one process, one stream, let the platform manage
retention. This is the standard twelve-factor arrangement and is correct as-is.

`settings.logs_dir` (`$AMPWELL_DATA_DIR/logs`, `config.py`) is created at startup
by `main.py` but **nothing writes to it today** — it is reserved for future
per-run artifacts (e.g. model-executor output). Do not assume backend logs live
there.

### 1.2 Bridge Agent — files next to the install

Two files, both under `logs/` in the agent install directory (a path the updater's
`_PROTECTED` set guarantees is never overwritten by an update package):

| File | Written by | Format |
|---|---|---|
| `logs/bridge_agent.log` | `run.py` (`basicConfig`, level INFO, FileHandler + stdout) | `%(asctime)s %(levelname)s %(name)s: %(message)s` |
| `logs/launcher.log` | `launcher.pyw` (same, name hardcoded to `launcher`) | `%(asctime)s %(levelname)s launcher: %(message)s` |

The split matters for debugging: the launcher log records update installs, hash
verification, and restarts; the agent log records everything the running agent
does. An update that bricks the agent leaves its evidence in `launcher.log`.

**Current gap: no rotation.** A plain `FileHandler` appends forever. An agent that
runs for months on a lab PC, logging every run start/finish and upload, will grow
this file indefinitely. When touching `run.py`, the best-practice upgrade is:

```python
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(
            _LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
```

~30 MB ceiling, zero external dependencies. Prefer size-based over
`TimedRotatingFileHandler` here: rollover happens on a write from the owning
process, which behaves better on Windows where an open file cannot be renamed by
anyone else. Apply the same to `launcher.pyw`.

### 1.3 Backend fix recipe — make `logger.info()` emit

Uvicorn's `uvicorn` / `uvicorn.access` loggers set `propagate=False`, so giving the
root logger a handler does **not** duplicate uvicorn's own lines. The minimal,
worker-safe fix is one call at the top of `app/main.py`, before the app is built:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
```

`basicConfig` is a no-op if the root logger already has handlers, so it composes
safely with test runners and scripts that configure logging themselves. Until this
lands, remember when debugging: absence of an INFO line in journald does not mean
the code path didn't run.

---

## 2. Log levels and what belongs at each

Measured usage today — backend: 38 `info`, 17 `warning`, 12 `exception`, 9 `error`,
1 `debug`; agent: 32 `info`, 38 `warning`, 9 `exception`, 6 `error`, 5 `debug`.
The rules below describe the conventions those call sites already follow.

| Level | Belongs here | AmpWell examples |
|---|---|---|
| `DEBUG` | High-volume or best-effort detail useful only when actively debugging. Off by default. | Tray-icon stop failure on shutdown (`agent.py:290`) |
| `INFO` | State transitions an operator would want in the timeline: lifecycle events, counts of work done, one line per meaningful unit. | `"Agent [%s] registered: %d new devices, %d new channels"`; `"Session cleanup: deleted %d expired session(s)"`; `"Update v%s staged and verified: %s"` |
| `WARNING` | Something failed but the system continues by design — transient network errors that will be retried, degraded modes, suspicious-but-handled input. | `"Command poll failed: %s; retrying in %ss"`; `"pystray/Pillow not installed; tray icon disabled"`; insecure-config findings in debug mode |
| `ERROR` | An operation definitively failed and will not be retried automatically, but the process survives. | `"Update download failed: %s; will retry next cycle"` (paired with a longer deferral); missing `config.json` at startup |
| `logger.exception(...)` | ERROR **plus traceback** — reserve for *unexpected* failures where the stack is the diagnostic. Always call from inside an `except` block. | `"Session cleanup task failed"`; `"Unhandled error dispatching command %s"` |

Two house rules worth internalizing:

- **Expected transient failures get `warning` with `%s` of the exception — not
  `exception()`.** The agent's poll loop fails every time the server restarts or
  the network blips; a full traceback per blip is noise that buries real faults.
  The stack trace is reserved for the cases where nobody predicted the failure.
- **The audit log is not the application log.** State-changing, security-relevant
  operations (login, lockout, secret rotation, artifact locking, ...) go through
  `record_audit()` into the hash-chained `audit_log` table — that is the
  compliance record, tenant-scoped and tamper-evident. `logging` output is for
  operators diagnosing the process. Adding a `logger.info` never substitutes for a
  missing audit entry, and vice versa. See `security_checklist.md`.

---

## 3. Message format conventions (the "structured" contract)

There is no JSON formatter, so the structure lives in message discipline. The
required "keys" of every log line are supplied by the formatter and by convention:

1. **Timestamp, level, logger name** come from the format string — which is why
   every module must open with:

   ```python
   logger = logging.getLogger(__name__)
   ```

   Never `logging.getLogger()` (root) and never a made-up name; `__name__` makes
   the emitting module greppable (`app.api.routers.bridge_agent`,
   `bridge_agent.status_loop`).

2. **Lazy `%` formatting, never f-strings.** Every existing call site does
   `logger.info("Agent %s downloading update package v%s", agent.id, version)`.
   This defers string interpolation until a handler accepts the record and keeps
   arguments machine-separable. Match it.

3. **Stable identifiers in every message.** A line is only useful if it can be
   correlated: include the entity IDs the reader will pivot on — agent id, run id,
   channel label, org slug, command id. For multi-field lines the codebase uses
   `key=value` pairs, which is the greppable middle ground short of JSON:

   ```python
   logger.info(
       "run_event agent=%s run=%s channel=%s type=%s",
       agent_id, body.run_id, body.channel_id, body.event_type,
   )
   ```

   Prefer this shape for any new line carrying more than two fields.

4. **Truncate untrusted payloads.** When echoing a server/client response body
   into a log, cap it — `response.text[:500]` in `http_client.py` is the
   precedent. Never log a full request/response dump at INFO.

**JSON upgrade path (future, not current).** If log aggregation ever demands it,
the right move is a `logging.Formatter` subclass (or `python-json-logger`) applied
in the §1.3 `basicConfig` — required keys would be `ts`, `level`, `logger`, `msg`,
plus `agent_id` / `run_id` / `org` via `extra={...}` where relevant. Because every
call site already uses lazy args rather than pre-baked f-strings, this swap is a
formatter-only change. Do not introduce it piecemeal in one module.

---

## 4. Bridge Agent: what to log vs. what to suppress

The agent log must stay readable over a multi-week run on a lab PC. The governing
question: *would an operator scrolling this file reconstruct what happened to their
runs?* Everything else is suppressed.

**Log (INFO):**
- Startup, config load, journal recovery (`"Recovered %d in-flight run(s)..."`),
  clean-shutdown initiation and completion.
- Equipment registration results.
- Update lifecycle: available → staged/verified (with version and package name) →
  handoff to launcher. Hash-mismatch re-downloads at WARNING.
- Run lifecycle: natural completion (`"Run %s on %s completed naturally (%s ->
  FINISHED)"`), interrupted finish, each successful upload
  (`"Uploaded %s -> file_id=%s"`).

**Suppress (no log line at all):**
- **Empty long-polls.** `run_command_poll_loop` logs nothing when
  `response["commands"]` is empty — at a ~1 s effective cadence, logging polls
  would render the file useless within a day. Same for every routine
  status-broadcast tick: `status_loop` logs state *transitions* and errors, never
  the periodic heartbeat itself.
- **Per-request HTTP successes.** `http_client.py` contains exactly one log call —
  the `_raise_for_status` failure path. Successful requests are silent.
- **Repeated identical failures at full volume.** The poll loop logs one WARNING
  per failed cycle without a traceback; if you add a new retry loop, follow suit
  (or log the first failure and then every Nth) rather than stacking
  `logger.exception` per attempt.

**Error-path discipline** (see `python_async_patterns.md` for the task rules):
`asyncio.CancelledError` is always re-raised, never logged as a failure; unexpected
crashes of a dispatched `cmd-*` task get one `logger.exception` at the dispatch
boundary (`command_loop.py:127`), not one per layer.

---

## 5. Adding a log entry without leaking secrets

The threat model: `bridge_agent.log` sits on a shared lab PC readable by anyone at
the bench; journald is readable by ops staff who are not supposed to hold tenant
credentials. A leaked HMAC secret means impersonating the agent until rotation.

**Never log, at any level, in any process:**

| Secret | Where it lives | Log instead |
|---|---|---|
| Raw HMAC secret / derived signing key | Windows Credential Manager (agent), `hmac_secret_enc` AEAD blob (server) | `agent.id` and the *event* — the precedent is `ui_equipment.py:324`: `"Rotated Bridge Agent secret id=%s org=%s"` |
| `X-AmpWell-Signature` / `X-AmpWell-Nonce` header values | per-request | method + path + status code |
| Session tokens / `Authorization` header / `ampwell_session` cookie | httpOnly cookie, hashed in DB | user id or login id |
| Signed-URL and email-link tokens (`?token=...`) | query strings | filename and expiry |
| Passwords, TOTP seeds, recovery codes | bcrypt / AEAD / SHA-256 in DB | nothing — log the outcome (`auth.lockout` goes to the **audit** log, not here) |
| `AMPWELL_AGENT_KEK`, `AMPWELL_SECRET_KEY`, `.env` contents | env only | the *name* of the missing/invalid variable (as `assert_secure_configuration` does) |

Rules that keep it that way:

1. **Log identities and events, never credentials.** If a message needs to prove
   *which* secret was involved, the entity ID (`agent.id`, key *name*) is always
   sufficient.
2. **Never log a headers dict or raw request.** `auth_headers(...)` output contains
   the signature; a casual `logger.debug("headers=%s", headers)` while debugging is
   the single most likely way to leak. Log `method, path` and let the server's 401
   detail tell you which check failed. Neither `security.py` (agent) nor
   `dependencies.py` (server) contains any logging today — keep it that way.
3. **Watch exception messages.** `logger.warning("... %s", exc)` prints `str(exc)`.
   Library exceptions can embed URLs — and a URL can embed a `?token=...` query
   param. When logging exceptions around signed-URL downloads, log the filename
   you requested, not the full URL from the exception.
4. **Truncated response bodies are still bodies.** `response.text[:500]` is
   acceptable because AmpWell error responses are structured `detail` strings; do
   not echo bodies from endpoints that might reflect input containing tokens.
5. **Known accepted exposure — signed-URL tokens in the access log.** Staging
   downloads pass `token` as a query parameter (`files.py`), and uvicorn's
   `--access-log` records full request paths into journald. This is accepted
   because the token is HMAC-scoped to one filename, expires in 5 minutes, and the
   endpoint is additionally HMAC-authenticated — but it is exactly why this
   pattern must **never** be extended to long-lived credentials. New endpoints
   carry tokens in headers or the body, not the query string.

---

## 6. Pre-PR checklist

- [ ] New module logs via `logger = logging.getLogger(__name__)`; no root logger,
      no `print()`.
- [ ] Lazy `%` args (`logger.info("x=%s", x)`), not f-strings.
- [ ] Level matches §2: retried/transient → WARNING without traceback; unexpected →
      `logger.exception` once, at the boundary; timeline events → INFO.
- [ ] Message carries the pivot IDs (agent/run/channel/org/command); multi-field
      lines use `key=value` form.
- [ ] Nothing from the §5 table appears in any message, including via `%s` of an
      exception, a headers dict, or an un-truncated response body.
- [ ] No per-poll / per-tick / per-request-success log lines in loops.
- [ ] State-changing security events also hit `record_audit()` — a log line is not
      an audit entry.
- [ ] No new `FileHandler` in backend code (journald owns retention); agent file
      handlers rotate (§1.2).
- [ ] If your change depends on seeing backend INFO output, remember it is
      currently dropped (§1.3) — configure logging or verify at WARNING.
