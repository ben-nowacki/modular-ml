# AmpWell Python Error Handling Guide

Reference material for agents and developers working on AmpWell's Python code
(the FastAPI backend, the Bridge Agent, and the standalone packages under
`packages/*`). It documents how errors are expected to flow through the system,
the conventions the existing code already follows, and the industry practices
those conventions are built on.

Read this alongside [`docstring_styling.md`](./docstring_styling.md) (every
exception you raise or catch should be documented in the `Raises:` section of
the enclosing function's docstring).

---

## 1. Core principle: raise low, decide high

AmpWell follows the standard layered rule:

> **Libraries raise. Boundaries decide.**

- **Raise** a specific exception at the point where you *detect* a problem you
  cannot sensibly recover from locally. Do not paper over it with a sentinel
  return (`None`, `-1`, `False`) that a caller might forget to check.
- **Let it bubble** through intermediate layers unchanged — do not catch-and-
  re-raise just to add a log line at every frame. That produces duplicate logs
  and loses the original traceback.
- **Catch and convert** only at a *boundary* — a place where a raw Python
  exception must become something the outside world understands: an HTTP
  response, a command-result payload, a `CallbackDecision`, or "log it and keep
  the loop alive."

There are exactly three boundary types in AmpWell. Almost every `except Exception`
in the codebase lives at one of them:

| Boundary | Location | Raw exception becomes... |
| --- | --- | --- |
| HTTP request/response | FastAPI route handlers + `Depends` | an `HTTPException` (status + detail) |
| Bridge Agent poll loop | `bridge_agent/command_loop.py` | a logged warning + retry, or a `{"success": False, "error": ...}` result |
| Callback orchestration | `callback_executor.py` → queue orchestrator | a `CallbackDecision.error(...)` + notification |

Everywhere *else*, prefer to let exceptions propagate.

### When to define a new custom exception

Define one when **a caller needs to distinguish this failure from others** to
react differently. `require_device` catches `KekNotConfiguredError` (→ 503,
"try again later") separately from `SecretDecryptionError` (→ 401, "your secret
is bad") precisely because those map to different HTTP outcomes. If every caller
would treat a failure identically, a built-in (`ValueError`, `RuntimeError`) is
fine.

### When *not* to

- Don't invent an exception you immediately catch in the same function.
- Don't subclass `Exception` when a more specific stdlib base fits (`ValueError`
  for bad input, `TimeoutError` for timeouts, `RuntimeError` for invalid state).
- Don't catch `Exception` to "be safe" mid-stack — you'll swallow bugs
  (`KeyError`, `AttributeError`) that should crash loudly in tests.

---

## 2. The AmpWell exception hierarchy

AmpWell does not use a single shared base exception. Instead each subsystem
defines its own small set, subclassing the **stdlib base whose semantics match**
so that generic `except ValueError` / `except RuntimeError` handlers behave
sensibly even if they don't know the concrete type.

| Exception | Base | Defined in | Meaning |
| --- | --- | --- | --- |
| `AwmParseError` | `ValueError` | `backend/app/model_utils.py` | Uploaded `.awm` bytes are not a valid model file (server-side parser). |
| `ModelExecutionError` | `RuntimeError` | `backend/app/model_executor.py` | A model's `forward()` failed, or its container/IPC could not be reached. |
| `CallbackError` | `RuntimeError` | `backend/app/callback_sandbox.py` | Callback code raised, was malformed, or violated the sandbox. |
| `CallbackTimeoutError` | `CallbackError` | `backend/app/callback_sandbox.py` | Callback exceeded its `timeout_s` budget. |
| `KekNotConfiguredError` | `RuntimeError` | `backend/app/auth/crypto.py` | `AMPWELL_AGENT_KEK` missing/invalid — secret crypto unavailable. |
| `SecretDecryptionError` | `RuntimeError` | `backend/app/auth/crypto.py` | An encrypted blob failed authentication/decryption. |
| `AwmFormatError` | `ValueError` | `packages/ampwell-models/.../serialization.py` | Client-side `.awm` (de)serialization failure. |
| `ProtocolError` | `Exception` | `packages/ampwell-model-harness/.../protocol.py` | Length-prefixed socket frame was truncated/invalid. |
| `RegistrationValidationError` | `ValueError` | `packages/equipment/.../models/equipment.py` | Equipment registration payload is invalid. |
| `ExpressionError` | `Exception` | `packages/protocol/.../expressions.py` | A `ValueExpr` failed to parse/evaluate in the sandbox. |
| `ResolutionError` | `Exception` | `packages/protocol/.../resolver.py` | A protocol could not be resolved to a `ResolvedProtocol`. |
| `CompilationError` | `Exception` | `packages/protocol/.../compiler/neware.py` | A resolved protocol could not be compiled for hardware. |

### `CallbackTimeoutError` — a hierarchy in miniature

`CallbackTimeoutError(CallbackError)` is the one intentional inheritance chain.
It lets the orchestrator write `except CallbackError` to handle *any* callback
failure uniformly, while code that cares specifically about the deadline can
catch `CallbackTimeoutError` first. This is the model to copy when you need a
family: **specific subclass, broad base, catch specific-before-broad.**

### A note on naming (`EquipmentOfflineError`, `HMACVerificationError`)

These names describe *conditions* that AmpWell absolutely handles, but there is
**no custom class** for them today — they are handled inline at the HTTP
boundary as `HTTPException`s (see §3). If you find yourself needing to
distinguish "equipment offline" or "bad HMAC signature" at multiple call sites,
that's the signal to promote them into real exceptions in this table. Until then,
don't reference classes that don't exist.

---

## 3. FastAPI: turning exceptions into HTTP responses

### There is no global exception handler

The backend registers **no** `@app.exception_handler` / `add_exception_handler`.
It relies on two things:

1. FastAPI's built-in handling of `HTTPException` (renders `{"detail": ...}` with
   the given status code) and of `RequestValidationError` (auto-422 for bad
   request bodies).
2. **Per-route conversion**: handlers and dependencies catch domain exceptions
   and re-raise them as `HTTPException`.

Any exception that escapes a route uncaught becomes a generic **500** with the
traceback in the server log — acceptable only for genuine bugs, never for
expected failure modes. If a failure is expected, convert it.

### The conversion pattern

Catch the specific domain exception, raise `HTTPException` with an accurate
status code. Keep the conversion in a tiny helper when it's reused. From the
model-upload pipeline (`backend/app/api/routers/models.py`):

```python
def _parse_or_400(file_bytes: bytes) -> AwmMetadata:
    """Parse .awm bytes, translating parse failures into HTTP 400."""
    try:
        return parse_awm(file_bytes)
    except AwmParseError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))
```

Three habits shown here that you should follow:

- **`raise ... from`? For HTTP conversions we deliberately omit the `from exc`**,
  because the chained traceback is noise once the failure is a clean 4xx the
  client caused. *Do* use `raise NewError(...) from exc` when converting between
  two internal exceptions where the cause aids debugging (see the callback
  example in §5).
- **Status code accuracy** — see the table below.
- **The `detail` can be structured.** When the client needs machine-readable
  context, pass a dict:

```python
raise HTTPException(
    status.HTTP_422_UNPROCESSABLE_ENTITY,
    detail={"message": "Requirements conflict with the execution harness",
            "conflicts": conflicts},
)
```

### Converting in a dependency (auth example)

`require_device` (`backend/app/auth/dependencies.py`) is the canonical example of
a dependency that maps many failure modes to precise statuses:

```python
try:
    key = crypto.derive_agent_hmac_key(agent.hmac_secret_enc, agent.id)
except crypto.KekNotConfiguredError:
    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Agent secret decryption is unavailable (KEK not configured)")
except crypto.SecretDecryptionError:
    raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                        detail="Bridge Agent secret could not be decrypted")
...
if not hmac.compare_digest(expected, signature):
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid HMAC signature")
```

Note the *server-fault* (`KekNotConfiguredError` → 503) vs *client-fault*
(`SecretDecryptionError`, bad signature → 401) split. Getting this right is the
whole point of separate exception types. Also note `hmac.compare_digest` — HMAC
comparisons must be constant-time; never use `==`.

### Status code conventions

| Situation | Status | Notes |
| --- | --- | --- |
| Malformed body/file the client sent | **400** | e.g. `AwmParseError`. |
| Missing/invalid auth, bad signature, expired timestamp | **401** | All HMAC failures land here. |
| Authenticated but lacks permission / no org context | **403** | `check_permission` raises this. |
| Target row not found or cross-org access | **404** | Cross-org is 404, *not* 403 — never confirm existence of another org's data. |
| Locked resource / illegal state transition | **400** | e.g. editing a locked model. |
| Uniqueness/race lost at commit | **409** | catch `IntegrityError`, `rollback()`, raise 409. |
| Semantically invalid but well-formed | **422** | e.g. harness conflicts; use the structured `detail`. |
| Dependency temporarily unavailable | **503** | e.g. KEK not configured. |

### Always roll back before converting a DB error

```python
db.add(entry)
try:
    db.commit()
except IntegrityError:
    db.rollback()
    raise HTTPException(status.HTTP_409_CONFLICT,
                        detail="A concurrent upload claimed this version — retry")
```

A converted `HTTPException` leaves the request's `Session` in whatever state it
was in; if you committed-and-failed you **must** `rollback()` first or the
session is poisoned for the rest of the request.

---

## 4. The Bridge Agent: never crash the poll loop

The Bridge Agent is an unattended process on a lab PC. Its cardinal rule:

> **A single failed poll, command, or upload must never terminate the agent.**

This is achieved with layered `try/except` that gets *broader* the closer you
get to the top-level loop, plus fire-and-forget task isolation.

### Layer 1 — the loop body catches everything, sleeps, continues

From `bridge_agent/command_loop.py`:

```python
while not shutdown_event.is_set():
    try:
        response = await http.poll_commands()
        ...
    except asyncio.CancelledError:
        raise                     # cooperative shutdown — NEVER swallow this
    except Exception as exc:
        logger.warning("Command poll failed: %s; retrying in %ss", exc, poll_interval_s)
        await asyncio.sleep(poll_interval_s)
```

Two non-negotiables:

- **Re-raise `asyncio.CancelledError` first.** Swallowing it breaks task
  cancellation and clean shutdown. Every broad `except Exception` in async code
  must be preceded by `except asyncio.CancelledError: raise`.
- **Broad `except Exception` is correct *here***, at the top of a daemon loop,
  because the whole purpose is resilience. This is the one place the "never
  catch bare Exception" rule is deliberately inverted — and it always pairs the
  catch with a log + backoff, never a silent `pass`.

### Layer 2 — isolate each command as its own task

Commands are dispatched as independent tasks so one hanging or crashing command
can't stall the poll loop:

```python
asyncio.create_task(_dispatch_and_report(dispatcher, http, command),
                    name=f"cmd-{command_id}")
```

`_dispatch_and_report` converts an exception into a *result payload* rather than
letting it escape — the agent's equivalent of the HTTP boundary:

```python
try:
    result = await dispatcher.dispatch(command)
except Exception as exc:
    logger.exception("Unhandled error dispatching command %s", command_id)
    result = {"success": False, "error": f"Unhandled dispatcher error: {exc}"}
await _safe_post_command_result(http, command_id, result)
```

### Layer 3 — retry, then give up gracefully

Delivering the result is itself allowed to fail. `_safe_post_command_result`
retries a bounded number of times, then **logs and returns** — it never raises,
because there is no higher layer that could do anything useful with the failure:

```python
for attempt in range(1, _COMMAND_RESULT_RETRIES + 1):
    try:
        await http.post_command_result(command_id, result)
        return
    except Exception as exc:
        logger.warning("... delivery attempt %d/%d failed: %s", attempt, _COMMAND_RESULT_RETRIES, exc)
        if attempt < _COMMAND_RESULT_RETRIES:
            await asyncio.sleep(_COMMAND_RESULT_RETRY_DELAY_S)
logger.error("Command result %s was not delivered after %d attempts", command_id, _COMMAND_RESULT_RETRIES)
```

### Layer 4 — the HTTP client raises; loops catch

The `AgentHTTPClient` does the opposite — it is a *library*, so it raises. Its
`_raise_for_status` logs context and re-raises `requests.HTTPError`; callers up
in the loops decide what to do. Its docstrings list `requests.HTTPError` /
`requests.RequestException` under `Raises:` so callers know what to expect.

The same pattern guards side activities: a failed update download is caught,
logged at `error`, and retried next cycle rather than crashing the agent.

---

## 5. Package-level exceptions (`packages/*`)

The standalone packages (`ampwell-protocol`, `ampwell-models`,
`ampwell-model-harness`, `ampwell-equipment`, `ampwell-bridge-agent`) are
**libraries**. They should almost never catch broadly and never know about HTTP.
They *raise* their typed exceptions (`ExpressionError`, `ResolutionError`,
`CompilationError`, `AwmFormatError`, `ProtocolError`, ...) and let the consuming
application decide the boundary behavior.

Example of the consumer converting between internal exceptions **with** cause
chaining (`callback_executor.py`), which — unlike HTTP conversions — *keeps* the
`from exc` so the model failure is visible in logs:

```python
try:
    proxies[arg_name] = get_model_proxy(model_name, model_version, db)
except ModelExecutionError as exc:
    raise CallbackError(
        f"Cannot load model '{model_name}' v{model_version} for "
        f"callback argument '{arg_name}': {exc}"
    ) from exc
```

Here `ModelExecutionError` (raised by the executor library) is caught at the
callback-orchestration boundary and re-cast as `CallbackError`, which the queue
orchestrator ultimately turns into a `CallbackDecision.error(...)` and a
`run_error` notification — not a 500. That is the third boundary type in action.

---

## 6. Logging conventions

Pick the level by *who needs to act* and whether a traceback helps:

| Call | Use when | Includes traceback? |
| --- | --- | --- |
| `logger.exception(msg)` | An unexpected error you're handling at a boundary and want the full stack for. Only valid **inside an `except`**. | Yes (automatic) |
| `logger.error(msg, ...)` | A failure the operator should notice, but the stack adds nothing (e.g. "gave up after N retries"). | No |
| `logger.warning(msg, ...)` | Transient/expected-ish failures the loop will retry past. | No |
| `logger.info(msg, ...)` | Normal lifecycle events (upload accepted, update staged). | No |

Rules of thumb:

- Use **`logger.exception`** exactly once per failure, at the boundary that
  handles it — not at every frame it passed through.
- Use lazy `%`-formatting (`logger.warning("failed: %s", exc)`), **not**
  f-strings, so the string isn't built when the level is disabled.
- **Never** log secrets: HMAC secrets, KEK, Neware passwords, TOTP seeds, or
  decrypted blobs must never reach a log line (see the security invariants in
  the Bridge Agent notes).

---

## 7. Quick checklist

When adding or reviewing error-handling code:

- [ ] Am I raising the **most specific** exception whose base semantics fit
      (`ValueError` for bad input, `TimeoutError` for timeouts, a custom class
      only if a caller must distinguish it)?
- [ ] Did I document it in the function's `Raises:` docstring section?
- [ ] Am I catching **only at a boundary** (HTTP handler, agent loop, callback
      orchestrator) — not mid-stack "to be safe"?
- [ ] If this is async, does my broad `except` re-raise `asyncio.CancelledError`
      first?
- [ ] Does a broad `except Exception` always pair with a log **and** a concrete
      recovery (retry/backoff, error payload, error decision) — never silent
      `pass`?
- [ ] At the HTTP boundary: correct status code, `rollback()` before converting
      a DB error, structured `detail` when the client needs context, cross-org
      access returns 404 not 403?
- [ ] Am I using `logger.exception` inside the `except`, and `%`-formatting?
- [ ] No secrets in any log line or exception message?
- [ ] For internal→internal conversions, did I keep `raise ... from exc`? For
      client-caused HTTP 4xx, did I omit it?
```
