# Python async patterns

Reference material for Claude Code agents and human developers working on
AmpWell's concurrent paths: the FastAPI backend, its lifespan background tasks,
the `asyncio.Event`-driven long-poll, and the Bridge Agent's polling loops.

> **Read this first — the one fact that governs everything below.**
> **The backend database layer is fully *synchronous* SQLAlchemy.** There is no
> `AsyncSession`, no `create_async_engine`, no `async_sessionmaker`.
> `app/database.py` builds a plain `create_engine` + `sessionmaker`, and the
> `get_db()` dependency is a **sync generator** yielding a `Session`. So the
> title of this domain notwithstanding, there is *no async session lifecycle to
> manage.* The real skill here is the opposite one: **running synchronous DB
> work without blocking the event loop.** §1 documents the session lifecycle we
> actually have and the rule that keeps it safe; the rest of the guide is about
> the genuinely-async surfaces (§2 background tasks, §3 the long-poll event, §4
> the agent loops) and how each one keeps blocking work off the loop.

---

## 1. The (synchronous) session lifecycle in FastAPI

### What the code actually does

`app/database.py` is the whole story:

```python
engine = create_engine(
    str(settings.database_url),
    pool_pre_ping=True,   # verify connections (important across Docker restarts)
    pool_size=5,
    max_overflow=10,
)

SessionFactory = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False,
)

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a database session."""
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()
```

A request-scoped session opens when the dependency is resolved and closes when
the request finishes. `expire_on_commit=False` means ORM instances stay usable
after `commit()` (you can serialize them into the response without a re-`SELECT`).

### Why "no async session" is fine — the threadpool

**Prefer a plain `def` route handler.** FastAPI runs every non-`async` handler
in its own threadpool (anyio worker threads), so a sync handler that blocks on
`Session.query(...)` blocks *a worker thread*, never the event loop. This is the
default across the codebase — e.g. `command_result` in
`app/api/routers/bridge_agent.py` is a plain `def`.

**The rule that keeps the loop healthy:**

> **Never touch a synchronous `Session` from inside an `async def` running on the
> event loop.** A blocking DB call there stalls *every* concurrent request on
> that worker. If a path must be `async def` (because it awaits something — see
> §3), wrap the DB work in `await asyncio.to_thread(...)` so it runs on a thread.

The long-poll endpoint is the canonical example of an `async def` that obeys
this rule: it does its DB reads in a helper run via `asyncio.to_thread` and
never calls the `Session` directly (§3).

### Session hygiene rules

- **Request handlers** take `db: Session = Depends(get_db)`; the dependency owns
  the lifecycle. Don't `close()` it yourself *unless* you are deliberately
  releasing the pooled connection early before a long wait — the long-poll does
  exactly this: `db.close()` right after auth, so a parked poll holds no pooled
  connection (§3).
- **Code that is not a request** (background loops, threads, `to_thread`
  helpers) must **not** use `get_db()`. Open a short-lived session directly:
  ```python
  db = SessionFactory()
  try:
      ...  # do the work, commit if writing
  finally:
      db.close()
  ```
  This is the pattern every lifespan `*_tick` uses (§2). One session per unit of
  work, always closed in `finally`.
- **One session is not thread-safe.** Never share a `Session` across threads or
  across `to_thread` calls. Each thread/task opens and closes its own.
- The pool is small (`pool_size=5`, `max_overflow=10`). Long-held connections
  starve it — this is *why* the long-poll releases its connection before waiting,
  and why background ticks use short-lived sessions instead of holding one open
  across an `await asyncio.sleep(3600)`.

---

## 2. Background task patterns (lifespan tasks & daemon threads)

AmpWell has no separate orchestrator *process*. The "background daemon" work is
a set of long-lived tasks the FastAPI **lifespan** owns, plus a couple of
`daemon=True` threads for genuinely-blocking listeners. All of it starts in
`lifespan()` in `app/main.py` and is torn down on shutdown.

### Pattern A — periodic maintenance loop (`asyncio.create_task`)

Every periodic job follows the same three-part shape: a **sync tick** that does
the DB work in its own session, an **async loop** that sleeps then offloads the
tick to a thread, and a **`create_task`** in the lifespan.

```python
# 1. Sync tick — its own short-lived session, closed in finally.
def _command_timeout_tick() -> int:
    db = SessionFactory()
    try:
        result = db.execute(update(PendingCommand)
            .where(PendingCommand.status == "delivered",
                   PendingCommand.delivered_at < sa_text("now() - interval '120 seconds'"))
            .values(status="pending", delivered_at=None))
        db.commit()
        return result.rowcount or 0
    finally:
        db.close()

# 2. Async loop — sleeps on the loop, runs the tick OFF the loop.
async def _command_timeout_loop() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            count = await asyncio.to_thread(_command_timeout_tick)
            if count:
                logger.info("reset %d stale delivered command(s)", count)
        except Exception:
            logger.exception("Command timeout recovery task failed")  # loop survives
```

```python
# 3. Lifespan — create on startup, cancel + await on shutdown.
async def lifespan(_app: FastAPI):
    ...
    cmd_timeout_task = asyncio.create_task(_command_timeout_loop(),
                                           name="command-timeout-recovery")
    yield
    for task in (cleanup_task, agent_cleanup_task, cmd_timeout_task, ...):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
```

The maintenance loops that exist today: `session-cleanup` (hourly),
`pending-agent-cleanup` (60 s), `command-timeout-recovery` (30 s),
`agent-disconnect-detection` (10 s), `model-container-eviction`, and a one-shot
`model-base-prebuild`.

**Rules for a new background loop:**

- **Name every task** (`asyncio.create_task(coro, name="...")`) so it shows up in
  tracebacks and `asyncio.all_tasks()`.
- **The loop body must never raise out of the `while`.** Wrap the work in
  `try/except Exception: logger.exception(...)` so one bad tick doesn't kill the
  loop for the process's lifetime. (Let `asyncio.CancelledError` propagate — see
  the agent loops in §4.)
- **All DB / blocking work goes through `asyncio.to_thread(...)`.** The `await
  asyncio.sleep(...)` is the *only* thing that runs on the loop.
- **Sleep first or sleep last, but sleep on the loop** — never `time.sleep()`.
- **Register teardown in the same lifespan.** Cancel the task and `await` it,
  swallowing `CancelledError`. Leaking tasks across reload/shutdown causes
  "Task was destroyed but it is pending" warnings and half-done writes.
- **Multi-worker awareness.** Each uvicorn worker runs its own copy of every
  lifespan task. For idempotent cleanup (delete-where-expired) that's harmless —
  workers race to no effect. If a job must run *once* cluster-wide, it needs a
  DB advisory lock or a claim row; don't assume single execution.

### Pattern B — blocking listener on a `daemon=True` thread

Some work can't be expressed as "sleep then offload": it blocks in C on a socket
until data arrives. The Postgres `LISTEN`/`NOTIFY` relay
(`app/services/command_events.py`) is the example — it holds a dedicated,
**non-pooled** psycopg2 connection and blocks in `select.select([conn], ...)`:

```python
def start_command_listener(loop: asyncio.AbstractEventLoop) -> None:
    global _listener_thread
    if _listener_thread is not None and _listener_thread.is_alive():
        return  # idempotent
    _listener_stop.clear()
    _listener_thread = threading.Thread(
        target=_listener_main, args=(loop,),
        name="command-notify-listener", daemon=True,
    )
    _listener_thread.start()
```

Crossing back from the thread into the event loop is done **only** via
`loop.call_soon_threadsafe(...)`:

```python
def _relay_notification(loop, agent_id: str) -> None:
    # dict access + Event.set() both run inside the loop thread → no race with
    # the poll endpoint over the shared events dict.
    loop.call_soon_threadsafe(lambda: get_command_event(agent_id).set())
```

**Rules for a listener thread:**

- Capture the running loop in the lifespan (`asyncio.get_running_loop()`) and
  pass it into `start_...(loop)`; a thread has no loop of its own.
- **Never touch loop objects (Events, futures) directly from the thread.** Marshal
  every interaction through `loop.call_soon_threadsafe`. Touching an
  `asyncio.Event` from another thread is undefined behavior.
- Use a **dedicated, non-pooled** DB connection for a long-lived `LISTEN`;
  parking a pooled connection forever shrinks the request pool.
- Make `start_...` **idempotent** and give `stop_...` a bounded `join(timeout=...)`.
- Reconnect with backoff and **degrade gracefully**: if the relay dies, delivery
  falls back to poll-timeout latency — it never loses data, because the poll
  always re-reads pending rows from the DB.

### Pattern C — CPU-bound / untrusted work → process, not thread

Callback model execution (`app/callback_sandbox.py`) and the model executor pool
run user code in **`multiprocessing` processes**, bridged by a short-lived
`daemon=True` thread pumping request/response queues. Threads share the GIL and
the address space; processes give real parallelism and isolation. Rule of thumb:
**I/O-bound blocking → thread (`to_thread`/`run_in_executor`); CPU-bound or
untrusted → process.**

---

## 3. `asyncio.Event` and the long-poll endpoint

The long-poll gives sub-second command delivery to Bridge Agents without a
persistent socket. It is the one place the backend is genuinely async, and it
combines every rule above.

### The moving parts

- **One `asyncio.Event` per agent per worker**, lazily created in a module dict
  (`app/services/command_events.py`):
  ```python
  _command_events: dict[str, asyncio.Event] = {}

  def get_command_event(agent_id: str) -> asyncio.Event:
      if agent_id not in _command_events:
          _command_events[agent_id] = asyncio.Event()
      return _command_events[agent_id]
  ```
- **The producer** (`POST /channels/{id}/command`, queue auto-dispatch) calls
  `notify_new_command(agent_id)` *after the command row is committed*. That sets
  the local event **and** emits `pg_notify` so the other workers' listener
  threads set theirs (§2 Pattern B).
- **The consumer** is the long-poll handler (`app/api/routers/bridge_agent.py`):

```python
@router.post("/equipment/{agent_id}/poll")
async def poll(agent_id, agent = Depends(require_device), db = Depends(get_db)):
    _assert_agent_id_matches(agent_id, agent)
    agent_pk = agent.id

    db.close()                       # release the pooled connection before waiting
    event = get_command_event(agent_id)
    event.clear()                    # clear BEFORE the first read (see ordering note)

    payload = await asyncio.to_thread(_poll_once, agent_pk)   # DB work off the loop
    if payload["commands"]:
        return payload               # fast path: something was already pending

    try:
        await asyncio.wait_for(event.wait(), timeout=LONG_POLL_TIMEOUT_S)  # 25 s
    except TimeoutError:
        pass                         # normal: nothing arrived in the window

    return await asyncio.to_thread(_poll_once, agent_pk)       # re-read after wake
```

`_poll_once` opens its **own** `SessionFactory()` session, marks pending commands
`delivered`, commits, and closes — so no pooled connection is held across the
wait.

### Why the ordering is what it is

The sequence **clear → read → wait → re-read** is deliberate and prevents the
classic lost-wakeup race:

1. `event.clear()` **before** the first read. If a command commits *after* our
   read but *before* our `wait()`, its `set()` leaves the event signaled, so
   `wait()` returns immediately instead of parking for the full 25 s.
2. First `_poll_once` handles the common case where a command is already pending
   — no waiting at all.
3. `asyncio.wait_for(event.wait(), timeout=...)` parks the coroutine cheaply (no
   thread, no connection) until either a wake or the timeout.
4. The **re-read after waking is mandatory.** The event is only a hint that
   "something may be ready" — the DB is the source of truth. A cross-worker
   `NOTIFY` or a coalesced double-set could wake us with nothing (or something)
   to deliver, so we always re-query rather than trusting the signal's payload.

### Long-poll rules

- **`LONG_POLL_TIMEOUT_S = 25`** stays *below* Cloudflare's edge timeout, and the
  agent's HTTP client uses a *longer* read timeout (~40 s) so the client never
  gives up before the server returns. Keep that inequality
  (`agent timeout > server hold > 0`) whenever you touch either number.
- **Commit the row before you notify.** Setting the event before the command is
  visible to a fresh session means the woken poll re-reads and finds nothing —
  a spurious wake at best, a silently dropped command at worst.
- **Release the connection before parking.** `db.close()` up front + short-lived
  sessions in `_poll_once` keep parked polls off the tiny pool. A design that
  held `get_db`'s connection across the 25 s wait would exhaust `pool_size` with
  a handful of idle agents.
- **Treat the event as edge-triggered and lossy.** It carries no payload and can
  coalesce multiple sets into one wake. Correctness must come from the re-read,
  never from the event.
- **Per-worker events are only half the mechanism.** The event alone would only
  wake polls parked on the *same* worker that enqueued the command. The
  `pg_notify` relay (§2) is what makes it work across workers; if you add another
  event-driven endpoint, replicate both halves.

---

## 4. The Bridge Agent polling loop (client side)

The agent (`packages/bridge-agent/`) is a desktop process with a system-tray UI.
Its concurrency problem is the mirror image of the server's: it must run several
network loops *and* a blocking GUI toolkit *and* blocking vendor SDK / `requests`
calls, without any of them stalling the others.

### Loop-in-a-thread architecture

`pystray` must own the main thread, so the agent runs asyncio on a **dedicated
daemon thread** (`agent.py`):

```python
def _run_async() -> None:
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)
    loop_ready.set()
    self._loop.run_until_complete(self._async_main())

thread = threading.Thread(target=_run_async, daemon=True, name="ampwell-async")
thread.start()
loop_ready.wait()          # don't build the tray until the loop exists
self._tray_icon = self._build_tray_icon()
self._tray_icon.run()      # blocks the MAIN thread (tray event loop)
```

`_async_main` builds the runtime objects once, then runs the two loops
concurrently under a single supervising `gather`:

```python
status_task = asyncio.create_task(run_status_broadcast_loop(...), name="status-broadcast")
poll_task   = asyncio.create_task(run_command_poll_loop(...),   name="command-poll")
try:
    await asyncio.gather(status_task, poll_task)
finally:
    for task in (status_task, poll_task):
        if not task.done():
            task.cancel()
    await asyncio.gather(status_task, poll_task, return_exceptions=True)
    await self._clean_shutdown()
```

### How the loop stays unblocked

1. **Every blocking network call is offloaded.** The HTTP client uses synchronous
   `requests` (one shared `requests.Session` for connection reuse), and *each*
   method wraps the blocking call in `run_in_executor` so the agent loop never
   stalls on the socket:
   ```python
   async def poll_commands(self) -> dict:
       resp = await asyncio.get_event_loop().run_in_executor(
           None, lambda: self._post_json(url, {}, timeout=40))
       self._raise_for_status(resp, "command poll")
       return resp.json()
   ```
   The 40 s client timeout deliberately exceeds the server's 25 s hold (§3).

2. **Command execution never blocks the poll.** A poll that returns commands does
   **not** await them inline — each is spun into its own named task so the loop
   immediately circles back to poll again:
   ```python
   for command in commands:
       asyncio.create_task(
           _dispatch_and_report(dispatcher, http, command),
           name=f"cmd-{command['command_id']}")
   ```
   A slow 10-second device command can't delay command #2 or the next poll. The
   `cmd-*` naming lets shutdown and update logic find in-flight work via
   `asyncio.all_tasks()`.

3. **Long file work is offloaded too.** The update downloader hashes and streams a
   zip via `run_in_executor(None, _blocking_download)` — file I/O and hashing off
   the loop, then verifies SHA-256 before staging.

4. **Cooperative shutdown via `asyncio.Event`, set across the thread boundary.**
   The tray "Quit" runs on the main thread and must reach the loop thread safely:
   ```python
   def _quit(self, icon, item):
       if self._loop and self._shutdown_event:
           self._loop.call_soon_threadsafe(self._shutdown_event.set)
       icon.stop()
   ```
   Both loops check `while not shutdown_event.is_set()` each cycle and treat their
   `poll_interval` / long-poll return as natural wake points, so they exit within
   one cycle. `_clean_shutdown` then `await`s in-flight `cmd-*` tasks with a
   bounded `asyncio.wait(..., timeout=10.0)` before cancelling stragglers.

### Cancellation discipline

Both agent loops re-raise `CancelledError` and swallow only real errors — the
inverse of a background maintenance loop's job is still to *stay alive on
transient failure* but *die promptly on cancel*:

```python
while not shutdown_event.is_set():
    try:
        response = await http.poll_commands()
        ...
        if not commands:
            await asyncio.sleep(poll_interval_s)   # only sleep when idle
    except asyncio.CancelledError:
        raise                                      # never swallow cancellation
    except Exception as exc:
        logger.warning("Command poll failed: %s; retrying in %ss", exc, poll_interval_s)
        await asyncio.sleep(poll_interval_s)       # transient: back off and retry
```

**Rules for agent loop work:**

- **Any blocking call** (`requests`, vendor SDK, filesystem, hashing) **must** go
  through `run_in_executor`/`asyncio.to_thread`. The loop drives *many* channels;
  one synchronous SDK call stalls all of them.
- **Fan out slow work into named tasks**; don't `await` it inside the poll cycle.
- **Always re-raise `CancelledError`.** Swallowing it defeats cooperative
  shutdown and leaks tasks.
- **Cross the thread boundary only via `call_soon_threadsafe`** — the tray thread
  must never call `Event.set()` on a loop object directly.
- **Sleep only when idle.** The command loop sleeps `poll_interval_s` after an
  *empty* long-poll (the server already held the connection); it loops
  immediately when commands arrive.

---

## Pre-merge checklist

**Sessions & the loop**
- [ ] New DB code in a request handler uses `Depends(get_db)`; code outside a
      request opens its own `SessionFactory()` and closes it in `finally`.
- [ ] No synchronous `Session` call happens directly inside an `async def` on the
      event loop. If the handler must be `async`, DB work goes through
      `asyncio.to_thread`.
- [ ] No session/connection is held across an `await` that can park (sleep, event
      wait, network). Release it first.

**Background tasks**
- [ ] Every `create_task` has a `name=`.
- [ ] The loop body cannot raise out of the `while` (wrapped in
      `try/except Exception: logger.exception`), and `CancelledError` propagates.
- [ ] Blocking/DB work is offloaded via `asyncio.to_thread`; only `asyncio.sleep`
      runs on the loop.
- [ ] The task is cancelled **and awaited** in the lifespan teardown.
- [ ] Multi-worker behavior considered: the job is idempotent, or guarded by a
      DB lock/claim if it must run once cluster-wide.
- [ ] New listener threads are `daemon=True`, idempotent to start, bounded to
      stop, use a non-pooled connection for `LISTEN`, and marshal into the loop
      only via `call_soon_threadsafe`.

**Long-poll / `asyncio.Event`**
- [ ] `notify_*` is called **after** the row is committed.
- [ ] The waiter clears the event before its first read and **re-reads the DB**
      after waking (never trusts the event as payload).
- [ ] Server hold < edge/proxy timeout < agent client timeout, and all three stay
      consistent if any is changed.
- [ ] Cross-worker fan-out (`pg_notify` + listener) is wired if delivery must span
      workers.

**Agent loops**
- [ ] Every blocking call is wrapped in `run_in_executor`/`to_thread`.
- [ ] Slow command/work is fanned out into a named task, not awaited inline.
- [ ] `CancelledError` is re-raised; transient errors back off and retry.
- [ ] Thread→loop signaling uses `call_soon_threadsafe`.
