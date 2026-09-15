# session-supervisor

A small synchronous supervisor for one replaceable, restartable, long-lived
resource session. The state machine remains in the parent process; each actual
session runs in an isolated child process.

This is useful for native SDKs such as camera/device libraries where a C/C++
access violation, SIGSEGV, SIGABRT, or hard process exit cannot be caught by a
Python thread or `try/except`.

The abstraction remains resource-agnostic. A session can be a camera, WebSocket,
serial device, robot controller, database connection, RPC client, or anything
else with a long-lived acquire/serve/release lifecycle.

## Core model

- `SessionSupervisor[T]`: reconciles desired state with one actual session.
- `SessionHandler[T]`: application-specific session implementation, executed in a child process.
- `SessionControl`: cancellation + READY handshake exposed to the handler.
- `SessionStatus[T]`: immutable observable state.
- `SessionState`: CLOSED / OPENING / ONLINE / CLOSING / RETRYING / FAILED / STOPPED.
- `RetryPolicy`: fixed or exponential retry/backoff.

Two counters have deliberately different meanings:

- `revision`: changes when user intent changes (open/switch/close).
- `generation`: increments only when a session successfully calls `mark_ready()`.

That makes late startup from a superseded session harmless.

## Handler contract

```python
class MyHandler:
    def run(self, target, control):
        # Import fragile/native libraries here when practical.
        resource = acquire(target)
        try:
            if not control.mark_ready():
                return

            while not control.cancelled:
                serve(resource)
        finally:
            release(resource)
```

A normal return after cancellation means intentional shutdown. A Python exception
is serialized back to the parent as text. An abnormal child-process exit is
reported as `WorkerProcessError` and enters the same retry policy.

## Native-crash isolation

```text
Parent process
  SessionSupervisor thread
        |
        +-- worker process
              |
              +-- SessionHandler.run()
                    |
                    +-- native SDK (DepthAI, etc.)
                          X native crash

Parent remains alive -> RETRYING -> new worker process
```

Large application data should not pass through the supervisor control pipe. Keep
using your own shared-memory/NNG/ZMQ/NPB transport for camera frames or other
large payloads.

## Lifecycle

```text
CLOSED -> OPENING -> ONLINE
             ^          |
             |          | failure / worker crash
             +-- RETRYING

OPENING/ONLINE/RETRYING -> CLOSING -> CLOSED
anything -> STOPPED  (shutdown)
```

Calling `open(B)` while `A` is active cancels A, advances `revision`, and starts B.
If A later calls `mark_ready()`, it is rejected because its revision is stale.

## Windows / `spawn` requirements

The default process start method is `spawn`, which is the correct isolation model
for Windows. Therefore:

1. The handler object and target must be picklable. Define handler classes at
   module scope; do not store `threading.Lock`, open sockets, DepthAI objects, or
   other unpicklable native objects on the handler before `open()`.
2. Create/start the application under the normal Windows guard:

```python
if __name__ == "__main__":
    main()
```

3. Create DepthAI `Device`, `Pipeline`, queues, and callbacks inside
   `SessionHandler.run()`, not in the parent process.

## Shutdown hardening

Cancellation is cooperative first. If the child ignores cancellation (for
example, a native SDK call is hung), the supervisor waits `cancel_grace_period`,
then calls `terminate()`. If needed, it escalates to `kill()` after
`terminate_grace_period`.

```python
supervisor = SessionSupervisor(
    handler,
    retry=RetryPolicy(initial_delay=1.0, max_delay=10.0),
    cancel_grace_period=2.0,
    terminate_grace_period=1.0,
)
```
