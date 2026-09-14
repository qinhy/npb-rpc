# session-supervisor

A small synchronous/thread-based supervisor for one replaceable, restartable,
long-lived resource session.

The abstraction is intentionally resource-agnostic. A session can be a camera,
WebSocket, serial device, robot controller, database connection, RPC client, or
anything else that has a long-lived acquire/serve/release lifecycle.

## Core model

- `SessionSupervisor[T]`: reconciles desired state with one actual session.
- `SessionHandler[T]`: application-specific session implementation.
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
        resource = acquire(target)
        try:
            if not control.mark_ready():
                return

            while not control.cancelled:
                serve(resource)
        finally:
            release(resource)
```

A normal return after cancellation means intentional shutdown. An exception while
the same session is still desired is a failure and enters the retry policy.

## Lifecycle

```text
CLOSED -> OPENING -> ONLINE
             ^          |
             |          | failure
             +-- RETRYING

OPENING/ONLINE/RETRYING -> CLOSING -> CLOSED
anything -> STOPPED  (shutdown)
```

Calling `open(B)` while `A` is active cancels A, advances `revision`, and starts B.
If A later calls `mark_ready()`, it is rejected because its revision is stale.
