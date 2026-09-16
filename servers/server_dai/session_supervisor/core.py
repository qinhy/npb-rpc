from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
from multiprocessing.connection import Connection
import random
import threading
import time
import traceback
from typing import Callable, Generic, Protocol, TypeVar

TTarget = TypeVar("TTarget")


class _EventLike(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class SessionState(str, Enum):
    CLOSED = "closed"
    OPENING = "opening"
    ONLINE = "online"
    CLOSING = "closing"
    RETRYING = "retrying"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class RetryPolicy:
    """Retry/backoff policy for failed sessions.

    ``attempt`` is 1-based and counts consecutive failures since the last
    successful ``mark_ready()`` or a new open request.
    """

    initial_delay: float = 1.0
    multiplier: float = 1.0
    max_delay: float = 30.0
    max_attempts: int | None = None
    jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_delay < 0:
            raise ValueError("initial_delay must be >= 0")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")
        if self.max_attempts is not None and self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0 or None")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")

    def allows(self, attempt: int) -> bool:
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        return self.max_attempts is None or attempt <= self.max_attempts

    def delay_for(self, attempt: int) -> float:
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        delay = min(
            self.max_delay,
            self.initial_delay * (self.multiplier ** (attempt - 1)),
        )
        if self.jitter and delay:
            spread = delay * self.jitter
            delay = random.uniform(max(0.0, delay - spread), delay + spread)
        return delay


@dataclass(frozen=True)
class SessionStatus(Generic[TTarget]):
    state: SessionState
    desired_open: bool
    desired_target: TTarget | None
    active_target: TTarget | None
    revision: int
    generation: int
    restart_count: int
    retry_attempt: int
    error: str | None

    @property
    def online(self) -> bool:
        return self.state is SessionState.ONLINE


class SessionControl:
    """Restricted control surface passed to one running session.

    A handler can observe cancellation and announce that startup completed via
    ``mark_ready()``.  It intentionally cannot mutate supervisor state directly.
    """

    def __init__(
        self,
        *,
        revision: int,
        cancel_event: _EventLike,
        mark_ready_cb: Callable[[int], bool],
        emit_cb: Callable[[int, object], bool] | None = None,
    ) -> None:
        self._revision = revision
        self._cancel_event = cancel_event
        self._mark_ready_cb = mark_ready_cb
        self._emit_cb = emit_cb
        self._ready = False
        self._ready_lock = threading.Lock()

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def ready(self) -> bool:
        with self._ready_lock:
            return self._ready

    def wait_cancelled(self, timeout: float | None = None) -> bool:
        return self._cancel_event.wait(timeout)

    def mark_ready(self) -> bool:
        """Mark this session ONLINE if it is still the requested session.

        Returns ``False`` when the session became stale while it was opening.
        Repeated calls are harmless.
        """
        with self._ready_lock:
            if self._ready:
                return not self.cancelled
            accepted = self._mark_ready_cb(self._revision)
            if accepted:
                self._ready = True
            return accepted

    def emit(self, payload: object) -> bool:
        """Send one application event from the isolated worker to the parent.

        The event travels over the worker's private pipe.  The pipe is created
        fresh for every session attempt, so a native worker crash cannot poison
        the IPC channel used by a later retry.
        """
        if self._emit_cb is None:
            return False
        return self._emit_cb(self._revision, payload)


class SessionHandler(Protocol[TTarget]):
    """Application-specific implementation of one resource session."""

    def run(self, target: TTarget, control: SessionControl) -> None:
        """Run one session until cancelled or failed.

        The handler should:
        1. acquire/connect the resource,
        2. call ``control.mark_ready()`` when usable,
        3. keep serving while ``not control.cancelled``, and
        4. release the resource before returning.

        Returning after cancellation is a normal shutdown.  Raising an
        exception while the session is still desired is treated as a failure
        and may be retried according to ``RetryPolicy``.
        """
        ...


class SessionSupersededError(RuntimeError):
    pass


class WorkerProcessError(RuntimeError):
    """A session worker process exited abnormally."""

    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode
        super().__init__(f"session worker exited abnormally ({_format_exitcode(exitcode)})")


class RemoteSessionError(RuntimeError):
    """Python exception raised by ``SessionHandler.run`` in the worker process."""

    def __init__(self, exc_type: str, message: str, remote_traceback: str) -> None:
        self.exc_type = exc_type
        self.message = message
        self.remote_traceback = remote_traceback
        text = f"{exc_type}: {message}" if message else exc_type
        super().__init__(text)


def _format_exitcode(exitcode: int | None) -> str:
    if exitcode is None:
        return "exitcode=None"
    # On Windows a native access violation is commonly 0xC0000005.  Depending
    # on the Python/runtime path it may be surfaced as a signed or unsigned int.
    unsigned = exitcode & 0xFFFFFFFF
    if unsigned >= 0x80000000:
        return f"exitcode={exitcode} (0x{unsigned:08X})"
    return f"exitcode={exitcode}"


def _session_process_main(
    handler: SessionHandler[TTarget],
    target: TTarget,
    revision: int,
    cancel_event: _EventLike,
    control_conn: Connection,
) -> None:
    """Run exactly one session in an isolated child process.

    This function intentionally lives at module scope so the ``spawn`` start
    method can import/pickle it on Windows.  Only small control messages cross
    the pipe; application data should use its own IPC/shared-memory transport.
    """

    send_lock = threading.Lock()

    def send_message(message: tuple[object, ...]) -> bool:
        try:
            with send_lock:
                control_conn.send(message)
            return True
        except (EOFError, BrokenPipeError, OSError):
            return False

    def mark_ready(rev: int) -> bool:
        if cancel_event.is_set():
            return False
        if not send_message(("ready", rev)):
            return False
        try:
            while not cancel_event.is_set():
                if control_conn.poll(0.1):
                    message = control_conn.recv()
                    if (
                        isinstance(message, tuple)
                        and len(message) >= 3
                        and message[0] == "ready_ack"
                        and message[1] == rev
                    ):
                        return bool(message[2])
            return False
        except (EOFError, BrokenPipeError, OSError):
            return False

    def emit(rev: int, payload: object) -> bool:
        return send_message(("event", rev, payload))

    control = SessionControl(
        revision=revision,
        cancel_event=cancel_event,
        mark_ready_cb=mark_ready,
        emit_cb=emit,
    )

    try:
        handler.run(target, control)
        try:
            send_message(("returned",))
        except (EOFError, BrokenPipeError, OSError):
            pass
    except BaseException as exc:
        # Send strings only: arbitrary exception objects are not guaranteed to be
        # picklable, especially when third-party native extensions are involved.
        try:
            send_message(
                (
                    "error",
                    type(exc).__name__,
                    str(exc),
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                )
            )
        except (EOFError, BrokenPipeError, OSError):
            pass
        # Preserve a non-zero exit status without dumping a duplicate traceback
        # to stderr; the parent already received the formatted remote traceback.
        raise SystemExit(1) from None
    finally:
        try:
            control_conn.close()
        except OSError:
            pass


class SessionSupervisor(Generic[TTarget]):
    """Supervise one replaceable, restartable long-lived session.

    The supervisor continuously reconciles *desired state* (open/closed and the
    requested target) in a background thread, while each actual session runs in
    an isolated child process.  Native crashes in third-party extensions therefore
    terminate only the session worker, not the parent application.
    """

    def __init__(
        self,
        handler: SessionHandler[TTarget],
        *,
        retry: RetryPolicy | None = None,
        auto_start: bool = False,
        process_start_method: str = "spawn",
        cancel_grace_period: float = 2.0,
        terminate_grace_period: float = 1.0,
        on_event: Callable[[TTarget, int, object], None] | None = None,
    ) -> None:
        self._handler = handler
        self._retry = retry or RetryPolicy()
        self._on_event = on_event
        self._mp_context = mp.get_context(process_start_method)
        self._cancel_grace_period = max(0.0, cancel_grace_period)
        self._terminate_grace_period = max(0.0, terminate_grace_period)

        self._cv = threading.Condition()
        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._desired_open = False
        self._desired_target: TTarget | None = None
        self._revision = 0

        self._state = SessionState.CLOSED
        self._active_target: TTarget | None = None
        self._session_active = False
        self._session_cancel: _EventLike | None = None

        self._generation = 0
        self._restart_count = 0
        self._retry_attempt = 0
        self._error: str | None = None

        self._thread = self._make_thread()
        if auto_start:
            self.start()

    def _make_thread(self) -> threading.Thread:
        return threading.Thread(
            target=self._thread_main,
            name="session-supervisor",
            daemon=True,
        )

    def start(self) -> None:
        """Start the persistent supervisor worker."""
        with self._thread_lock:
            if self._stop_event.is_set():
                raise RuntimeError("session supervisor has been permanently stopped")
            if self._thread.is_alive():
                return
            if self._thread.ident is not None:
                self._thread = self._make_thread()
            self._thread.start()

    def open(self, target: TTarget, *, timeout: float | None = None) -> SessionStatus[TTarget]:
        """Request ``target`` and optionally wait until that exact request is ONLINE.

        ``timeout=None`` waits indefinitely.  ``timeout=0`` is a non-blocking
        request and immediately returns the current status.
        """
        if self._stop_event.is_set():
            raise RuntimeError("session supervisor has been permanently stopped")
        if not self._thread.is_alive():
            self.start()

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)

        with self._cv:
            # An already-online identical target is idempotent.
            if (
                self._desired_open
                and self._desired_target == target
                and self._state is SessionState.ONLINE
                and self._active_target == target
            ):
                return self._status_unlocked()

            # If this same target is currently being opened/retried, join that
            # request.  FAILED is intentionally excluded: calling open() again
            # creates a fresh revision and resets its retry budget.
            joining_existing = (
                self._desired_open
                and self._desired_target == target
                and self._state in {
                    SessionState.OPENING,
                    SessionState.RETRYING,
                }
            )

            if joining_existing:
                revision = self._revision
            else:
                self._desired_open = True
                self._desired_target = target
                self._revision += 1
                revision = self._revision
                self._retry_attempt = 0
                self._error = None
                self._state = SessionState.OPENING
                if self._session_cancel is not None:
                    self._session_cancel.set()
                self._cv.notify_all()

            if timeout == 0:
                return self._status_unlocked()

            while True:
                if self._revision != revision:
                    raise SessionSupersededError(
                        f"open request revision {revision} was superseded by {self._revision}"
                    )
                if (
                    self._desired_open
                    and self._desired_target == target
                    and self._state is SessionState.ONLINE
                    and self._active_target == target
                ):
                    return self._status_unlocked()
                if self._state is SessionState.FAILED:
                    return self._status_unlocked()
                if self._state is SessionState.STOPPED:
                    return self._status_unlocked()

                if deadline is None:
                    self._cv.wait()
                    continue

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"session did not become online within {timeout:g}s; "
                        f"state={self._state.value}, error={self._error!r}"
                    )
                self._cv.wait(remaining)

    def close_session(self, *, timeout: float | None = None) -> SessionStatus[TTarget]:
        """Close the active session while keeping the supervisor reusable."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)

        with self._cv:
            if self._state is SessionState.STOPPED:
                return self._status_unlocked()

            self._desired_open = False
            self._revision += 1
            self._retry_attempt = 0
            self._error = None

            if self._session_active:
                self._state = SessionState.CLOSING
                if self._session_cancel is not None:
                    self._session_cancel.set()
            else:
                self._state = SessionState.CLOSED
                self._active_target = None
            self._cv.notify_all()

            if timeout == 0:
                return self._status_unlocked()

            while self._session_active:
                if deadline is None:
                    self._cv.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("session shutdown is still in progress")
                self._cv.wait(remaining)

            self._state = SessionState.CLOSED
            self._active_target = None
            self._cv.notify_all()
            return self._status_unlocked()

    def shutdown(self, *, timeout: float | None = 5.0) -> SessionStatus[TTarget]:
        """Permanently stop the supervisor.  It cannot be restarted afterwards."""
        self._stop_event.set()
        with self._cv:
            self._desired_open = False
            self._revision += 1
            if self._session_cancel is not None:
                self._session_cancel.set()
            self._cv.notify_all()

        if self._thread.ident is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("session supervisor worker did not stop in time")

        with self._cv:
            self._state = SessionState.STOPPED
            self._active_target = None
            self._session_active = False
            self._session_cancel = None
            self._cv.notify_all()
            return self._status_unlocked()

    def status(self) -> SessionStatus[TTarget]:
        with self._cv:
            return self._status_unlocked()

    def _status_unlocked(self) -> SessionStatus[TTarget]:
        return SessionStatus(
            state=self._state,
            desired_open=self._desired_open,
            desired_target=self._desired_target,
            active_target=self._active_target,
            revision=self._revision,
            generation=self._generation,
            restart_count=self._restart_count,
            retry_attempt=self._retry_attempt,
            error=self._error,
        )

    def _mark_ready(self, revision: int) -> bool:
        with self._cv:
            if (
                self._stop_event.is_set()
                or not self._desired_open
                or revision != self._revision
                or not self._session_active
            ):
                return False

            self._state = SessionState.ONLINE
            self._generation += 1
            self._retry_attempt = 0
            self._error = None
            self._cv.notify_all()
            return True

    def _thread_main(self) -> None:
        # Last-resort protection for failures in the supervision machinery itself.
        # It cannot protect the process from native SIGSEGV/SIGABRT failures.
        while not self._stop_event.is_set():
            try:
                self._run()
                return
            except BaseException as exc:
                if self._stop_event.is_set():
                    return
                with self._cv:
                    self._restart_count += 1
                    self._session_active = False
                    self._session_cancel = None
                    self._active_target = None
                    self._error = f"supervisor error: {type(exc).__name__}: {exc}"
                    self._state = (
                        SessionState.OPENING if self._desired_open else SessionState.CLOSED
                    )
                    self._cv.notify_all()
                time.sleep(min(1.0, self._retry.initial_delay))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._cv:
                while not self._desired_open and not self._stop_event.is_set():
                    self._state = SessionState.CLOSED
                    self._active_target = None
                    self._cv.wait()
                if self._stop_event.is_set():
                    break

                revision = self._revision
                target = self._desired_target
                cancel = self._mp_context.Event()
                self._session_cancel = cancel
                self._session_active = True
                self._active_target = target
                self._state = SessionState.OPENING
                self._error = None
                self._cv.notify_all()

            parent_conn, child_conn = self._mp_context.Pipe(duplex=True)
            process = self._mp_context.Process(
                target=_session_process_main,
                args=(self._handler, target, revision, cancel, child_conn),
                name=f"session-worker-{revision}",
                daemon=True,
            )

            failure: BaseException | None = None
            process_started = False
            worker_returned = False
            cancel_seen_at: float | None = None
            terminate_seen_at: float | None = None
            terminated_by_supervisor = False

            try:
                try:
                    process.start()
                    process_started = True
                except BaseException as exc:
                    failure = exc
                finally:
                    # The parent must not keep the child's pipe endpoint open;
                    # otherwise EOF cannot be observed when the child dies.
                    try:
                        child_conn.close()
                    except OSError:
                        pass

                while failure is None and process.is_alive():
                    # Handle all currently queued child -> parent messages.
                    while parent_conn.poll(0):
                        try:
                            message = parent_conn.recv()
                        except (EOFError, OSError):
                            break

                        if not isinstance(message, tuple) or not message:
                            continue

                        kind = message[0]
                        if kind == "ready" and len(message) >= 2:
                            ready_revision = int(message[1])
                            accepted = self._mark_ready(ready_revision)
                            try:
                                parent_conn.send(("ready_ack", ready_revision, accepted))
                            except (EOFError, BrokenPipeError, OSError):
                                pass
                        elif kind == "event" and len(message) >= 3:
                            event_revision = int(message[1])
                            if event_revision == revision and self._on_event is not None:
                                self._on_event(target, event_revision, message[2])
                        elif kind == "error" and len(message) >= 4:
                            failure = RemoteSessionError(
                                str(message[1]),
                                str(message[2]),
                                str(message[3]),
                            )
                        elif kind == "returned":
                            worker_returned = True

                    # open(B), close_session(), or shutdown() sets this event.
                    # Give the handler a chance to release its resource cleanly,
                    # then force the worker down if a native call is hung.
                    if cancel.is_set():
                        now = time.monotonic()
                        if cancel_seen_at is None:
                            cancel_seen_at = now
                        elif (
                            not terminated_by_supervisor
                            and now - cancel_seen_at >= self._cancel_grace_period
                        ):
                            process.terminate()
                            terminated_by_supervisor = True
                            terminate_seen_at = now
                        elif (
                            terminated_by_supervisor
                            and terminate_seen_at is not None
                            and now - terminate_seen_at >= self._terminate_grace_period
                            and process.is_alive()
                        ):
                            kill = getattr(process, "kill", None)
                            if kill is not None:
                                kill()
                            terminate_seen_at = now

                    process.join(timeout=0.05)

                # Process may have exited between poll cycles.  Drain any final
                # READY/ERROR/RETURNED message before classifying the exit.
                while failure is None and parent_conn.poll(0):
                    try:
                        message = parent_conn.recv()
                    except (EOFError, OSError):
                        break
                    if not isinstance(message, tuple) or not message:
                        continue
                    kind = message[0]
                    if kind == "ready" and len(message) >= 2:
                        ready_revision = int(message[1])
                        accepted = self._mark_ready(ready_revision)
                        try:
                            parent_conn.send(("ready_ack", ready_revision, accepted))
                        except (EOFError, BrokenPipeError, OSError):
                            pass
                    elif kind == "event" and len(message) >= 3:
                        event_revision = int(message[1])
                        if event_revision == revision and self._on_event is not None:
                            self._on_event(target, event_revision, message[2])
                    elif kind == "error" and len(message) >= 4:
                        failure = RemoteSessionError(
                            str(message[1]), str(message[2]), str(message[3])
                        )
                    elif kind == "returned":
                        worker_returned = True

                if process_started:
                    process.join(timeout=0)

                with self._cv:
                    intentional = (
                        self._stop_event.is_set()
                        or cancel.is_set()
                        or not self._desired_open
                        or self._revision != revision
                        or self._desired_target != target
                    )

                if not intentional and failure is None:
                    if process.exitcode not in (0, None):
                        failure = WorkerProcessError(process.exitcode)
                    elif worker_returned or process.exitcode == 0:
                        failure = RuntimeError("session handler returned unexpectedly")
                    else:
                        failure = RuntimeError("session worker ended unexpectedly")

            finally:
                try:
                    parent_conn.close()
                except OSError:
                    pass
                if process_started and process.is_alive():
                    # This path is mainly for unexpected errors in the supervisor
                    # itself.  Never leave an orphan resource process behind.
                    try:
                        process.terminate()
                    except (OSError, ValueError):
                        pass
                    process.join(timeout=self._terminate_grace_period)
                    if process.is_alive():
                        kill = getattr(process, "kill", None)
                        if kill is not None:
                            try:
                                kill()
                            except (OSError, ValueError):
                                pass
                        process.join(timeout=self._terminate_grace_period)
                try:
                    process.close()
                except (OSError, ValueError):
                    pass

            with self._cv:
                intentional = (
                    self._stop_event.is_set()
                    or cancel.is_set()
                    or not self._desired_open
                    or self._revision != revision
                    or self._desired_target != target
                )

                self._session_active = False
                if self._session_cancel is cancel:
                    self._session_cancel = None

                if intentional:
                    if not self._desired_open:
                        self._state = SessionState.CLOSED
                        self._active_target = None
                        self._error = None
                    elif self._revision != revision or self._desired_target != target:
                        # A replacement target is already desired.
                        self._state = SessionState.OPENING
                        self._active_target = None
                    self._cv.notify_all()
                    continue

                if failure is None:
                    failure = RuntimeError("session ended unexpectedly")

                self._restart_count += 1
                self._retry_attempt += 1
                attempt = self._retry_attempt
                self._error = f"{type(failure).__name__}: {failure}"
                if isinstance(failure, RemoteSessionError) and failure.remote_traceback:
                    self._error += f"\n{failure.remote_traceback}"
                self._active_target = None

                if not self._retry.allows(attempt):
                    self._state = SessionState.FAILED
                    self._cv.notify_all()
                    while (
                        self._desired_open
                        and self._revision == revision
                        and self._desired_target == target
                        and not self._stop_event.is_set()
                    ):
                        self._cv.wait()
                    continue

                self._state = SessionState.RETRYING
                delay = self._retry.delay_for(attempt)
                self._cv.notify_all()

                if delay > 0:
                    self._cv.wait(delay)

        with self._cv:
            self._state = SessionState.STOPPED
            self._active_target = None
            self._session_active = False
            self._session_cancel = None
            self._cv.notify_all()
