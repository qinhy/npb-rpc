from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
import threading
import time
from typing import Callable, Generic, Protocol, TypeVar

TTarget = TypeVar("TTarget")


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
        cancel_event: threading.Event,
        mark_ready_cb: Callable[[int], bool],
    ) -> None:
        self._revision = revision
        self._cancel_event = cancel_event
        self._mark_ready_cb = mark_ready_cb
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


class SessionSupervisor(Generic[TTarget]):
    """Supervise one replaceable, restartable long-lived session.

    The supervisor continuously reconciles *desired state* (open/closed and the
    requested target) with one actual session owned by a background thread.
    """

    def __init__(
        self,
        handler: SessionHandler[TTarget],
        *,
        retry: RetryPolicy | None = None,
        auto_start: bool = False,
    ) -> None:
        self._handler = handler
        self._retry = retry or RetryPolicy()

        self._cv = threading.Condition()
        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._desired_open = False
        self._desired_target: TTarget | None = None
        self._revision = 0

        self._state = SessionState.CLOSED
        self._active_target: TTarget | None = None
        self._session_active = False
        self._session_cancel: threading.Event | None = None

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
                cancel = threading.Event()
                self._session_cancel = cancel
                self._session_active = True
                self._active_target = target
                self._state = SessionState.OPENING
                self._error = None
                self._cv.notify_all()

            control = SessionControl(
                revision=revision,
                cancel_event=cancel,
                mark_ready_cb=self._mark_ready,
            )

            failure: BaseException | None = None
            try:
                # target cannot be None here unless the caller intentionally uses
                # None as TTarget.  Runtime behavior is still valid in that case.
                self._handler.run(target, control)  # type: ignore[arg-type]
                with self._cv:
                    still_desired = (
                        self._desired_open
                        and self._revision == revision
                        and self._desired_target == target
                        and not cancel.is_set()
                        and not self._stop_event.is_set()
                    )
                if still_desired:
                    failure = RuntimeError("session handler returned unexpectedly")
            except BaseException as exc:
                failure = exc

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
                    # Defensive fallback; normally converted to RuntimeError above.
                    failure = RuntimeError("session ended unexpectedly")

                self._restart_count += 1
                self._retry_attempt += 1
                attempt = self._retry_attempt
                self._error = f"{type(failure).__name__}: {failure}"
                self._active_target = None

                if not self._retry.allows(attempt):
                    self._state = SessionState.FAILED
                    self._cv.notify_all()
                    # Stay idle until open() creates a fresh revision, target changes,
                    # close_session(), or shutdown().
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

                # Condition.wait() makes retry sleep interruptible by close/switch.
                if delay > 0:
                    self._cv.wait(delay)

        with self._cv:
            self._state = SessionState.STOPPED
            self._active_target = None
            self._session_active = False
            self._session_cancel = None
            self._cv.notify_all()
