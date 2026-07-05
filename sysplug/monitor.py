"""Online training monitor running in a background thread.

The :class:`Monitor` polls GPU metrics and checks for training instability
at a configurable interval.  It is designed to be used as a context manager
wrapping the training loop:

    with advisor.monitor(check_interval_steps=100) as mon:
        for step, batch in loader:
            loss = train_step(batch)
            mon.record(step=step, loss=loss.item())

All shared state is protected by a :class:`threading.Lock` so ``record()``
never blocks the training loop.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from sysplug.advisor import Advisor


class EventType(str, Enum):
    """Type of monitor event."""
    OOM_RISK = "oom_risk"
    DIVERGING_LOSS = "diverging_loss"
    OSCILLATING_LOSS = "oscillating_loss"
    GRAD_NORM_SPIKE = "grad_norm_spike"
    RECONFIG_SUGGESTED = "reconfig_suggested"
    RECONFIG_APPLIED = "reconfig_applied"
    WARNING = "warning"
    INFO = "info"


@dataclass
class MonitorEvent:
    """A single event emitted by the :class:`Monitor`.

    Attributes:
        event_type: Category of event.
        step: Training step at which the event was recorded.
        message: Human-readable description.
        details: Optional dict with additional structured data.
        timestamp: Unix timestamp of the event.
    """
    event_type: EventType
    step: int
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class _StepRecord:
    """Internal record passed through the step queue."""
    step: int
    loss: float
    grad_norm: Optional[float] = None
    custom_metrics: Optional[Dict[str, Any]] = None


class Monitor:
    """Background-thread training monitor.

    Args:
        advisor: The parent :class:`~sysplug.advisor.Advisor` instance.
        check_interval_steps: How often (in steps) to run the full check.
        reconfig_policy: What to do when a reconfiguration is recommended:
            - ``"suggest"`` — print a rich-formatted suggestion.
            - ``"auto-apply"`` — directly update ``advisor.current_config``.
            - ``"warn-only"`` — log a warning without suggesting changes.
        verbose: Whether to emit output to the console.

    Examples:
        >>> # Used as a context manager from Advisor.monitor()
        >>> with advisor.monitor(check_interval_steps=50) as mon:
        ...     for step in range(100):
        ...         mon.record(step=step, loss=0.5)
    """

    def __init__(
        self,
        advisor: "Advisor",
        check_interval_steps: int = 50,
        reconfig_policy: str = "suggest",
        verbose: bool = True,
    ) -> None:
        self._advisor = advisor
        self._check_interval = check_interval_steps
        self._reconfig_policy = reconfig_policy
        self._verbose = verbose

        self._step_queue: queue.Queue[_StepRecord] = queue.Queue()
        self._events: List[MonitorEvent] = []
        self._events_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._step_buffer: List[_StepRecord] = []
        self._last_check_step = -1
        self._last_checked_at_step = -1

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Monitor":
        """Start the background monitoring thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._check_loop,
            name="sysplug-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Stop the background thread and drain remaining records."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Training-loop API (called from main thread)
    # ------------------------------------------------------------------

    def record(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        custom_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a training step metric (non-blocking, thread-safe).

        This method is designed to be called from the hot training loop.
        It enqueues the record and returns immediately.

        Args:
            step: Global training step index.
            loss: Training loss value.
            grad_norm: Optional gradient L2-norm.
            custom_metrics: Optional dict of additional metrics.
        """
        try:
            self._step_queue.put_nowait(
                _StepRecord(
                    step=step,
                    loss=loss,
                    grad_norm=grad_norm,
                    custom_metrics=custom_metrics,
                )
            )
        except queue.Full:
            pass  # Never block training; silently drop if queue full

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _check_loop(self) -> None:
        """Main loop running in the background thread."""
        from sysplug.hardware import HardwareProfiler
        from sysplug.stability import StabilitySignal
        from sysplug.utils.logging import get_console

        stability = StabilitySignal(window_size=50)
        profiler = HardwareProfiler(verbose=False)
        console = get_console(verbose=self._verbose)

        while not self._stop_event.is_set():
            # Drain the step queue
            drained: List[_StepRecord] = []
            try:
                while True:
                    drained.append(self._step_queue.get_nowait())
            except queue.Empty:
                pass

            for record in drained:
                stability.record_loss(record.step, record.loss)
                if record.grad_norm is not None:
                    stability.record_grad_norm(record.step, record.grad_norm)
                self._last_check_step = record.step

            # Run checks whenever we've advanced at least check_interval steps
            # since the last check (handles batched drains correctly)
            if drained and (
                self._last_check_step - self._last_checked_at_step >= self._check_interval
            ):
                self._run_checks(profiler, stability, console)
                self._last_checked_at_step = self._last_check_step

            time.sleep(0.1)

    def _run_checks(
        self,
        profiler: Any,
        stability: Any,
        console: Any,
    ) -> None:
        """Run hardware and stability checks and emit events."""
        # Hardware snapshot
        hw = profiler.snapshot()
        step = self._last_check_step

        # OOM risk
        if hw.gpus:
            gpu = hw.gpus[0]
            used_frac = gpu.used_memory_mb / max(gpu.total_memory_mb, 1)
            if used_frac > 0.90:
                msg = (
                    f"[bold red]OOM RISK[/bold red]: GPU {gpu.device_id} memory at "
                    f"{used_frac*100:.1f}% "
                    f"({gpu.used_memory_mb:.0f}/{gpu.total_memory_mb:.0f} MiB)"
                )
                self._emit_event(EventType.OOM_RISK, step, msg)
                if self._verbose:
                    console.print(f"[SysPlug step={step}] {msg}")

        # Stability signal
        report = stability.check()
        if report.is_diverging:
            msg = f"Loss is diverging at step {step}: {report.message}"
            self._emit_event(EventType.DIVERGING_LOSS, step, msg)
            if self._verbose:
                console.print(f"[SysPlug step={step}] [yellow]{msg}[/yellow]")

            if self._reconfig_policy in {"suggest", "auto-apply"}:
                self._suggest_reconfig(step, report, console)

        elif report.is_oscillating:
            msg = f"Loss is oscillating at step {step}: {report.message}"
            self._emit_event(EventType.OSCILLATING_LOSS, step, msg)
            if self._verbose:
                console.print(f"[SysPlug step={step}] [yellow]{msg}[/yellow]")

        if report.gradient_norm_spike:
            msg = f"Gradient norm spike at step {step}: {report.message}"
            self._emit_event(EventType.GRAD_NORM_SPIKE, step, msg)
            if self._verbose:
                console.print(f"[SysPlug step={step}] [yellow]{msg}[/yellow]")

    def _suggest_reconfig(
        self,
        step: int,
        report: Any,
        console: Any,
    ) -> None:
        """Emit a reconfig suggestion based on the stability report."""
        action = report.recommended_action
        current = self._advisor.current_config

        change: dict[str, Any] = {}
        if action == "reduce_lr":
            new_lr = current.learning_rate * 0.5
            change["learning_rate"] = new_lr
        elif action == "reduce_batch":
            new_bs = max(1, current.batch_size // 2)
            change["batch_size"] = new_bs

        if not change:
            return

        try:
            new_cfg = self._advisor.what_if(change, current_config=current)
        except Exception:
            return

        new_config = new_cfg.new_config if hasattr(new_cfg, "new_config") else new_cfg

        if self._reconfig_policy == "auto-apply":
            self._advisor._current_config = new_config
            msg = f"Auto-applied reconfig at step {step}: {change}"
            self._emit_event(EventType.RECONFIG_APPLIED, step, msg, {"change": change})
            if self._verbose:
                console.print(f"[SysPlug] [green]{msg}[/green]")
        else:
            msg = f"Suggested reconfig at step {step}: {change}"
            self._emit_event(
                EventType.RECONFIG_SUGGESTED, step, msg, {"change": change, "new_config": change}
            )
            if self._verbose:
                console.print(f"[SysPlug] Suggestion: {new_config.summary(verbose=False)}")

    def _emit_event(
        self,
        event_type: EventType,
        step: int,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Thread-safely append an event to the event list."""
        event = MonitorEvent(
            event_type=event_type,
            step=step,
            message=message,
            details=details or {},
        )
        with self._events_lock:
            self._events.append(event)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_events(self) -> List[MonitorEvent]:
        """Return a snapshot of all events recorded so far.

        Returns:
            List of :class:`MonitorEvent` instances, ordered by timestamp.
        """
        with self._events_lock:
            return list(self._events)

    def clear_events(self) -> None:
        """Remove all recorded events."""
        with self._events_lock:
            self._events.clear()
