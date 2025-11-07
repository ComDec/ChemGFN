"""
Schedulers for dynamic parameter adjustment during training.

This module provides a unified scheduler class that supports multiple schedule types:
- linear: Linear interpolation
- cosine: Single cosine annealing (no restart)
- cosine_restart: Cosine annealing with warm restarts
"""

from typing import Any, Literal

import numpy as np


class Scheduler:
    """
    Unified scheduler for parameter adjustment during training.

    Supports three schedule types:
    - linear: Constant rate of change
    - cosine: Smooth cosine decay (single cycle)
    - cosine_restart: Cosine decay with periodic restarts
    """

    def __init__(
        self,
        schedule_type: Literal["linear", "cosine", "cosine_restart"],
        start: float,
        end: float,
        horizon: int,
        restart_period: int = None,
        t_mult: float = 1.0,
        restart_decay: float = 1.0,
    ):
        """
        Initialize scheduler.

        Args:
            schedule_type: Type of schedule ("linear", "cosine", or "cosine_restart")
            start: Initial value
            end: Final value
            horizon: Number of steps for the schedule
            restart_period: Period for restarts (only for cosine_restart)
            t_mult: Period multiplication factor after each restart (only for cosine_restart)
            restart_decay: Amplitude decay factor after each restart (only for cosine_restart)
        """
        self.schedule_type = schedule_type
        self.start = float(start)
        self.end = float(end)
        self.horizon = float(horizon)
        self.restart_period = (
            float(restart_period) if restart_period is not None else float(horizon)
        )
        self.t_mult = float(t_mult)
        self.restart_decay = float(restart_decay)

        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        if self.schedule_type not in ["linear", "cosine", "cosine_restart"]:
            raise ValueError(
                f"Invalid schedule_type: {schedule_type}. "
                f"Must be one of: 'linear', 'cosine', 'cosine_restart'"
            )

    def __call__(self, step: Any) -> float:
        """
        Get the scheduled value at a given step.

        Args:
            step: Current training step

        Returns:
            Scheduled value at the given step
        """
        step_value = self._normalize_scalar(step)

        if self.schedule_type == "linear":
            return self._linear(step_value)
        elif self.schedule_type == "cosine":
            return self._cosine(step_value)
        else:  # cosine_restart
            return self._cosine_restart(step_value)

    def _normalize_scalar(self, value: Any) -> float:
        """Normalize a scalar value to float."""
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _linear(self, step: float) -> float:
        """Linear schedule."""
        progress = min(1.0, step / self.horizon)
        return self.start + (self.end - self.start) * progress

    def _cosine(self, step: float) -> float:
        """Single cosine annealing (no restart)."""
        progress = min(1.0, step / self.horizon)
        # Use concave cosine decay: 1 - ((1 + cos(pi * progress)) / 2)
        # This gives a concave curve (slow decay at start, fast decay at end)
        cosine_decay = 1 - ((1 + np.cos(np.pi * progress)) / 2)
        return self.start + (self.end - self.start) * cosine_decay

    def _cosine_restart(self, step: float) -> float:
        """Cosine annealing with warm restarts."""
        if self.restart_period <= 0:
            return self.end

        # Find which restart cycle we're in and position within that cycle
        current_period = self.restart_period
        accumulated_steps = 0.0
        current_amplitude = self.start - self.end

        while step >= accumulated_steps + current_period:
            accumulated_steps += current_period
            current_period *= self.t_mult
            current_amplitude *= self.restart_decay

        # Position within current cycle [0, 1]
        cycle_progress = (step - accumulated_steps) / current_period

        # Cosine annealing with concave decay
        current_start = self.end + current_amplitude
        cosine_decay = 1 - ((1 + np.cos(np.pi * cycle_progress)) / 2)
        value = current_start + (self.end - current_start) * cosine_decay

        return float(value)

    def __repr__(self) -> str:
        """String representation of the scheduler."""
        if self.schedule_type == "cosine_restart":
            return (
                f"Scheduler(type={self.schedule_type}, start={self.start}, end={self.end}, "
                f"restart_period={self.restart_period}, t_mult={self.t_mult}, "
                f"restart_decay={self.restart_decay})"
            )
        else:
            return (
                f"Scheduler(type={self.schedule_type}, start={self.start}, end={self.end}, "
                f"horizon={self.horizon})"
            )
