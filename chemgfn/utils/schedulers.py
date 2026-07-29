"""Step-indexed schedules for the training factors of :class:`ChemGFNModule`.

Factors such as the reward temperature, the replay-buffer mixing ratio and the
SubTB/RapTB window bounds ``k_min``/``k_max`` are supplied as callables that map
the current training step to a float. :class:`Scheduler` is the callable used by
every experiment config.
"""

from __future__ import annotations

from typing import Any, Literal


class Scheduler:
    """Linear schedule from ``start`` to ``end`` over ``horizon`` steps.

    The value at step ``t`` is ``start + (end - start) * min(1, t / horizon)``
    and is clamped to ``end`` for ``t >= horizon``.
    """

    def __init__(
        self,
        schedule_type: Literal["linear"],
        start: float,
        end: float,
        horizon: int,
    ) -> None:
        """Build a linear schedule.

        Args:
            schedule_type: Schedule family; only ``"linear"`` is supported.
            start: Value at step 0.
            end: Value reached at ``horizon`` and held afterwards.
            horizon: Number of steps over which the value is interpolated.

        Raises:
            ValueError: If ``horizon`` is not positive or ``schedule_type`` is unknown.
        """
        self.schedule_type = schedule_type
        self.start = float(start)
        self.end = float(end)
        self.horizon = float(horizon)

        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if self.schedule_type != "linear":
            raise ValueError(f"Invalid schedule_type: {schedule_type}. Must be 'linear'.")

    def __call__(self, step: Any) -> float:
        """Return the scheduled value at ``step``.

        Args:
            step: Current training step; any int, float or 0-d tensor.

        Returns:
            The interpolated value at ``step``.
        """
        step_value = max(0.0, self._to_float(step))
        progress = min(1.0, step_value / self.horizon)
        return self.start + (self.end - self.start) * progress

    @staticmethod
    def _to_float(value: Any) -> float:
        """Convert an int, float or 0-d tensor step index to ``float``."""
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def __repr__(self) -> str:
        return (
            f"Scheduler(type={self.schedule_type}, start={self.start}, "
            f"end={self.end}, horizon={self.horizon})"
        )
