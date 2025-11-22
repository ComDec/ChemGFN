"""
Unit tests for cosine restart scheduler.

Run with:
    pytest tests/test_cosine_restart_schedule.py -v
"""

import numpy as np
import pytest

from chemgfn.models.gfn import GFlowNetLightningModule


class TestCosineRestartSchedule:
    """Test suite for _build_cosine_schedule_with_restart method."""

    def test_basic_cosine_with_restart(self):
        """Test basic cosine schedule with constant restarts."""
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0, end=0.0, restart_period=100, t_mult=1.0, restart_decay=1.0
        )

        # Test start of first cycle
        assert np.isclose(schedule(0), 1.0, atol=1e-6)

        # Test middle of first cycle (should be around 0.5)
        mid_value = schedule(50)
        assert 0.4 < mid_value < 0.6

        # Test near end of first cycle
        assert schedule(99) < 0.1

        # Test restart (beginning of second cycle)
        assert np.isclose(schedule(100), 1.0, atol=1e-6)

        # Test middle of second cycle
        mid_value_2 = schedule(150)
        assert 0.4 < mid_value_2 < 0.6

    def test_increasing_period(self):
        """Test schedule with increasing period length (t_mult > 1)."""
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0, end=0.0, restart_period=100, t_mult=2.0, restart_decay=1.0
        )

        # First cycle: steps 0-99
        assert np.isclose(schedule(0), 1.0, atol=1e-6)
        assert schedule(99) < 0.1

        # Second cycle starts at step 100, length 200 (100 * 2.0)
        assert np.isclose(schedule(100), 1.0, atol=1e-6)

        # Middle of second cycle (step 200)
        mid_value = schedule(200)
        assert 0.4 < mid_value < 0.6

        # Second cycle ends at step 299
        assert schedule(299) < 0.1

        # Third cycle starts at step 300, length 400 (200 * 2.0)
        assert np.isclose(schedule(300), 1.0, atol=1e-6)

    def test_decaying_amplitude(self):
        """Test schedule with decaying amplitude (restart_decay < 1)."""
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0, end=0.0, restart_period=100, t_mult=1.0, restart_decay=0.5
        )

        # First cycle: amplitude = 1.0
        assert np.isclose(schedule(0), 1.0, atol=1e-6)

        # Second cycle: amplitude = 0.5 (1.0 * 0.5)
        assert np.isclose(schedule(100), 0.5, atol=1e-6)

        # Third cycle: amplitude = 0.25 (0.5 * 0.5)
        assert np.isclose(schedule(200), 0.25, atol=1e-6)

        # Fourth cycle: amplitude = 0.125 (0.25 * 0.5)
        assert np.isclose(schedule(300), 0.125, atol=1e-6)

    def test_combined_growth_and_decay(self):
        """Test schedule with both period growth and amplitude decay."""
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0, end=0.1, restart_period=100, t_mult=2.0, restart_decay=0.8
        )

        # First cycle starts at 1.0
        assert np.isclose(schedule(0), 1.0, atol=1e-6)

        # Second cycle starts at 0.1 + 0.9 * 0.8 = 0.82
        expected_start_2 = 0.1 + (1.0 - 0.1) * 0.8
        assert np.isclose(schedule(100), expected_start_2, atol=1e-6)

        # Third cycle starts at 0.1 + 0.9 * 0.8 * 0.8 = 0.676
        expected_start_3 = 0.1 + (1.0 - 0.1) * 0.8 * 0.8
        # Third cycle starts after first cycle (100) + second cycle (200) = 300
        assert np.isclose(schedule(300), expected_start_3, atol=1e-6)

    def test_non_zero_end_value(self):
        """Test schedule with non-zero end value."""
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0, end=0.2, restart_period=100, t_mult=1.0, restart_decay=1.0
        )

        # Start should be 1.0
        assert np.isclose(schedule(0), 1.0, atol=1e-6)

        # End should approach 0.2
        assert schedule(99) > 0.15
        assert schedule(99) < 0.25

        # Restart should go back to 1.0
        assert np.isclose(schedule(100), 1.0, atol=1e-6)

    def test_zero_or_negative_period(self):
        """Test that zero or negative period returns end value."""
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0, end=0.0, restart_period=0, t_mult=1.0, restart_decay=1.0
        )

        # Should always return end value
        assert schedule(0) == 0.0
        assert schedule(100) == 0.0
        assert schedule(1000) == 0.0

    def test_monotonic_decrease_within_cycle(self):
        """Test that values decrease monotonically within each cycle."""
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0, end=0.0, restart_period=100, t_mult=1.0, restart_decay=1.0
        )

        # Check monotonic decrease in first cycle
        prev_value = schedule(0)
        for step in range(1, 100):
            current_value = schedule(step)
            assert current_value <= prev_value, f"Non-monotonic at step {step}"
            prev_value = current_value

    def test_cosine_shape(self):
        """Test that the schedule follows a cosine curve."""
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0, end=0.0, restart_period=100, t_mult=1.0, restart_decay=1.0
        )

        # Sample points across the cycle
        steps = [0, 25, 50, 75, 99]
        values = [schedule(s) for s in steps]

        # Check approximate cosine shape
        # At t=0: cos(0) = 1 → value = 1.0
        # At t=0.25: cos(π/4) ≈ 0.707 → value ≈ 0.854
        # At t=0.5: cos(π/2) = 0 → value = 0.5
        # At t=0.75: cos(3π/4) ≈ -0.707 → value ≈ 0.146
        # At t=1: cos(π) = -1 → value ≈ 0

        expected = [1.0, 0.854, 0.5, 0.146, 0.0]
        for i, (actual, exp) in enumerate(zip(values, expected)):
            assert np.isclose(
                actual, exp, atol=0.05
            ), f"Step {steps[i]}: expected {exp}, got {actual}"

    def test_backward_compatibility(self):
        """Test that method handles different input types."""
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0, end=0.0, restart_period=100, t_mult=1.0, restart_decay=1.0
        )

        # Test with int
        val_int = schedule(50)

        # Test with float
        val_float = schedule(50.0)

        # Test with numpy scalar
        val_numpy = schedule(np.int64(50))

        # All should give the same result
        assert np.isclose(val_int, val_float, atol=1e-10)
        assert np.isclose(val_int, val_numpy, atol=1e-10)


class TestScheduleIntegration:
    """Integration tests for schedule usage in model."""

    def test_default_parameters_fallback(self):
        """Test that missing parameters fall back to sensible defaults."""
        # This test would require mocking the full GFlowNetLightningModule
        # For now, we just verify the method signature allows default values
        schedule = GFlowNetLightningModule._build_cosine_schedule_with_restart(
            start=1.0,
            end=0.0,
            restart_period=1000
            # t_mult and restart_decay should use defaults
        )

        # Should work without errors
        assert schedule(0) == 1.0
        assert schedule(1000) == 1.0  # Restart with default parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
