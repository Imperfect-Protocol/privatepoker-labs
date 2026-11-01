"""Control surfaces: four channels, rate-limited slew toward commanded targets.

The brain commands surface targets; physics obeys with real actuator limits.
Deflections teleport nowhere: a servo moves at a bounded slew rate toward its
target and clamps at its mechanical limits. These are airframe properties,
not control logic.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

AILERON, ELEVATOR, RUDDER, THROTTLE = 0, 1, 2, 3
CHANNEL_NAMES = ("aileron", "elevator", "rudder", "throttle")

# Mechanical limits per channel: (min, max), radians for the three surfaces,
# [0, 1] for throttle.
SURFACE_LIMITS = ((-0.35, 0.35), (-0.35, 0.35), (-0.35, 0.35), (0.0, 1.0))
# Slew rate: rad/s for surfaces, full range per second for throttle.
SURFACE_SLEW_RATE = 6.0


class SurfacesState:
    """Physical surface positions. Owned by the airframe, updated per tick."""

    def __init__(self) -> None:
        self.positions = np.zeros(4, dtype=np.float64)

    def command(self, targets: Sequence[float], dt: float) -> None:
        """Slew toward targets, rate-limited and clamped. Deterministic."""
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        for channel in range(4):
            low, high = SURFACE_LIMITS[channel]
            target = min(high, max(low, float(targets[channel])))
            delta = target - self.positions[channel]
            max_step = SURFACE_SLEW_RATE * dt
            step = max(-max_step, min(max_step, delta))
            self.positions[channel] += step

    def reset(self) -> None:
        self.positions[:] = 0.0
